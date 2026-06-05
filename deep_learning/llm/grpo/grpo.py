import re
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from dataclasses import dataclass
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from math_verify import parse, verify


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class GRPOConfig:
    model_name: str = "Qwen/Qwen3-0.6B"
    num_rollouts: int = 4
    max_new_tokens: int = 1024
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    clip_epsilon: float = 0.2
    kl_coeff: float = 0.01
    learning_rate: float = 1e-6
    num_ppo_epochs: int = 1
    batch_size: int = 1
    max_grad_norm: float = 1.0


# =============================================================================
# Reward Functions
# =============================================================================
def cleanup_reasoning(response: str) -> str:
    """Remove the <think>...</think> reasoning block from a response."""
    response = response.strip()
    think_pattern = re.compile(r'(<think>)?(.+)(</think>)', re.DOTALL)
    matched = re.search(think_pattern, response)
    if matched:
        answer = response[matched.end(0):]
    else:
        answer = response
    return answer.strip()


def compute_score(solution_str, ground_truth):
    """Binary reward: 1 if the parsed answer matches ground truth, 0 otherwise.

    We rely on math-verify, which offers more flexibility in terms
    of mathematical equivalence answers instead of strict answer match
    https://github.com/huggingface/Math-Verify
    """
    solution_str_cleaned = cleanup_reasoning(solution_str)
    gold = parse(ground_truth)
    answer = parse(solution_str_cleaned)
    score = int(verify(gold, answer))
    return score


# =============================================================================
# GRPO Core Functions
# =============================================================================
def compute_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-token log probabilities.

    Uses cross_entropy with ignore_index=-100 so that masked positions
    (pad tokens) contribute zero to the log-prob tensor.

    Returns: (batch_size, seq_len - 1) log probabilities
    """
    logits = model(input_ids, attention_mask, use_cache=False).logits

    # mark positions to ignore: where attention_mask == 0
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    # shift: logits[t] predicts token[t+1]
    logits_shifted = logits[:, :-1, :].contiguous()
    labels_shifted = labels[:, 1:].contiguous()

    # per-token negative cross entropy = log probability
    log_probs = -F.cross_entropy(
        logits_shifted.view(-1, logits_shifted.shape[-1]),
        labels_shifted.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(labels_shifted.shape)

    return log_probs


def compute_group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """
    Z-score normalize rewards within each group.

    Args:
        rewards: (batch_size, num_rollouts)

    Returns:
        advantages: (batch_size, num_rollouts)
    """
    mean_rewards = rewards.mean(dim=-1, keepdim=True)
    std_rewards = rewards.std(dim=-1, keepdim=True)
    advantages = (rewards - mean_rewards) / (std_rewards + 1e-8)
    return advantages


def compute_kl_divergence(
    log_probs_current: torch.Tensor,
    log_probs_ref: torch.Tensor,
) -> torch.Tensor:
    """Per-token KL divergence using the k3 estimator (reverse KL approximation)."""
    log_ratio = log_probs_ref - log_probs_current
    return torch.exp(log_ratio) - log_ratio - 1


def compute_grpo_loss(
    log_probs_current: torch.Tensor,
    log_probs_old: torch.Tensor,
    log_probs_ref: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_epsilon: float = 0.2,
    kl_coeff: float = 0.01,
) -> dict:
    """
    Compute the GRPO loss (negated objective, to be minimized).

    Returns a dict with loss and diagnostic metrics.
    """
    # per-token importance sampling ratio
    log_ratio = log_probs_current - log_probs_old
    ratio = torch.exp(log_ratio)

    # broadcast sequence-level advantages to all token positions
    advantages_expanded = advantages.unsqueeze(-1)

    # clipped surrogate objective
    surrogate_unclipped = ratio * advantages_expanded
    surrogate_clipped = (
        torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        * advantages_expanded
    )
    surrogate_loss = torch.min(surrogate_unclipped, surrogate_clipped)

    # KL divergence penalty, masked to response tokens only
    kl_penalty = compute_kl_divergence(log_probs_current, log_probs_ref)
    per_token_objective = (surrogate_loss - kl_coeff * kl_penalty) * response_mask

    # average over valid response tokens per completion, then over batch
    response_lengths = response_mask.sum(dim=-1).clamp(min=1)
    per_completion_objective = per_token_objective.sum(dim=-1) / response_lengths
    loss = -per_completion_objective.mean()

    # diagnostics
    with torch.no_grad():
        clip_fraction = (
            ((ratio > 1.0 + clip_epsilon) | (ratio < 1.0 - clip_epsilon)).float()
            * response_mask
        ).sum() / response_mask.sum()
        mean_kl = (kl_penalty * response_mask).sum() / response_mask.sum()

    return {
        "loss": loss,
        "clip_fraction": clip_fraction,
        "mean_kl": mean_kl,
    }


# =============================================================================
# Data
# =============================================================================
def extract_hash_answer(text: str) -> str:
    return text.split("####")[1].strip()


def collate_fn(batch):
    prompts = []
    answers = []
    for example in batch:
        prompt = [{"role": "user", "content": example["question"]}]
        prompts.append(prompt)
        answers.append(extract_hash_answer(example["answer"]))
    return {"prompts": prompts, "answers": answers}


# =============================================================================
# Lightning Module
# =============================================================================
class GRPOLightningModule(pl.LightningModule):

    ptl_module_prefix = "policy_model"
    tokenizer_prefix = "tokenizer"
    
    def __init__(self, config: GRPOConfig):
        super().__init__()
        self.config = config
        # disable Lightning's automatic optimization since GRPO has a
        # non-standard training loop (rollout -> reward -> update)
        self.automatic_optimization = False

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # policy model (trainable)
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        # reference model (frozen copy of initial policy)
        # not wrapped by DeepSpeed since it's inference-only
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=self.config.learning_rate,
        )

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        config = self.config

        # ==================================================================
        # Phase A: Rollout Generation (no gradients)
        # ==================================================================
        generated_ids, prompt_len = self._generate_rollouts(batch)

        # build attention mask: 1 for real tokens, 0 for pad
        gen_attention_mask = (generated_ids != self.tokenizer.pad_token_id).long()

        # decode response-only text for reward computation
        response_only_ids = generated_ids[:, prompt_len:]
        completion_texts = self.tokenizer.batch_decode(
            response_only_ids, skip_special_tokens=True
        )

        # ==================================================================
        # Phase B: Reward & Advantage Computation (no gradients)
        # ==================================================================
        advantages, rewards = self._compute_advantages(
            batch, completion_texts
        )

        # compute frozen log-probs for importance sampling and KL
        with torch.no_grad():
            log_probs_old = compute_logprobs(
                self.policy_model, generated_ids, gen_attention_mask
            )
            log_probs_ref = compute_logprobs(
                self.ref_model, generated_ids, gen_attention_mask
            )

        # response mask aligned with shifted log-prob shape (seq_len - 1)
        response_mask = gen_attention_mask.clone()
        response_mask[:, :prompt_len] = 0
        response_mask = response_mask[:, 1:]

        # ==================================================================
        # Phase C: Policy Update (with gradients)
        # ==================================================================
        self.policy_model.train()
        for _ in range(config.num_ppo_epochs):
            log_probs_current = compute_logprobs(
                self.policy_model, generated_ids, gen_attention_mask
            )
    
            results = compute_grpo_loss(
                log_probs_current=log_probs_current,
                log_probs_old=log_probs_old,
                log_probs_ref=log_probs_ref,
                advantages=advantages,
                response_mask=response_mask,
                clip_epsilon=config.clip_epsilon,
                kl_coeff=config.kl_coeff,
            )
    
            # manual backward + optimizer step
            loss = results["loss"]
            self.manual_backward(loss)
            self.clip_gradients(
                optimizer, gradient_clip_val=config.max_grad_norm, gradient_clip_algorithm="norm"
            )
            optimizer.step()
            optimizer.zero_grad()

        # logging
        self.log_dict(
            {
                "train/loss": loss,
                "train/mean_reward": rewards.mean(),
                "train/std_reward": rewards.std(),
                "train/clip_fraction": results["clip_fraction"],
                "train/mean_kl": results["mean_kl"],
            },
            prog_bar=True,
            sync_dist=True,
        )

    def _generate_rollouts(self, batch) -> tuple[torch.Tensor, int]:
        """
        Generate G completions per prompt using the current policy.

        Returns:
            generated_ids: (batch_size * num_rollouts, total_seq_len)
            prompt_len: int, uniform prompt length after left-padding
        """
        config = self.config

        # apply chat template and left-pad for causal generation
        prompts = self.tokenizer.apply_chat_template(
            batch["prompts"],
            tokenize=False,
            enable_thinking=False,
            add_generation_prompt=True,
        )
        prompt_encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
        ).to(self.device)

        input_ids = prompt_encoded["input_ids"]
        attention_mask = prompt_encoded["attention_mask"]
        prompt_len = input_ids.shape[1]

        # ZeRO-2 partitions optimizer states and gradients but keeps full
        # model parameters on each GPU, so generate() works directly.
        # (ZeRO-3 would require deepspeed.zero.GatheredParameters context)
        self.policy_model.eval()
        with torch.no_grad():
            generated_ids = self.policy_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.max_new_tokens,
                do_sample=True,
                num_return_sequences=config.num_rollouts,
                top_p=config.top_p,
                top_k=config.top_k,
                temperature=config.temperature,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        return generated_ids, prompt_len

    def _compute_advantages(
        self, batch, completion_texts: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rewards and group-relative advantages.

        Returns:
            advantages: (batch_size * num_rollouts,)
            rewards: (batch_size * num_rollouts,)
        """
        config = self.config

        # repeat answers to match rollout dimension
        answers_repeated = [
            answer
            for answer in batch["answers"]
            for _ in range(config.num_rollouts)
        ]

        rewards = torch.tensor(
            [
                compute_score(completion, ground_truth)
                for completion, ground_truth in zip(completion_texts, answers_repeated)
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # temporarily reshape to [batch_size, num_rollouts] for
        # computing batch group-relative advantages
        # the last batch in an epoch might be smaller than config's batch size
        rewards_grouped = rewards.view(len(answers), num_rollouts)
        advantages_grouped = compute_group_advantages(rewards_grouped)
        advantages = advantages_grouped.view(-1)
        return advantages, rewards


def get_deepspeed_config() -> dict:
    """
    DeepSpeed ZeRO Stage 2 configuration.
    """
    return {
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 5e8,
            "reduce_bucket_size": 5e8,
        },
        "bf16": {
            "enabled": True,
        },
        "gradient_clipping": 1.0,
    }


class HuggingfaceModelCheckpoint(pl.callbacks.ModelCheckpoint):
    """
    Model checkpoint callback that saves a pytorch lightning module
    in huggingface compatible format via save_pretrained. This callback only saves huggingface
    model checkpoint, i.e. no optimizer/scheduler state, etc for full resume training.

    Works for DDP, and Deepspeed Zero 2 (ZeRO-2 partitions optimizer states and gradients
    across GPUs, but keeps full model parameters on each rank). 
    """

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        if trainer.is_global_zero:
            # trainer.lightning_module is always the pure LightningModule that gets
            # passed to Trainer.fit by end user, whereas .model might
            # get wrapped by different training strategy
            huggingface_model = getattr(
                trainer.lightning_module, trainer.lightning_module.ptl_module_prefix
            )
            huggingface_tokenizer = getattr(
                trainer.lightning_module, trainer.lightning_module.tokenizer_prefix
            )

            hf_save_dst_dir = filepath.rstrip("/") + ".huggingface"
            huggingface_model.save_pretrained(hf_save_dst_dir)
            if huggingface_tokenizer:
                huggingface_tokenizer.save_pretrained(hf_save_dst_dir)

        trainer.strategy.barrier()
        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath


# =============================================================================
# Training Entry Point
# =============================================================================
def main():
    config = GRPOConfig()

    # data
    train_data = load_dataset("openai/gsm8k", "main")["train"]
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=0,
    )

    # model
    module = GRPOLightningModule(config)

    # trainer
    strategy = pl.strategies.DeepSpeedStrategy(
        config=get_deepspeed_config(),
    )
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=8,
        strategy=strategy,
        precision="bf16-mixed",
        gradient_clip_val=None,  # handled by DeepSpeed config
        log_every_n_steps=1,
        enable_checkpointing=True,
        callbacks=[
            HuggingfaceModelCheckpoint(
                dirpath="grpo_checkpoints",
                every_n_train_steps=25,
            ),
        ],
    )

    trainer.fit(module, train_loader)


if __name__ == "__main__":
    main()

