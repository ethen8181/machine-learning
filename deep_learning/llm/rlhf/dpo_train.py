"""Relies on liger kernel, flash attention 2, bfloat16,
requires A100 GPU and torch > 2.3"""
import os
import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig


def create_preference_triplets(example):
    """
    Create preference triplets:

    - `prompt`: prompt that is given to a model for text generation.
    - `chosen`: preferred generated response for the corresponding prompt.
    - `rejected`: response that is not preferred.
    """
    chosen = extract_assistant_messages(example["chosen"], index=-1)
    rejected = extract_assistant_messages(example["rejected"], index=-1)

    return {
        "prompt": example["prompt"],
        "chosen": chosen,
        "rejected": rejected
    }


def extract_assistant_messages(messages, index=-1):
    """Recursively extract the last assistant messages from the end of the conversation."""
    if messages[index]["role"] == "assistant":
        return messages[index]["content"]
    else:
        extract_assistant_messages(messages, index - 1)


if __name__ == "__main__":
    dataset = load_dataset(
        "argilla/ultrafeedback-binarized-preferences-cleaned",
        split="train",
        verification_mode="no_checks",
        cache_dir="/data"
    )
    dataset_dict = dataset.train_test_split(test_size=0.01, seed=54321)
    dataset_dict_preprocessed = dataset_dict.map(
        create_preference_triplets,
        num_proc=8
    )

    model_name_or_path = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir="/data")
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoLigerKernelForCausalLM.from_pretrained(
        model_name_or_path,
        cache_dir="/data",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.train()
    print(model)

    hf_save_tmp_dir = "dpo_model"
    training_args = DPOConfig(
        output_dir=hf_save_tmp_dir,
        bf16=True,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_steps=5000,
        logging_steps=50,
        learning_rate=0.0001,
        beta=0.1,
        max_length=1024,
        max_prompt_length=512,
        remove_unused_columns=False,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
    )
    print(training_args)
    peft_config = LoraConfig(
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "down_proj",
            "up_proj",
            "gate_proj"
        ],
        modules_to_save=[
            "embed_tokens",
            "lm_head"
        ]
    )
    print(peft_config)
    dpo_trainer = DPOTrainer(
        model,
        train_dataset=dataset_dict_preprocessed["train"],
        eval_dataset=dataset_dict_preprocessed["test"],
        tokenizer=tokenizer,
        args=training_args,
        peft_config=peft_config,
    )
    dpo_trainer.train()
