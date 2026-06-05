import re
import vllm
from typing import Optional
from datasets import load_dataset
from math_verify import parse, verify
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def extract_hash_answer(text: str) -> str:
    return text.split("####")[1].strip()


def cleanup_reasoning(response: str) -> str:
    """Remove the <think> process."""
    response = response.strip()
    think_pattern = re.compile(r'(<think>)?(.+)(</think>)', re.DOTALL)
    matched = re.search(think_pattern, response)
    if matched:
        # remove think tags
        answer = response[matched.end(0):]
    else:
        answer = response

    answer = answer.strip()
    return answer


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


def evaluate_gsm8k(
    model_path: str,
    tokenizer_path: Optional[str] = None,
    num_samples: Optional[int] = None,
    system_prompt: Optional[str] = None,
    llm_kwargs: Optional[dict] = None,
    sampling_kwargs: Optional[dict] = None,
) -> dict:
    """
    Evaluate a GRPO-trained model on the GSM8K test set.

    Args:
        model_path: Path to the model checkpoint (HF format).
        tokenizer_path: Path to the model's tokenizer (defaults to model_path).
        num_samples: Number of test samples to evaluate (None = full test set).
        system_prompt: Custom system prompt (None = no system prompt).
        llm_kwargs: Additional keyword arguments passed to vllm.LLM().
            e.g. {"tensor_parallel_size": 2, "gpu_memory_utilization": 0.9}
        sampling_kwargs: Additional keyword arguments passed to vllm.SamplingParams().
            e.g. {"temperature": 0.6, "top_p": 0.95, "max_tokens": 1024}

    Returns:
        Dictionary with accuracy, total, correct count, and per-sample results.
    """
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} samples from GSM8K test set...")

    # Prepare prompts
    tokenizer_path = tokenizer_path if tokenizer_path else model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    prompts = []
    for sample in dataset:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": sample["question"]})

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        prompts.append(prompt)

    # Extract ground truth answers
    ground_truths = [extract_hash_answer(sample["answer"]) for sample in dataset]

    # Initialize vLLM engine        
    llm = LLM(
        tokenizer=tokenizer_path,
        model=model_path,
        **llm_kwargs
    )

    sampling_params = SamplingParams(**sampling_kwargs)

    print("Running inference...")
    outputs = llm.generate(prompts, sampling_params)

    # Evaluate results
    results = []
    correct_count = 0

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        solution_str = ground_truths[i]
        correct = compute_score(solution_str, generated_text)

        if correct:
            correct_count += 1

        results.append({
            "idx": i,
            "prompt": prompts[i],
            "ground_truth": solution_str,
            "generated_text": generated_text,
            "correct": correct,
        })

    accuracy = correct_count / len(dataset) * 100

    summary = {
        "accuracy": accuracy,
        "correct": correct_count,
        "total": len(dataset),
        "model_path": model_path,
        "sampling_kwargs": sampling_kwargs,
        "results": results,
    }

    print(f"\n{'='*50}")
    print(f"GSM8K Evaluation Results")
    print(f"{'='*50}")
    print(f"Model: {model_path}")
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{len(dataset)})")
    print(f"{'='*50}")
    return summary



tokenizer_path = "Qwen/Qwen3-0.6B"
model_path = "Qwen/Qwen3-0.6B"
summary = evaluate_gsm8k(
    model_path,
    tokenizer_path,
    llm_kwargs={
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.4,
        "seed": 42,
    },
    sampling_kwargs={
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 1024,
    },
)
