"""
Baselines: LLM zero-shot and few-shot prediction.

- GPT-4o zero-shot
- GPT-4o few-shot (5-shot)
- LLaMA-3.1-8B zero-shot (pre-SFT baseline)

Usage:
    python baselines/llm_zeroshot.py --task media --model gpt4o --mode zero_shot
    python baselines/llm_zeroshot.py --task policy --model gpt4o --mode few_shot
    python baselines/llm_zeroshot.py --task media --model llama --mode zero_shot
"""
import os
import sys
import json
import argparse
import random
from pathlib import Path
from tqdm import tqdm

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROCESSED_DATA_DIR, OUTPUTS_DIR, PRIMARY_FIELD_NAME,
    MEDIA_INSTRUCTION, POLICY_INSTRUCTION, RANDOM_SEED,
)


def build_prompt(title: str, abstract: str, task: str) -> str:
    instruction = MEDIA_INSTRUCTION if task == "media" else POLICY_INSTRUCTION
    return f"{instruction}\n\nTitle: {title}\nAbstract: {abstract}"


def build_few_shot_prompt(title: str, abstract: str, task: str, examples: list) -> str:
    instruction = MEDIA_INSTRUCTION if task == "media" else POLICY_INSTRUCTION
    parts = [instruction, "\nHere are some examples:\n"]
    for ex in examples:
        label = "Yes" if ex["label"] == 1 else "No"
        parts.append(f"Title: {ex['title']}\nAbstract: {ex['abstract'][:300]}...\nPrediction: {label}\n")
    parts.append(f"\nNow predict for this paper:\nTitle: {title}\nAbstract: {abstract}")
    return "\n".join(parts)


def predict_gpt4o(prompt: str) -> str:
    import openai
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=512,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def predict_llama(prompt: str, model=None, tokenizer=None) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=False)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response


def parse_prediction(response: str) -> int:
    """Extract Yes/No prediction from model response."""
    response_lower = response.lower()
    # Check after "prediction:" first
    if "prediction:" in response_lower:
        after = response_lower.split("prediction:")[-1].strip()
        if after.startswith("yes"):
            return 1
        elif after.startswith("no"):
            return 0
    # Fallback: check last occurrence
    last_yes = response_lower.rfind("yes")
    last_no = response_lower.rfind("no")
    if last_yes > last_no:
        return 1
    elif last_no > last_yes:
        return 0
    return 0  # Default to negative


def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["media", "policy"], required=True)
    parser.add_argument("--model", choices=["gpt4o", "llama"], required=True)
    parser.add_argument("--mode", choices=["zero_shot", "few_shot"], default="zero_shot")
    parser.add_argument("--field_name", default=PRIMARY_FIELD_NAME)
    parser.add_argument("--max_test", type=int, default=None, help="Limit test samples (for cost control)")
    parser.add_argument("--n_shots", type=int, default=5)
    args = parser.parse_args()

    field_dir = args.field_name.lower().replace(" ", "_")
    task_dir = PROCESSED_DATA_DIR / field_dir / args.task

    # Load test data
    with open(task_dir / "test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if args.max_test:
        test_data = test_data[:args.max_test]

    # Prepare few-shot examples from train
    few_shot_examples = []
    if args.mode == "few_shot":
        with open(task_dir / "train.json", "r", encoding="utf-8") as f:
            train_data = json.load(f)
        rng = random.Random(RANDOM_SEED)
        pos = [s for s in train_data if s["label"] == 1]
        neg = [s for s in train_data if s["label"] == 0]
        few_shot_examples = rng.sample(pos, args.n_shots // 2 + 1)[:args.n_shots // 2 + args.n_shots % 2]
        few_shot_examples += rng.sample(neg, args.n_shots // 2)
        rng.shuffle(few_shot_examples)

    # Load LLaMA model if needed
    llama_model, llama_tokenizer = None, None
    if args.model == "llama":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from config import BASE_MODEL
        print(f"Loading {BASE_MODEL}...")
        llama_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        llama_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype="auto", device_map="auto"
        )
        print("Model loaded.")

    print(f"\nRunning {args.model} {args.mode} for {args.task} task")
    print(f"Test samples: {len(test_data)}")

    predictions = []
    labels = []

    for sample in tqdm(test_data, desc="Predicting"):
        if args.mode == "few_shot":
            prompt = build_few_shot_prompt(
                sample["title"], sample["abstract"], args.task, few_shot_examples
            )
        else:
            prompt = build_prompt(sample["title"], sample["abstract"], args.task)

        try:
            if args.model == "gpt4o":
                response = predict_gpt4o(prompt)
            else:
                response = predict_llama(prompt, llama_model, llama_tokenizer)

            pred = parse_prediction(response)
            predictions.append(pred)
            labels.append(sample["label"])
        except Exception as e:
            print(f"\n  Error: {e}")
            continue

    # Evaluate
    results = evaluate(np.array(labels), np.array(predictions))
    print(f"\nResults:")
    print(json.dumps(results, indent=2))

    # Save
    output_dir = OUTPUTS_DIR / "baselines" / field_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.task}_{args.model}_{args.mode}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
