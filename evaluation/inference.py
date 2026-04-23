"""
Batch inference using SFT-trained LoRA model.

Usage:
    python evaluation/inference.py --task media --adapter_path outputs/media_lora_sft
    python evaluation/inference.py --task policy --adapter_path outputs/policy_lora_sft
"""
import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BASE_MODEL, PROCESSED_DATA_DIR, SFT_DATA_DIR, OUTPUTS_DIR,
    PRIMARY_FIELD_NAME, MEDIA_INSTRUCTION, POLICY_INSTRUCTION,
)


def load_model(base_model_path: str, adapter_path: str):
    """Load base model + LoRA adapter."""
    print(f"Loading base model: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype="auto", device_map="auto"
    )
    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def predict_single(model, tokenizer, title: str, abstract: str, task: str) -> dict:
    """Run prediction on a single paper."""
    instruction = MEDIA_INSTRUCTION if task == "media" else POLICY_INSTRUCTION
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\nTitle: {title}\nAbstract: {abstract}\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=512,
            temperature=0.1, do_sample=False,
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    # Parse prediction
    pred = 0
    response_lower = response.lower()
    if "prediction:" in response_lower:
        after = response_lower.split("prediction:")[-1].strip()
        pred = 1 if after.startswith("yes") else 0
    else:
        pred = 1 if response_lower.rstrip().endswith("yes") else 0

    # Extract explanation
    explanation = ""
    if "prediction:" in response.lower():
        explanation = response[:response.lower().rfind("prediction:")].strip()

    return {
        "prediction": pred,
        "raw_response": response,
        "explanation": explanation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["media", "policy"], required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--base_model", default=BASE_MODEL)
    parser.add_argument("--field_name", default=PRIMARY_FIELD_NAME)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    field_dir = args.field_name.lower().replace(" ", "_")
    task_dir = PROCESSED_DATA_DIR / field_dir / args.task

    # Load test data
    with open(task_dir / f"{args.split}.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if args.max_samples:
        test_data = test_data[:args.max_samples]

    print(f"Loaded {len(test_data)} {args.split} samples for {args.task} task")

    # Load model
    model, tokenizer = load_model(args.base_model, args.adapter_path)

    # Run inference
    results = []
    for sample in tqdm(test_data, desc="Inference"):
        pred_result = predict_single(
            model, tokenizer, sample["title"], sample["abstract"], args.task
        )
        results.append({
            "id": sample["id"],
            "title": sample["title"],
            "label": sample["label"],
            "prediction": pred_result["prediction"],
            "explanation": pred_result["explanation"],
            "raw_response": pred_result["raw_response"],
            "subfield": sample.get("subfield", ""),
            "year": sample.get("publication_date", "")[:4],
        })

    # Save predictions
    output_dir = OUTPUTS_DIR / "predictions" / field_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.task}_{args.split}_predictions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nPredictions saved to {out_path}")

    # Quick summary
    correct = sum(1 for r in results if r["prediction"] == r["label"])
    print(f"Accuracy: {correct}/{len(results)} = {correct/len(results):.4f}")


if __name__ == "__main__":
    main()
