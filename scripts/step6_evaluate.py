"""
Step 6: Evaluate SFT model on test set.

Usage:
    python scripts/step6_evaluate.py --adapter_path /mnt/nvme1/lcx/academic_social_impact/media_medicine_lora --field medicine
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    classification_report, confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SFT_DATA_DIR, MEDIA_INSTRUCTION


def load_model(base_model_path, adapter_path):
    print(f"Loading base model: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, title, abstract):
    messages = [
        {"role": "system", "content": MEDIA_INSTRUCTION},
        {"role": "user", "content": f"Title: {title}\nAbstract: {abstract}"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=32,
            temperature=0.1, do_sample=False,
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    # Parse prediction
    response_lower = response.lower().strip()
    if "yes" in response_lower:
        return 1, response
    else:
        return 0, response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--base_model", default="/mnt/nvme1/hf-model/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--field", default="medicine")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    # Load test data
    test_path = SFT_DATA_DIR / args.field / "test.json"
    with open(test_path) as f:
        test_data = json.load(f)

    if args.max_samples:
        test_data = test_data[:args.max_samples]

    print(f"Test samples: {len(test_data)}")

    # Load model
    model, tokenizer = load_model(args.base_model, args.adapter_path)

    # Run predictions
    labels = []
    preds = []
    responses = []

    for sample in tqdm(test_data, desc="Evaluating"):
        # Extract title and abstract from input field
        input_text = sample["input"]
        title = ""
        abstract = ""
        if "Title:" in input_text and "Abstract:" in input_text:
            parts = input_text.split("Abstract:", 1)
            title = parts[0].replace("Title:", "").strip()
            abstract = parts[1].strip()

        true_label = 1 if "Yes" in sample["output"] else 0
        pred, response = predict(model, tokenizer, title, abstract)

        labels.append(true_label)
        preds.append(pred)
        responses.append(response)

    # Metrics
    print(f"\n{'='*60}")
    print("RESULTS")
    print("=" * 60)

    results = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "mcc": matthews_corrcoef(labels, preds),
    }

    for k, v in results.items():
        print(f"  {k:15s}: {v:.4f}")

    print(f"\n{classification_report(labels, preds, target_names=['No', 'Yes'])}")

    cm = confusion_matrix(labels, preds)
    print(f"Confusion Matrix:")
    print(f"  TN={cm[0][0]:5d}  FP={cm[0][1]:5d}")
    print(f"  FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")

    # Save results
    output_dir = Path(args.adapter_path) / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"{args.field}_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save predictions
    pred_data = []
    for i, sample in enumerate(test_data):
        pred_data.append({
            "input": sample["input"][:200],
            "true_label": labels[i],
            "predicted": preds[i],
            "response": responses[i],
        })
    with open(output_dir / f"{args.field}_predictions.json", "w", encoding="utf-8") as f:
        json.dump(pred_data, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
