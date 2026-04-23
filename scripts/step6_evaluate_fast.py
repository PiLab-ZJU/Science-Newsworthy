"""
Step 6: Fast evaluation using vLLM or multi-GPU batch inference.

Splits test data across 4 GPUs for parallel inference.

Usage:
    python scripts/step6_evaluate_fast.py --adapter_path ... --field combined --gpus 0,1,2,3
"""
import os
import sys
import json
import argparse
import torch
import multiprocessing as mp
from pathlib import Path
from functools import partial

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, classification_report, confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SFT_DATA_DIR, MEDIA_INSTRUCTION


def worker(gpu_id, samples, base_model_path, adapter_path, result_queue):
    """Worker process for one GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from tqdm import tqdm

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for s in tqdm(samples, desc=f"GPU {gpu_id}", position=gpu_id):
        input_text = s["input"]
        title, abstract = "", ""
        if "Title:" in input_text and "Abstract:" in input_text:
            parts = input_text.split("Abstract:", 1)
            title = parts[0].replace("Title:", "").strip()
            abstract = parts[1].strip()

        messages = [
            {"role": "system", "content": MEDIA_INSTRUCTION},
            {"role": "user", "content": f"Title: {title}\nAbstract: {abstract}"},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        resp = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        true_label = 1 if s["output"].strip().lower().startswith("yes") else 0
        pred = 1 if resp.strip().lower().startswith("yes") else 0

        results.append({"true": true_label, "pred": pred, "response": resp[:200]})

    result_queue.put(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--base_model", default="/mnt/nvme1/hf-model/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--field", default="combined")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    n_gpus = len(gpu_ids)

    # Load test data
    test_path = SFT_DATA_DIR / args.field / "test.json"
    with open(test_path) as f:
        test_data = json.load(f)
    if args.max_samples:
        test_data = test_data[:args.max_samples]
    print(f"Test samples: {len(test_data)}, GPUs: {gpu_ids}")

    # Split data across GPUs
    chunks = [test_data[i::n_gpus] for i in range(n_gpus)]

    # Launch workers
    mp.set_start_method("spawn", force=True)
    result_queue = mp.Queue()
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        p = mp.Process(target=worker, args=(gpu_id, chunks[i], args.base_model, args.adapter_path, result_queue))
        p.start()
        processes.append(p)

    # Collect results
    all_results = []
    for _ in processes:
        all_results.extend(result_queue.get())

    for p in processes:
        p.join()

    labels = [r["true"] for r in all_results]
    preds = [r["pred"] for r in all_results]

    # Metrics
    print(f"\n{'='*60}")
    print("RESULTS")
    print("=" * 60)
    results = {
        "accuracy": round(accuracy_score(labels, preds), 4),
        "precision": round(precision_score(labels, preds, zero_division=0), 4),
        "recall": round(recall_score(labels, preds, zero_division=0), 4),
        "f1": round(f1_score(labels, preds, zero_division=0), 4),
        "mcc": round(matthews_corrcoef(labels, preds), 4),
    }
    for k, v in results.items():
        print(f"  {k:15s}: {v}")

    print(f"\n{classification_report(labels, preds, target_names=['No', 'Yes'])}")
    cm = confusion_matrix(labels, preds)
    print(f"Confusion Matrix:")
    print(f"  TN={cm[0][0]:5d}  FP={cm[0][1]:5d}")
    print(f"  FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")

    # Save
    output_dir = Path(args.adapter_path) / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{args.field}_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
