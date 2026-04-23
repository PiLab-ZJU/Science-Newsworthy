"""
Analysis 1: Batch generate prediction explanations for test set.
Uses 4-GPU parallel inference. Two-step: predict Yes/No, then ask why.

Usage:
    python scripts/analysis_1_generate_explanations.py --gpus 0,1,2,3
"""
import os, sys, json, torch
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SFT_DATA_DIR, PROCESSED_DATA_DIR

INSTRUCTION = (
    "Based on the following academic paper's title and abstract, "
    "predict whether this paper will receive mainstream media news coverage. "
    "Answer Yes or No."
)


def worker(gpu_id, samples, base_model_path, adapter_path, result_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    results = []
    for s in tqdm(samples, desc=f"GPU {gpu_id}", position=gpu_id):
        title = s.get("title", "")
        abstract = s.get("abstract", "")
        label = s["label"]
        doi = s.get("doi", "")
        field = s.get("field", "")

        # Step 1: Predict
        msgs = [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": f"Title: {title}\nAbstract: {abstract}"},
        ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        pred_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        pred = 1 if pred_text.lower().startswith("yes") else 0

        # Step 2: Ask why (only for predicted Yes)
        explanation = ""
        if pred == 1:
            ask = "Why do you think this paper will receive news coverage? Explain in 2-3 sentences."
        else:
            ask = "Why do you think this paper will NOT receive news coverage? Explain in 2-3 sentences."

        msgs2 = msgs + [
            {"role": "assistant", "content": pred_text},
            {"role": "user", "content": ask},
        ]
        text2 = tokenizer.apply_chat_template(msgs2, tokenize=False, add_generation_prompt=True)
        inputs2 = tokenizer(text2, return_tensors="pt", truncation=True, max_length=1200).to(model.device)
        with torch.no_grad():
            out2 = model.generate(**inputs2, max_new_tokens=150, do_sample=False)
        explanation = tokenizer.decode(out2[0][inputs2.input_ids.shape[1]:], skip_special_tokens=True).strip()

        results.append({
            "doi": doi,
            "title": title,
            "field": field,
            "true_label": label,
            "predicted": pred,
            "prediction_text": pred_text,
            "explanation": explanation,
        })

    result_queue.put(results)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="/mnt/nvme1/hf-model/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter_path", default="/mnt/nvme1/lcx/academic_social_impact/media_combined_lora_r32")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    # Load test data (processed, with all metadata)
    test_path = PROCESSED_DATA_DIR / "combined" / "test.json"
    with open(test_path) as f:
        test_data = json.load(f)
    if args.max_samples:
        test_data = test_data[:args.max_samples]
    print(f"Test samples: {len(test_data)}")

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    chunks = [test_data[i::len(gpu_ids)] for i in range(len(gpu_ids))]

    mp.set_start_method("spawn", force=True)
    result_queue = mp.Queue()
    procs = []
    for i, gid in enumerate(gpu_ids):
        p = mp.Process(target=worker, args=(gid, chunks[i], args.base_model, args.adapter_path, result_queue))
        p.start()
        procs.append(p)

    all_results = []
    for _ in procs:
        all_results.extend(result_queue.get())
    for p in procs:
        p.join()

    # Save
    output_dir = Path("/mnt/nvme1/lcx/academic_social_impact/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "test_predictions_with_explanations.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Quick stats
    correct = sum(1 for r in all_results if r["true_label"] == r["predicted"])
    total = len(all_results)
    tp = sum(1 for r in all_results if r["true_label"] == 1 and r["predicted"] == 1)
    fp = sum(1 for r in all_results if r["true_label"] == 0 and r["predicted"] == 1)
    fn = sum(1 for r in all_results if r["true_label"] == 1 and r["predicted"] == 0)
    tn = sum(1 for r in all_results if r["true_label"] == 0 and r["predicted"] == 0)

    print(f"\nAccuracy: {correct}/{total} = {correct/total:.4f}")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"\nSample explanations:")
    for r in all_results[:3]:
        tag = "CORRECT" if r["true_label"] == r["predicted"] else "WRONG"
        print(f"  [{tag}] True={r['true_label']} Pred={r['predicted']}")
        print(f"    Title: {r['title'][:80]}...")
        print(f"    Explanation: {r['explanation'][:150]}...")
        print()

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
