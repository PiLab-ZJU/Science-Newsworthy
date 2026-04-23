"""Evaluate CoT model with JSON output format."""
import os, sys, json, torch
import multiprocessing as mp
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, classification_report, confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SFT_DATA_DIR

INSTRUCTION = (
    "Based on the following academic paper's title and abstract, "
    "predict whether this paper will receive mainstream media news coverage. "
    "If yes, briefly explain why and predict the news angle. "
    "Answer with your analysis and prediction."
)


def parse_prediction(resp):
    """Extract prediction from JSON or raw response."""
    try:
        obj = json.loads(resp)
        pred = obj.get("prediction", "").strip().lower()
        return 1 if pred.startswith("yes") else 0
    except:
        r = resp.lower()
        if '"prediction"' in r:
            after = r.split('"prediction"')[-1]
            return 1 if "yes" in after[:20] else 0
        return 1 if "yes" in r[:50] else 0


def worker(gpu_id, samples, base_model_path, adapter_path, result_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from tqdm import tqdm

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    results = []
    for s in tqdm(samples, desc=f"GPU {gpu_id}", position=gpu_id):
        inp = s["input"]
        title, abstract = "", ""
        if "Title:" in inp and "Abstract:" in inp:
            parts = inp.split("Abstract:", 1)
            title = parts[0].replace("Title:", "").strip()
            abstract = parts[1].strip()

        msgs = [{"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": f"Title: {title}\nAbstract: {abstract}"}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        resp = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        true_out = s["output"]
        true_label = parse_prediction(true_out)
        pred_label = parse_prediction(resp)

        results.append({"true": true_label, "pred": pred_label, "response": resp[:300]})
    result_queue.put(results)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--base_model", default="/mnt/nvme1/hf-model/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--field", default="combined_cot_balanced")
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    test_path = SFT_DATA_DIR / args.field / "test.json"
    with open(test_path) as f:
        test_data = json.load(f)
    print(f"Test: {len(test_data)}, GPUs: {gpu_ids}")

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

    labels = [r["true"] for r in all_results]
    preds = [r["pred"] for r in all_results]

    print(f"\n{'='*60}\nRESULTS\n{'='*60}")
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
    print(f"Confusion Matrix:\n  TN={cm[0][0]:5d}  FP={cm[0][1]:5d}\n  FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")

    # Save
    out_dir = Path(args.adapter_path) / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save sample outputs
    with open(out_dir / "sample_outputs.json", "w", encoding="utf-8") as f:
        json.dump(all_results[:50], f, ensure_ascii=False, indent=2)

    print(f"\nSample outputs:")
    for r in all_results[:5]:
        tag = "OK" if r["true"] == r["pred"] else "WRONG"
        print(f"  [{tag}] True={r['true']} Pred={r['pred']} -> {r['response'][:150]}")


if __name__ == "__main__":
    main()
