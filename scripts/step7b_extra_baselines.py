"""
Extra baselines: LLaMA zero-shot, GPT zero-shot, Logistic Regression.

Usage:
    python scripts/step7b_extra_baselines.py --only lr          # Logistic Regression (no GPU)
    python scripts/step7b_extra_baselines.py --only gpt         # GPT via API
    python scripts/step7b_extra_baselines.py --only llama_zero  # LLaMA zero-shot (GPU)
"""
import os, sys, json, argparse, time, random
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, RANDOM_SEED

INSTRUCTION = (
    "Based on the following academic paper's title and abstract, "
    "predict whether this paper will receive mainstream media news coverage. "
    "Answer Yes or No."
)

def load_data():
    splits = {}
    for s in ["train", "test"]:
        with open(PROCESSED_DATA_DIR / "combined" / f"{s}.json") as f:
            splits[s] = json.load(f)
    return splits

def evaluate(labels, preds, name):
    r = {
        "accuracy": round(accuracy_score(labels, preds), 4),
        "precision": round(precision_score(labels, preds, zero_division=0), 4),
        "recall": round(recall_score(labels, preds, zero_division=0), 4),
        "f1": round(f1_score(labels, preds, zero_division=0), 4),
        "mcc": round(matthews_corrcoef(labels, preds), 4),
    }
    print(f"\n  {name}:")
    for k, v in r.items():
        print(f"    {k:15s}: {v}")
    return r

# ============================================================
# Logistic Regression
# ============================================================
def run_lr(splits):
    print("\n" + "="*50)
    print("Baseline: Logistic Regression + TF-IDF")
    print("="*50)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    train_texts = [f"{s.get('title','')} [SEP] {s.get('abstract','')}" for s in splits["train"]]
    test_texts = [f"{s.get('title','')} [SEP] {s.get('abstract','')}" for s in splits["test"]]
    train_labels = np.array([s["label"] for s in splits["train"]])
    test_labels = np.array([s["label"] for s in splits["test"]])

    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)

    model = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_SEED)
    model.fit(X_train, train_labels)
    preds = model.predict(X_test)
    return evaluate(test_labels, preds, "Logistic Regression + TF-IDF")

# ============================================================
# GPT zero-shot via API
# ============================================================
def run_gpt(splits, model_name="gpt-4o-mini", max_samples=500):
    print("\n" + "="*50)
    print(f"Baseline: {model_name} zero-shot (n={max_samples})")
    print("="*50)
    import requests

    API_URL = "https://api.xty.app/v1/chat/completions"
    API_KEY = "sk-YizsA7d2z2nA9vA2qt31T2xeqEDYcTGm8HumXSd5o3hEV3km"

    test = splits["test"]
    rng = random.Random(RANDOM_SEED)
    subset = rng.sample(test, min(max_samples, len(test)))

    labels, preds = [], []
    from tqdm import tqdm
    for s in tqdm(subset, desc=model_name):
        title = s.get("title", "")
        abstract = s.get("abstract", "")

        try:
            resp = requests.post(API_URL, headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            }, json={
                "model": model_name,
                "max_tokens": 10,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": INSTRUCTION},
                    {"role": "user", "content": f"Title: {title}\nAbstract: {abstract[:800]}"},
                ],
            }, timeout=30)
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"].strip().lower()
            pred = 1 if answer.startswith("yes") else 0
        except Exception as e:
            pred = 0

        labels.append(s["label"])
        preds.append(pred)
        time.sleep(0.05)

    return evaluate(np.array(labels), np.array(preds), f"{model_name} zero-shot")

# ============================================================
# LLaMA zero-shot (no SFT)
# ============================================================
def run_llama_zero(splits, max_samples=500):
    print("\n" + "="*50)
    print(f"Baseline: LLaMA-3.1-8B zero-shot (n={max_samples})")
    print("="*50)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = "/mnt/nvme1/hf-model/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    test = splits["test"]
    rng = random.Random(RANDOM_SEED)
    subset = rng.sample(test, min(max_samples, len(test)))

    labels, preds = [], []
    from tqdm import tqdm
    for s in tqdm(subset, desc="LLaMA zero-shot"):
        title = s.get("title", "")
        abstract = s.get("abstract", "")

        msgs = [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": f"Title: {title}\nAbstract: {abstract}"},
        ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        resp = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
        pred = 1 if resp.startswith("yes") else 0

        labels.append(s["label"])
        preds.append(pred)

    return evaluate(np.array(labels), np.array(preds), "LLaMA-3.1-8B zero-shot")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="all", choices=["all", "lr", "gpt", "llama_zero"])
    parser.add_argument("--gpt_model", default="gpt-4o-mini")
    parser.add_argument("--gpt_samples", type=int, default=500)
    parser.add_argument("--llama_samples", type=int, default=500)
    args = parser.parse_args()

    splits = load_data()
    results = {}

    if args.only in ["all", "lr"]:
        results["logistic_regression"] = run_lr(splits)

    if args.only in ["all", "gpt"]:
        results[f"{args.gpt_model}_zeroshot"] = run_gpt(splits, args.gpt_model, args.gpt_samples)

    if args.only in ["all", "llama_zero"]:
        results["llama_zeroshot"] = run_llama_zero(splits, args.llama_samples)

    # Save
    out = Path("/mnt/nvme1/lcx/academic_social_impact/analysis/extra_baselines.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        with open(out) as f:
            existing = json.load(f)
        existing.update(results)
        results = existing
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"{'Model':<35s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'MCC':>7s}")
    print(f"{'-'*70}")
    for name, r in results.items():
        print(f"  {name:<33s} {r['accuracy']:>7.4f} {r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f} {r['mcc']:>7.4f}")
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
