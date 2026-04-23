"""
Step 7: Run all baseline experiments.

Baselines:
1. Random
2. XGBoost + TF-IDF
3. XGBoost (metadata only)
4. LambdaMART + n-gram (LightGBM)
5. SciBERT + classification head
6. LLaMA-3.1-8B zero-shot

Usage:
    python scripts/step7_baselines.py --field medicine
    python scripts/step7_baselines.py --field medicine --only scibert
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, classification_report,
)
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, MEDIA_INSTRUCTION, RANDOM_SEED


def load_data(field):
    field_dir = PROCESSED_DATA_DIR / field
    splits = {}
    for split in ["train", "val", "test"]:
        with open(field_dir / f"{split}.json") as f:
            splits[split] = json.load(f)
    return splits


def get_texts_labels(data):
    texts = [f"{s.get('title', '')} [SEP] {s.get('abstract', '')}" for s in data]
    labels = np.array([s["label"] for s in data])
    return texts, labels


def evaluate(y_true, y_pred, name=""):
    results = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "mcc": round(matthews_corrcoef(y_true, y_pred), 4),
    }
    print(f"\n  {name}:")
    for k, v in results.items():
        print(f"    {k:15s}: {v}")
    return results


# ============================================================
# Baseline 1: Random
# ============================================================
def run_random(train_labels, test_labels):
    print("\n" + "=" * 50)
    print("Baseline: Random")
    print("=" * 50)
    rng = np.random.RandomState(RANDOM_SEED)
    pos_ratio = train_labels.mean()
    y_pred = rng.choice([0, 1], size=len(test_labels), p=[1 - pos_ratio, pos_ratio])
    return evaluate(test_labels, y_pred, "Random")


# ============================================================
# Baseline 2: XGBoost + TF-IDF
# ============================================================
def run_xgboost_tfidf(train_texts, train_labels, test_texts, test_labels):
    print("\n" + "=" * 50)
    print("Baseline: XGBoost + TF-IDF")
    print("=" * 50)
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)

    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        eval_metric="logloss", random_state=RANDOM_SEED,
        tree_method="hist",
    )
    model.fit(X_train, train_labels)
    y_pred = model.predict(X_test)
    return evaluate(test_labels, y_pred, "XGBoost + TF-IDF")


# ============================================================
# Baseline 3: XGBoost (metadata only)
# ============================================================
def run_xgboost_metadata(train_data, test_data, train_labels, test_labels):
    print("\n" + "=" * 50)
    print("Baseline: XGBoost (metadata only)")
    print("=" * 50)
    import pandas as pd

    def extract_meta(data):
        rows = []
        for s in data:
            rows.append({
                "cited_by_count": s.get("cited_by_count", 0),
                "title_len": len(s.get("title", "")),
                "abstract_len": len(s.get("abstract", "")),
                "year": int(s.get("publication_date", "2020")[:4]),
            })
        return pd.DataFrame(rows).fillna(0)

    X_train = extract_meta(train_data)
    X_test = extract_meta(test_data)

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        eval_metric="logloss", random_state=RANDOM_SEED,
    )
    model.fit(X_train, train_labels)
    y_pred = model.predict(X_test)
    return evaluate(test_labels, y_pred, "XGBoost (metadata)")


# ============================================================
# Baseline 4: LambdaMART + n-gram (LightGBM)
# ============================================================
def run_lambdamart(train_texts, train_labels, test_texts, test_labels):
    print("\n" + "=" * 50)
    print("Baseline: LightGBM + n-gram")
    print("=" * 50)
    try:
        import lightgbm as lgb
    except ImportError:
        print("  SKIPPED (lightgbm not installed)")
        return None

    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), stop_words="english")
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)

    model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        num_leaves=31, random_state=RANDOM_SEED, verbose=-1,
    )
    model.fit(X_train, train_labels)
    y_pred = model.predict(X_test)
    return evaluate(test_labels, y_pred, "LightGBM + n-gram")


# ============================================================
# Baseline 5: SciBERT
# ============================================================
def run_scibert(train_data, val_data, test_data, field, output_dir):
    print("\n" + "=" * 50)
    print("Baseline: SciBERT")
    print("=" * 50)
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer,
    )

    class PaperDataset(Dataset):
        def __init__(self, samples, tokenizer, max_length=512):
            self.samples = samples
            self.tokenizer = tokenizer
            self.max_length = max_length
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            s = self.samples[idx]
            text = f"{s.get('title', '')} [SEP] {s.get('abstract', '')}"
            enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                 padding="max_length", return_tensors="pt")
            return {
                "input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels": torch.tensor(s["label"], dtype=torch.long),
            }

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, zero_division=0),
            "mcc": matthews_corrcoef(labels, preds),
        }

    model_name = "allenai/scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_ds = PaperDataset(train_data, tokenizer)
    val_ds = PaperDataset(val_data, tokenizer)
    test_ds = PaperDataset(test_data, tokenizer)

    scibert_output = output_dir / f"scibert_{field}"
    training_args = TrainingArguments(
        output_dir=str(scibert_output),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=RANDOM_SEED,
        logging_steps=100,
        save_total_limit=1,
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # Test
    preds_output = trainer.predict(test_ds)
    preds = np.argmax(preds_output.predictions, axis=-1)
    labels = np.array([s["label"] for s in test_data])
    return evaluate(labels, preds, "SciBERT")


# ============================================================
# Baseline 6: LLaMA-3.1-8B zero-shot
# ============================================================
def run_llama_zeroshot(test_data, max_samples=500):
    print("\n" + "=" * 50)
    print(f"Baseline: LLaMA-3.1-8B zero-shot (n={max_samples})")
    print("=" * 50)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model_path = "/mnt/nvme1/hf-model/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Sample subset for speed
    rng = np.random.RandomState(RANDOM_SEED)
    indices = rng.choice(len(test_data), size=min(max_samples, len(test_data)), replace=False)
    subset = [test_data[i] for i in indices]

    labels = []
    preds = []
    from tqdm import tqdm
    for s in tqdm(subset, desc="LLaMA zero-shot"):
        title = s.get("title", "")
        abstract = s.get("abstract", "")

        messages = [
            {"role": "system", "content": MEDIA_INSTRUCTION},
            {"role": "user", "content": f"Title: {title}\nAbstract: {abstract}"},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).lower()

        labels.append(s["label"])
        preds.append(1 if "yes" in response else 0)

    return evaluate(np.array(labels), np.array(preds), "LLaMA-3.1-8B zero-shot")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", default="medicine")
    parser.add_argument("--only", default=None,
                        help="Run only one baseline: random/xgboost/metadata/lgbm/scibert/llama")
    parser.add_argument("--llama_samples", type=int, default=500)
    args = parser.parse_args()

    output_dir = Path("/mnt/nvme1/lcx/academic_social_impact/baselines")
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = load_data(args.field)
    train_texts, train_labels = get_texts_labels(splits["train"])
    test_texts, test_labels = get_texts_labels(splits["test"])

    print(f"Field: {args.field}")
    print(f"Train: {len(train_labels)} (pos={train_labels.sum()})")
    print(f"Test:  {len(test_labels)} (pos={test_labels.sum()})")

    all_results = {}
    run_all = args.only is None

    if run_all or args.only == "random":
        all_results["random"] = run_random(train_labels, test_labels)

    if run_all or args.only == "xgboost":
        all_results["xgboost_tfidf"] = run_xgboost_tfidf(
            train_texts, train_labels, test_texts, test_labels)

    if run_all or args.only == "metadata":
        all_results["xgboost_metadata"] = run_xgboost_metadata(
            splits["train"], splits["test"], train_labels, test_labels)

    if run_all or args.only == "lgbm":
        r = run_lambdamart(train_texts, train_labels, test_texts, test_labels)
        if r:
            all_results["lightgbm_ngram"] = r

    if run_all or args.only == "scibert":
        all_results["scibert"] = run_scibert(
            splits["train"], splits["val"], splits["test"], args.field, output_dir)

    if run_all or args.only == "llama":
        all_results["llama_zeroshot"] = run_llama_zeroshot(
            splits["test"], max_samples=args.llama_samples)

    # Save all results
    results_path = output_dir / f"{args.field}_baselines.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY: {args.field}")
    print(f"{'='*70}")
    print(f"  {'Model':<30s} {'Acc':>8s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'MCC':>8s}")
    print(f"  {'-'*62}")
    for name, r in all_results.items():
        print(f"  {name:<30s} {r['accuracy']:>8.4f} {r['precision']:>8.4f} "
              f"{r['recall']:>8.4f} {r['f1']:>8.4f} {r['mcc']:>8.4f}")
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
