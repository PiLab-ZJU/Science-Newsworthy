"""
Baseline experiments: Traditional ML models.

- XGBoost + TF-IDF
- XGBoost (metadata only)
- lambdaMART + n-gram (replicating Piotrkowicz et al., ICWSM 2018)

Usage:
    python baselines/traditional_ml.py --task media
    python baselines/traditional_ml.py --task policy
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    classification_report,
)
from xgboost import XGBClassifier

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, OUTPUTS_DIR, PRIMARY_FIELD_NAME


def load_split(task_dir: Path, split: str):
    path = task_dir / f"{split}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [f"{s['title']} [SEP] {s['abstract']}" for s in data]
    labels = [s["label"] for s in data]
    metadata = pd.DataFrame([{
        "subfield": s.get("subfield", ""),
        "year": s.get("publication_date", "")[:4],
        "cited_by_count": s.get("cited_by_count", 0),
    } for s in data])
    return texts, np.array(labels), metadata


def evaluate(y_true, y_pred, y_prob=None):
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_prob is not None:
        results["auc_roc"] = roc_auc_score(y_true, y_prob)
    return results


def run_xgboost_tfidf(train_texts, train_labels, test_texts, test_labels):
    """XGBoost + TF-IDF baseline."""
    print("\n--- XGBoost + TF-IDF ---")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, train_labels)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results = evaluate(test_labels, y_pred, y_prob)
    print(json.dumps(results, indent=2))
    return results


def run_xgboost_metadata(train_meta, train_labels, test_meta, test_labels):
    """XGBoost with metadata only (replicating Bailey 2017 style)."""
    print("\n--- XGBoost (metadata only) ---")

    # Encode categorical features
    for col in ["subfield", "year"]:
        if col in train_meta.columns:
            combined = pd.concat([train_meta[col], test_meta[col]])
            codes = combined.astype("category").cat.codes
            train_meta[col] = codes[:len(train_meta)].values
            test_meta[col] = codes[len(train_meta):].values

    train_meta = train_meta.fillna(0)
    test_meta = test_meta.fillna(0)

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42,
    )
    model.fit(train_meta, train_labels)

    y_pred = model.predict(test_meta)
    y_prob = model.predict_proba(test_meta)[:, 1]
    results = evaluate(test_labels, y_pred, y_prob)
    print(json.dumps(results, indent=2))
    return results


def run_lambdamart_ngram(train_texts, train_labels, test_texts, test_labels):
    """LambdaMART + n-gram (Piotrkowicz et al., ICWSM 2018 replication)."""
    if not HAS_LIGHTGBM:
        print("\n--- LambdaMART + n-gram: SKIPPED (lightgbm not installed) ---")
        return None

    print("\n--- LambdaMART + n-gram ---")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), stop_words="english")
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()

    # LambdaMART is a ranking model; we use LGBMClassifier with lambdarank-style objective
    # For binary classification, we approximate with binary objective
    model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        num_leaves=31, random_state=42, verbose=-1,
    )
    model.fit(X_train, train_labels)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results = evaluate(test_labels, y_pred, y_prob)
    print(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["media", "policy"], required=True)
    parser.add_argument("--field_name", default=PRIMARY_FIELD_NAME)
    args = parser.parse_args()

    field_dir = args.field_name.lower().replace(" ", "_")
    task_dir = PROCESSED_DATA_DIR / field_dir / args.task

    print(f"Running traditional ML baselines for {args.task} task ({args.field_name})")
    print(f"Data dir: {task_dir}")

    train_texts, train_labels, train_meta = load_split(task_dir, "train")
    test_texts, test_labels, test_meta = load_split(task_dir, "test")

    print(f"Train: {len(train_labels)} samples, Test: {len(test_labels)} samples")

    all_results = {}

    all_results["xgboost_tfidf"] = run_xgboost_tfidf(
        train_texts, train_labels, test_texts, test_labels
    )

    all_results["xgboost_metadata"] = run_xgboost_metadata(
        train_meta.copy(), train_labels, test_meta.copy(), test_labels
    )

    result = run_lambdamart_ngram(train_texts, train_labels, test_texts, test_labels)
    if result:
        all_results["lambdamart_ngram"] = result

    # Random baseline
    print("\n--- Random Baseline ---")
    rng = np.random.RandomState(42)
    pos_ratio = train_labels.mean()
    y_rand = rng.choice([0, 1], size=len(test_labels), p=[1 - pos_ratio, pos_ratio])
    all_results["random"] = evaluate(test_labels, y_rand)
    print(json.dumps(all_results["random"], indent=2))

    # Save results
    output_dir = OUTPUTS_DIR / "baselines" / field_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.task}_traditional_ml.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {out_path}")


if __name__ == "__main__":
    main()
