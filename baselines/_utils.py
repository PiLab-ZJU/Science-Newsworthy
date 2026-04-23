"""
Shared utilities for baseline experiments.

All baselines use the same load_split / evaluate / metadata feature builder
to guarantee comparable numbers across Table X in the paper.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
)


# Fields in the JSON samples that must NEVER be used as features.
# `news_count` is the OpenAlex cited_by_news_count — the label itself.
LEAKAGE_FIELDS = {"news_count", "label"}


def load_split(task_dir: Path, split: str) -> list:
    """Load a JSON split file and return list of sample dicts."""
    path = Path(task_dir) / f"{split}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_text(sample: dict) -> str:
    """Standard text concatenation used by all text baselines."""
    return f"{sample.get('title', '')} [SEP] {sample.get('abstract', '')}"


def evaluate(y_true, y_pred, y_prob=None) -> dict:
    """Unified metrics. y_prob optional for AUC."""
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_prob is not None:
        try:
            m["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            m["auc_roc"] = None
    return m


def build_metadata_frame(samples: list, author_freq: dict = None) -> pd.DataFrame:
    """
    Turn a list of samples into a metadata DataFrame using only fields
    that exist in the current data (no journal / authorships available).

    Columns:
        - cited_by_count:        int
        - year:                  int (from publication_date)
        - month:                 int
        - subfield:              str (~100-200 values)
        - field:                 str (~26 values)
        - topic:                 str (~thousands; consumer should hash/topK)
        - type:                  str (article / preprint / review ...)
        - title_len:             int (chars)
        - abstract_len:          int (chars)
        - first_author_freq:     int (count in train set, 0 for unseen)
    """
    rows = []
    for s in samples:
        pub = s.get("publication_date", "") or ""
        year = int(pub[:4]) if pub[:4].isdigit() else 0
        month = int(pub[5:7]) if pub[5:7].isdigit() else 0
        author_id = s.get("first_author_id", "") or ""
        rows.append({
            "cited_by_count": int(s.get("cited_by_count", 0) or 0),
            "year": year,
            "month": month,
            "subfield": s.get("subfield", "") or "UNK",
            "field": s.get("field", "") or "UNK",
            "topic": s.get("topic", "") or "UNK",
            "type": s.get("type", "") or "UNK",
            "title_len": len(s.get("title", "") or ""),
            "abstract_len": len(s.get("abstract", "") or ""),
            "first_author_freq": (author_freq or {}).get(author_id, 0),
        })
    return pd.DataFrame(rows)


def compute_author_freq(train_samples: list) -> dict:
    """Count first-author appearances in train set."""
    c = Counter(s.get("first_author_id", "") or "" for s in train_samples)
    c.pop("", None)
    return dict(c)


def encode_categorical(train_df: pd.DataFrame, test_df: pd.DataFrame,
                       columns: list, topic_topk: int = 500) -> tuple:
    """
    Encode categorical columns jointly (fit on train, apply to test).

    `topic` column is special-cased: keep top-K most frequent values in train,
    fold everything else into 'OTHER' (topic has thousands of values).
    Other columns are encoded as pandas category codes.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    for col in columns:
        if col == "topic":
            top = set(train_df[col].value_counts().head(topic_topk).index)
            train_df[col] = train_df[col].where(train_df[col].isin(top), "OTHER")
            test_df[col] = test_df[col].where(test_df[col].isin(top), "OTHER")

        combined = pd.concat([train_df[col], test_df[col]])
        codes = combined.astype("category").cat.codes
        train_df[col] = codes[:len(train_df)].values
        test_df[col] = codes[len(train_df):].values

    return train_df, test_df


def save_results(results: dict, out_path: Path):
    """Save with stable schema and print."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n=> saved: {out_path}")
    print(json.dumps(results, indent=2, default=str))
