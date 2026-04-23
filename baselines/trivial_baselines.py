"""
Trivial single-feature baselines.

These sanity-check whether the task is learnable from one obvious signal alone
(cited_by_count, year, subfield, field, topic, author productivity). The fear
is that a reviewer says: "your 0.80 accuracy is just citation count proxied by
a fancy LLM." If citation-count alone gets 0.75, the concern is legit.

Outputs one JSON with all single-feature results plus a pooled "all metadata"
row for reference.

Usage:
    python baselines/trivial_baselines.py --task media
    python baselines/trivial_baselines.py --task media \
        --data_dir /root/pilab_jiang/cxlin/academic_new_policy/data/processed/combined
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baselines._utils import (
    load_split, evaluate, build_metadata_frame,
    compute_author_freq, encode_categorical, save_results,
)
from config import OUTPUTS_DIR, RANDOM_SEED


def run_single_numeric(name, train_df, train_y, test_df, test_y, col):
    X_tr = train_df[[col]].values.astype(float)
    X_te = test_df[[col]].values.astype(float)
    # log1p for heavy-tailed counts
    if col in {"cited_by_count", "first_author_freq",
               "title_len", "abstract_len"}:
        X_tr = np.log1p(X_tr)
        X_te = np.log1p(X_te)
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    clf.fit(X_tr, train_y)
    pred = clf.predict(X_te)
    prob = clf.predict_proba(X_te)[:, 1]
    print(f"[{name}] acc={evaluate(test_y, pred, prob)['accuracy']:.4f}")
    return evaluate(test_y, pred, prob)


def run_single_categorical(name, train_df, train_y, test_df, test_y, col):
    tr_enc, te_enc = encode_categorical(train_df, test_df, [col])
    X_tr = tr_enc[[col]].values
    X_te = te_enc[[col]].values
    # One-hot is cleaner than ordinal codes for LR
    uniq = np.unique(np.concatenate([X_tr.ravel(), X_te.ravel()]))
    idx = {v: i for i, v in enumerate(uniq)}
    def onehot(X):
        out = np.zeros((len(X), len(uniq)))
        for i, v in enumerate(X.ravel()):
            out[i, idx[v]] = 1
        return out
    clf = LogisticRegression(
        max_iter=1000, random_state=RANDOM_SEED, C=1.0,
    )
    clf.fit(onehot(X_tr), train_y)
    pred = clf.predict(onehot(X_te))
    prob = clf.predict_proba(onehot(X_te))[:, 1]
    print(f"[{name}] acc={evaluate(test_y, pred, prob)['accuracy']:.4f}")
    return evaluate(test_y, pred, prob)


def run_all_metadata(train_df, train_y, test_df, test_y):
    cat_cols = ["subfield", "field", "topic", "type"]
    tr_enc, te_enc = encode_categorical(train_df, test_df, cat_cols, topic_topk=500)
    num_cols = ["cited_by_count", "year", "month",
                "title_len", "abstract_len", "first_author_freq"]
    for c in num_cols:
        if c in {"cited_by_count", "first_author_freq",
                 "title_len", "abstract_len"}:
            tr_enc[c] = np.log1p(tr_enc[c].astype(float))
            te_enc[c] = np.log1p(te_enc[c].astype(float))
    feat_cols = num_cols + cat_cols
    X_tr = tr_enc[feat_cols].values.astype(float)
    X_te = te_enc[feat_cols].values.astype(float)
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
    clf = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
    clf.fit(X_tr, train_y)
    pred = clf.predict(X_te)
    prob = clf.predict_proba(X_te)[:, 1]
    print(f"[all_metadata_LR] acc={evaluate(test_y, pred, prob)['accuracy']:.4f}")
    return evaluate(test_y, pred, prob)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="combined",
                   help="subdirectory under data/processed containing train/val/test.json")
    p.add_argument("--data_dir", default=None,
                   help="full path to folder containing train.json / test.json. "
                        "If omitted, inferred from --task as data/processed/<task>/")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    if args.data_dir:
        task_dir = Path(args.data_dir)
        task_name = task_dir.name
    else:
        task_name = args.task
        task_dir = Path("data/processed") / task_name

    print(f"Loading from: {task_dir}")
    train = load_split(task_dir, "train")
    test = load_split(task_dir, "test")
    print(f"train={len(train)}, test={len(test)}")

    author_freq = compute_author_freq(train)
    train_df = build_metadata_frame(train, author_freq)
    test_df = build_metadata_frame(test, author_freq)
    train_y = np.array([s["label"] for s in train])
    test_y = np.array([s["label"] for s in test])

    print(f"\nPositive rate: train={train_y.mean():.3f}, test={test_y.mean():.3f}")

    results = {}

    # Single-feature baselines
    results["citation_count_only"] = run_single_numeric(
        "citation_count_only", train_df, train_y, test_df, test_y, "cited_by_count")
    results["year_only"] = run_single_categorical(
        "year_only", train_df.assign(year=train_df["year"].astype(str)),
        train_y,
        test_df.assign(year=test_df["year"].astype(str)),
        test_y, "year")
    results["subfield_only"] = run_single_categorical(
        "subfield_only", train_df, train_y, test_df, test_y, "subfield")
    results["field_only"] = run_single_categorical(
        "field_only", train_df, train_y, test_df, test_y, "field")
    results["topic_only"] = run_single_categorical(
        "topic_only", train_df, train_y, test_df, test_y, "topic")
    results["first_author_freq_only"] = run_single_numeric(
        "first_author_freq_only", train_df, train_y, test_df, test_y, "first_author_freq")
    results["abstract_len_only"] = run_single_numeric(
        "abstract_len_only", train_df, train_y, test_df, test_y, "abstract_len")

    # Pooled LR on all metadata (for comparison with Enhanced XGBoost)
    results["all_metadata_LR"] = run_all_metadata(train_df, train_y, test_df, test_y)

    out_path = Path(args.output) if args.output else (
        OUTPUTS_DIR / "baselines" / task_name / "trivial_baselines.json"
    )
    save_results(results, out_path)


if __name__ == "__main__":
    main()
