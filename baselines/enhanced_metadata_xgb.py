"""
Enhanced XGBoost metadata baseline.

Replaces the original minimal xgboost_metadata baseline (subfield/year/citation)
with every metadata feature available in the current data. Because the raw
OpenAlex pull never kept host_venue / authorships lists, we cannot do
"journal tier" or "author count" features; the paper's Limitations section
must acknowledge this.

Available metadata:
    cited_by_count, year, month, subfield, field, topic, type,
    title_len, abstract_len, first_author_freq

Usage:
    python baselines/enhanced_metadata_xgb.py --task combined
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baselines._utils import (
    load_split, evaluate, build_metadata_frame,
    compute_author_freq, encode_categorical, save_results,
)
from config import OUTPUTS_DIR, RANDOM_SEED


CAT_COLS = ["subfield", "field", "topic", "type"]
# cited_by_count excluded: post-hoc information (citations accumulated partly
# BECAUSE of news coverage). This baseline is strictly pre-publication.
NUM_COLS = ["year", "month",
            "title_len", "abstract_len", "first_author_freq"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="combined")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--n_estimators", type=int, default=500)
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--topic_topk", type=int, default=500)
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

    # Encode categoricals as ordinal codes (XGBoost handles these fine)
    train_df, test_df = encode_categorical(
        train_df, test_df, CAT_COLS, topic_topk=args.topic_topk,
    )

    feat_cols = NUM_COLS + CAT_COLS
    X_tr = train_df[feat_cols].values.astype(float)
    X_te = test_df[feat_cols].values.astype(float)

    print(f"\nFeatures ({len(feat_cols)}): {feat_cols}")

    clf = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    clf.fit(X_tr, train_y)
    pred = clf.predict(X_te)
    prob = clf.predict_proba(X_te)[:, 1]

    metrics = evaluate(test_y, pred, prob)

    # Feature importance for the paper
    importance = {
        f: float(v) for f, v in sorted(
            zip(feat_cols, clf.feature_importances_),
            key=lambda kv: -kv[1]
        )
    }

    results = {
        "model": "XGBoost_enhanced_metadata",
        "features": feat_cols,
        "metrics": metrics,
        "feature_importance": importance,
        "hyperparameters": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.lr,
            "topic_topk": args.topic_topk,
        },
    }

    out_path = Path(args.output) if args.output else (
        OUTPUTS_DIR / "baselines" / task_name / "enhanced_metadata_xgb.json"
    )
    save_results(results, out_path)


if __name__ == "__main__":
    main()
