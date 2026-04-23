"""
Wallace-style baseline: Logistic Regression on TF-IDF(title + abstract)
concatenated with engineered metadata features.

Follows the modelling approach of Wallace et al. (2015) — logistic regression
on textual n-grams plus paper metadata — scaled up to the full 112K dataset.
This serves as the "strong classical baseline" that reviewers from
scientometrics will expect.

Usage:
    python baselines/wallace_lr.py --task combined
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baselines._utils import (
    load_split, evaluate, build_metadata_frame,
    compute_author_freq, encode_categorical, save_results, get_text,
)
from config import OUTPUTS_DIR, RANDOM_SEED


CAT_COLS = ["subfield", "field", "topic", "type"]
# cited_by_count excluded: post-hoc information (pre-publication setup only).
NUM_COLS = ["year", "month",
            "title_len", "abstract_len", "first_author_freq"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="combined")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--max_features", type=int, default=50000,
                   help="TF-IDF vocabulary size")
    p.add_argument("--ngram_max", type=int, default=2)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--topic_topk", type=int, default=500)
    args = p.parse_args()

    if args.data_dir:
        task_dir = Path(args.data_dir)
        task_name = task_dir.name
    else:
        task_name = args.task
        task_dir = Path("data/processed") / task_name

    train = load_split(task_dir, "train")
    test = load_split(task_dir, "test")
    print(f"train={len(train)}, test={len(test)}")

    train_y = np.array([s["label"] for s in train])
    test_y = np.array([s["label"] for s in test])

    # ---------- TF-IDF on title + abstract ----------
    train_texts = [get_text(s) for s in train]
    test_texts = [get_text(s) for s in test]
    vec = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        stop_words="english",
        min_df=5,
        sublinear_tf=True,
    )
    X_tr_text = vec.fit_transform(train_texts)
    X_te_text = vec.transform(test_texts)
    print(f"TF-IDF shape: train={X_tr_text.shape}, test={X_te_text.shape}")

    # ---------- Metadata ----------
    author_freq = compute_author_freq(train)
    train_df = build_metadata_frame(train, author_freq)
    test_df = build_metadata_frame(test, author_freq)
    train_df, test_df = encode_categorical(
        train_df, test_df, CAT_COLS, topic_topk=args.topic_topk,
    )

    for c in ("first_author_freq", "title_len", "abstract_len"):
        train_df[c] = np.log1p(train_df[c].astype(float))
        test_df[c] = np.log1p(test_df[c].astype(float))

    feat_cols = NUM_COLS + CAT_COLS
    X_tr_meta = train_df[feat_cols].values.astype(float)
    X_te_meta = test_df[feat_cols].values.astype(float)
    scaler = StandardScaler().fit(X_tr_meta)
    X_tr_meta = scaler.transform(X_tr_meta)
    X_te_meta = scaler.transform(X_te_meta)

    # ---------- Concat ----------
    X_tr = hstack([X_tr_text, csr_matrix(X_tr_meta)]).tocsr()
    X_te = hstack([X_te_text, csr_matrix(X_te_meta)]).tocsr()
    print(f"Combined shape: train={X_tr.shape}, test={X_te.shape}")

    clf = LogisticRegression(
        max_iter=2000, C=args.C, random_state=RANDOM_SEED, n_jobs=-1,
        solver="liblinear",
    )
    clf.fit(X_tr, train_y)
    pred = clf.predict(X_te)
    prob = clf.predict_proba(X_te)[:, 1]

    metrics = evaluate(test_y, pred, prob)

    results = {
        "model": "Wallace_style_LR_TFIDF_plus_metadata",
        "hyperparameters": {
            "max_features": args.max_features,
            "ngram_range": [1, args.ngram_max],
            "C": args.C,
            "topic_topk": args.topic_topk,
        },
        "n_text_features": X_tr_text.shape[1],
        "n_metadata_features": len(feat_cols),
        "metrics": metrics,
    }

    out_path = Path(args.output) if args.output else (
        OUTPUTS_DIR / "baselines" / task_name / "wallace_lr.json"
    )
    save_results(results, out_path)


if __name__ == "__main__":
    main()
