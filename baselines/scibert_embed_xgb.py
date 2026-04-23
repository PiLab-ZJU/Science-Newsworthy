"""
SciBERT-embedding + metadata + XGBoost baseline (Wang et al. PLOS One 2024).

Step 1: Extract frozen SciBERT [CLS] embedding for every sample (title + abstract).
Step 2: Concatenate with engineered metadata features.
Step 3: Train XGBoost on the concatenated vector.

Embeddings are cached on disk as .npy — re-running on the same splits is cheap.

Usage:
    python baselines/scibert_embed_xgb.py --task combined \
        --model_path /root/pilab_jiang/hf-model/scibert_scivocab_uncased
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from xgboost import XGBClassifier

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


class TextDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=512):
        self.samples = samples
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        enc = self.tok(
            get_text(s), truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


@torch.no_grad()
def extract_embeddings(samples, tokenizer, model, batch_size=64, device="cuda"):
    model.eval().to(device)
    ds = TextDataset(samples, tokenizer)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    feats = []
    for batch in dl:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out = model(input_ids=ids, attention_mask=mask)
        # [CLS] pooling (first token of last hidden state)
        cls = out.last_hidden_state[:, 0, :].float().cpu().numpy()
        feats.append(cls)
        if len(feats) % 20 == 0:
            print(f"  embedded {len(feats) * batch_size} / {len(samples)}")
    return np.concatenate(feats, axis=0)


def get_or_build_embeddings(samples, tokenizer, model, cache_path,
                            batch_size=64, device="cuda"):
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"[cache hit] {cache_path}")
        return np.load(cache_path)
    print(f"[cache miss] extracting {len(samples)} embeddings -> {cache_path}")
    feats = extract_embeddings(samples, tokenizer, model, batch_size, device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, feats)
    return feats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="combined")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--model_path", required=True,
                   help="Local path to SciBERT (e.g. "
                        "/root/pilab_jiang/hf-model/scibert_scivocab_uncased)")
    p.add_argument("--cache_dir", default=None,
                   help="Where to cache [CLS] embeddings. Defaults to outputs/baselines/<task>/embed_cache/scibert/")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--n_estimators", type=int, default=500)
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--topic_topk", type=int, default=500)
    p.add_argument("--no_metadata", action="store_true",
                   help="Ablation: use only SciBERT embedding, no metadata")
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

    # ---------- SciBERT embeddings ----------
    tok = AutoTokenizer.from_pretrained(args.model_path)
    bert = AutoModel.from_pretrained(args.model_path, torch_dtype=torch.float16)

    cache_dir = Path(args.cache_dir) if args.cache_dir else (
        OUTPUTS_DIR / "baselines" / task_name / "embed_cache" / "scibert"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_tr_emb = get_or_build_embeddings(
        train, tok, bert, cache_dir / "train.npy",
        batch_size=args.batch_size, device=device)
    X_te_emb = get_or_build_embeddings(
        test, tok, bert, cache_dir / "test.npy",
        batch_size=args.batch_size, device=device)
    print(f"SciBERT embeddings: train={X_tr_emb.shape}, test={X_te_emb.shape}")

    # ---------- Metadata ----------
    if args.no_metadata:
        X_tr = X_tr_emb
        X_te = X_te_emb
        feat_note = "embedding_only"
    else:
        author_freq = compute_author_freq(train)
        train_df = build_metadata_frame(train, author_freq)
        test_df = build_metadata_frame(test, author_freq)
        train_df, test_df = encode_categorical(
            train_df, test_df, CAT_COLS, topic_topk=args.topic_topk)
        for c in ("first_author_freq", "title_len", "abstract_len"):
            train_df[c] = np.log1p(train_df[c].astype(float))
            test_df[c] = np.log1p(test_df[c].astype(float))
        feat_cols = NUM_COLS + CAT_COLS
        X_tr_meta = train_df[feat_cols].values.astype(float)
        X_te_meta = test_df[feat_cols].values.astype(float)
        X_tr = np.concatenate([X_tr_emb, X_tr_meta], axis=1)
        X_te = np.concatenate([X_te_emb, X_te_meta], axis=1)
        feat_note = f"embedding+metadata ({len(feat_cols)} meta cols)"
    print(f"Concat shape: train={X_tr.shape}, test={X_te.shape} [{feat_note}]")

    # ---------- XGBoost ----------
    clf = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
    )
    clf.fit(X_tr, train_y)
    pred = clf.predict(X_te)
    prob = clf.predict_proba(X_te)[:, 1]

    metrics = evaluate(test_y, pred, prob)

    results = {
        "model": "Wang2024_SciBERT_embedding_plus_metadata_XGBoost"
                 + ("_ablation_embedding_only" if args.no_metadata else ""),
        "feature_composition": feat_note,
        "n_features": int(X_tr.shape[1]),
        "hyperparameters": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.lr,
            "topic_topk": args.topic_topk,
            "scibert_max_length": args.max_length,
        },
        "metrics": metrics,
    }

    out_name = ("scibert_embed_xgb_embedonly.json" if args.no_metadata
                else "scibert_embed_xgb.json")
    out_path = Path(args.output) if args.output else (
        OUTPUTS_DIR / "baselines" / task_name / out_name
    )
    save_results(results, out_path)


if __name__ == "__main__":
    main()
