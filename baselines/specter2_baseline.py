"""
SPECTER2 baseline: citation-informed scientific paper embedding + Logistic Regression.

Loads `allenai/specter2_base` + the `[PRX]` proximity adapter
(`allenai/specter2` in the SPECTER2 release). Then trains a linear head.

Two fallbacks if the adapters library isn't available:
  --no_adapter   : use only the base model [CLS] embedding (still a valid baseline)

Embeddings are cached per split to .npy.

Usage (standard):
    python baselines/specter2_baseline.py --task combined \
        --base_path /root/pilab_jiang/hf-model/specter2_base \
        --adapter_path /root/pilab_jiang/hf-model/specter2

Usage (fallback, no adapter library):
    python baselines/specter2_baseline.py --task combined \
        --base_path /root/pilab_jiang/hf-model/specter2_base --no_adapter
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baselines._utils import (
    load_split, evaluate, save_results,
)
from config import OUTPUTS_DIR, RANDOM_SEED


class SpecterTextDataset(Dataset):
    """SPECTER/SPECTER2 expects title + SEP + abstract."""
    def __init__(self, samples, tokenizer, max_length=512):
        self.samples = samples
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        text = (s.get("title", "") or "") + self.tok.sep_token \
               + (s.get("abstract", "") or "")
        enc = self.tok(
            text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def load_specter2(base_path: str, adapter_path: str = None,
                  dtype=torch.float16):
    """Return (tokenizer, model). Adapter is optional.
    Everything is cast to `dtype` AFTER adapter load to keep matmul
    operands consistent (adapter weights ship as fp32 by default).
    """
    tok = AutoTokenizer.from_pretrained(base_path)
    if adapter_path:
        try:
            from adapters import AutoAdapterModel
        except ImportError as e:
            raise ImportError(
                "Install adapter support: pip install adapters\n"
                "Or rerun with --no_adapter to skip."
            ) from e
        # Load in fp32, then cast the whole model (base + adapter) together.
        model = AutoAdapterModel.from_pretrained(base_path)
        model.load_adapter(adapter_path, source=None,
                           load_as="specter2", set_active=True)
        try:
            model.set_active_adapters("specter2")
        except Exception:
            pass
        if dtype == torch.float16:
            model = model.half()
        elif dtype == torch.bfloat16:
            model = model.to(dtype=torch.bfloat16)
    else:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(base_path, torch_dtype=dtype)
    return tok, model


@torch.no_grad()
def extract_embeddings(samples, tokenizer, model, batch_size=64, device="cuda"):
    model.eval().to(device)
    ds = SpecterTextDataset(samples, tokenizer)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    feats = []
    for batch in dl:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out = model(input_ids=ids, attention_mask=mask)
        # For SPECTER2, take the [CLS] token from last hidden state
        if hasattr(out, "last_hidden_state"):
            cls = out.last_hidden_state[:, 0, :]
        else:
            # AutoAdapterModel forward may return a different structure;
            # fall back to the first tensor
            cls = out[0][:, 0, :]
        feats.append(cls.float().cpu().numpy())
        if len(feats) % 20 == 0:
            print(f"  embedded {len(feats) * batch_size} / {len(samples)}")
    return np.concatenate(feats, axis=0)


def get_or_build(samples, tokenizer, model, cache_path,
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
    p.add_argument("--base_path", required=True)
    p.add_argument("--adapter_path", default=None)
    p.add_argument("--no_adapter", action="store_true")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--C", type=float, default=1.0)
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

    adapter_path = None if args.no_adapter else args.adapter_path
    variant = "specter2_base_only" if args.no_adapter or not args.adapter_path \
              else "specter2_base_plus_PRX_adapter"
    print(f"Loading SPECTER2 from {args.base_path}, variant={variant}")

    tok, model = load_specter2(args.base_path, adapter_path)

    cache_dir = Path(args.cache_dir) if args.cache_dir else (
        OUTPUTS_DIR / "baselines" / task_name / "embed_cache" / variant
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_tr = get_or_build(train, tok, model, cache_dir / "train.npy",
                        batch_size=args.batch_size, device=device)
    X_te = get_or_build(test, tok, model, cache_dir / "test.npy",
                        batch_size=args.batch_size, device=device)
    print(f"SPECTER2 embeddings: train={X_tr.shape}, test={X_te.shape}")

    # L2-normalise (standard for SPECTER-family)
    X_tr = X_tr / (np.linalg.norm(X_tr, axis=1, keepdims=True) + 1e-9)
    X_te = X_te / (np.linalg.norm(X_te, axis=1, keepdims=True) + 1e-9)

    clf = LogisticRegression(
        max_iter=2000, C=args.C, random_state=RANDOM_SEED,
        n_jobs=-1, solver="liblinear",
    )
    clf.fit(X_tr, train_y)
    pred = clf.predict(X_te)
    prob = clf.predict_proba(X_te)[:, 1]

    metrics = evaluate(test_y, pred, prob)
    results = {
        "model": f"SPECTER2_{variant}_LR",
        "base_path": args.base_path,
        "adapter_path": adapter_path,
        "n_features": int(X_tr.shape[1]),
        "hyperparameters": {"C": args.C, "max_length": args.max_length},
        "metrics": metrics,
    }

    out_path = Path(args.output) if args.output else (
        OUTPUTS_DIR / "baselines" / task_name / f"specter2_{variant}.json"
    )
    save_results(results, out_path)


if __name__ == "__main__":
    main()
