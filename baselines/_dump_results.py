"""Dump every baseline metric to stdout as a simple table."""
import json
from pathlib import Path

p = Path("outputs/baselines/combined")


def load(f):
    with open(f) as fp:
        return json.load(fp)


rows = []
for k, m in load(p / "trivial_baselines.json").items():
    rows.append((k, m))

for name in [
    "enhanced_metadata_xgb",
    "wallace_lr",
    "scibert_embed_xgb_embedonly",
    "scibert_embed_xgb",
    "specter2_specter2_base_plus_PRX_adapter",
]:
    d = load(p / (name + ".json"))
    rows.append((name, d.get("metrics", d)))


def fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) and x is not None else "-"


print(f"{'baseline':<50} {'acc':>7} {'prec':>7} {'rec':>7} {'f1':>7} {'mcc':>7} {'auc':>7}")
print("-" * 100)
for k, m in rows:
    acc = fmt(m.get("accuracy"))
    pr = fmt(m.get("precision"))
    re = fmt(m.get("recall"))
    f1 = fmt(m.get("f1"))
    mcc = fmt(m.get("mcc"))
    auc = fmt(m.get("auc_roc"))
    print(f"{k:<50} {acc:>7} {pr:>7} {re:>7} {f1:>7} {mcc:>7} {auc:>7}")
