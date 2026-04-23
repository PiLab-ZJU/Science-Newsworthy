"""
Publication-grade figures from analysis/*.json.

Render order matches narrative flow:
  1. baseline_comparison       — leaderboard vs SFT
  2. per_field_performance     — disciplinary spread
  3. error_by_field            — volume vs difficulty
  4. signal_analysis           — 12-theme dumbbell + 8-Galtung slope (merged)
  5. discriminative_keywords   — gold-vs-model lexicon dumbbell
  6. bertopic_topics           — corpus-share-normalized topic bars

Usage:
    python analysis/visualize.py
    python analysis/visualize.py --only newsvalue baseline_comparison
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUTS_DIR, PROJECT_ROOT

ANALYSIS_DIR = PROJECT_ROOT / "analysis"
FIG_DIR = OUTPUTS_DIR / "figures"

COLOR = {
    "baseline":  "#94A3B8",
    "ours":      "#F97316",
    "abstract":  "#9CA3AF",
    "model":     "#2563EB",
    "news":      "#DC2626",
    "gold":      "#10B981",
    "pred":      "#F59E0B",
    "tp":        "#16A34A",
    "tn":        "#0EA5E9",
    "fp":        "#F59E0B",
    "fn":        "#DC2626",
    "f1":        "#2563EB",
    "acc":       "#10B981",
    "mcc":       "#7C3AED",
    "muted":     "#E5E7EB",
    "ink":       "#1F2937",
    "subtitle":  "#6B7280",
    "good":      "#16A34A",
    "bad":       "#DC2626",
}


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#374151",
        "axes.labelcolor": COLOR["ink"],
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "axes.grid": True,
        "grid.alpha": 0.16,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
    })


def _load(name: str) -> Any:
    with open(ANALYSIS_DIR / name) as f:
        return json.load(f)


def _save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{name}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.relative_to(PROJECT_ROOT)}")
    return out


def _title(fig: plt.Figure, title: str, subtitle: str | None = None,
           x: float = 0.06, y: float = 0.985) -> None:
    fig.text(x, y, title, fontsize=14.5, fontweight="bold",
             color=COLOR["ink"], ha="left", va="top")
    if subtitle:
        fig.text(x, y - 0.038, subtitle, fontsize=10.5,
                 color=COLOR["subtitle"], ha="left", va="top", style="italic")


def _footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.99, 0.005, text, fontsize=8, color=COLOR["subtitle"],
             ha="right", va="bottom")


# ---------------------------------------------------------------------------
# Cache: SFT overall metrics from raw predictions
# ---------------------------------------------------------------------------

_SFT_CACHE: dict[str, float] | None = None


def sft_overall() -> dict[str, float]:
    global _SFT_CACHE
    if _SFT_CACHE is not None:
        return _SFT_CACHE
    rows = _load("test_predictions_with_explanations.json")
    y_true = [r["true_label"] for r in rows]
    y_pred = [r["predicted"] for r in rows]
    _SFT_CACHE = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "mcc":       matthews_corrcoef(y_true, y_pred),
    }
    return _SFT_CACHE


# ===========================================================================
# 1. baseline_comparison
# ===========================================================================

def baseline_comparison() -> None:
    extras = _load("extra_baselines.json")
    sft = sft_overall()

    label_map = {
        "logistic_regression":   "Logistic Regression\n(TF-IDF, supervised)",
        "llama_zeroshot":        "LLaMA-3.1-8B\n(zero-shot)",
        "gpt-4o-mini_zeroshot":  "GPT-4o-mini\n(zero-shot)",
    }
    rows = [(label_map.get(k, k), v) for k, v in extras.items()]
    rows.append(("LLaMA-3.1-8B SFT\n(Ours)", sft))

    metrics = [("f1", "F1", COLOR["f1"]),
               ("accuracy", "Accuracy", COLOR["acc"]),
               ("mcc", "MCC", COLOR["mcc"])]

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    n_models = len(rows)
    y = np.arange(n_models)

    ours_idx = n_models - 1
    ax.axhspan(ours_idx - 0.42, ours_idx + 0.42,
               color=COLOR["ours"], alpha=0.08, zorder=0)

    offsets = [0.22, 0.0, -0.22]
    for (key, label, color), off in zip(metrics, offsets):
        vals = [r[1].get(key, 0) for r in rows]
        ax.scatter(vals, y + off, s=170, color=color, zorder=3,
                   edgecolor="white", linewidth=1.2, label=label)
        for yi, v in zip(y + off, vals):
            ax.text(v + 0.012, yi, f"{v:.3f}", va="center",
                    fontsize=9.5, color=COLOR["ink"])

    best_baseline_f1 = max(r[1].get("f1", 0) for r in rows[:-1])
    best_baseline_mcc = max(r[1].get("mcc", 0) for r in rows[:-1])
    delta_f1 = sft["f1"] - best_baseline_f1
    delta_mcc = sft["mcc"] - best_baseline_mcc

    ax.axvline(best_baseline_f1, color=COLOR["baseline"], linestyle=":",
               linewidth=1.2, alpha=0.6, zorder=1)
    ax.text(best_baseline_f1, -0.55,
            f"best baseline F1 = {best_baseline_f1:.3f}",
            fontsize=8.5, color=COLOR["baseline"], ha="center", va="top")

    ax.annotate(
        "", xy=(sft["f1"], ours_idx + 0.22),
        xytext=(best_baseline_f1, ours_idx + 0.22),
        arrowprops=dict(arrowstyle="->", color=COLOR["ours"],
                        lw=1.6, alpha=0.75),
    )
    ax.annotate(
        f"+{delta_f1:.3f} F1\n+{delta_mcc:.3f} MCC\nvs. best baseline",
        xy=(sft["f1"], ours_idx + 0.22),
        xytext=(0.93, ours_idx),
        fontsize=10, fontweight="bold", color=COLOR["ours"],
        ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color=COLOR["ours"], lw=1.0, alpha=0.5),
    )

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Score")
    ax.legend(loc="lower right", frameon=False, ncol=3, fontsize=10,
              bbox_to_anchor=(1.0, -0.18))
    ax.grid(axis="x", alpha=0.18)
    ax.grid(axis="y", visible=False)

    _title(fig,
           "News-Coverage Prediction: Fine-Tuned LLaMA Beats All Baselines",
           f"4 models × 3 metrics. SFT model gains +{delta_f1:.3f} F1 and "
           f"+{delta_mcc:.3f} MCC over the strongest non-SFT baseline.")
    _footer(fig, "n=24,093 papers · source: extra_baselines.json + test_predictions_with_explanations.json")
    fig.subplots_adjust(top=0.84, left=0.18, right=0.95, bottom=0.16)
    _save(fig, "baseline_comparison")


# ===========================================================================
# 2. per_field_performance
# ===========================================================================

def per_field_performance() -> None:
    rows = _load("per_field_metrics.json")
    rows = [r for r in rows if r.get("n", 0) >= 100]
    rows.sort(key=lambda r: r["f1"], reverse=True)

    names = [r["field"]
             .replace("Biochemistry, Genetics and Molecular Biology",
                      "Biochem, Genetics & Mol. Biology")
             .replace("Economics, Econometrics and Finance",
                      "Economics & Finance")
             .replace("Pharmacology, Toxicology and Pharmaceutics",
                      "Pharmacology & Toxicology")
             .replace("Business, Management and Accounting",
                      "Business & Management")
             .replace("Agricultural and Biological Sciences",
                      "Agricultural & Biological Sci.")
             .replace("Earth and Planetary Sciences",
                      "Earth & Planetary Sciences")
             .replace(" and ", " & ")
             for r in rows]
    f1 = np.array([r["f1"] for r in rows])
    mcc = np.array([r["mcc"] for r in rows])
    n = np.array([r["n"] for r in rows])
    sizes = np.clip(50 + np.log10(n) * 50, 70, 280)

    fig, ax = plt.subplots(figsize=(11.5, max(6.5, 0.36 * len(rows))))
    y = np.arange(len(rows))

    for yi, a, b in zip(y, f1, mcc):
        ax.plot([a, b], [yi, yi], color="#CBD5E1", linewidth=2.2, zorder=1)

    ax.scatter(f1, y, s=sizes, color=COLOR["f1"], zorder=3,
               edgecolor="white", linewidth=1.0, label="F1")
    ax.scatter(mcc, y, s=sizes, color=COLOR["mcc"], zorder=3,
               edgecolor="white", linewidth=1.0, label="MCC")

    ax.axvline(f1.mean(), color=COLOR["f1"], alpha=0.35, linestyle=":",
               linewidth=1.3)
    ax.axvline(mcc.mean(), color=COLOR["mcc"], alpha=0.35, linestyle=":",
               linewidth=1.3)
    ax.text(f1.mean(), -0.9,
            f"mean F1 = {f1.mean():.3f}",
            color=COLOR["f1"], fontsize=9, ha="center", va="bottom",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=COLOR["f1"], alpha=0.95))
    ax.text(mcc.mean(), -0.9,
            f"mean MCC = {mcc.mean():.3f}",
            color=COLOR["mcc"], fontsize=9, ha="center", va="bottom",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=COLOR["mcc"], alpha=0.95))

    for yi, a, b in zip(y, f1, mcc):
        ax.text(max(a, b) + 0.012, yi, f"F1={a:.3f}  MCC={b:.3f}",
                va="center", fontsize=8.8, color=COLOR["ink"])

    ax.set_yticks(y)
    ax.set_yticklabels([f"{nm}  (n={ni:,})" for nm, ni in zip(names, n)],
                       fontsize=9.8)
    ax.set_xlim(0.40, 1.02)
    ax.set_xticks(np.arange(0.4, 1.01, 0.1))
    ax.set_ylim(len(rows) - 0.5, -1.8)
    ax.set_xlabel("Metric value")
    ax.legend(loc="lower right", frameon=False, fontsize=10,
              labelspacing=0.4, markerscale=0.7)
    ax.grid(axis="x", alpha=0.18)
    ax.grid(axis="y", visible=False)
    for ref in [0.6, 0.7, 0.8]:
        ax.axvline(ref, color="#D1D5DB", linewidth=0.7, zorder=0)

    fig.subplots_adjust(top=0.95, left=0.28, right=0.97, bottom=0.06)
    _save(fig, "per_field_performance")


# ===========================================================================
# 3. confusion_matrix (with marginals)
# ===========================================================================

def confusion_matrix() -> None:
    rows = _load("test_predictions_with_explanations.json")
    tp = sum(1 for r in rows if r["true_label"] == 1 and r["predicted"] == 1)
    tn = sum(1 for r in rows if r["true_label"] == 0 and r["predicted"] == 0)
    fp = sum(1 for r in rows if r["true_label"] == 0 and r["predicted"] == 1)
    fn = sum(1 for r in rows if r["true_label"] == 1 and r["predicted"] == 0)
    total = tp + tn + fp + fn

    acc  = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    npv  = tn / (tn + fn) if (tn + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    mcc_den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = ((tp * tn) - (fp * fn)) / mcc_den if mcc_den else 0

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    cells = np.array([[tn, fp], [fn, tp]])
    cell_labels = np.array([["TN", "FP"], ["FN", "TP"]])
    cell_face = np.array([["#DBEAFE", "#FECACA"],
                          ["#FECACA", "#DBEAFE"]])

    for i in range(2):
        for j in range(2):
            v = cells[i, j]
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1,
                                        facecolor=cell_face[i, j],
                                        edgecolor="white", linewidth=2.0))
            ax.text(j + 0.5, 1 - i + 0.5,
                    f"{cell_labels[i, j]}\n{v:,}\n{v / total:.1%}",
                    ha="center", va="center",
                    fontsize=15, fontweight="bold",
                    color=COLOR["ink"])

    ax.text(2.06, 1.5, f"Specificity\n{spec:.3f}",
            ha="left", va="center", fontsize=10.5, color=COLOR["subtitle"])
    ax.text(2.06, 0.5, f"Recall\n{rec:.3f}",
            ha="left", va="center", fontsize=10.5, color=COLOR["subtitle"])
    ax.text(0.5, -0.10, f"NPV\n{npv:.3f}",
            ha="center", va="top", fontsize=10.5, color=COLOR["subtitle"])
    ax.text(1.5, -0.10, f"Precision\n{prec:.3f}",
            ha="center", va="top", fontsize=10.5, color=COLOR["subtitle"])

    ax.text(2.06, -0.10,
            f"Accuracy: {acc:.3f}\nF1:        {f1:.3f}\nMCC:       {mcc:.3f}",
            ha="left", va="top", fontsize=10.5, fontweight="bold",
            color=COLOR["ink"], family="monospace",
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="#F3F4F6", edgecolor="#D1D5DB"))

    ax.set_xlim(-0.05, 3.30)
    ax.set_ylim(-0.55, 2.05)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Pred No", "Pred Yes"], fontsize=11)
    ax.set_yticks([1.5, 0.5])
    ax.set_yticklabels(["True No", "True Yes"], fontsize=11)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    ax.set_aspect("equal")

    _title(fig,
           "Confusion Matrix with Per-Class Rates",
           f"n = {total:,} test papers. Balanced model: precision {prec:.3f}, "
           f"recall {rec:.3f}, MCC {mcc:.3f}.")
    _footer(fig, "source: analysis/test_predictions_with_explanations.json")
    fig.subplots_adjust(top=0.86, left=0.10, right=0.96, bottom=0.10)
    _save(fig, "confusion_matrix")


# ===========================================================================
# 4. error_by_field (counts + rates)
# ===========================================================================

def error_by_field() -> None:
    """FPR vs FNR per field — diagnostic scatter (miss-heavy vs alarm-heavy)."""
    from collections import defaultdict

    preds_path = ANALYSIS_DIR / "test_predictions_with_explanations.json"
    with open(preds_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])  # tp, tn, fp, fn
    for p in preds:
        y, yh = p["true_label"], p["predicted"]
        if   y == 1 and yh == 1: agg[p["field"]][0] += 1
        elif y == 0 and yh == 0: agg[p["field"]][1] += 1
        elif y == 0 and yh == 1: agg[p["field"]][2] += 1
        else:                    agg[p["field"]][3] += 1

    rows = []
    for f_, (tp, tn, fp_c, fn_c) in agg.items():
        pos, neg = tp + fn_c, tn + fp_c
        n = pos + neg
        if n < 100:
            continue
        rows.append({
            "field": f_, "n": n,
            "fpr": fp_c / neg if neg else 0.0,
            "fnr": fn_c / pos if pos else 0.0,
        })

    short = lambda f: (
        f.replace("Biochemistry, Genetics and Molecular Biology", "Biochem & Mol. Bio")
         .replace("Economics, Econometrics and Finance",          "Economics & Finance")
         .replace("Pharmacology, Toxicology and Pharmaceutics",   "Pharmacology")
         .replace("Business, Management and Accounting",          "Business & Mgmt")
         .replace("Agricultural and Biological Sciences",         "Agric. & Bio. Sci.")
         .replace("Earth and Planetary Sciences",                 "Earth & Planetary")
         .replace("Immunology and Microbiology",                  "Immunology & Micro")
         .replace("Physics and Astronomy",                        "Physics & Astro")
         .replace("Health Professions",                           "Health Prof.")
         .replace(" and ", " & ")
    )
    fpr = np.array([r["fpr"] for r in rows])
    fnr = np.array([r["fnr"] for r in rows])
    n_v = np.array([r["n"]   for r in rows])
    names = [short(r["field"]) for r in rows]
    sizes = np.clip(np.log10(n_v) * 170 - 200, 80, 900)

    # overall
    FP_all = sum(v[2] for v in agg.values()); TN_all = sum(v[1] for v in agg.values())
    FN_all = sum(v[3] for v in agg.values()); TP_all = sum(v[0] for v in agg.values())
    ov_fpr = FP_all / (FP_all + TN_all)
    ov_fnr = FN_all / (FN_all + TP_all)

    fig, ax = plt.subplots(figsize=(11, 8.5))
    lim = max(fpr.max(), fnr.max()) * 1.12

    ax.plot([0, lim], [0, lim], color="#CBD5E1", linestyle="--",
            linewidth=1.2, zorder=1, label="balanced (FPR = FNR)")
    ax.axvline(ov_fpr, color="#94A3B8", linewidth=0.8, alpha=0.6)
    ax.axhline(ov_fnr, color="#94A3B8", linewidth=0.8, alpha=0.6)
    ax.text(ov_fpr + 0.003, lim * 0.98, f"overall FPR = {ov_fpr:.2f}",
            fontsize=9, color="#64748B", va="top")
    ax.text(lim * 0.98, ov_fnr - 0.008, f"overall FNR = {ov_fnr:.2f}",
            fontsize=9, color="#64748B", ha="right", va="top")

    colors = ["#DC2626" if fn_i > fp_i else "#F59E0B"
              for fp_i, fn_i in zip(fpr, fnr)]
    ax.scatter(fpr, fnr, s=sizes, c=colors, alpha=0.55,
               edgecolor="white", linewidth=1.2, zorder=3)

    for fp_i, fn_i, name in zip(fpr, fnr, names):
        ax.annotate(name, (fp_i, fn_i), fontsize=8.8,
                    ha="left", va="bottom",
                    xytext=(4, 4), textcoords="offset points",
                    color=COLOR["ink"])

    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("FPR  (false-alarm rate on true-negatives)", fontsize=11)
    ax.set_ylabel("FNR  (miss rate on true-positives)", fontsize=11)
    ax.grid(alpha=0.18)

    ax.text(0.02, lim * 0.93,
            "Miss-heavy\n(academic framing\nfails to flag news)",
            fontsize=9.5, color="#B91C1C", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor="#FEE2E2", edgecolor="#FCA5A5"))
    ax.text(lim * 0.98, 0.01,
            "Alarm-heavy\n(model over-\npredicts news)",
            fontsize=9.5, color="#B45309", fontweight="bold", ha="right",
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor="#FEF3C7", edgecolor="#FCD34D"))

    for sz, lab in zip([200, 1000, 3000],
                       ["n = 200", "n = 1,000", "n = 3,000"]):
        ax.scatter([], [], s=np.clip(np.log10(sz) * 170 - 200, 80, 900),
                   c="#CBD5E1", alpha=0.55, edgecolor="white", linewidth=1.2,
                   label=lab)
    ax.legend(loc="lower left", frameon=False, fontsize=9.5,
              labelspacing=1.0, borderpad=0.6, handletextpad=0.8)

    fig.subplots_adjust(top=0.97, left=0.08, right=0.97, bottom=0.08)
    _save(fig, "error_by_field")


# ===========================================================================
# 5. signal_taxonomy_heatmap (with precision/recall strip)
# ===========================================================================

def signal_taxonomy_heatmap() -> None:
    tax = _load("signal_taxonomy.json")
    buckets = ["tp_signals", "tn_signals", "fp_signals", "fn_signals"]
    bucket_labels = ["TP", "TN", "FP", "FN"]

    signals = sorted(
        {s for b in buckets for s in tax.get(b, {})},
        key=lambda s: -sum(tax.get(b, {}).get(s, 0) for b in buckets),
    )
    matrix = np.array(
        [[tax.get(b, {}).get(s, 0) for s in signals] for b in buckets],
        dtype=float,
    )
    row_norm = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)

    tp, tn, fp, fn = matrix
    prec = np.where((tp + fp) > 0, tp / np.maximum(tp + fp, 1), np.nan)
    rec  = np.where((tp + fn) > 0, tp / np.maximum(tp + fn, 1), np.nan)

    fig = plt.figure(figsize=(13.5, 7))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[2.2, 1], hspace=0.10)
    ax_h = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    annot = np.array([[f"{int(v):,}" for v in row] for row in matrix])
    sns.heatmap(
        row_norm, annot=annot, fmt="",
        xticklabels=False, yticklabels=bucket_labels,
        cmap="YlOrRd", linewidths=0.4, linecolor="white",
        cbar_kws={"label": "Row-normalized share", "shrink": 0.8},
        annot_kws={"fontsize": 9}, ax=ax_h,
    )
    ax_h.set_yticklabels(bucket_labels, rotation=0, fontsize=11,
                          fontweight="bold")
    ax_h.tick_params(left=False, length=0)

    x = np.arange(len(signals)) + 0.5
    w = 0.36
    ax_b.bar(x - w / 2, np.nan_to_num(prec, nan=0), w,
             color=COLOR["model"],
             label="Precision = TP / (TP+FP)",
             edgecolor="white", linewidth=0.6)
    ax_b.bar(x + w / 2, np.nan_to_num(rec, nan=0), w,
             color=COLOR["news"],
             label="Recall = TP / (TP+FN)",
             edgecolor="white", linewidth=0.6)
    for xi, p in zip(x, prec):
        if np.isnan(p):
            ax_b.text(xi - w / 2, 0.04, "n/a", ha="center",
                      fontsize=8, color=COLOR["subtitle"], rotation=90)
    ax_b.set_xlim(0, len(signals))
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([s.replace("_", " ").title() for s in signals],
                          rotation=30, ha="right", fontsize=9.5)
    ax_b.set_ylim(0, 1.15)
    ax_b.set_ylabel("Rate")
    ax_b.legend(loc="upper center", frameon=False, fontsize=9.5, ncol=2,
                bbox_to_anchor=(0.5, 1.14))
    ax_b.grid(axis="y", alpha=0.18)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    _title(fig,
           "Signal Taxonomy: Counts by Outcome and Per-Signal Reliability",
           f"{len(signals)} signals × 4 outcomes (sorted by total mention). "
           "Bottom strip: Precision (low → signal causes FP) and Recall "
           "(low → model misses cases that have this signal).")
    _footer(fig, "source: analysis/signal_taxonomy.json")
    fig.subplots_adjust(top=0.89, left=0.07, right=0.96, bottom=0.18)
    _save(fig, "signal_taxonomy_heatmap")


# ===========================================================================
# 6. signal_analysis (merged: 12-theme dumbbell + 8-Galtung slope)
# ===========================================================================

def signal_analysis() -> None:
    """Two-panel lexical/news-value gap: custom themes (left) + Galtung (right)."""
    themes = _load("signal_ratios.json")
    themes.sort(key=lambda r: r["ratio"])
    t_names  = [r["signal"] for r in themes]
    t_model  = np.array([r["model_count"] for r in themes], dtype=float)
    t_news   = np.array([r["news_count"]  for r in themes], dtype=float)
    t_ratio  = np.array([r["ratio"]       for r in themes], dtype=float)

    # normalize counts per row to max so the two dots share a common scale
    t_max = np.maximum(t_model, t_news)
    mn = t_model / t_max
    nn = t_news  / t_max

    galt = _load("newsvalue_signals.json")
    galt.sort(key=lambda r: r["news_pct"] - r["model_pct"])
    g_names = [r["signal"] for r in galt]
    g_abs   = np.array([r["abstract_pct"] for r in galt])
    g_mod   = np.array([r["model_pct"]    for r in galt])
    g_news  = np.array([r["news_pct"]     for r in galt])
    g_gap   = g_news - g_mod

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(15.8, 6.8),
        gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.42},
    )

    # ---------- LEFT: 12-theme dumbbell (model vs news, normalized) ----------
    y = np.arange(len(t_names))
    for yi, m, n_ in zip(y, mn, nn):
        ax_l.plot([m, n_], [yi, yi], color="#CBD5E1", linewidth=2.2, zorder=1)
    ax_l.scatter(mn, y, s=150, color=COLOR["model"],
                 edgecolor="white", linewidth=1.0, zorder=3,
                 label="Model explanation")
    ax_l.scatter(nn, y, s=150, color=COLOR["news"],
                 edgecolor="white", linewidth=1.0, zorder=3,
                 label="News coverage")

    for yi, m, n_, r in zip(y, mn, nn, t_ratio):
        x_lab = max(m, n_) + 0.03
        color = COLOR["good"] if r >= 1 else COLOR["bad"]
        ax_l.text(x_lab, yi, f"r={r:.2f}", va="center",
                  fontsize=9, fontweight="bold", color=color)

    ax_l.set_yticks(y)
    ax_l.set_yticklabels(t_names, fontsize=10)
    ax_l.invert_yaxis()
    ax_l.set_xlim(0, 1.32)
    ax_l.set_xticks(np.arange(0, 1.01, 0.25))
    ax_l.set_xlabel("Normalized presence  (per-theme max = 1.0)")
    ax_l.set_title("(a) Custom themes (n = 12)",
                   fontsize=11.5, color=COLOR["ink"], pad=8)
    ax_l.legend(loc="lower right", frameon=False, fontsize=9.5,
                bbox_to_anchor=(1.0, -0.015))
    ax_l.grid(axis="x", alpha=0.18)
    ax_l.grid(axis="y", visible=False)

    # ---------- RIGHT: 8-Galtung slope (abstract → model → news) ----------
    x_pos = [0, 1, 2]
    x_labels = ["Abstract", "Model", "News"]
    cmap = plt.get_cmap("RdYlGn_r")
    norm_gap = (g_gap - g_gap.min()) / max(1e-9, g_gap.max() - g_gap.min())

    for x in x_pos:
        ax_r.axvline(x, color="#E5E7EB", linewidth=1.0, zorder=0)

    for i, s in enumerate(g_names):
        is_top = i >= len(g_names) - 3  # top-3 largest news-model gap
        color = cmap(norm_gap[i])
        ax_r.plot(x_pos, [g_abs[i], g_mod[i], g_news[i]],
                  color=color,
                  linewidth=2.8 if is_top else 1.4,
                  alpha=0.95 if is_top else 0.55,
                  marker="o", markersize=9 if is_top else 5,
                  markeredgecolor="white", markeredgewidth=1.0,
                  zorder=4 if is_top else 2)

    label_order = sorted(range(len(g_names)), key=lambda i: -g_news[i])
    used_y: list[float] = []
    MIN_GAP = 5.0
    for i in label_order:
        is_top = i >= len(g_names) - 3
        color = cmap(norm_gap[i])
        ly = g_news[i]
        for _ in range(30):
            collided = False
            for prev in used_y:
                if abs(ly - prev) < MIN_GAP:
                    ly = prev - MIN_GAP
                    collided = True
                    break
            if not collided:
                break
        used_y.append(ly)
        label = f"{g_names[i]} +{g_gap[i]:.0f}pp" if is_top else g_names[i]
        ax_r.text(2.05, ly, " " + label, va="center",
                  fontsize=10 if is_top else 9.2,
                  fontweight="bold" if is_top else "normal",
                  color=color if is_top else COLOR["subtitle"])

    ax_r.set_xticks(x_pos)
    ax_r.set_xticklabels(x_labels, fontsize=10.5, fontweight="bold")
    ax_r.set_xlim(-0.15, 2.95)
    ax_r.set_ylim(0, 100)
    ax_r.set_ylabel("% of items mentioning signal")
    ax_r.set_title("(b) Galtung news values (n = 8)",
                   fontsize=11.5, color=COLOR["ink"], pad=8)
    ax_r.grid(axis="y", alpha=0.18)
    ax_r.grid(axis="x", visible=False)

    n_under = int((t_ratio < 1).sum())
    _title(fig,
           "News-Value Gap: Where Model Explanations Diverge from News Coverage",
           f"(a) 12 themes: model under-uses {n_under} of them. "
           f"(b) 8 Galtung values along Abstract → Model → News trajectories; "
           f"top-3 widening gaps highlighted.")
    _footer(fig, "source: analysis/signal_ratios.json + newsvalue_signals.json")
    fig.subplots_adjust(top=0.87, left=0.17, right=0.95, bottom=0.10)
    _save(fig, "signal_analysis")


# ===========================================================================
# 9. discriminative_keywords (gold vs pred dumbbell)
# ===========================================================================

def discriminative_keywords() -> None:
    mb = _load("model_behavior.json")

    def _panel(ax, true_list, pred_list, title, x_sign):
        true_d = {w: info for w, info in true_list}
        pred_d = {w: info for w, info in pred_list}
        all_words = set(true_d) | set(pred_d)
        word_max = {
            w: max(abs(true_d.get(w, {}).get("z", 0)),
                   abs(pred_d.get(w, {}).get("z", 0)))
            for w in all_words
        }
        top = sorted(all_words, key=lambda w: -word_max[w])[:22]
        top.sort(key=lambda w: word_max[w], reverse=True)
        y = np.arange(len(top))
        n_agree = 0

        for yi, word in zip(y, top):
            tz = true_d.get(word, {}).get("z")
            pz = pred_d.get(word, {}).get("z")
            both = tz is not None and pz is not None
            if both:
                n_agree += 1
                gap = abs(tz - pz)
                line_c = "#94A3B8" if gap > 3 else "#CBD5E1"
                ax.plot([tz, pz], [yi, yi], color=line_c,
                        linewidth=1.6, zorder=1)
            if tz is not None:
                ax.scatter(tz, yi, s=150 if both else 110,
                           color=COLOR["gold"],
                           edgecolor="white", linewidth=1.0, zorder=3)
            if pz is not None:
                ax.scatter(pz, yi, s=150 if both else 110,
                           color=COLOR["pred"],
                           marker="s", edgecolor="white", linewidth=1.0,
                           zorder=3)

            label_color = COLOR["ink"] if both else COLOR["subtitle"]
            label_weight = "bold" if both else "normal"
            ax.text(-0.01 if x_sign > 0 else 0.01, yi, word,
                    transform=ax.get_yaxis_transform(),
                    fontsize=10, ha="right" if x_sign > 0 else "left",
                    va="center", color=label_color, fontweight=label_weight)

        ax.set_yticks(y)
        ax.set_yticklabels(["" for _ in top])
        ax.invert_yaxis()
        ax.set_title(f"{title}   ({n_agree}/{len(top)} shared)",
                     fontsize=11, color=COLOR["ink"], pad=10)
        ax.grid(axis="x", alpha=0.18)
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color="#9CA3AF", linewidth=0.8)
        if x_sign < 0:
            ax.invert_xaxis()
        ax.set_xlabel(f"z-score  ({'positive' if x_sign>0 else 'negative'})")

    fig, axes = plt.subplots(1, 2, figsize=(14, 8.5))
    _panel(axes[0], mb["true_yes_top30"], mb["pred_yes_top30"],
           "YES discriminators (predict news coverage)", x_sign=+1)
    _panel(axes[1], mb["true_no_top30"], mb["pred_no_top30"],
           "NO discriminators (predict no coverage)", x_sign=-1)

    j_y = mb.get("jaccard_yes", 0); j_n = mb.get("jaccard_no", 0)
    legend_handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=COLOR["gold"],
                   markeredgecolor="white", markersize=11,
                   label="Gold (true labels)"),
        plt.Line2D([], [], marker="s", linestyle="", color=COLOR["pred"],
                   markeredgecolor="white", markersize=11,
                   label="Predicted (model)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", frameon=False,
               fontsize=10.5, bbox_to_anchor=(0.5, -0.01), ncol=2)

    _title(fig,
           "Lexical Fingerprint: Gold Labels vs. Model Predictions",
           f"Top-22 discriminators per side (union of both top-30s). "
           f"Jaccard overlap: yes={j_y:.2f}, no={j_n:.2f}. "
           "Single dot = word appears in only one set; long connector = lexical disagreement.")
    _footer(fig, "source: analysis/model_behavior.json")
    fig.subplots_adjust(top=0.88, left=0.08, right=0.96, bottom=0.10,
                         wspace=0.35)
    _save(fig, "discriminative_keywords")


# ===========================================================================
# 10. bertopic_topics (corpus-share-normalized, shared x)
# ===========================================================================

def bertopic_topics() -> None:
    bt = _load("bertopic_results.json")
    panels = [
        ("Model YES topics", bt.get("yes_topics", []),  COLOR["model"]),
        ("Model NO topics",  bt.get("no_topics", []),   "#7C3AED"),
        ("News topics",      bt.get("news_topics", []), COLOR["news"]),
    ]
    for _, topics, _ in panels:
        total = sum(t.get("count", 0) for t in topics)
        for t in topics:
            t["pct"] = 100.0 * t["count"] / total if total else 0
    max_pct = max((t.get("pct", 0)
                   for _, topics, _ in panels for t in topics), default=10)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7),
                              sharex=True,
                              gridspec_kw={"wspace": 0.55})
    for ax, (title, topics, color) in zip(axes, panels):
        if not topics:
            ax.set_visible(False); continue
        ts = sorted(topics, key=lambda t: t.get("count", 0), reverse=True)[:10]
        labels = [f"T{t['id']}: " + ", ".join(t["words"][:3]) for t in ts]
        pcts   = [t["pct"] for t in ts]
        counts = [t["count"] for t in ts]
        y = np.arange(len(ts))
        ax.barh(y, pcts[::-1], color=color, alpha=0.85,
                edgecolor="white", linewidth=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(labels[::-1], fontsize=9.5)
        for yi, p, c in zip(y, pcts[::-1], counts[::-1]):
            ax.text(p + max_pct * 0.012, yi, f"{p:.1f}% ({c:,})",
                    va="center", fontsize=8.8, color=COLOR["ink"])
        ax.set_title(title, fontsize=11.5, color=color, fontweight="bold",
                     pad=10)
        ax.set_xlim(0, max_pct * 1.45)
        ax.set_xlabel("% of corpus")
        ax.grid(axis="x", alpha=0.18)
        ax.grid(axis="y", visible=False)

    _title(fig,
           "BERTopic Topic Sizes: Model vs. Real News Coverage",
           "Top-10 topics per corpus, normalized to corpus share. "
           "Compare Model-YES with News to see thematic overlap and divergence.",
           x=0.015)
    _footer(fig, "source: analysis/bertopic_results.json")
    fig.subplots_adjust(top=0.84, left=0.08, right=0.985, bottom=0.09)
    _save(fig, "bertopic_topics")


# ===========================================================================
# 11. model_ladder (full baseline + SFT ladder)
# ===========================================================================

def model_ladder() -> None:
    rows = _load("extended_baselines.json")

    groups = {
        "Trivial":           ["Random"],
        "Classical ML":      ["XGBoost + TF-IDF", "LightGBM + n-gram",
                              "Logistic Regression + TF-IDF"],
        "Zero-shot LLM":     ["LLaMA-3.1-8B zero-shot",
                              "GPT-4o-mini zero-shot",
                              "Qwen2.5-7B zero-shot"],
        "SFT (ours)":        ["LLaMA SFT r8 (Prediction:Yes/No)",
                              "LLaMA SFT r32 (Yes/No)",
                              "LLaMA SFT r32 CoT",
                              "Qwen2.5-7B SFT r32"],
    }
    group_color = {
        "Trivial":         "#9CA3AF",
        "Classical ML":    "#0EA5E9",
        "Zero-shot LLM":   "#A855F7",
        "SFT (ours)":      COLOR["ours"],
    }

    by_model = {r["model"]: r for r in rows}
    ordered, colors, groups_list = [], [], []
    for g, names in groups.items():
        for n in names:
            if n in by_model:
                ordered.append(by_model[n])
                colors.append(group_color[g])
                groups_list.append(g)

    labels = []
    for r in ordered:
        name = r["model"]
        name = (name.replace("LLaMA SFT r8 (Prediction:Yes/No)",
                             "LLaMA SFT r8 (Yes/No)")
                    .replace("Logistic Regression + TF-IDF",
                             "LogReg + TF-IDF"))
        ep = r.get("epoch")
        if ep and ep not in ("-", "nan", "None") and "SFT" in name:
            try:
                name = f"{name}  ep{int(float(ep))}"
            except (TypeError, ValueError):
                pass
        labels.append(name)

    f1  = np.array([r["f1"] or 0 for r in ordered])
    mcc = np.array([r["mcc"] or 0 for r in ordered])

    fig, ax = plt.subplots(figsize=(12, max(6.5, 0.44 * len(ordered))))
    y = np.arange(len(ordered))

    for yi, a, b in zip(y, f1, mcc):
        ax.plot([a, b], [yi, yi], color="#CBD5E1", linewidth=2.2, zorder=1)
    ax.scatter(f1, y, s=170, color=[COLOR["f1"]]*len(y),
               edgecolor="white", linewidth=1.0, zorder=3, label="F1")
    ax.scatter(mcc, y, s=170, color=[COLOR["mcc"]]*len(y),
               edgecolor="white", linewidth=1.0, zorder=3, label="MCC")

    for yi, a, b in zip(y, f1, mcc):
        ax.text(max(a, b) + 0.012, yi,
                f"F1={a:.3f}  MCC={b:.3f}",
                va="center", fontsize=9, color=COLOR["ink"])

    for g, c in group_color.items():
        idxs = [i for i, gi in enumerate(groups_list) if gi == g]
        if not idxs:
            continue
        ax.axhspan(min(idxs) - 0.45, max(idxs) + 0.45,
                   color=c, alpha=0.07, zorder=0)
        ax.text(-0.34, (min(idxs) + max(idxs)) / 2, g,
                transform=ax.get_yaxis_transform(),
                ha="right", va="center",
                fontsize=10.5, fontweight="bold", color=c)

    best_f1_idx = int(np.argmax(f1))
    ax.scatter(f1[best_f1_idx], best_f1_idx, s=280,
               facecolor="none", edgecolor=COLOR["ours"], linewidth=2.0,
               zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_xlabel("Metric value")
    ax.legend(loc="lower right", frameon=False, fontsize=10, ncol=2,
              bbox_to_anchor=(1.0, -0.12))
    ax.grid(axis="x", alpha=0.18)
    ax.grid(axis="y", visible=False)
    for ref in [0.5, 0.7, 0.8]:
        ax.axvline(ref, color="#D1D5DB", linewidth=0.7, zorder=0)

    _title(fig,
           f"Model Ladder: {len(ordered)} systems across 4 method families",
           f"Best F1 = {f1[best_f1_idx]:.3f} ({ordered[best_f1_idx]['model']}). "
           "CoT variant of the same SFT model underperforms Yes/No by −0.27 F1 — "
           "a flag for chain-of-thought training on short-context classification.")
    _footer(fig, "source: experiment_results.xlsx → extended_baselines.json")
    fig.subplots_adjust(top=0.91, left=0.40, right=0.97, bottom=0.11)
    _save(fig, "model_ladder")


# ===========================================================================
# 12. ablation_overview (rank / CoT / cross-model)
# ===========================================================================

def ablation_overview() -> None:
    rows = _load("extended_baselines.json")
    by_model = {r["model"]: r for r in rows}

    panels = [
        {
            "title": "LoRA rank",
            "subtitle": "r8 vs r32 (epoch 3, Yes/No head)",
            "bars": [
                ("r=8",  by_model.get("LLaMA SFT r8 (Prediction:Yes/No)")),
                ("r=32", by_model.get("LLaMA SFT r32 (Yes/No)")),
            ],
            "color": ["#94A3B8", COLOR["ours"]],
        },
        {
            "title": "Output format",
            "subtitle": "Yes/No vs Chain-of-Thought (r=32)",
            "bars": [
                ("Yes/No", by_model.get("LLaMA SFT r32 (Yes/No)")),
                ("CoT",    by_model.get("LLaMA SFT r32 CoT")),
            ],
            "color": [COLOR["ours"], "#94A3B8"],
        },
        {
            "title": "Base model",
            "subtitle": "LLaMA-3.1-8B vs Qwen2.5-7B (r=32, Yes/No)",
            "bars": [
                ("LLaMA-3.1-8B", by_model.get("LLaMA SFT r32 (Yes/No)")),
                ("Qwen2.5-7B",   by_model.get("Qwen2.5-7B SFT r32")),
            ],
            "color": [COLOR["ours"], "#0EA5E9"],
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, p in zip(axes, panels):
        labels, f1_vals, mcc_vals, accs = [], [], [], []
        for lab, r in p["bars"]:
            if r is None:
                continue
            labels.append(lab)
            f1_vals.append(r["f1"])
            mcc_vals.append(r["mcc"])
            accs.append(r["acc"])

        x = np.arange(len(labels))
        w = 0.28
        ax.bar(x - w, f1_vals,  w, color=COLOR["f1"],  label="F1",
               edgecolor="white", linewidth=0.6)
        ax.bar(x,     accs,     w, color=COLOR["acc"], label="Acc",
               edgecolor="white", linewidth=0.6)
        ax.bar(x + w, mcc_vals, w, color=COLOR["mcc"], label="MCC",
               edgecolor="white", linewidth=0.6)

        for xi, v in zip(x - w, f1_vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
        for xi, v in zip(x,     accs):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
        for xi, v in zip(x + w, mcc_vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

        d_f1 = f1_vals[1] - f1_vals[0] if len(f1_vals) == 2 else 0
        arrow_color = COLOR["good"] if d_f1 > 0 else COLOR["bad"]
        ax.annotate(f"ΔF1 = {d_f1:+.2f}",
                    xy=(0.5, 0.97), xycoords="axes fraction",
                    ha="center", va="top",
                    fontsize=10, fontweight="bold", color=arrow_color,
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white",
                              edgecolor=arrow_color, alpha=0.9))

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_title(p["title"], fontsize=11.5, fontweight="bold", pad=6)
        ax.text(0.5, -0.19, p["subtitle"], transform=ax.transAxes,
                ha="center", fontsize=9, color=COLOR["subtitle"],
                style="italic")
        ax.grid(axis="y", alpha=0.18)
        ax.grid(axis="x", visible=False)

    axes[0].legend(loc="lower left", frameon=False, fontsize=9,
                   bbox_to_anchor=(0.0, -0.32), ncol=3)

    _title(fig,
           "Ablations: What Moves the Needle",
           "Three controlled comparisons at 24,093-paper test set. "
           "Rank and Yes/No format matter; base model does not.")
    _footer(fig, "source: experiment_results.xlsx → extended_baselines.json")
    fig.subplots_adjust(top=0.82, left=0.05, right=0.98, bottom=0.22,
                         wspace=0.22)
    _save(fig, "ablation_overview")


# ===========================================================================
# 13. case_studies (FN qualitative comparison)
# ===========================================================================

def case_studies() -> None:
    cases = _load("case_studies.json")

    def _wrap(text: str, width: int = 55) -> str:
        words = text.split()
        lines, line = [], ""
        for w in words:
            if len(line) + len(w) + 1 <= width:
                line = (line + " " + w).strip()
            else:
                lines.append(line); line = w
        if line:
            lines.append(line)
        return "\n".join(lines)

    n = len(cases)
    fig, ax = plt.subplots(figsize=(14, 1.7 * n + 2))
    ax.set_xlim(0, 10); ax.set_ylim(0, n)
    ax.axis("off")

    col_x = [0.05, 3.65, 6.85]
    headers = ["Paper  &  Field",
               "Model said NO — reason",
               "Real news headline / lede"]
    header_colors = [COLOR["ink"], COLOR["bad"], COLOR["good"]]

    ax.axhline(n - 0.05, color="#E5E7EB", linewidth=1.0, xmin=0, xmax=1)
    for hx, ht, hc in zip(col_x, headers, header_colors):
        ax.text(hx, n - 0.05, ht, fontsize=11, fontweight="bold",
                color=hc, va="bottom", ha="left")

    for i, c in enumerate(cases):
        row_y = n - 1 - i + 0.5
        ax.axhspan(n - 1 - i, n - i, facecolor="#F9FAFB" if i % 2 else "white",
                   edgecolor="none", zorder=0)

        ax.text(col_x[0], row_y + 0.32,
                _wrap(c["title"], 40),
                fontsize=9.8, fontweight="bold", color=COLOR["ink"],
                va="top", ha="left")
        ax.text(col_x[0], row_y - 0.32,
                c["field"], fontsize=9, color=COLOR["subtitle"],
                style="italic", va="top", ha="left")

        ax.text(col_x[1], row_y + 0.32,
                _wrap(c["model_explanation"].rstrip(".") + "…", 40),
                fontsize=9.5, color=COLOR["bad"], va="top", ha="left")

        ax.text(col_x[2], row_y + 0.32,
                _wrap(c["news_coverage"].rstrip(".") + "…", 40),
                fontsize=9.5, color=COLOR["good"], va="top", ha="left")

        ax.axhline(n - 1 - i, color="#E5E7EB", linewidth=0.5,
                   xmin=0.01, xmax=0.99)

    _title(fig,
           "False-Negative Case Studies: Where the Model's Framing Fails",
           f"{n} FN examples. Left: paper. Middle: model's (incorrect) reason for predicting "
           "NO. Right: how news actually covered it — consistently more "
           "human-interest and relatable than the model's academic framing.")
    _footer(fig, "source: experiment_results.xlsx → case_studies.json")
    fig.subplots_adjust(top=0.88, left=0.01, right=0.99, bottom=0.04)
    _save(fig, "case_studies")


# ===========================================================================
# Entry
# ===========================================================================

REGISTRY = {
    "baseline_comparison":     baseline_comparison,
    "per_field":               per_field_performance,
    "error_by_field":          error_by_field,
    "signal_analysis":         signal_analysis,
    "discriminative_keywords": discriminative_keywords,
    "bertopic":                bertopic_topics,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=list(REGISTRY),
                        default=None,
                        help="Render only the named figures (default: all).")
    args = parser.parse_args()

    _style()
    targets = args.only or list(REGISTRY)
    print(f"Rendering {len(targets)} figure(s) → "
          f"{FIG_DIR.relative_to(PROJECT_ROOT)}/")
    for name in targets:
        print(f"[{name}]")
        try:
            REGISTRY[name]()
        except FileNotFoundError as e:
            print(f"  SKIP — missing input: {e.filename}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERROR — {type(e).__name__}: {e}")
    print("Done.")


if __name__ == "__main__":
    main()
