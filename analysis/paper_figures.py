"""
Paper-ready PDFs for paper-workflow/latex-temple/figures/.

These are the figures actually cited in the LaTeX source:
    fig_baseline_comparison.pdf  — 9-model F1/Acc/MCC leaderboard
    fig_per_field.pdf            — per-field F1+MCC across 25 disciplines
    fig_error_by_field.pdf       — FPR/FNR scatter with volume sizing
    fig_signal_analysis.pdf      — 12-theme dumbbell + 8-Galtung slope
    fig_discriminative_keywords.pdf — gold-vs-predicted lexical fingerprint
    fig_bertopic.pdf             — BERTopic corpus-share topic panels
    fig_mn_radar.pdf             — Galtung 8-category M/N radar
    fig_vocabulary.pdf           — log-frequency scatter with Jaccard

Style: minimal chrome — no figure titles, no subtitles, no source footer.
All chart labels live inside axes. Vector PDF at publication size.

Usage:
    python analysis/paper_figures.py
    python analysis/paper_figures.py --only mn_radar vocabulary
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROJECT_ROOT

ANALYSIS_DIR = PROJECT_ROOT / "analysis"
PAPER_FIG_DIR = PROJECT_ROOT / "paper-workflow" / "latex-temple" / "figures"

COLOR = {
    "ink":       "#1F2937",
    "muted":     "#6B7280",
    "edge":      "#374151",
    "grid":      "#E5E7EB",
    "model":     "#2563EB",
    "news":      "#DC2626",
    "abstract":  "#9CA3AF",
    "ref":       "#94A3B8",
    "good":      "#16A34A",
    "bad":       "#DC2626",
    "warn":      "#F59E0B",
    "ours":      "#B91C1C",
    "baseline":  "#93C5FD",
    "trad":      "#60A5FA",
    "zero":      "#FBBF24",
    "sft":       "#B91C1C",
}


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": COLOR["edge"],
        "axes.labelcolor": COLOR["ink"],
        "xtick.color": COLOR["edge"],
        "ytick.color": COLOR["edge"],
        "axes.grid": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "pdf.fonttype": 42,
    })


def _load(name: str) -> Any:
    with open(ANALYSIS_DIR / name) as f:
        return json.load(f)


def _save(fig: plt.Figure, name: str) -> Path:
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = PAPER_FIG_DIR / f"{name}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.relative_to(PROJECT_ROOT)}")
    return out


# ===========================================================================
# fig_mn_radar.pdf — Galtung 8-category Model/News ratio radar
# ===========================================================================

def fig_mn_radar() -> None:
    rows = _load("newsvalue_signals.json")

    GALTUNG_ORDER = [
        "Magnitude", "Good News", "Entertainment", "Power Elite",
        "Conflict", "Relevance", "Bad News", "Surprise",
    ]
    by_name = {r["signal"]: r for r in rows}
    signals = [s for s in GALTUNG_ORDER if s in by_name]
    ratios = np.array([by_name[s]["ratio"] for s in signals])

    n = len(signals)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])
    ratios_closed = np.concatenate([ratios, ratios[:1]])

    fig = plt.figure(figsize=(6.2, 5.8))
    ax = fig.add_subplot(111, polar=True)

    for r in (0.2, 0.4, 0.6, 0.8, 1.0):
        ax.plot(np.linspace(0, 2 * np.pi, 200),
                np.full(200, r),
                color=COLOR["grid"], linewidth=0.8, zorder=1)
    ax.plot(np.linspace(0, 2 * np.pi, 200), np.full(200, 1.0),
            color=COLOR["muted"], linewidth=1.0, linestyle="--",
            zorder=2, label="Perfect alignment (1.0)")

    ax.fill(angles_closed, ratios_closed, color=COLOR["ours"],
            alpha=0.18, zorder=3)
    ax.plot(angles_closed, ratios_closed, color=COLOR["ours"],
            linewidth=2.2, zorder=4, label="Model M/N ratio")
    ax.scatter(angles, ratios, s=55, color=COLOR["ours"],
               edgecolor="white", linewidth=1.2, zorder=5)

    for ang, r, s in zip(angles, ratios, signals):
        offset = 0.08
        ha = "center"
        va = "center"
        if np.cos(ang) > 0.3:
            ha = "left"
        elif np.cos(ang) < -0.3:
            ha = "right"
        ax.text(ang, r + offset, f"{r:.2f}",
                ha=ha, va=va, fontsize=10, fontweight="bold",
                color=COLOR["ours"], zorder=6)

    ax.set_xticks(angles)
    ax.set_xticklabels(signals, fontsize=10.5)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"],
                       fontsize=8, color=COLOR["muted"])
    ax.set_rlabel_position(90)
    ax.spines["polar"].set_color(COLOR["muted"])
    ax.spines["polar"].set_linewidth(0.8)
    ax.grid(False)

    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.10),
              frameon=False, fontsize=9, handlelength=2.0)

    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.08, right=0.88)
    _save(fig, "fig_mn_radar")


# ===========================================================================
# fig_vocabulary.pdf — log-frequency scatter of model vs news vocabulary
# ===========================================================================

def fig_vocabulary() -> None:
    cs = _load("contrastive_signals.json")
    model_top = cs["model_yes_top30"]        # positive LOR = model-favoring
    news_top  = cs["model_no_top30"]         # negative LOR = news-favoring
    jaccard   = cs["jaccard"]

    pts = [(w, m["freq_a"], m["freq_b"], m["lor"])
           for w, m in model_top + news_top]
    words = [p[0] for p in pts]
    fm = np.array([p[1] for p in pts])
    fn = np.array([p[2] for p in pts])
    lor = np.array([p[3] for p in pts])

    xs = np.log10(fm + 1)
    ys = np.log10(fn + 1)
    colors = [COLOR["model"] if l > 0 else COLOR["news"] for l in lor]
    sizes  = 28 + 3.2 * np.abs(lor)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    lim = max(xs.max(), ys.max()) + 0.3

    ax.plot([0, lim], [0, lim], color=COLOR["grid"], linewidth=1.1,
            linestyle="--", zorder=1)

    ax.scatter(xs, ys, s=sizes, c=colors, alpha=0.78,
               edgecolor="white", linewidth=0.8, zorder=3)

    deviations = ys - xs
    order = np.argsort(deviations)
    label_idx = list(order[:6]) + list(order[-6:])
    for i in label_idx:
        is_model = colors[i] == COLOR["model"]
        dx = -0.07 if is_model else 0.07
        ha = "right" if is_model else "left"
        ax.annotate(words[i], (xs[i], ys[i]),
                    xytext=(xs[i] + dx, ys[i]),
                    fontsize=8.8, color=colors[i], fontweight="bold",
                    ha=ha, va="center", zorder=5)

    ax.set_xlabel(r"$\log_{10}$(frequency in model explanations $+ 1$)",
                  fontsize=10)
    ax.set_ylabel(r"$\log_{10}$(frequency in news headlines $+ 1$)",
                  fontsize=10)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid(alpha=0.18, linestyle="-")
    ax.set_axisbelow(True)

    ax.scatter([], [], s=55, c=COLOR["model"], label="Model-favoring word")
    ax.scatter([], [], s=55, c=COLOR["news"],  label="News-favoring word")
    ax.plot([], [], color=COLOR["grid"], linestyle="--", label="Equal usage")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=9)

    ax.text(0.98, 0.03,
            f"Vocabulary overlap (all terms)\nJaccard = {jaccard:.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.5, fontweight="bold", color=COLOR["ink"],
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#FEF3C7",
                      edgecolor="#F59E0B", linewidth=1.0))

    fig.subplots_adjust(top=0.96, bottom=0.11, left=0.11, right=0.97)
    _save(fig, "fig_vocabulary")


# ===========================================================================
# fig_baseline_comparison.pdf — 9-model F1/Acc/MCC dot plot
# ===========================================================================

_SFT_CACHE: dict[str, float] | None = None


def _sft_overall() -> dict[str, float]:
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


def fig_baseline_comparison() -> None:
    rows = _load("extended_baselines.json")
    by_model = {r["model"]: r for r in rows}

    groups = [
        ("Classical ML",  ["XGBoost + TF-IDF", "LightGBM + n-gram",
                            "Logistic Regression + TF-IDF"],    COLOR["trad"]),
        ("Zero-shot LLM", ["LLaMA-3.1-8B zero-shot",
                            "Qwen2.5-7B zero-shot",
                            "GPT-4o-mini zero-shot"],            COLOR["zero"]),
        ("Supervised LLM",["Qwen2.5-7B SFT r32",
                            "LLaMA SFT r32 (Yes/No)"],           COLOR["sft"]),
    ]

    ordered: list[tuple[str, dict, str, bool]] = []
    for _, names, color in groups:
        for name in names:
            if name in by_model:
                r = by_model[name]
                is_best = (name == "LLaMA SFT r32 (Yes/No)")
                ordered.append((name, r, color, is_best))

    labels = [
        n.replace("Logistic Regression + TF-IDF", "LogReg + TF-IDF")
         .replace("LLaMA SFT r32 (Yes/No)",        "LLaMA-3.1-8B SFT (Ours)")
         .replace("Qwen2.5-7B SFT r32",             "Qwen2.5-7B SFT")
        for n, _, _, _ in ordered
    ]
    f1  = np.array([r["f1"]  or 0 for _, r, _, _ in ordered])
    acc = np.array([r["acc"] or 0 for _, r, _, _ in ordered])
    mcc = np.array([r["mcc"] or 0 for _, r, _, _ in ordered])

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    y = np.arange(len(ordered))

    for gi, (_, names, color) in enumerate(groups):
        idxs = [i for i, (n, _, _, _) in enumerate(ordered) if n in names]
        if idxs:
            ax.axhspan(min(idxs) - 0.48, max(idxs) + 0.48,
                       color=color, alpha=0.07, zorder=0)

    for yi, (_, _, color, is_best) in zip(y, ordered):
        ax.plot([f1[yi], acc[yi], mcc[yi]], [yi, yi, yi],
                color="#CBD5E1", linewidth=1.5, zorder=1)
        if is_best:
            ax.scatter([f1[yi]], [yi], s=320, facecolor="none",
                       edgecolor=color, linewidth=1.8, zorder=2)

    ax.scatter(f1,  y, s=110, color=[o[2] for o in ordered],
               edgecolor="white", linewidth=1.0, zorder=3, label="F1")
    ax.scatter(acc, y, s=80, facecolor="white",
               edgecolor=[o[2] for o in ordered], linewidth=1.6,
               zorder=3, label="Accuracy")
    ax.scatter(mcc, y, s=80, marker="s", color=[o[2] for o in ordered],
               alpha=0.5, edgecolor="white", linewidth=0.8,
               zorder=3, label="MCC")

    for yi in y:
        ax.text(max(f1[yi], acc[yi]) + 0.015, yi,
                f"F1={f1[yi]:.3f}", va="center",
                fontsize=8.8, color=COLOR["ink"])

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Metric value")
    ax.legend(loc="lower right", frameon=False, fontsize=9,
              ncol=3, bbox_to_anchor=(1.0, -0.14))
    ax.grid(axis="x", alpha=0.18)
    ax.grid(axis="y", visible=False)
    for ref in [0.5, 0.7, 0.8]:
        ax.axvline(ref, color="#E5E7EB", linewidth=0.7, zorder=0)

    fig.subplots_adjust(top=0.97, left=0.34, right=0.97, bottom=0.14)
    _save(fig, "fig_baseline_comparison")


# ===========================================================================
# fig_per_field.pdf — per-field F1 with MCC dumbbell
# ===========================================================================

def fig_per_field() -> None:
    rows = _load("per_field_metrics.json")
    rows = [r for r in rows if r.get("n", 0) >= 100]
    rows.sort(key=lambda r: r["f1"], reverse=True)

    short = lambda f: (
        f.replace("Biochemistry, Genetics and Molecular Biology",
                  "Biochem. & Mol. Bio.")
         .replace("Economics, Econometrics and Finance",
                  "Economics & Finance")
         .replace("Pharmacology, Toxicology and Pharmaceutics",
                  "Pharmacology")
         .replace("Business, Management and Accounting",
                  "Business & Mgmt")
         .replace("Agricultural and Biological Sciences",
                  "Agric. & Bio. Sci.")
         .replace("Earth and Planetary Sciences",
                  "Earth & Planetary Sci.")
         .replace("Immunology and Microbiology",
                  "Immunology & Micro.")
         .replace("Physics and Astronomy",
                  "Physics & Astronomy")
         .replace("Health Professions",
                  "Health Professions")
         .replace(" and ", " & ")
    )
    names = [short(r["field"]) for r in rows]
    f1  = np.array([r["f1"]  for r in rows])
    mcc = np.array([r["mcc"] for r in rows])
    n   = np.array([r["n"]   for r in rows])

    fig, ax = plt.subplots(figsize=(7.8, 0.28 * len(rows) + 1.2))
    y = np.arange(len(rows))

    for yi, a, b in zip(y, f1, mcc):
        ax.plot([b, a], [yi, yi], color="#CBD5E1", linewidth=1.8, zorder=1)
    ax.scatter(f1,  y, s=95,  color=COLOR["model"],
               edgecolor="white", linewidth=0.9, zorder=3, label="F1")
    ax.scatter(mcc, y, s=75, color="#9CA3AF",
               edgecolor="white", linewidth=0.9, zorder=3, label="MCC")

    ax.axvline(0.77, color="#F59E0B", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.text(0.77, -1.1, "F1 = 0.77", fontsize=8.5, color="#B45309",
            ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="#FCD34D"))

    for yi, a in zip(y, f1):
        ax.text(a + 0.012, yi, f"{a:.3f}",
                va="center", fontsize=8.4, color=COLOR["ink"])

    ax.set_yticks(y)
    ax.set_yticklabels([f"{nm}  (n={ni:,})" for nm, ni in zip(names, n)],
                        fontsize=8.8)
    ax.set_xlim(0.40, 1.00)
    ax.set_xticks(np.arange(0.4, 1.01, 0.1))
    ax.set_ylim(len(rows) - 0.5, -1.8)
    ax.set_xlabel("Metric value")
    ax.legend(loc="lower right", frameon=False, fontsize=9,
              bbox_to_anchor=(1.0, -0.06))
    ax.grid(axis="x", alpha=0.18)
    ax.grid(axis="y", visible=False)

    fig.subplots_adjust(top=0.98, left=0.36, right=0.97, bottom=0.07)
    _save(fig, "fig_per_field")


# ===========================================================================
# fig_error_by_field.pdf — FPR vs FNR scatter, per field
# ===========================================================================

def fig_error_by_field() -> None:
    preds = _load("test_predictions_with_explanations.json")
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for p in preds:
        y, yh = p["true_label"], p["predicted"]
        if   y == 1 and yh == 1: agg[p["field"]][0] += 1
        elif y == 0 and yh == 0: agg[p["field"]][1] += 1
        elif y == 0 and yh == 1: agg[p["field"]][2] += 1
        else:                     agg[p["field"]][3] += 1

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
        f.replace("Biochemistry, Genetics and Molecular Biology", "Biochem")
         .replace("Economics, Econometrics and Finance",          "Economics")
         .replace("Pharmacology, Toxicology and Pharmaceutics",   "Pharmacology")
         .replace("Business, Management and Accounting",          "Business")
         .replace("Agricultural and Biological Sciences",         "Agric/Bio")
         .replace("Earth and Planetary Sciences",                 "Earth Sci")
         .replace("Immunology and Microbiology",                  "Immunology")
         .replace("Physics and Astronomy",                        "Physics")
         .replace("Health Professions",                           "Health Prof.")
         .replace("Environmental Science",                        "Environ. Sci")
         .replace("Materials Science",                            "Materials")
         .replace("Computer Science",                             "CS")
         .replace("Social Sciences",                              "Soc. Sci")
         .replace("Decision Sciences",                            "Decision Sci")
         .replace("Arts and Humanities",                          "Arts/Hum.")
         .replace(" and ", " & ")
    )
    fpr = np.array([r["fpr"] for r in rows])
    fnr = np.array([r["fnr"] for r in rows])
    n_v = np.array([r["n"]   for r in rows])
    names = [short(r["field"]) for r in rows]
    sizes = np.clip(np.log10(n_v) * 130 - 140, 60, 700)

    FP_all = sum(v[2] for v in agg.values()); TN_all = sum(v[1] for v in agg.values())
    FN_all = sum(v[3] for v in agg.values()); TP_all = sum(v[0] for v in agg.values())
    ov_fpr = FP_all / (FP_all + TN_all)
    ov_fnr = FN_all / (FN_all + TP_all)

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    lim = max(fpr.max(), fnr.max()) * 1.15

    ax.plot([0, lim], [0, lim], color="#CBD5E1", linestyle="--",
            linewidth=1.0, zorder=1, label="balanced (FPR = FNR)")
    ax.axvline(ov_fpr, color="#94A3B8", linewidth=0.7, alpha=0.6)
    ax.axhline(ov_fnr, color="#94A3B8", linewidth=0.7, alpha=0.6)
    ax.text(ov_fpr + 0.003, lim * 0.97,
            f"overall FPR = {ov_fpr:.2f}",
            fontsize=8, color="#64748B", va="top")
    ax.text(lim * 0.98, ov_fnr - 0.006,
            f"overall FNR = {ov_fnr:.2f}",
            fontsize=8, color="#64748B", ha="right", va="top")

    colors = [COLOR["bad"] if fn_i > fp_i else COLOR["warn"]
              for fp_i, fn_i in zip(fpr, fnr)]
    ax.scatter(fpr, fnr, s=sizes, c=colors, alpha=0.55,
               edgecolor="white", linewidth=0.9, zorder=3)

    try:
        from adjustText import adjust_text
        texts = [ax.text(fp_i, fn_i, name, fontsize=7.8,
                         color=COLOR["ink"], zorder=5)
                 for fp_i, fn_i, name in zip(fpr, fnr, names)]
        adjust_text(
            texts, ax=ax,
            expand_points=(1.25, 1.35),
            expand_text=(1.05, 1.12),
            force_points=0.4,
            force_text=0.5,
            arrowprops=dict(arrowstyle="-", color="#9CA3AF",
                            lw=0.45, alpha=0.55),
        )
    except ImportError:
        for fp_i, fn_i, name in zip(fpr, fnr, names):
            ax.annotate(name, (fp_i, fn_i), fontsize=7.5,
                        ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points",
                        color=COLOR["ink"])

    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("FPR (false-alarm rate on true-negatives)", fontsize=9.5)
    ax.set_ylabel("FNR (miss rate on true-positives)", fontsize=9.5)
    ax.grid(alpha=0.18)

    ax.text(0.02, lim * 0.95, "Miss-heavy",
            fontsize=9, color="#B91C1C", fontweight="bold")
    ax.text(lim * 0.98, 0.01, "Alarm-heavy",
            fontsize=9, color="#B45309", fontweight="bold", ha="right")

    for sz, lab in zip([200, 1000, 3000],
                       ["n = 200", "n = 1,000", "n = 3,000"]):
        ax.scatter([], [], s=np.clip(np.log10(sz) * 130 - 140, 60, 700),
                   c="#CBD5E1", alpha=0.55, edgecolor="white", linewidth=0.9,
                   label=lab)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5,
              labelspacing=0.9, borderpad=0.4, handletextpad=0.6)

    fig.subplots_adjust(top=0.97, left=0.12, right=0.97, bottom=0.10)
    _save(fig, "fig_error_by_field")


# ===========================================================================
# fig_signal_analysis.pdf — 12-theme dumbbell + 8-Galtung slope
# ===========================================================================

def fig_signal_analysis() -> None:
    themes = _load("signal_ratios.json")
    themes.sort(key=lambda r: r["ratio"])
    t_names  = [r["signal"] for r in themes]
    t_model  = np.array([r["model_count"] for r in themes], dtype=float)
    t_news   = np.array([r["news_count"]  for r in themes], dtype=float)
    t_ratio  = np.array([r["ratio"]       for r in themes], dtype=float)
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
        1, 2, figsize=(12.6, 5.5),
        gridspec_kw={"width_ratios": [1.1, 1.0], "wspace": 0.35},
    )

    # ---- LEFT: 12-theme dumbbell ----
    y = np.arange(len(t_names))
    for yi, m, n_ in zip(y, mn, nn):
        ax_l.plot([m, n_], [yi, yi], color="#CBD5E1", linewidth=1.6, zorder=1)
    ax_l.scatter(mn, y, s=95, color=COLOR["model"],
                 edgecolor="white", linewidth=0.8, zorder=3,
                 label="Model explanation")
    ax_l.scatter(nn, y, s=95, color=COLOR["news"],
                 edgecolor="white", linewidth=0.8, zorder=3,
                 label="News coverage")

    for yi, m, n_, r in zip(y, mn, nn, t_ratio):
        x_lab = max(m, n_) + 0.03
        color = COLOR["good"] if r >= 1 else COLOR["bad"]
        ax_l.text(x_lab, yi, f"r={r:.2f}", va="center",
                  fontsize=8.5, fontweight="bold", color=color)

    ax_l.set_yticks(y)
    ax_l.set_yticklabels(t_names, fontsize=9)
    ax_l.invert_yaxis()
    ax_l.set_xlim(0, 1.32)
    ax_l.set_xticks(np.arange(0, 1.01, 0.25))
    ax_l.set_xlabel("Normalized presence (per-theme max = 1.0)", fontsize=9.5)
    ax_l.set_title("(a) Custom themes (n = 12)", fontsize=10.5,
                    color=COLOR["ink"], pad=6)
    ax_l.legend(loc="lower right", frameon=False, fontsize=8.5,
                bbox_to_anchor=(1.0, -0.01))
    ax_l.grid(axis="x", alpha=0.18)
    ax_l.grid(axis="y", visible=False)

    # ---- RIGHT: 8-Galtung slope ----
    x_pos = [0, 1, 2]
    x_labels = ["Abstract", "Model", "News"]
    cmap = plt.get_cmap("RdYlGn_r")
    norm_gap = (g_gap - g_gap.min()) / max(1e-9, g_gap.max() - g_gap.min())

    for x in x_pos:
        ax_r.axvline(x, color="#E5E7EB", linewidth=1.0, zorder=0)

    for i, s in enumerate(g_names):
        is_top = i >= len(g_names) - 3
        color = cmap(norm_gap[i])
        ax_r.plot(x_pos, [g_abs[i], g_mod[i], g_news[i]],
                  color=color,
                  linewidth=2.4 if is_top else 1.2,
                  alpha=0.95 if is_top else 0.55,
                  marker="o", markersize=7 if is_top else 4,
                  markeredgecolor="white", markeredgewidth=0.8,
                  zorder=4 if is_top else 2)

    label_order = sorted(range(len(g_names)), key=lambda i: -g_news[i])
    used_y: list[float] = []
    MIN_GAP = 5.5
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
        label = f"{g_names[i]}  +{g_gap[i]:.0f}pp" if is_top else g_names[i]
        ax_r.text(2.05, ly, " " + label, va="center",
                  fontsize=9 if is_top else 8.5,
                  fontweight="bold" if is_top else "normal",
                  color=color if is_top else COLOR["muted"])

    ax_r.set_xticks(x_pos)
    ax_r.set_xticklabels(x_labels, fontsize=10, fontweight="bold")
    ax_r.set_xlim(-0.12, 2.95)
    ax_r.set_ylim(0, 100)
    ax_r.set_ylabel("% of items mentioning signal", fontsize=9.5)
    ax_r.set_title("(b) Galtung news values (n = 8)", fontsize=10.5,
                    color=COLOR["ink"], pad=6)
    ax_r.grid(axis="y", alpha=0.18)
    ax_r.grid(axis="x", visible=False)

    fig.subplots_adjust(top=0.94, left=0.19, right=0.94, bottom=0.10)
    _save(fig, "fig_signal_analysis")


# ===========================================================================
# fig_discriminative_keywords.pdf — dumbbell, gold vs predicted
# ===========================================================================

def fig_discriminative_keywords() -> None:
    mb = _load("model_behavior.json")
    GOLD = "#16A34A"
    PRED = "#F59E0B"

    def _panel(ax, true_list, pred_list, title, x_sign):
        true_d = {w: info for w, info in true_list}
        pred_d = {w: info for w, info in pred_list}
        all_words = set(true_d) | set(pred_d)
        word_max = {
            w: max(abs(true_d.get(w, {}).get("z", 0)),
                   abs(pred_d.get(w, {}).get("z", 0)))
            for w in all_words
        }
        top = sorted(all_words, key=lambda w: -word_max[w])[:20]
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
                        linewidth=1.4, zorder=1)
            if tz is not None:
                ax.scatter(tz, yi, s=95 if both else 70,
                           color=GOLD,
                           edgecolor="white", linewidth=0.9, zorder=3)
            if pz is not None:
                ax.scatter(pz, yi, s=95 if both else 70,
                           color=PRED, marker="s",
                           edgecolor="white", linewidth=0.9, zorder=3)

            label_color = COLOR["ink"] if both else COLOR["muted"]
            label_weight = "bold" if both else "normal"
            ax.text(-0.01 if x_sign > 0 else 0.01, yi, word,
                    transform=ax.get_yaxis_transform(),
                    fontsize=9, ha="right" if x_sign > 0 else "left",
                    va="center", color=label_color, fontweight=label_weight)

        ax.set_yticks(y)
        ax.set_yticklabels(["" for _ in top])
        ax.invert_yaxis()
        ax.set_title(f"{title}   ({n_agree}/{len(top)} shared)",
                     fontsize=10.5, color=COLOR["ink"], pad=8)
        ax.grid(axis="x", alpha=0.18)
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color="#9CA3AF", linewidth=0.7)
        if x_sign < 0:
            ax.invert_xaxis()
        ax.set_xlabel(f"z-score  ({'positive' if x_sign>0 else 'negative'})",
                      fontsize=9)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.8))
    _panel(axes[0], mb["true_yes_top30"], mb["pred_yes_top30"],
           "YES discriminators", x_sign=+1)
    _panel(axes[1], mb["true_no_top30"], mb["pred_no_top30"],
           "NO discriminators", x_sign=-1)

    legend_handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=GOLD,
                   markeredgecolor="white", markersize=9,
                   label="Gold (true labels)"),
        plt.Line2D([], [], marker="s", linestyle="", color=PRED,
                   markeredgecolor="white", markersize=9,
                   label="Predicted (model)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 0.005), ncol=2)

    fig.subplots_adjust(top=0.94, left=0.08, right=0.97, bottom=0.12,
                        wspace=0.38)
    _save(fig, "fig_discriminative_keywords")


# ===========================================================================
# fig_bertopic.pdf — BERTopic panels, corpus-share
# ===========================================================================

def fig_bertopic() -> None:
    # Prefer v2 (extended-stopword) results if present
    v2 = ANALYSIS_DIR / "bertopic_results_v2.json"
    src = "bertopic_results_v2.json" if v2.exists() else "bertopic_results.json"
    bt = _load(src)

    def _clean_topic(t):
        # filter artefact topics whose first word is the single literal "yes"
        # (arises from rationales that start with "Yes, ..."—not a real theme)
        if not t.get("words"):
            return False
        first = t["words"][0].strip().lower()
        return first not in {"yes", "no", ""}

    panels = [
        ("Model YES topics",
         [t for t in bt.get("yes_topics", []) if _clean_topic(t)],
         COLOR["model"]),
        ("Model NO topics",
         [t for t in bt.get("no_topics", []) if _clean_topic(t)],
         "#7C3AED"),
        ("News topics",
         [t for t in bt.get("news_topics", []) if _clean_topic(t)],
         COLOR["news"]),
    ]
    # Percentages relative to that panel's own kept-topics sum
    for _, topics, _ in panels:
        total = sum(t.get("count", 0) for t in topics)
        for t in topics:
            t["pct"] = 100.0 * t["count"] / total if total else 0
    max_pct = max((t.get("pct", 0)
                   for _, topics, _ in panels for t in topics), default=10)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5),
                              sharex=True,
                              gridspec_kw={"wspace": 0.55})
    for ax, (title, topics, color) in zip(axes, panels):
        if not topics:
            ax.set_visible(False); continue
        ts = sorted(topics, key=lambda t: t.get("count", 0), reverse=True)[:8]
        labels = [f"T{t['id']}: " + ", ".join(t["words"][:3]) for t in ts]
        pcts   = [t["pct"] for t in ts]
        counts = [t["count"] for t in ts]
        y = np.arange(len(ts))
        ax.barh(y, pcts[::-1], color=color, alpha=0.85,
                edgecolor="white", linewidth=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(labels[::-1], fontsize=8.6)
        for yi, p, c in zip(y, pcts[::-1], counts[::-1]):
            ax.text(p + max_pct * 0.012, yi, f"{p:.1f}%",
                    va="center", fontsize=8.2, color=COLOR["ink"])
        ax.set_title(title, fontsize=10.5, color=color, fontweight="bold",
                     pad=8)
        ax.set_xlim(0, max_pct * 1.25)
        ax.set_xlabel("% of corpus", fontsize=9)
        ax.grid(axis="x", alpha=0.18)
        ax.grid(axis="y", visible=False)

    fig.subplots_adjust(top=0.92, left=0.06, right=0.99, bottom=0.10)
    _save(fig, "fig_bertopic")


# ===========================================================================
# fig_diagnostic_summary.pdf — three-panel synthesis (category/topic/word)
# ===========================================================================

def fig_diagnostic_summary() -> None:
    GALTUNG_ORDER = [
        "Entertainment", "Good News", "Magnitude", "Surprise",
        "Bad News", "Relevance", "Conflict", "Power Elite",
    ]
    rows = _load("newsvalue_signals.json")
    by_name = {r["signal"]: r for r in rows}
    cat_names  = [s for s in GALTUNG_ORDER if s in by_name][::-1]
    cat_ratios = [by_name[s]["ratio"] for s in cat_names]

    def _cat_color(r: float) -> str:
        if r >= 0.60: return COLOR["good"]
        if r >= 0.40: return COLOR["warn"]
        return COLOR["bad"]
    cat_colors = [_cat_color(r) for r in cat_ratios]

    cs = _load("contrastive_signals.json")
    word_vs_news = cs["jaccard"]
    word_vs_gold_yes = 0.492
    word_vs_gold_no  = 0.591

    fig = plt.figure(figsize=(12.4, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.00], wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    y_pos = np.arange(len(cat_names))
    ax1.barh(y_pos, cat_ratios, color=cat_colors, alpha=0.88,
             edgecolor="white", linewidth=0.6)
    ax1.axvline(1.0, color=COLOR["muted"], linestyle="--",
                linewidth=1.0, zorder=1)
    for yi, r in zip(y_pos, cat_ratios):
        ax1.text(r + 0.018, yi, f"{r:.2f}", va="center",
                 fontsize=9.2, color=COLOR["ink"], fontweight="bold")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(cat_names, fontsize=10)
    ax1.set_xlim(0, 1.15)
    ax1.set_xlabel("Model / News rate ratio (M/N)", fontsize=10)
    ax1.set_title("A. News-value categories (8 Galtung dimensions)",
                  fontsize=10.8, color=COLOR["ink"], pad=8)
    ax1.grid(axis="x", alpha=0.2, linestyle="-")
    ax1.set_axisbelow(True)

    h_good = plt.Rectangle((0, 0), 1, 1, color=COLOR["good"], alpha=0.88)
    h_warn = plt.Rectangle((0, 0), 1, 1, color=COLOR["warn"], alpha=0.88)
    h_bad  = plt.Rectangle((0, 0), 1, 1, color=COLOR["bad"],  alpha=0.88)
    ax1.legend(
        [h_good, h_warn, h_bad],
        ["M/N $\\geq$ 0.60",
         "0.40 $\\leq$ M/N $<$ 0.60",
         "M/N $<$ 0.40"],
        loc="lower right", frameon=True, framealpha=0.95,
        fontsize=8.5, handlelength=1.2,
    )

    ax3 = fig.add_subplot(gs[0, 1])
    labels3 = [
        "vs journalist\nvocabulary",
        "vs gold labels\n(YES signals)",
        "vs gold labels\n(NO signals)",
    ]
    values3 = [word_vs_news, word_vs_gold_yes, word_vs_gold_no]
    colors3 = [COLOR["news"], COLOR["good"], COLOR["model"]]
    x3 = np.arange(len(labels3))
    ax3.bar(x3, values3, color=colors3, alpha=0.88,
            edgecolor="white", linewidth=0.6, width=0.6)
    for xi, v in zip(x3, values3):
        ax3.text(xi, v + 0.015, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=10,
                 fontweight="bold", color=COLOR["ink"])
    ax3.set_xticks(x3)
    ax3.set_xticklabels(labels3, fontsize=9.2)
    ax3.set_ylim(0, 0.75)
    ax3.set_ylabel("Jaccard overlap", fontsize=10)
    ax3.set_title("B. Vocabulary (word level)",
                  fontsize=10.8, color=COLOR["ink"], pad=8)
    ax3.grid(axis="y", alpha=0.2, linestyle="-")
    ax3.set_axisbelow(True)

    fig.subplots_adjust(top=0.90, bottom=0.17, left=0.10, right=0.97)
    _save(fig, "fig_diagnostic_summary")


# ===========================================================================
# fig_ceiling.pdf — sorted M/N bars with external-information annotations
# ===========================================================================

def fig_ceiling() -> None:
    data = _load("newsvalue_signals.json")
    data = sorted(data, key=lambda r: r["ratio"])
    names  = [r["signal"] for r in data]
    ratios = [r["ratio"] for r in data]

    # Per-category palette, consistent with fig_category_vocabulary raincloud
    CATEGORY_COLOR = {
        "Surprise":      "#0072B2",
        "Bad News":      "#D55E00",
        "Good News":     "#009E73",
        "Magnitude":     "#CC79A7",
        "Relevance":     "#56B4E9",
        "Power Elite":   "#E69F00",
        "Entertainment": "#7B3FA5",
        "Conflict":      "#B22222",
    }
    colors = [CATEGORY_COLOR.get(n, COLOR["muted"]) for n in names]

    # ---- waffle grid: 20 cells per row, each = 5% ----
    N_CELLS = 20
    cell_w, cell_h = 0.9, 0.72
    gap = 0.08          # horizontal gap between cells
    row_gap = 0.5       # vertical gap between category rows

    n_rows = len(names)
    fig_h = max(3.0, 0.72 * n_rows + 1.2)
    fig, ax = plt.subplots(figsize=(11.5, fig_h))

    for yi, (r, name, col) in enumerate(zip(ratios, names, colors)):
        filled = int(round(r * N_CELLS))  # number of filled cells
        for ci in range(N_CELLS):
            x = ci * (cell_w + gap)
            y = yi * (cell_h + row_gap)
            is_filled = ci < filled
            ax.add_patch(plt.Rectangle(
                (x, y), cell_w, cell_h,
                facecolor=col if is_filled else "#F3F4F6",
                edgecolor="white",
                linewidth=1.0,
                zorder=2,
            ))

        # category label (left)
        ax.text(-0.3, yi * (cell_h + row_gap) + cell_h / 2, name,
                ha="right", va="center", fontsize=10.2, color=COLOR["ink"])

        # M/N value (right)
        end_x = N_CELLS * (cell_w + gap) + 0.3
        ax.text(end_x, yi * (cell_h + row_gap) + cell_h / 2,
                f"{r:.2f}",
                ha="left", va="center",
                fontsize=10.5, fontweight="bold",
                color=col)

    # axes cosmetics
    total_w = N_CELLS * (cell_w + gap)
    total_h = n_rows * (cell_h + row_gap)
    ax.set_xlim(-5.0, total_w + 2.5)
    ax.set_ylim(-0.4, total_h + 0.1)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    # x-axis-like tick strip beneath the waffle (0%, 50%, 100%)
    base_y = total_h - 0.05
    for pct in (0, 0.25, 0.5, 0.75, 1.0):
        xpos = pct * N_CELLS * (cell_w + gap)
        ax.plot([xpos, xpos], [base_y, base_y + 0.12],
                color="#9CA3AF", linewidth=0.6)
        ax.text(xpos, base_y + 0.22, f"{pct:.2f}",
                ha="center", va="top",
                fontsize=8.5, color="#6B7280")
    ax.text(total_w / 2, base_y + 0.75,
            "Model / News rate ratio (M/N)",
            ha="center", va="top", fontsize=10, color=COLOR["ink"])

    # Tier boundary markers at 0.40 and 0.60 (beneath the waffle)
    def _tier_marker(pct, label, color):
        xpos = pct * N_CELLS * (cell_w + gap)
        ax.plot([xpos, xpos], [-0.55, base_y], color=color,
                linewidth=0.8, linestyle=":", alpha=0.6, zorder=1)
        ax.text(xpos, -0.75, label, ha="center", va="top",
                fontsize=8.5, color=color, style="italic")
    _tier_marker(0.40, "partial", "#9CA3AF")
    _tier_marker(0.60, "recoverable", "#9CA3AF")

    fig.subplots_adjust(top=0.97, bottom=0.10, left=0.14, right=0.98)
    _save(fig, "fig_ceiling")


# ===========================================================================
# Entry
# ===========================================================================

REGISTRY = {
    "baseline_comparison":     fig_baseline_comparison,
    "per_field":               fig_per_field,
    "error_by_field":          fig_error_by_field,
    "signal_analysis":         fig_signal_analysis,
    "discriminative_keywords": fig_discriminative_keywords,
    "bertopic":                fig_bertopic,
    "mn_radar":                fig_mn_radar,
    "vocabulary":              fig_vocabulary,
    "diagnostic_summary":      fig_diagnostic_summary,
    "ceiling":                 fig_ceiling,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=list(REGISTRY),
                        default=None,
                        help="Render only named figures (default: all).")
    args = parser.parse_args()

    _style()
    targets = args.only or list(REGISTRY)
    print(f"Rendering {len(targets)} PDF(s) → "
          f"{PAPER_FIG_DIR.relative_to(PROJECT_ROOT)}/")
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
