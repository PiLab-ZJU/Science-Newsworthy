"""
§¶2 figure: Gold-label vs model-predicted lexical discriminators.

Each word is plotted at ( gold_z, pred_z ) where
  gold_z = z-score from LOR(true_YES abstracts vs true_NO abstracts)
  pred_z = z-score from LOR(predicted_YES abstracts vs predicted_NO abstracts)

What each region tells the reader:
  diagonal y=x                — model perfectly matches gold
  upper-right cluster         — shared YES discriminators (model learned)
  lower-left  cluster         — shared NO discriminators  (model learned)
  near y-axis, far from x=0   — MODEL-ONLY (model added; gold neutral)
  near x-axis, far from y=0   — GOLD-ONLY  (model missed; gold signals)

A thin side panel lists the top Model-only and Gold-only words so the
"health/behavior bias + ecology/neuroscience omission" story is visible.
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROJECT_ROOT

ANALYSIS_DIR = PROJECT_ROOT / "analysis"
PAPER_FIG_DIR = PROJECT_ROOT / "paper-workflow" / "latex-temple" / "figures"
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

PRED_PATH = ANALYSIS_DIR / "test_predictions_with_explanations.json"
TEST_PATH = Path("/Volumes/Lin_SSD/lcx/academic_new_policy/data/processed/"
                 "combined/test.json")

Z_THRESH = 3.0
MIN_FREQ = 10

STOPWORDS = {
    "the","a","an","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","can","shall",
    "i","we","they","he","she","it","you","me","him","her","this","that","these",
    "those","my","your","his","its","our","of","in","to","for","with","on","at",
    "by","from","as","into","about","between","through","during","before","after",
    "and","or","but","not","if","than","so","because","while","also","just","very",
    "more","most","only","even","still","such","each","both","all","any","some",
    "no","other","what","which","who","whom","how","when","where","why","there",
    "here","then","now","well","however","although","whether","since","until",
    "using","used","based","show","shown","found","results","method","approach",
}


def tokenize(text: str):
    return [w for w in re.findall(r"\b[a-z]{3,}\b", text.lower())
            if w not in STOPWORDS]


def log_odds(cnt_a, cnt_b, n_a, n_b, min_freq=MIN_FREQ):
    """Laplace-smoothed log-odds + z approximation."""
    out = {}
    for w in set(cnt_a) | set(cnt_b):
        a = cnt_a.get(w, 0)
        b = cnt_b.get(w, 0)
        if a + b < min_freq:
            continue
        fa = a + 1
        fb = b + 1
        lor = math.log2((fa / (n_a - fa + 1)) / (fb / (n_b - fb + 1)))
        z = lor / math.sqrt(1 / fa + 1 / fb)
        out[w] = {"z": z, "a": a, "b": b}
    return out


def doc_freq(texts):
    c = Counter()
    for t in texts:
        c.update(set(tokenize(t)))
    return c


def main():
    # ----- load data -----
    preds = json.loads(PRED_PATH.read_text())
    test = json.loads(TEST_PATH.read_text())
    doi_abs = {d["doi"]: d.get("abstract", "") for d in test}

    # attach abstracts
    for p in preds:
        p["abstract"] = doi_abs.get(p.get("doi", ""), "")
    preds = [p for p in preds if p["abstract"]]
    print(f"preds with abstract: {len(preds)}")

    true_yes_abs = [p["abstract"] for p in preds if p["true_label"] == 1]
    true_no_abs  = [p["abstract"] for p in preds if p["true_label"] == 0]
    pred_yes_abs = [p["abstract"] for p in preds if p["predicted"] == 1]
    pred_no_abs  = [p["abstract"] for p in preds if p["predicted"] == 0]
    print(f"true YES/NO = {len(true_yes_abs)}/{len(true_no_abs)}")
    print(f"pred YES/NO = {len(pred_yes_abs)}/{len(pred_no_abs)}")

    gold_lor = log_odds(doc_freq(true_yes_abs), doc_freq(true_no_abs),
                        len(true_yes_abs), len(true_no_abs))
    pred_lor = log_odds(doc_freq(pred_yes_abs), doc_freq(pred_no_abs),
                        len(pred_yes_abs), len(pred_no_abs))
    print(f"|gold vocab| = {len(gold_lor)}   |pred vocab| = {len(pred_lor)}")

    # ----- build (gold_z, pred_z) table -----
    all_words = set(gold_lor) | set(pred_lor)
    points = []
    bg_points = []   # sub-threshold background points
    for w in all_words:
        gz = gold_lor.get(w, {}).get("z", 0.0)
        pz = pred_lor.get(w, {}).get("z", 0.0)
        if abs(gz) < Z_THRESH and abs(pz) < Z_THRESH:
            bg_points.append({"word": w, "gz": gz, "pz": pz})
            continue
        in_gold = abs(gz) >= Z_THRESH
        in_pred = abs(pz) >= Z_THRESH
        category = None
        if in_gold and in_pred:
            sign_match = (gz > 0) == (pz > 0)
            if not sign_match:
                category = "disagree"
            elif gz > 0:
                category = "shared_yes"
            else:
                category = "shared_no"
        elif in_pred:
            category = "pred_only_yes" if pz > 0 else "pred_only_no"
        else:
            category = "gold_only_yes" if gz > 0 else "gold_only_no"
        points.append({"word": w, "gz": gz, "pz": pz, "cat": category})
    print(f"|significant points| = {len(points)}   "
          f"|background points| = {len(bg_points)}")

    # Jaccards
    gold_yes = {w for w in gold_lor if gold_lor[w]["z"] >= Z_THRESH}
    pred_yes = {w for w in pred_lor if pred_lor[w]["z"] >= Z_THRESH}
    gold_no  = {w for w in gold_lor if gold_lor[w]["z"] <= -Z_THRESH}
    pred_no  = {w for w in pred_lor if pred_lor[w]["z"] <= -Z_THRESH}
    J_yes = len(gold_yes & pred_yes) / max(len(gold_yes | pred_yes), 1)
    J_no  = len(gold_no  & pred_no)  / max(len(gold_no  | pred_no),  1)
    print(f"Jaccard_YES = {J_yes:.3f}   Jaccard_NO = {J_no:.3f}")

    # ----- plot -----
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#374151",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })
    COLOR = {
        "shared_yes":    "#16A34A",
        "shared_no":     "#16A34A",
        "pred_only_yes": "#D97706",
        "pred_only_no":  "#D97706",
        "gold_only_yes": "#2563EB",
        "gold_only_no":  "#2563EB",
        "disagree":      "#9CA3AF",
    }

    fig, ax = plt.subplots(figsize=(10.5, 8.5))

    # axis range
    all_gz = np.array([p["gz"] for p in points])
    all_pz = np.array([p["pz"] for p in points])
    lim = max(abs(all_gz).max(), abs(all_pz).max()) * 1.08

    # Subtle shaded square for the non-discriminative zone (replaces 4 dashed lines)
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle(
        (-Z_THRESH, -Z_THRESH), 2 * Z_THRESH, 2 * Z_THRESH,
        facecolor="#F3F4F6", edgecolor="none", alpha=0.7, zorder=1,
    ))
    # a single thin y=x reference
    ax.plot([-lim, lim], [-lim, lim], color="#D1D5DB",
            linewidth=0.8, linestyle="--", alpha=0.8, zorder=1)
    # axis crosshairs (very light)
    ax.axhline(0, color="#E5E7EB", linewidth=0.5, zorder=1)
    ax.axvline(0, color="#E5E7EB", linewidth=0.5, zorder=1)

    # Background: sub-threshold words (|z|<3 on both axes), faint grey dots
    if bg_points:
        bg_gz = np.array([p["gz"] for p in bg_points])
        bg_pz = np.array([p["pz"] for p in bg_points])
        ax.scatter(bg_gz, bg_pz, s=2.5, c="#9CA3AF",
                   alpha=0.22, edgecolor="none", zorder=2)
        ax.text(0, 0, "non-discriminative\n($|z|<3$)",
                ha="center", va="center",
                fontsize=9, color="#6B7280", style="italic", zorder=2)

    # scatter each category
    draw_order = ["disagree", "shared_no", "shared_yes",
                  "pred_only_no", "pred_only_yes",
                  "gold_only_no", "gold_only_yes"]
    for cat in draw_order:
        pts = [p for p in points if p["cat"] == cat]
        if not pts:
            continue
        gz = [p["gz"] for p in pts]
        pz = [p["pz"] for p in pts]
        ax.scatter(gz, pz, s=22, c=COLOR[cat],
                   alpha=0.55, edgecolor="white", linewidth=0.35,
                   zorder=3)

    # --- selective labels ---
    def _rank_key(p, axis):
        # max deviation from diagonal toward the axis of interest
        if axis == "pred_only":
            # high |pz| but small |gz|
            return abs(p["pz"]) - abs(p["gz"])
        if axis == "gold_only":
            return abs(p["gz"]) - abs(p["pz"])
        if axis == "shared":
            return min(abs(p["gz"]), abs(p["pz"]))
        return 0

    label_plan = [
        ("shared_yes",    "shared",    4, COLOR["shared_yes"]),
        ("shared_no",     "shared",    4, COLOR["shared_no"]),
        ("pred_only_yes", "pred_only", 4, COLOR["pred_only_yes"]),
        ("pred_only_no",  "pred_only", 3, COLOR["pred_only_no"]),
        ("gold_only_yes", "gold_only", 4, COLOR["gold_only_yes"]),
        ("gold_only_no",  "gold_only", 3, COLOR["gold_only_no"]),
    ]
    labelled = []
    for cat, axis, n_take, color in label_plan:
        pts = sorted([p for p in points if p["cat"] == cat],
                     key=lambda x: -_rank_key(x, axis))[:n_take]
        for p in pts:
            labelled.append((p, color))

    try:
        from adjustText import adjust_text
        texts = [ax.text(p["gz"], p["pz"], p["word"],
                         fontsize=9, color=col, fontweight="bold", zorder=6)
                 for p, col in labelled]
        adjust_text(texts, ax=ax,
                    expand_points=(1.15, 1.2),
                    expand_text=(1.05, 1.1),
                    force_points=0.35,
                    force_text=0.4,
                    arrowprops=dict(arrowstyle="-", color="#9CA3AF",
                                    lw=0.45, alpha=0.5))
    except ImportError:
        for p, col in labelled:
            ax.annotate(p["word"], (p["gz"], p["pz"]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=9, color=col, fontweight="bold", zorder=6)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"Gold $z$-score  "
                  r"(LOR$_\mathrm{true\ YES\ vs\ NO}$)  $\longrightarrow$  more NEWS-covered",
                  fontsize=11, color="#111827")
    ax.set_ylabel(r"Model-predicted $z$-score  "
                  r"(LOR$_\mathrm{pred\ YES\ vs\ NO}$)  $\longrightarrow$  more YES-predicted",
                  fontsize=11, color="#111827")

    # legend
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=COLOR["shared_yes"], alpha=0.7,
              label=f"Shared  (Jaccard$_\\mathrm{{YES}}$={J_yes:.2f}, "
                    f"Jaccard$_\\mathrm{{NO}}$={J_no:.2f})"),
        Patch(facecolor=COLOR["pred_only_yes"], alpha=0.7,
              label="Model-only  (added; model bias)"),
        Patch(facecolor=COLOR["gold_only_yes"], alpha=0.7,
              label="Gold-only  (missed by model)"),
        Patch(facecolor=COLOR["disagree"], alpha=0.7,
              label="Sign disagreement"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True,
              framealpha=0.95, fontsize=10)

    ax.grid(alpha=0.1, linestyle="--", linewidth=0.4)
    ax.set_axisbelow(True)

    fig.subplots_adjust(top=0.97, bottom=0.09, left=0.09, right=0.97)
    out = PAPER_FIG_DIR / "fig_discriminative_scatter.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"\nsaved {out}")

    # dump side-panel word lists for paper text
    def _top_words(cat, axis, n=15):
        pts = sorted([p for p in points if p["cat"] == cat],
                     key=lambda x: -_rank_key(x, axis))
        return [p["word"] for p in pts[:n]]

    side = {
        "jaccard_yes": float(J_yes),
        "jaccard_no":  float(J_no),
        "n_points":    len(points),
        "counts": {c: sum(1 for p in points if p["cat"] == c)
                   for c in set(p["cat"] for p in points)},
        "model_only_yes_top15": _top_words("pred_only_yes", "pred_only"),
        "model_only_no_top15":  _top_words("pred_only_no",  "pred_only"),
        "gold_only_yes_top15":  _top_words("gold_only_yes", "gold_only"),
        "gold_only_no_top15":   _top_words("gold_only_no",  "gold_only"),
        "shared_yes_top15":     _top_words("shared_yes",    "shared"),
        "shared_no_top15":      _top_words("shared_no",     "shared"),
    }
    (ANALYSIS_DIR / "discriminative_scatter_side.json").write_text(
        json.dumps(side, indent=2))
    print(f"saved stats: analysis/discriminative_scatter_side.json")


if __name__ == "__main__":
    main()
