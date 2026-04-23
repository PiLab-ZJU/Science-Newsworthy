"""
Two alternative visualisations for §¶2 (model vs gold discriminator
comparison), both foregrounding the *bias* (off-diagonal) rather than
the "everyone agrees" diagonal.

    --method ma      Bland-Altman / MA plot: x = average z, y = pred-gold
    --method tornado Divergent bar chart: pred_z - gold_z per word
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


def log_odds(cnt_a, cnt_b, n_a, n_b, min_freq=10):
    out = {}
    for w in set(cnt_a) | set(cnt_b):
        a = cnt_a.get(w, 0); b = cnt_b.get(w, 0)
        if a + b < min_freq:
            continue
        fa, fb = a + 1, b + 1
        lor = math.log2((fa / (n_a - fa + 1)) / (fb / (n_b - fb + 1)))
        z = lor / math.sqrt(1 / fa + 1 / fb)
        out[w] = {"z": z, "a": a, "b": b}
    return out


def doc_freq(texts):
    c = Counter()
    for t in texts:
        c.update(set(tokenize(t)))
    return c


def compute_scores():
    preds = json.loads(PRED_PATH.read_text())
    test = json.loads(TEST_PATH.read_text())
    doi_abs = {d["doi"]: d.get("abstract", "") for d in test}
    for p in preds:
        p["abstract"] = doi_abs.get(p.get("doi", ""), "")
    preds = [p for p in preds if p["abstract"]]

    true_yes = [p["abstract"] for p in preds if p["true_label"] == 1]
    true_no  = [p["abstract"] for p in preds if p["true_label"] == 0]
    pred_yes = [p["abstract"] for p in preds if p["predicted"] == 1]
    pred_no  = [p["abstract"] for p in preds if p["predicted"] == 0]

    gold = log_odds(doc_freq(true_yes), doc_freq(true_no),
                    len(true_yes), len(true_no))
    pred = log_odds(doc_freq(pred_yes), doc_freq(pred_no),
                    len(pred_yes), len(pred_no))

    rows = []
    for w in set(gold) | set(pred):
        gz = gold.get(w, {}).get("z", 0.0)
        pz = pred.get(w, {}).get("z", 0.0)
        if abs(gz) < 1.5 and abs(pz) < 1.5:
            continue
        rows.append({
            "word": w, "gz": gz, "pz": pz,
            "mean_z": 0.5 * (gz + pz),
            "delta": pz - gz,
            "abs_delta": abs(pz - gz),
        })
    print(f"|words with |z|>=1.5| = {len(rows)}")
    return rows


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 22,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#374151",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })


# =====================================================================
# Method A: MA plot
# =====================================================================

def fig_ma(rows):
    INK   = "#111827"
    MUTED = "#6B7280"
    C_POS = "#D97706"   # model over-emphasizes
    C_NEG = "#2563EB"   # model under-emphasizes
    C_YES = "#16A34A"   # on-diagonal YES signal
    C_NO  = "#9CA3AF"   # on-diagonal low-interest

    fig, ax = plt.subplots(figsize=(16, 10.5))

    xs = np.array([r["mean_z"] for r in rows])
    ys = np.array([r["delta"]  for r in rows])

    # color by deviation magnitude and direction
    colors = []
    for r in rows:
        if abs(r["delta"]) < 2:
            colors.append("#D1D5DB")
        elif r["delta"] > 0:
            colors.append(C_POS)
        else:
            colors.append(C_NEG)

    ax.axhline(0, color="#9CA3AF", linewidth=0.8, linestyle="--", zorder=1)
    ax.axvline(0, color="#E5E7EB", linewidth=0.6, zorder=1)
    ax.scatter(xs, ys, s=18, c=colors, alpha=0.55,
               edgecolor="white", linewidth=0.3, zorder=2)

    # label top-8 in each deviation direction (and both signs)
    sorted_pos_yes = sorted([r for r in rows if r["mean_z"] > 0 and r["delta"] > 2],
                            key=lambda r: -r["delta"])[:8]
    sorted_pos_no  = sorted([r for r in rows if r["mean_z"] < 0 and r["delta"] > 2],
                            key=lambda r: -r["delta"])[:6]
    sorted_neg_yes = sorted([r for r in rows if r["mean_z"] > 0 and r["delta"] < -2],
                            key=lambda r: r["delta"])[:8]
    sorted_neg_no  = sorted([r for r in rows if r["mean_z"] < 0 and r["delta"] < -2],
                            key=lambda r: r["delta"])[:6]
    labelled = sorted_pos_yes + sorted_pos_no + sorted_neg_yes + sorted_neg_no
    try:
        from adjustText import adjust_text
        texts = []
        for r in labelled:
            c = C_POS if r["delta"] > 0 else C_NEG
            t = ax.text(r["mean_z"], r["delta"], r["word"],
                        fontsize=17, color=c, fontweight="bold", zorder=5)
            texts.append(t)
        adjust_text(texts, ax=ax, expand_points=(1.1, 1.2),
                    arrowprops=dict(arrowstyle="-", color="#9CA3AF",
                                    lw=0.4, alpha=0.5))
    except ImportError:
        for r in labelled:
            c = C_POS if r["delta"] > 0 else C_NEG
            ax.annotate(r["word"], (r["mean_z"], r["delta"]),
                        xytext=(4, 3), textcoords="offset points",
                        fontsize=17, color=c, fontweight="bold", zorder=5)

    # Leave xlabel empty, then place the two components as annotations
    # so that the "|" is pinned to x=0 while "Mean z-score..." stays left.
    ax.set_xlabel("")
    ax.annotate(
        r"Mean $z$-score  $(z_\mathrm{gold} + z_\mathrm{pred})/2$",
        xy=(0, 0), xytext=(0.0, -0.09),
        xycoords="axes fraction",
        ha="left", va="top",
        fontsize=21, color=INK,
    )
    ax.annotate(
        r"$\longleftarrow$ NO signal  $\;\mid\;$  YES signal $\longrightarrow$",
        xy=(0, 0), xytext=(0, -0.09),
        xycoords=("data", "axes fraction"),
        ha="center", va="top",
        fontsize=21, color=INK,
    )
    ax.set_ylabel(r"$z_\mathrm{pred} - z_\mathrm{gold}$    "
                  r"(+ = over-emphasises, $-$ = misses)",
                  fontsize=21, color=INK)
    ax.grid(alpha=0.12, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    # Quadrant annotations — small grey italic text, sitting just off the
    # y=0 axis so they label each quadrant without occupying corners.
    GREY = "#6B7280"

    def _quad(ax_x, ax_y, ha, va, text):
        ax.text(ax_x, ax_y, text, transform=ax.transAxes,
                ha=ha, va=va, fontsize=17, color=GREY,
                style="italic", zorder=6)

    _quad(0.985, 0.535, "right", "bottom", "Q1.  Model over-emphasises YES")
    _quad(0.015, 0.535, "left",  "bottom", "Q2.  Model weakens NO")
    _quad(0.015, 0.465, "left",  "top",    "Q3.  Model over-emphasises NO")
    _quad(0.985, 0.465, "right", "top",    "Q4.  Model under-emphasises YES")

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=C_POS, alpha=0.7,
              label=r"Model over-emphasises ($\Delta z > 2$)"),
        Patch(facecolor=C_NEG, alpha=0.7,
              label=r"Model under-emphasises ($\Delta z < -2$)"),
        Patch(facecolor="#D1D5DB", alpha=0.7,
              label=r"Close agreement ($|\Delta z| \leq 2$)"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True,
              framealpha=0.95, fontsize=19)

    fig.subplots_adjust(top=0.97, bottom=0.11, left=0.08, right=0.97)
    out = PAPER_FIG_DIR / "fig_discriminative_ma.pdf"
    fig.savefig(out); plt.close(fig)
    print(f"saved {out}")


# =====================================================================
# Method C: Tornado (divergent bar chart)
# =====================================================================

def fig_tornado(rows, K=20):
    INK   = "#111827"
    MUTED = "#6B7280"
    C_POS = "#D97706"
    C_NEG = "#2563EB"

    # Top K by deviation in each direction
    over = sorted([r for r in rows if r["delta"] > 0],
                  key=lambda r: -r["delta"])[:K]
    under = sorted([r for r in rows if r["delta"] < 0],
                   key=lambda r: r["delta"])[:K]

    # We lay them out as one figure with two horizontal bar charts side-by-side
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.5, 7.0),
                                      gridspec_kw={"wspace": 0.42})

    def _bars(ax, items, color, leftward, header):
        y = np.arange(len(items))[::-1]
        deltas = np.array([r["delta"] for r in items])
        sign_mag = np.abs(deltas) if not leftward else -np.abs(deltas)
        ax.barh(y, sign_mag, height=0.72, color=color,
                alpha=0.85, edgecolor="white", linewidth=0.5, zorder=2)
        ax.axvline(0, color="#374151", linewidth=0.8, zorder=1)
        for yi, r, m in zip(y, items, sign_mag):
            # word label on the "inside" of the bar
            if leftward:
                ax.text(m - 0.15, yi, r["word"], ha="right", va="center",
                        fontsize=9.2, color="white", fontweight="bold",
                        zorder=4)
                ax.text(0.1, yi, f"g={r['gz']:+.1f}  p={r['pz']:+.1f}",
                        ha="left", va="center",
                        fontsize=8.2, color=MUTED, zorder=4)
            else:
                ax.text(m + 0.15, yi, r["word"], ha="left", va="center",
                        fontsize=9.2, color="white", fontweight="bold",
                        zorder=4)
                # put stats outside, right of the bar end
                ax.text(-0.1, yi, f"g={r['gz']:+.1f}  p={r['pz']:+.1f}",
                        ha="right", va="center",
                        fontsize=8.2, color=MUTED, zorder=4)
        ax.set_yticks([])
        ax.set_xlabel(r"$\Delta z = z_\mathrm{pred} - z_\mathrm{gold}$",
                      fontsize=10.5, color=INK)
        ax.set_title(header, fontsize=11.5, color=color,
                     fontweight="bold", pad=8)
        ax.grid(axis="x", alpha=0.12, linestyle="--", linewidth=0.4)
        ax.set_axisbelow(True)

    max_mag = max(abs(over[0]["delta"]), abs(under[0]["delta"])) * 1.08
    _bars(ax_l, under, C_NEG, leftward=True,
          header=f"Model under-emphasises  (top {K})")
    _bars(ax_r, over, C_POS, leftward=False,
          header=f"Model over-emphasises  (top {K})")
    ax_l.set_xlim(-max_mag, 0.3)
    ax_r.set_xlim(-0.3, max_mag)

    # For each row, write the gold word a bit differently — already inline
    fig.subplots_adjust(top=0.93, bottom=0.08, left=0.04, right=0.98)
    out = PAPER_FIG_DIR / "fig_discriminative_tornado.pdf"
    fig.savefig(out); plt.close(fig)
    print(f"saved {out}")


def main():
    style()
    method = sys.argv[1] if len(sys.argv) > 1 else "both"
    rows = compute_scores()
    if method in ("ma", "both"):
        fig_ma(rows)
    if method in ("tornado", "both"):
        fig_tornado(rows)


if __name__ == "__main__":
    main()
