"""
Butterfly / tornado figure for §5.4.2 Model vs Journalist Vocabulary.

Single figure answering all three questions:
  (1) Jaccard number (shown in a prominent box at top)
  (2) Which words are in each camp (labels on bars)
  (3) Distribution pattern (visual symmetry of two disjoint sides)

Left half  — top-25 news-favoring words (from news-vs-abstract LOR)
Right half — top-25 model-favoring words (from model-YES-vs-NO LOR)
Shared words are drawn twice and linked with a grey connector line.

Reads directly from analysis/contrastive_signals.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROJECT_ROOT

ANALYSIS_DIR = PROJECT_ROOT / "analysis"
PAPER_FIG_DIR = PROJECT_ROOT / "paper-workflow" / "latex-temple" / "figures"
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

K = 25

COLOR = {
    "model": "#2563EB",     # blue — model-favoring
    "news":  "#DC2626",     # red — news-favoring
    "shared": "#6B7280",    # grey — appears in both top-25
    "edge":  "#1F2937",
    "ink":   "#111827",
    "muted": "#6B7280",
    "grid":  "#E5E7EB",
    "accent":"#F59E0B",
}


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })


def main() -> None:
    _style()
    d = json.loads((ANALYSIS_DIR / "contrastive_signals.json").read_text())

    model_top = [(w, float(m["z"])) for w, m in d["model_yes_top30"][:K]]
    news_top  = [(w, float(m["z"])) for w, m in d["news_added_top30"][:K]]
    shared = set([w for w, _ in model_top]) & set([w for w, _ in news_top])
    union = set([w for w, _ in model_top]) | set([w for w, _ in news_top])
    jaccard = len(shared) / len(union)
    print(f"top-{K} Jaccard = {jaccard:.3f}   shared = {sorted(shared)}")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(10.5, 8.5))

    y = np.arange(K)[::-1]  # row 0 at top

    # Left side — news-favoring (bars go leftwards)
    news_z = np.array([z for _, z in news_top])
    news_w = [w for w, _ in news_top]
    z_max_n = news_z.max()
    for yi, (wi, zi) in zip(y, zip(news_w, news_z)):
        c = COLOR["shared"] if wi in shared else COLOR["news"]
        ax.barh(yi, -zi / z_max_n, height=0.70,
                color=c, edgecolor="white", linewidth=0.6, zorder=2)
        ax.text(-zi / z_max_n - 0.01, yi, f"{wi}",
                ha="right", va="center",
                fontsize=9.5, color=c,
                fontweight="bold" if wi in shared else "normal", zorder=4)
        ax.text(-0.004, yi, f"{zi:.0f}",
                ha="right", va="center",
                fontsize=7.6, color="white", zorder=5)

    # Right side — model-favoring (bars go rightwards)
    model_z = np.array([z for _, z in model_top])
    model_w = [w for w, _ in model_top]
    z_max_m = model_z.max()
    for yi, (wi, zi) in zip(y, zip(model_w, model_z)):
        c = COLOR["shared"] if wi in shared else COLOR["model"]
        ax.barh(yi, zi / z_max_m, height=0.70,
                color=c, edgecolor="white", linewidth=0.6, zorder=2)
        ax.text(zi / z_max_m + 0.01, yi, f"{wi}",
                ha="left", va="center",
                fontsize=9.5, color=c,
                fontweight="bold" if wi in shared else "normal", zorder=4)
        ax.text(0.004, yi, f"{zi:.0f}",
                ha="left", va="center",
                fontsize=7.6, color="white", zorder=5)

    # Connector for shared words
    for wi in shared:
        yi_n = K - 1 - news_w.index(wi)
        yi_m = K - 1 - model_w.index(wi)
        ax.annotate("", xy=(0.02, yi_m), xytext=(-0.02, yi_n),
                    arrowprops=dict(arrowstyle="-", color=COLOR["shared"],
                                    lw=1.0, linestyle=":"))

    # Centre divider
    ax.axvline(0, color=COLOR["edge"], linewidth=0.9, zorder=1)

    # Column headers
    ax.text(-0.5, K + 0.7,
            "News-favoring signals\n"
            r"(top-25 from $\mathrm{LOR}_\mathrm{news\ vs\ abstract}$)",
            ha="center", va="bottom", fontsize=10.5,
            color=COLOR["news"], fontweight="bold")
    ax.text(0.5, K + 0.7,
            "Model-favoring signals\n"
            r"(top-25 from $\mathrm{LOR}_\mathrm{model\ YES\ vs\ NO}$)",
            ha="center", va="bottom", fontsize=10.5,
            color=COLOR["model"], fontweight="bold")

    # Jaccard header banner
    shared_str = ", ".join(f"\\textit{{{w}}}" for w in sorted(shared))
    ax.text(0.0, K + 3.4,
            f"Top-{K} Jaccard = {jaccard:.3f}"
            + (f"     (only shared word: \"{list(shared)[0]}\")"
               if len(shared) == 1 else
               f"     ({len(shared)} shared of {len(union)} unique)"),
            ha="center", va="center",
            fontsize=12, fontweight="bold", color=COLOR["ink"],
            bbox=dict(boxstyle="round,pad=0.55", facecolor="#FEF3C7",
                      edgecolor=COLOR["accent"], linewidth=1.3))

    # x-axis meaning
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.8, K + 4.0)
    ax.set_xticks([-1.0, -0.5, 0, 0.5, 1.0])
    ax.set_xticklabels([
        f"z = {z_max_n:.0f}", f"z = {z_max_n/2:.0f}",
        "0",
        f"z = {z_max_m/2:.0f}", f"z = {z_max_m:.0f}",
    ], fontsize=8.5, color=COLOR["muted"])
    ax.tick_params(axis='x', length=0, pad=2)
    ax.tick_params(axis='y', length=0)
    ax.set_yticks([])

    # Note at bottom
    ax.text(0.0, -0.55,
            "Bar length encodes the word's $z$-score within its own LOR comparison "
            "(normalised to its side's max).\nLeft = words used more in "
            "news articles than in matched abstracts; "
            "Right = words used more in the model's YES-prediction rationales "
            "than in NO-prediction rationales.",
            ha="center", va="top", fontsize=8.6,
            color=COLOR["muted"], style="italic")

    fig.subplots_adjust(top=0.93, bottom=0.09, left=0.03, right=0.97)
    out = PAPER_FIG_DIR / "fig_vocabulary_butterfly.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
