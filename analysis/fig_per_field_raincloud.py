"""
Raincloud per-field figure, Allen-et-al.-2019 style.

Layout: one raincloud PER FIELD, stacked horizontally as columns.
Each column has (left→right): half-violin | box | jitter points.

Bootstrap: 1000 resamples per field → MCC distribution.

Usage:
    python analysis/fig_per_field_raincloud.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps
from sklearn.metrics import matthews_corrcoef, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROJECT_ROOT

ANALYSIS_DIR = PROJECT_ROOT / "analysis"
PAPER_FIG_DIR = PROJECT_ROOT / "paper-workflow" / "latex-temple" / "figures"
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 42
N_BOOT = 1000
MIN_N = 100
METRIC = "mcc"


# Allen-et-al. style: soft pastel fills, crisp dark strokes
BLUE_FILL = "#A9C4E8"
BLUE_EDGE = "#3B6BB0"
BLUE_BOX_FILL = "#6E95C6"
BLUE_LABEL = "#1E4A8C"

INK = "#1F2937"
MUTED = "#6B7280"
REF = "#D97706"


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#374151",
        "axes.labelcolor": INK,
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })


def shorten(field: str) -> str:
    table = {
        "Biochemistry, Genetics and Molecular Biology": "Biochem.\n& Mol. Bio.",
        "Economics, Econometrics and Finance": "Economics\n& Finance",
        "Pharmacology, Toxicology and Pharmaceutics": "Pharmacology",
        "Business, Management and Accounting": "Business\n& Mgmt",
        "Agricultural and Biological Sciences": "Agric.\n& Bio. Sci.",
        "Earth and Planetary Sciences": "Earth\n& Planetary",
        "Immunology and Microbiology": "Immunology\n& Micro.",
        "Physics and Astronomy": "Physics\n& Astron.",
        "Environmental Science": "Environ. Sci.",
        "Health Professions": "Health Prof.",
        "Materials Science": "Materials",
        "Computer Science": "Computer Sci.",
        "Decision Sciences": "Decision Sci.",
        "Arts and Humanities": "Arts & Hum.",
        "Social Sciences": "Social Sci.",
        "Neuroscience": "Neurosci.",
    }
    return table.get(field, field)


def compute_metric(y, yhat, metric: str) -> float:
    if metric == "mcc":
        return float(matthews_corrcoef(y, yhat))
    if metric == "f1":
        return float(f1_score(y, yhat, zero_division=0))
    raise ValueError(metric)


def bootstrap_metric(y, yhat, metric: str, n_boot: int,
                     rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    n = len(y)
    out = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        out[i] = compute_metric(y[idx], yhat[idx], metric)
    return out


def draw_raincloud(ax, x_center, boot, obs, color_fill, color_edge,
                   color_boxfill, color_label, rng,
                   half_width=0.34, box_width=0.14, jitter_width=0.28,
                   jitter_n=80):
    """Draw one raincloud at horizontal position x_center.

    left:   half-violin (KDE), mirrored left
    center: box (IQR) with white median line
    right:  jitter scatter
    """
    # ----- half-violin on the LEFT of center -----
    kde = sps.gaussian_kde(boot, bw_method="scott")
    y_grid = np.linspace(boot.min() - 0.01, boot.max() + 0.01, 220)
    density = kde(y_grid)
    density = density / density.max() * half_width
    x_left = x_center - density
    ax.fill_betweenx(
        y_grid, x_center, x_left,
        facecolor=color_fill, edgecolor=color_edge,
        linewidth=0.9, alpha=0.85, zorder=2,
    )

    # ----- box (IQR) at slightly right of center -----
    q1, med, q3 = np.percentile(boot, [25, 50, 75])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    box_x = x_center + 0.02
    # whisker
    ax.plot([box_x, box_x], [lo, hi],
            color=color_edge, linewidth=1.1, zorder=3,
            solid_capstyle="round")
    # box
    ax.add_patch(mpatches.FancyBboxPatch(
        (box_x - box_width / 2, q1), box_width, q3 - q1,
        boxstyle="round,pad=0,rounding_size=0.003",
        facecolor=color_boxfill, edgecolor=color_edge,
        linewidth=0.9, zorder=4,
    ))
    # median line (white)
    ax.plot([box_x - box_width / 2, box_x + box_width / 2],
            [med, med], color="white", linewidth=1.6, zorder=5,
            solid_capstyle="round")
    # observed value marker (large white dot)
    ax.plot(box_x, obs, "o", color="white", markersize=5.5,
            markeredgecolor=color_edge, markeredgewidth=1.2, zorder=6)

    # ----- jitter on the RIGHT of center -----
    sub = rng.choice(boot, size=min(jitter_n, len(boot)), replace=False)
    jx = x_center + 0.12 + rng.uniform(0, jitter_width, size=len(sub))
    ax.scatter(sub, jx * 0 + sub * 0 + sub * 0,  # placeholder avoid warning
               s=0)
    ax.scatter(jx, sub, s=6, color=color_fill,
               alpha=0.5, edgecolor="none", zorder=2)

    # ----- value label (right of jitter cloud) -----
    ax.text(x_center + 0.42, obs, f"{obs:.3f}",
            fontsize=8.6, color=color_label,
            fontweight="bold", va="center", ha="left", zorder=7)


def main() -> None:
    _style()
    preds = json.loads(
        (ANALYSIS_DIR / "test_predictions_with_explanations.json").read_text())
    print(f"loaded {len(preds)} predictions")

    by_field = defaultdict(lambda: {"y": [], "yhat": []})
    for p in preds:
        f = p.get("field", "")
        if not f:
            continue
        by_field[f]["y"].append(p["true_label"])
        by_field[f]["yhat"].append(p["predicted"])

    rng = np.random.default_rng(RNG_SEED)

    rows = []
    for f, d in by_field.items():
        if len(d["y"]) < MIN_N:
            continue
        obs = compute_metric(d["y"], d["yhat"], METRIC)
        boot = bootstrap_metric(d["y"], d["yhat"], METRIC, N_BOOT, rng)
        rows.append({
            "field": f, "n": len(d["y"]),
            "obs": obs, "boot": boot,
        })
    rows.sort(key=lambda r: r["obs"], reverse=True)

    overall = compute_metric(
        [p["true_label"] for p in preds],
        [p["predicted"] for p in preds],
        METRIC,
    )

    n_fields = len(rows)
    # Horizontal figure: fields along X
    fig_w = max(10, 0.65 * n_fields)
    fig, ax = plt.subplots(figsize=(fig_w, 5.6))

    x_positions = np.arange(n_fields) * 1.2  # spacing
    for xi, r in zip(x_positions, rows):
        draw_raincloud(
            ax, xi, r["boot"], r["obs"],
            color_fill=BLUE_FILL, color_edge=BLUE_EDGE,
            color_boxfill=BLUE_BOX_FILL, color_label=BLUE_LABEL,
            rng=rng,
        )

    # overall reference
    ax.axhline(overall, color=REF, linestyle=":", linewidth=1.2, alpha=0.8)
    ax.text(x_positions[-1] + 0.9, overall,
            f"overall\n{METRIC.upper()}={overall:.3f}",
            fontsize=8.4, color=REF, va="center", ha="left",
            fontweight="bold")

    # x ticks
    ax.set_xticks(x_positions)
    labels = [f"{shorten(r['field'])}\n(n={r['n']:,})" for r in rows]
    ax.set_xticklabels(labels, fontsize=8.4, rotation=0, ha="center")
    ax.set_xlim(x_positions[0] - 0.7, x_positions[-1] + 1.6)

    # y axis
    all_vals = np.concatenate([r["boot"] for r in rows])
    y_lo = max(0.0, all_vals.min() - 0.03)
    y_hi = min(1.0, max(r["obs"] for r in rows) + 0.05)
    ax.set_ylim(y_lo, y_hi)
    ax.set_ylabel(f"{METRIC.upper()}  (bootstrap 95% CI)", fontsize=10.5)
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
    ax.grid(axis="x", visible=False)

    # cleanup x ticks
    for t in ax.get_xticklabels():
        t.set_color(INK)

    fig.subplots_adjust(top=0.95, left=0.06, right=0.98, bottom=0.18)

    out = PAPER_FIG_DIR / "fig_per_field_raincloud.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
