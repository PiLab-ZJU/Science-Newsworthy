"""
Grouped horizontal bar chart of per-field MCC, coloured by OpenAlex domain.

Fields are bucketed into four OpenAlex domains — Health Sciences, Life
Sciences, Physical Sciences, Social Sciences — and sorted by observed MCC
within each bucket. Each bar shows observed MCC; thin whiskers show the
bootstrap 95% CI (1000 resamples).

Usage:
    python analysis/fig_per_field_grouped.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import matthews_corrcoef

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROJECT_ROOT

ANALYSIS_DIR = PROJECT_ROOT / "analysis"
PAPER_FIG_DIR = PROJECT_ROOT / "paper-workflow" / "latex-temple" / "figures"
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 42
N_BOOT = 1000
MIN_N = 100

# --- OpenAlex field → domain mapping ---
DOMAIN_MAP = {
    # Health Sciences
    "Medicine": "Health Sciences",
    "Nursing": "Health Sciences",
    "Dentistry": "Health Sciences",
    "Veterinary": "Health Sciences",
    "Health Professions": "Health Sciences",
    # Life Sciences
    "Agricultural and Biological Sciences": "Life Sciences",
    "Biochemistry, Genetics and Molecular Biology": "Life Sciences",
    "Immunology and Microbiology": "Life Sciences",
    "Neuroscience": "Life Sciences",
    "Pharmacology, Toxicology and Pharmaceutics": "Life Sciences",
    # Physical Sciences
    "Chemistry": "Physical Sciences",
    "Chemical Engineering": "Physical Sciences",
    "Computer Science": "Physical Sciences",
    "Earth and Planetary Sciences": "Physical Sciences",
    "Energy": "Physical Sciences",
    "Engineering": "Physical Sciences",
    "Environmental Science": "Physical Sciences",
    "Materials Science": "Physical Sciences",
    "Mathematics": "Physical Sciences",
    "Physics and Astronomy": "Physical Sciences",
    # Social Sciences
    "Arts and Humanities": "Social Sciences",
    "Business, Management and Accounting": "Social Sciences",
    "Decision Sciences": "Social Sciences",
    "Economics, Econometrics and Finance": "Social Sciences",
    "Psychology": "Social Sciences",
    "Social Sciences": "Social Sciences",
}

DOMAIN_ORDER = ["Health Sciences", "Life Sciences",
                "Physical Sciences", "Social Sciences"]
DOMAIN_COLOR = {
    "Health Sciences":   "#DC2626",  # red
    "Life Sciences":     "#16A34A",  # green
    "Physical Sciences": "#2563EB",  # blue
    "Social Sciences":   "#D97706",  # amber
}

INK = "#1F2937"
EDGE = "#374151"
REF = "#F59E0B"


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": EDGE,
        "axes.labelcolor": INK,
        "xtick.color": EDGE,
        "ytick.color": EDGE,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })


def shorten(field: str) -> str:
    return {
        "Biochemistry, Genetics and Molecular Biology": "Biochem. & Mol. Bio.",
        "Economics, Econometrics and Finance": "Economics & Finance",
        "Pharmacology, Toxicology and Pharmaceutics": "Pharmacology",
        "Business, Management and Accounting": "Business & Mgmt",
        "Agricultural and Biological Sciences": "Agric. & Bio. Sci.",
        "Earth and Planetary Sciences": "Earth & Planetary Sci.",
        "Immunology and Microbiology": "Immunology & Micro.",
        "Physics and Astronomy": "Physics & Astronomy",
    }.get(field, field)


def bootstrap_mcc(y, yhat, n_boot, rng):
    y = np.asarray(y); yhat = np.asarray(yhat); n = len(y)
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        out[i] = matthews_corrcoef(y[idx], yhat[idx])
    return out


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
        domain = DOMAIN_MAP.get(f, "Social Sciences")
        obs = float(matthews_corrcoef(d["y"], d["yhat"]))
        boot = bootstrap_mcc(d["y"], d["yhat"], N_BOOT, rng)
        rows.append({
            "field": f, "domain": domain, "n": len(d["y"]),
            "obs": obs,
            "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5)),
        })

    # group + sort inside group by MCC desc
    by_domain = defaultdict(list)
    for r in rows:
        by_domain[r["domain"]].append(r)
    for dom in by_domain:
        by_domain[dom].sort(key=lambda r: r["obs"], reverse=True)

    # assemble final list + y positions; domain_spans indexes `ordered`
    ordered = []
    y_positions = []
    domain_spans = []  # (start_idx_in_ordered, end_idx_in_ordered, domain)
    y = 0
    for dom in DOMAIN_ORDER:
        if dom not in by_domain:
            continue
        start_idx = len(ordered)
        for r in by_domain[dom]:
            ordered.append(r)
            y_positions.append(y)
            y += 1
        end_idx = len(ordered) - 1
        domain_spans.append((start_idx, end_idx, dom))
        y += 1  # gap between groups
    y_positions = np.array(y_positions)

    overall = float(matthews_corrcoef(
        [p["true_label"] for p in preds],
        [p["predicted"] for p in preds]))

    fig, ax = plt.subplots(figsize=(7.8, 0.30 * len(ordered) + 1.6))

    # bars
    for yp, r in zip(y_positions, ordered):
        c = DOMAIN_COLOR[r["domain"]]
        ax.barh(yp, r["obs"], height=0.7,
                color=c, edgecolor="none", alpha=0.92, zorder=2)
        # 95% CI whisker (right side only for clarity)
        ax.plot([r["lo"], r["hi"]], [yp, yp],
                color=EDGE, linewidth=0.9, alpha=0.8, zorder=3)
        ax.plot([r["lo"], r["lo"]], [yp - 0.14, yp + 0.14],
                color=EDGE, linewidth=0.9, alpha=0.8, zorder=3)
        ax.plot([r["hi"], r["hi"]], [yp - 0.14, yp + 0.14],
                color=EDGE, linewidth=0.9, alpha=0.8, zorder=3)
        # value label
        ax.text(r["hi"] + 0.008, yp, f"{r['obs']:.3f}",
                va="center", ha="left", fontsize=8.4,
                color=c, fontweight="bold", zorder=4)

    # domain group brackets on the far left
    for start, end, dom in domain_spans:
        s = y_positions[start]; e = y_positions[end]
        xb = -0.05  # left margin for bracket
        ax.annotate("", xy=(xb, e + 0.3), xytext=(xb, s - 0.3),
                    arrowprops=dict(arrowstyle="-",
                                    color=DOMAIN_COLOR[dom],
                                    linewidth=2.5),
                    xycoords=("axes fraction", "data"),
                    textcoords=("axes fraction", "data"),
                    annotation_clip=False, zorder=5)
        ax.text(xb - 0.02, (s + e) / 2, dom,
                ha="right", va="center", rotation=90,
                fontsize=9, color=DOMAIN_COLOR[dom],
                fontweight="bold",
                transform=ax.get_yaxis_transform(),
                clip_on=False)

    # overall ref line
    ax.axvline(overall, color=REF, linestyle=":", linewidth=1.1, alpha=0.85)
    ax.text(overall, y_positions[-1] + 0.9,
            f"overall MCC = {overall:.3f}",
            color="#B45309", fontsize=8.4, ha="center", va="center",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", edgecolor="#FCD34D", linewidth=0.7))

    # y labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [f"{shorten(r['field'])}  (n={r['n']:,})" for r in ordered],
        fontsize=8.6,
    )
    ax.invert_yaxis()

    # x axis
    xmin = 0.45
    xmax = min(1.0, max(r["hi"] for r in ordered) + 0.07)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("MCC (error bar = bootstrap 95% CI)", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.3, linewidth=0.6)

    # legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=DOMAIN_COLOR[d], label=d)
               for d in DOMAIN_ORDER if d in by_domain]
    ax.legend(handles=handles, loc="lower right", frameon=False,
              fontsize=9, ncol=1)

    fig.subplots_adjust(top=0.96, left=0.36, right=0.97, bottom=0.08)

    out = PAPER_FIG_DIR / "fig_per_field_grouped.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
