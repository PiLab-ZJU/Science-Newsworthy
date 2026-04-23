"""
Semantic-space figure for §5.4.2 Model vs Journalist Vocabulary.

Top-25 signal words from each corpus are embedded with sentence-transformers
MiniLM (384-d), projected to 2D with UMAP, and plotted as a scatter with
per-group KDE density contours and per-group convex hulls. All 49 words are
labelled. A small centre-top banner reports the lexical Jaccard.

Message: even when the two word sets are put into a continuous semantic
space, they occupy distinct regions — the model's signals cluster around
newsworthiness-meta concepts, journalists' signals cluster around
attribution/provenance concepts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROJECT_ROOT

ANALYSIS_DIR = PROJECT_ROOT / "analysis"
PAPER_FIG_DIR = PROJECT_ROOT / "paper-workflow" / "latex-temple" / "figures"
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

K = 25
SEED = 42

COLOR = {
    "model":  "#2563EB",
    "news":   "#DC2626",
    "shared": "#7C3AED",
    "ink":    "#111827",
    "muted":  "#6B7280",
}


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 20,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })


def main() -> None:
    _style()
    d = json.loads((ANALYSIS_DIR / "contrastive_signals.json").read_text())
    model_top = [(w, float(m["z"])) for w, m in d["model_yes_top30"][:K]]
    news_top  = [(w, float(m["z"])) for w, m in d["news_added_top30"][:K]]

    model_set = {w for w, _ in model_top}
    news_set  = {w for w, _ in news_top}
    shared = model_set & news_set
    union = model_set | news_set
    jaccard = len(shared) / len(union)
    print(f"top-{K} Jaccard = {jaccard:.3f}   shared = {sorted(shared)}")

    # ---- embed ----
    from sentence_transformers import SentenceTransformer
    model_st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    words = sorted(union)
    emb = model_st.encode(words, normalize_embeddings=True)
    print(f"embedded {len(words)} words into {emb.shape[1]}-d")

    # ---- project to 2D (method chosen by argv[1], default umap) ----
    method = sys.argv[1] if len(sys.argv) > 1 else "umap"
    variance = None
    if method == "umap":
        import umap
        reducer = umap.UMAP(
            n_neighbors=10, min_dist=0.35, metric="cosine",
            random_state=SEED, n_components=2,
        )
        xy = reducer.fit_transform(emb)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=2, perplexity=8, metric="cosine",
            random_state=SEED, init="pca", learning_rate="auto",
        )
        xy = reducer.fit_transform(emb)
    elif method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=SEED)
        xy = reducer.fit_transform(emb)
        variance = reducer.explained_variance_ratio_
        print(f"PCA explained variance: PC1={variance[0]:.1%}  PC2={variance[1]:.1%}")
    elif method == "mds":
        from sklearn.manifold import MDS
        reducer = MDS(
            n_components=2, random_state=SEED,
            dissimilarity="precomputed", n_init=8, max_iter=500,
        )
        # cosine distance matrix
        sim = emb @ emb.T
        dist = 1.0 - sim
        np.fill_diagonal(dist, 0.0)
        dist = np.clip(dist, 0.0, 2.0)
        xy = reducer.fit_transform(dist)
    else:
        raise ValueError(f"unknown method: {method}")
    print(f"projected via {method.upper()}")

    # per-group masks
    def group_of(w):
        if w in shared:
            return "shared"
        if w in model_set:
            return "model"
        return "news"
    groups = np.array([group_of(w) for w in words])

    # ---- draw ----
    fig, ax = plt.subplots(figsize=(16, 13))

    # Soft KDE fill + a few solid contour lines per group
    def draw_territory(mask, color):
        pts = xy[mask]
        if len(pts) < 3:
            return
        try:
            kde = sps.gaussian_kde(pts.T, bw_method=0.45)
        except np.linalg.LinAlgError:
            return
        margin = 1.5
        xg = np.linspace(xy[:, 0].min() - margin, xy[:, 0].max() + margin, 220)
        yg = np.linspace(xy[:, 1].min() - margin, xy[:, 1].max() + margin, 220)
        X, Y = np.meshgrid(xg, yg)
        Z = kde(np.stack([X.ravel(), Y.ravel()])).reshape(X.shape)
        # Soft fill (one level, very light)
        ax.contourf(X, Y, Z,
                    levels=[Z.max() * 0.12, Z.max() * 1.01],
                    colors=[color], alpha=0.14, zorder=1)
        # Solid contour line at 1 density level (tight around the core)
        ax.contour(X, Y, Z,
                   levels=[Z.max() * 0.55],
                   colors=[color], linewidths=1.0, alpha=0.75, zorder=2)

    draw_territory(groups == "model", COLOR["model"])
    draw_territory(groups == "news",  COLOR["news"])

    # Centroid marker + connecting arrow (subtle)
    m_centroid = xy[groups == "model"].mean(axis=0)
    n_centroid = xy[groups == "news"].mean(axis=0)
    ax.plot(*m_centroid, marker="+", ms=13, mew=2,
            color=COLOR["model"], zorder=4)
    ax.plot(*n_centroid, marker="+", ms=13, mew=2,
            color=COLOR["news"], zorder=4)
    ax.annotate("", xy=m_centroid, xytext=n_centroid,
                arrowprops=dict(arrowstyle="->",
                                color=COLOR["muted"],
                                alpha=0.45, lw=1.0))
    mid = (m_centroid + n_centroid) / 2
    ax.text(mid[0], mid[1] + 0.35,
            f"Δ = {np.linalg.norm(m_centroid - n_centroid):.2f}",
            ha="center", va="bottom",
            fontsize=16, color=COLOR["muted"], style="italic",
            fontweight="bold", zorder=3)

    # Scatter points
    for g, c in [("news", COLOR["news"]),
                 ("model", COLOR["model"]),
                 ("shared", COLOR["shared"])]:
        mask = groups == g
        ax.scatter(xy[mask, 0], xy[mask, 1],
                   s=60 if g != "shared" else 120,
                   c=c, edgecolor="white", linewidth=0.9,
                   zorder=5, alpha=0.9,
                   marker="D" if g == "shared" else "o")

    # Labels — simple placement with small offset, adjust manually for overlaps
    for w, (x, y), g in zip(words, xy, groups):
        c = COLOR[g]
        ax.annotate(w, (x, y),
                    xytext=(6, 5),
                    textcoords="offset points",
                    fontsize=17, color=c,
                    fontweight="bold",
                    zorder=6)

    # Legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=COLOR["news"], markersize=9,
                   label=f"News-favoring ({K - len(shared)})"),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=COLOR["model"], markersize=9,
                   label=f"Model-favoring ({K - len(shared)})"),
        plt.Line2D([0], [0], marker='D', color='w',
                   markerfacecolor=COLOR["shared"], markersize=10,
                   label=f"Shared ({len(shared)})"),
        plt.Line2D([0], [0], marker='+', color=COLOR["muted"],
                   markersize=10, linestyle="", mew=2,
                   label="Group centroid"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True,
              framealpha=0.93, fontsize=18, prop={"weight": "bold", "size": 18})

    # (title suppressed; Jaccard reported in caption / text)

    xlabel = f"{method.upper()}-1"
    ylabel = f"{method.upper()}-2"
    if variance is not None:
        xlabel += f"  ({variance[0]:.1%} var.)"
        ylabel += f"  ({variance[1]:.1%} var.)"
    ax.set_xlabel(xlabel, fontsize=19, color=COLOR["muted"], fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=19, color=COLOR["muted"], fontweight="bold")
    ax.tick_params(axis='both', length=0, labelsize=0, colors="white")
    ax.grid(alpha=0.12, linestyle="--", linewidth=0.5)

    fig.subplots_adjust(top=0.88, bottom=0.06, left=0.04, right=0.98)
    out = PAPER_FIG_DIR / f"fig_vocabulary_semantic_{method}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
