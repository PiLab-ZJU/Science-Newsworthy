"""
Semantic-overlap figure: 2D UMAP of the joint (YES ∪ NO ∪ News) corpus,
colored by provenance. Visually shows the three text types collapsing into
the same semantic cluster (→ the joint BERTopic finding).

Pairs with §5.4.2 (vocabulary Jaccard 0.02): semantic overlap + lexical
divergence is the §5.4 headline.

Output: paper-workflow/latex-temple/figures/fig_semantic_overlap.pdf
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROJECT_ROOT

ANALYSIS_DIR = PROJECT_ROOT / "analysis"
PAPER_FIG_DIR = PROJECT_ROOT / "paper-workflow" / "latex-temple" / "figures"
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

PRED_PATH = ANALYSIS_DIR / "test_predictions_with_explanations.json"
NEWS_PATH = Path("/Volumes/Lin_SSD/lcx/academic_new_policy/data/raw/news_text/"
                 "news_articles.json")

SEED = 42
ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

COLOR = {
    "yes":  "#2563EB",   # blue  — model YES rationales (TP)
    "no":   "#DC2626",   # red   — model NO rationales (FN, missed)
    "news": "#16A34A",   # green — news articles
}
LABEL = {
    "yes":  "Model YES rationale (TP, n = {n})",
    "no":   "Model NO rationale (FN, n = {n})",
    "news": "News article (n = {n})",
}


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })


def build_corpus():
    preds = json.loads(PRED_PATH.read_text())
    print(f"predictions: {len(preds)}")
    print("loading news_articles.json ...")
    news = json.loads(NEWS_PATH.read_text())
    doi_news = {a["doi"]: a for a in news if a.get("success")}

    # verified-news DOIs (same filter as §5.4.2/§5.4.3)
    verified = {}
    for p in preds:
        if p["true_label"] != 1:
            continue
        art = doi_news.get(p["doi"])
        if not art:
            continue
        text = art.get("text", "") or ""
        title = (p.get("title") or "").strip()
        if not (p["doi"].lower() in text.lower() or
                (title and title.lower() in text.lower())):
            continue
        if text:
            verified[p["doi"]] = text

    tp_matched = [p for p in preds
                  if p["true_label"] == 1 and p["predicted"] == 1
                  and p["doi"] in verified]
    fn_matched = [p for p in preds
                  if p["true_label"] == 1 and p["predicted"] == 0
                  and p["doi"] in verified]

    texts, labels = [], []
    for p in tp_matched:
        if p.get("explanation"):
            texts.append(p["explanation"])
            labels.append("yes")
    for p in fn_matched:
        if p.get("explanation"):
            texts.append(p["explanation"])
            labels.append("no")
    for p in tp_matched + fn_matched:
        t = verified.get(p["doi"], "")
        if not t:
            continue
        texts.append(" ".join(t.split()[:500]))
        labels.append("news")

    return texts, np.array(labels)


def main():
    _style()
    texts, labels = build_corpus()
    cnt = Counter(labels)
    print(f"corpus={len(texts)}  "
          f"(yes={cnt['yes']}, no={cnt['no']}, news={cnt['news']})")

    # cache embeddings
    cache_emb = ANALYSIS_DIR / "semantic_overlap_emb.npy"
    cache_lab = ANALYSIS_DIR / "semantic_overlap_labels.npy"
    if cache_emb.exists() and cache_lab.exists():
        emb = np.load(cache_emb)
        cached_labels = np.load(cache_lab, allow_pickle=True)
        if emb.shape[0] == len(texts) and list(cached_labels) == list(labels):
            print(f"[cache hit] {cache_emb}")
        else:
            emb = None
    else:
        emb = None
    if emb is None:
        from sentence_transformers import SentenceTransformer
        print("encoding with MiniLM (CPU, ~3-5 min)...")
        st = SentenceTransformer(ST_MODEL)
        emb = st.encode(texts, batch_size=128,
                        show_progress_bar=True, normalize_embeddings=True)
        np.save(cache_emb, emb)
        np.save(cache_lab, np.array(labels))
        print(f"[cache save] {cache_emb}")

    method = sys.argv[1] if len(sys.argv) > 1 else "umap"
    print(f"projecting with {method.upper()} (2D)...")
    if method == "umap":
        import umap
        reducer = umap.UMAP(
            n_neighbors=50, min_dist=0.5, spread=2.0,
            metric="cosine", random_state=SEED, n_components=2,
        )
        xy = reducer.fit_transform(emb)
    elif method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=SEED)
        xy = reducer.fit_transform(emb)
        print(f"explained variance: PC1={reducer.explained_variance_ratio_[0]:.1%}  "
              f"PC2={reducer.explained_variance_ratio_[1]:.1%}")
    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=2, perplexity=50, metric="cosine",
            random_state=SEED, init="pca", learning_rate="auto",
            max_iter=1000,
        )
        xy = reducer.fit_transform(emb)
    else:
        raise ValueError(f"unknown method: {method}")
    # Center and z-score so the figure fills its canvas
    xy = xy - xy.mean(axis=0)
    std = xy.std()
    if std > 1e-6:
        xy = xy / std
    print(f"xy shape: {xy.shape}  "
          f"xrange=[{xy[:,0].min():.2f}, {xy[:,0].max():.2f}]  "
          f"yrange=[{xy[:,1].min():.2f}, {xy[:,1].max():.2f}]")

    # -------- plot: 1x3 small multiples + overlay panel --------
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(15, 5.5))
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 1.15], wspace=0.12)

    # Compute shared axes range
    margin = 0.8
    xmin, xmax = xy[:, 0].min() - margin, xy[:, 0].max() + margin
    ymin, ymax = xy[:, 1].min() - margin, xy[:, 1].max() + margin

    # Pre-compute KDE grids per group (reuse for overlay panel)
    xg = np.linspace(xmin, xmax, 220)
    yg = np.linspace(ymin, ymax, 220)
    X, Y = np.meshgrid(xg, yg)
    kdes = {}
    for lab in ["yes", "no", "news"]:
        pts = xy[labels == lab]
        try:
            k = sps.gaussian_kde(pts.T, bw_method=0.35)
            kdes[lab] = k(np.stack([X.ravel(), Y.ravel()])).reshape(X.shape)
        except np.linalg.LinAlgError:
            kdes[lab] = None

    panel_order = ["yes", "no", "news"]
    panel_titles = {
        "yes":  "Model YES rationale  (TP, n=2{,}825)",
        "no":   "Model NO rationale  (FN, n=929)",
        "news": "News article  (n=3{,}754)",
    }
    # Use plain title strings without LaTeX
    panel_titles = {
        "yes":  "Model YES rationale  (TP, n = 2,825)",
        "no":   "Model NO rationale  (FN, n = 929)",
        "news": "News article  (n = 3,754)",
    }

    for i, lab in enumerate(panel_order):
        ax = fig.add_subplot(gs[0, i])
        pts = xy[labels == lab]
        ax.scatter(pts[:, 0], pts[:, 1], s=3.5, c=COLOR[lab],
                   alpha=0.30, edgecolor="none", zorder=2)
        if kdes[lab] is not None:
            Z = kdes[lab]
            ax.contour(X, Y, Z, levels=[Z.max() * 0.30, Z.max() * 0.60],
                       colors=[COLOR[lab]], linewidths=1.1,
                       alpha=0.9, zorder=3)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ["top", "right", "bottom", "left"]:
            ax.spines[s].set_color("#D1D5DB")
        ax.set_title(panel_titles[lab], fontsize=11,
                     color=COLOR[lab], fontweight="bold", pad=6)
        if i == 0:
            ax.set_ylabel("UMAP-2", fontsize=10, color="#6B7280")
        ax.set_xlabel("UMAP-1", fontsize=10, color="#6B7280")

    # Overlay panel: three KDE contours on one axis
    ax_over = fig.add_subplot(gs[0, 3])
    for lab in ["news", "yes", "no"]:  # draw news first (underneath)
        if kdes[lab] is None:
            continue
        Z = kdes[lab]
        # Filled density at one low threshold for territory
        ax_over.contourf(X, Y, Z, levels=[Z.max() * 0.30, Z.max() * 1.01],
                         colors=[COLOR[lab]], alpha=0.22, zorder=1)
        ax_over.contour(X, Y, Z, levels=[Z.max() * 0.60],
                        colors=[COLOR[lab]], linewidths=1.3,
                        alpha=0.9, zorder=2)
    ax_over.set_xlim(xmin, xmax)
    ax_over.set_ylim(ymin, ymax)
    ax_over.set_xticks([]); ax_over.set_yticks([])
    for s in ["top", "right", "bottom", "left"]:
        ax_over.spines[s].set_color("#D1D5DB")
    ax_over.set_title("Three provenances overlaid", fontsize=11,
                      color="#111827", fontweight="bold", pad=6)
    ax_over.set_xlabel("UMAP-1", fontsize=10, color="#6B7280")

    # Manual legend on overlay
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(facecolor=COLOR[lab], alpha=0.35,
                              edgecolor=COLOR[lab], linewidth=1.3,
                              label={"yes":"YES", "no":"NO",
                                     "news":"News"}[lab])
               for lab in ["yes", "no", "news"]]
    ax_over.legend(handles=handles, loc="upper right", frameon=True,
                   framealpha=0.95, fontsize=10)

    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.03, right=0.99)
    out = PAPER_FIG_DIR / f"fig_semantic_overlap_{method}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
