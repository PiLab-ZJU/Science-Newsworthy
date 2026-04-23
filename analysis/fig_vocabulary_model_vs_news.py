"""
Vocabulary scatter: model YES-rationales vs actual news-article text.

For every TP paper (true_label=1, predicted=1) whose news URL was crawled
successfully AND whose scraped text verifiably mentions the paper's DOI or
title, we compare:

  model-side : document-frequency of each token in the model's explanation
  news-side  : document-frequency of each token in the matched news article

Log-odds ratio (prior=1.0) with symmetric Dirichlet smoothing gives us the
per-word direction; |z|-ranked words are labeled on the scatter.

Output: paper-workflow/latex-temple/figures/fig_vocabulary_model_vs_news.pdf
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
NEWS_PATH = Path(
    "/Volumes/Lin_SSD/lcx/academic_new_policy/data/raw/news_text/"
    "news_articles.json"
)

# ---- tokenization & stoplist (identical to analysis_3g_contrastive.py) ----
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "must", "need",
    "i", "we", "they", "he", "she", "it", "you", "me", "him", "her",
    "this", "that", "these", "those", "my", "your", "his", "its", "our",
    "of", "in", "to", "for", "with", "on", "at", "by", "from", "as",
    "into", "about", "between", "through", "during", "before", "after",
    "and", "or", "but", "not", "if", "than", "so", "because", "while",
    "also", "just", "very", "more", "most", "only", "even", "still",
    "such", "each", "both", "all", "any", "some", "no", "other",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "there", "here", "then", "now", "well", "however", "although",
    "whether", "since", "until", "unless", "yet", "already",
    # Common filler
    "paper", "study", "research", "article", "work", "results",
    "using", "used", "based", "method", "approach", "analysis",
    "data", "model", "figure", "table", "shown", "found", "reported",
}


def tokenize(text: str):
    return [w for w in re.findall(r"\b[a-z]{3,}\b", text.lower())
            if w not in STOPWORDS]


def log_odds_ratio(counts_a, counts_b, n_a, n_b, prior=1.0, min_total=20):
    out = {}
    vocab = set(counts_a) | set(counts_b)
    for w in vocab:
        total = counts_a.get(w, 0) + counts_b.get(w, 0)
        if total < min_total:
            continue
        fa = counts_a.get(w, 0) + prior
        fb = counts_b.get(w, 0) + prior
        odds_a = fa / (n_a - fa + prior)
        odds_b = fb / (n_b - fb + prior)
        lor = math.log2(odds_a / odds_b)
        var = 1.0 / fa + 1.0 / fb
        z = lor / math.sqrt(var)
        out[w] = {
            "lor": lor, "z": z,
            "freq_a": counts_a.get(w, 0),
            "freq_b": counts_b.get(w, 0),
            "total": total,
        }
    return out


def main():
    # ---- load ----
    preds = json.loads(PRED_PATH.read_text())
    print(f"test predictions: {len(preds)}")

    print("loading news_articles.json (this may take ~20s)...")
    news = json.loads(NEWS_PATH.read_text())
    doi_news = {a["doi"]: a for a in news if a.get("success")}
    print(f"news articles (success=True): {len(doi_news)}")

    # ---- match TP with verified news text ----
    model_texts, news_texts = [], []
    n_tp = n_matched = 0
    for p in preds:
        if p["true_label"] != 1 or p["predicted"] != 1:
            continue
        n_tp += 1
        art = doi_news.get(p["doi"])
        if not art:
            continue
        text = art.get("text", "")
        title = (p.get("title") or "").strip()
        doi_l = p["doi"].lower()
        if doi_l in text.lower() or (title and title.lower() in text.lower()):
            expl = (p.get("explanation") or "").strip()
            if not expl:
                continue
            model_texts.append(expl)
            news_texts.append(text[:2000])  # same cap as analysis_3g
            n_matched += 1
    print(f"TP total: {n_tp}  |  matched & verified: {n_matched}")

    # ---- TOKEN-level counts (fair across different doc-length distributions) ----
    model_counts = Counter()
    news_counts = Counter()
    for t in model_texts:
        model_counts.update(tokenize(t))      # all tokens, not set()
    for t in news_texts:
        news_counts.update(tokenize(t))
    n_tok_model = sum(model_counts.values())
    n_tok_news = sum(news_counts.values())
    print(f"tokens  model={n_tok_model:,}  news={n_tok_news:,}  "
          f"(news/model = {n_tok_news/max(n_tok_model,1):.2f}x)")

    # ---- log-odds on token proportions ----
    lor = log_odds_ratio(
        model_counts, news_counts,
        n_a=n_tok_model, n_b=n_tok_news,
        prior=1.0, min_total=20,
    )
    # significance threshold (for scatter colouring)
    z_thresh = 3.0
    model_fav = {w for w, m in lor.items() if m["z"] > z_thresh}
    news_fav = {w for w, m in lor.items() if m["z"] < -z_thresh}

    # Vocabulary-overlap Jaccard: top-K by token proportion in each corpus.
    K_OVERLAP = 200
    top_model_words = {w for w, _ in model_counts.most_common(K_OVERLAP)}
    top_news_words = {w for w, _ in news_counts.most_common(K_OVERLAP)}
    jaccard = (
        len(top_model_words & top_news_words)
        / len(top_model_words | top_news_words)
    ) if (top_model_words | top_news_words) else 0.0
    print(f"vocab with min_total>=20: {len(lor)}")
    print(f"model-favoring words  (z>{z_thresh}): {len(model_fav)}")
    print(f"news-favoring words   (z<{-z_thresh}): {len(news_fav)}")
    print(f"Jaccard of top-{K_OVERLAP} by df each side: {jaccard:.4f}")

    # ---- pick top-K by |z| from each side for plotting ----
    K = 30
    sorted_by_z = sorted(lor.items(), key=lambda kv: kv[1]["z"])
    top_news = sorted_by_z[:K]              # most negative z
    top_model = sorted_by_z[-K:][::-1]      # most positive z
    pts = top_model + top_news

    words = [w for w, _ in pts]
    fm = np.array([m["freq_a"] for _, m in pts], dtype=float)
    fn = np.array([m["freq_b"] for _, m in pts], dtype=float)
    z = np.array([m["z"] for _, m in pts])
    # Axes: log10 of frequency per million tokens (IPM), +1 floor
    ipm_m = fm / max(n_tok_model, 1) * 1_000_000
    ipm_n = fn / max(n_tok_news, 1) * 1_000_000
    xs = np.log10(ipm_m + 1)
    ys = np.log10(ipm_n + 1)
    colors = ["#2563EB" if zi > 0 else "#DC2626" for zi in z]
    sizes = 32 + 1.1 * np.abs(z)

    # ---- plot ----
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(7.4, 6.0))

    lim = max(xs.max(), ys.max()) + 0.3
    ax.plot([0, lim], [0, lim], color="#D1D5DB",
            linewidth=1.1, linestyle="--", zorder=1)

    ax.scatter(xs, ys, s=sizes, c=colors, alpha=0.78,
               edgecolor="white", linewidth=0.8, zorder=3)

    # label words most off-diagonal
    deviations = ys - xs
    order = np.argsort(deviations)
    label_idx = list(order[:8]) + list(order[-8:])
    for i in label_idx:
        is_model = colors[i] == "#2563EB"
        dx = -0.06 if is_model else 0.06
        ha = "right" if is_model else "left"
        ax.annotate(words[i], (xs[i], ys[i]),
                    xytext=(xs[i] + dx, ys[i]),
                    fontsize=8.6, color=colors[i],
                    fontweight="bold", ha=ha, va="center", zorder=5)

    ax.set_xlabel(r"$\log_{10}$(occurrences per million tokens in model explanations $+1$)",
                  fontsize=10)
    ax.set_ylabel(r"$\log_{10}$(occurrences per million tokens in news articles $+1$)",
                  fontsize=10)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid(alpha=0.18, linestyle="-")
    ax.set_axisbelow(True)

    ax.scatter([], [], s=55, c="#2563EB", label="Model-favoring word")
    ax.scatter([], [], s=55, c="#DC2626", label="News-favoring word")
    ax.plot([], [], color="#D1D5DB", linestyle="--", label="Equal usage")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=9)

    ax.text(0.98, 0.03,
            f"Model (TP rationales) vs. news articles\n"
            f"n={len(model_texts):,} paired docs   "
            f"Jaccard$_{{top{K_OVERLAP}}}$ = {jaccard:.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.4, fontweight="bold", color="#1F2937",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#FEF3C7",
                      edgecolor="#F59E0B", linewidth=1.0))

    fig.subplots_adjust(top=0.97, bottom=0.10, left=0.11, right=0.97)
    out = PAPER_FIG_DIR / "fig_vocabulary_model_vs_news.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"\nsaved {out}")

    # dump the word lists & stats too (for captions / appendix)
    stats = {
        "n_tp": n_tp,
        "n_matched_verified": n_matched,
        "n_vocab_min_freq": len(lor),
        "z_threshold": z_thresh,
        "n_model_favoring": len(model_fav),
        "n_news_favoring": len(news_fav),
        "jaccard_topK_by_df": jaccard,
        "K_for_jaccard": K_OVERLAP,
        "top_model_words": [(w, {k: (float(v) if isinstance(v, (int, float)) else v)
                                  for k, v in m.items()})
                            for w, m in top_model[:20]],
        "top_news_words": [(w, {k: (float(v) if isinstance(v, (int, float)) else v)
                                 for k, v in m.items()})
                           for w, m in top_news[:20]],
    }
    side_out = ANALYSIS_DIR / "model_vs_news_vocab.json"
    side_out.write_text(json.dumps(stats, indent=2, default=str))
    print(f"stats  {side_out}")


if __name__ == "__main__":
    main()
