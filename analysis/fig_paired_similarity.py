"""
Paired-vs-random similarity figure.

For every TP paper whose news URL is verified (N = 2,825), we have a
(model_rationale, news_article) pair produced for the SAME paper.

Panel A — Semantic similarity
    MiniLM sentence embedding of each rationale and news article; report
    cosine similarity for true pairs and for a random-shuffle baseline.

Panel B — Lexical similarity
    Token-set Jaccard of each (rationale, news) pair (same stopword list as
    §5.4.2); compare true pairs with random baseline.

Reveals the §5.4 headline:
    semantic alignment (A panels separated)  +
    lexical divergence (B panels overlap near zero)
"""
from __future__ import annotations

import json
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
NEWS_PATH = Path("/Volumes/Lin_SSD/lcx/academic_new_policy/data/raw/news_text/"
                 "news_articles.json")
ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42
N_RANDOM = 2000  # random pairs for baseline

COLOR_TRUE = "#2563EB"
COLOR_RAND = "#9CA3AF"
INK = "#111827"
MUTED = "#6B7280"

STOPWORDS = {
    "the","a","an","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could",
    "should","may","might","can","shall","must","need",
    "i","we","they","he","she","it","you","me","him","her",
    "this","that","these","those","my","your","his","its","our",
    "of","in","to","for","with","on","at","by","from","as",
    "into","about","between","through","during","before","after",
    "and","or","but","not","if","than","so","because","while",
    "also","just","very","more","most","only","even","still",
    "such","each","both","all","any","some","no","other",
    "what","which","who","whom","how","when","where","why",
    "there","here","then","now","well","however","although",
    "whether","since","until","unless","yet","already",
    "paper","study","research","article","work","results",
    "using","used","based","method","approach","analysis",
    "data","model","figure","table","shown","found","reported",
}


def tokenize_set(text):
    return {w for w in re.findall(r"\b[a-z]{3,}\b", text.lower())
            if w not in STOPWORDS}


def build_pairs():
    preds = json.loads(PRED_PATH.read_text())
    print("loading news_articles.json ...")
    news = json.loads(NEWS_PATH.read_text())
    doi_news = {a["doi"]: a for a in news if a.get("success")}

    pairs = []
    for p in preds:
        if p["true_label"] != 1 or p["predicted"] != 1:
            continue  # TP only for the paired analysis
        art = doi_news.get(p["doi"])
        if not art:
            continue
        ntext = art.get("text", "") or ""
        title = (p.get("title") or "").strip()
        if not (p["doi"].lower() in ntext.lower() or
                (title and title.lower() in ntext.lower())):
            continue
        expl = (p.get("explanation") or "").strip()
        if not (expl and ntext):
            continue
        pairs.append({
            "rationale": expl,
            "news": " ".join(ntext.split()[:500]),
        })
    return pairs


def main():
    pairs = build_pairs()
    print(f"paired triplets (TP matched): {len(pairs)}")

    rationales = [p["rationale"] for p in pairs]
    news_texts = [p["news"] for p in pairs]
    n = len(pairs)

    # ---------- SEMANTIC ----------
    from sentence_transformers import SentenceTransformer
    print("encoding with MiniLM ...")
    st = SentenceTransformer(ST_MODEL)
    rat_emb = st.encode(rationales, batch_size=128,
                        show_progress_bar=True, normalize_embeddings=True)
    news_emb = st.encode(news_texts, batch_size=128,
                         show_progress_bar=True, normalize_embeddings=True)

    # true paired cosine
    sem_true = (rat_emb * news_emb).sum(axis=1)
    # random baseline
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    # ensure no self-pair after permute
    for i in range(n):
        if perm[i] == i:
            perm[i] = (i + 1) % n
    sem_rand = (rat_emb * news_emb[perm]).sum(axis=1)

    # effect sizes
    cohen_d_sem = (sem_true.mean() - sem_rand.mean()) / \
                  np.sqrt((sem_true.var() + sem_rand.var()) / 2)

    print(f"\nSemantic cosine")
    print(f"  true : mean={sem_true.mean():.3f}  sd={sem_true.std():.3f}")
    print(f"  rand : mean={sem_rand.mean():.3f}  sd={sem_rand.std():.3f}")
    print(f"  Cohen's d = {cohen_d_sem:.2f}")

    # ---------- LEXICAL ----------
    rat_sets = [tokenize_set(t) for t in rationales]
    news_sets = [tokenize_set(t) for t in news_texts]
    def jacc(a, b):
        u = a | b
        return (len(a & b) / len(u)) if u else 0.0
    lex_true = np.array([jacc(a, b) for a, b in zip(rat_sets, news_sets)])
    lex_rand = np.array([jacc(rat_sets[i], news_sets[perm[i]])
                         for i in range(n)])

    cohen_d_lex = (lex_true.mean() - lex_rand.mean()) / \
                  np.sqrt((lex_true.var() + lex_rand.var()) / 2)
    print(f"\nLexical Jaccard")
    print(f"  true : mean={lex_true.mean():.3f}  sd={lex_true.std():.3f}")
    print(f"  rand : mean={lex_rand.mean():.3f}  sd={lex_rand.std():.3f}")
    print(f"  Cohen's d = {cohen_d_lex:.2f}")

    # ---------- plot: two separate figures ----------
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

    def draw(true_v, rand_v, xlabel, out_path, bins=40):
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        bins_edges = np.linspace(
            min(true_v.min(), rand_v.min()) - 0.01,
            max(true_v.max(), rand_v.max()) + 0.01, bins)
        ax.hist(rand_v, bins=bins_edges, color=COLOR_RAND, alpha=0.55,
                edgecolor="white", linewidth=0.4,
                label=f"Random pairs   μ = {rand_v.mean():.3f}")
        ax.hist(true_v, bins=bins_edges, color=COLOR_TRUE, alpha=0.70,
                edgecolor="white", linewidth=0.4,
                label=f"True pairs      μ = {true_v.mean():.3f}")
        ax.axvline(rand_v.mean(), color=COLOR_RAND, linestyle="--",
                   linewidth=1.1, alpha=0.9)
        ax.axvline(true_v.mean(), color=COLOR_TRUE, linestyle="--",
                   linewidth=1.1, alpha=0.9)
        d = (true_v.mean() - rand_v.mean()) / \
            np.sqrt((true_v.var() + rand_v.var()) / 2)
        ax.text(0.98, 0.98,
                f"Cohen's $d$ = {d:.2f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=11, color=INK, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=11, color=INK)
        ax.set_ylabel("Count (pairs)", fontsize=11, color=MUTED)
        ax.grid(axis="y", alpha=0.15, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", frameon=False, fontsize=10.5)
        fig.subplots_adjust(top=0.97, bottom=0.14, left=0.12, right=0.97)
        fig.savefig(out_path)
        plt.close(fig)
        print(f"saved {out_path}")

    draw(sem_true, sem_rand,
         "cosine($\\,$rationale,$\\,$news$\\,$)",
         PAPER_FIG_DIR / "fig_paired_similarity_semantic.pdf",
         bins=40)
    draw(lex_true, lex_rand,
         "Jaccard($\\,$rationale,$\\,$news$\\,$)",
         PAPER_FIG_DIR / "fig_paired_similarity_lexical.pdf",
         bins=30)

    # dump numbers for the paper
    stats = {
        "n_pairs": n,
        "semantic": {"true_mean": float(sem_true.mean()),
                     "true_std": float(sem_true.std()),
                     "rand_mean": float(sem_rand.mean()),
                     "rand_std": float(sem_rand.std()),
                     "cohen_d": float(cohen_d_sem)},
        "lexical": {"true_mean": float(lex_true.mean()),
                    "true_std": float(lex_true.std()),
                    "rand_mean": float(lex_rand.mean()),
                    "rand_std": float(lex_rand.std()),
                    "cohen_d": float(cohen_d_lex)},
    }
    (ANALYSIS_DIR / "paired_similarity.json").write_text(
        json.dumps(stats, indent=2))
    print(f"stats saved: analysis/paired_similarity.json")


if __name__ == "__main__":
    main()
