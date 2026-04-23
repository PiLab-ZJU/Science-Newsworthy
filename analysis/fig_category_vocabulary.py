"""
§5.4.3 Category-level vocabulary overlap between model and journalist.

For each of the 8 Harcup news-value categories, we (a) select the papers
whose triplet (model rationale OR news article) mentions any of the
category's keywords, (b) extract the top-25 most frequent *non-category*
content words from model rationales and news articles within that subset,
and (c) report the per-category Jaccard of the two top-25 lists. Category
keywords are excluded to avoid trivial overlap from the assignment rule.

Figure: 2D scatter that links §5.4.1 and §5.4.3 —
    x-axis = Model/News mention-rate ratio (M/N, from §5.4.1)
    y-axis = Top-25 Jaccard within category (this analysis)
    Each point = one Harcup category.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
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
TEST_PATH = Path("/Volumes/Lin_SSD/lcx/academic_new_policy/data/processed/"
                 "combined/test.json")
NEWSVALUE_JSON = ANALYSIS_DIR / "newsvalue_signals.json"

K = 25
N_BOOT = 200    # LOR is ~10x more expensive than raw-df; keep modest
RNG_SEED = 42
GLOBAL_JACCARD = 0.020   # from §5.4.2 (top-25 model-vs-news)

# ---- Harcup categories (same keywords as analysis_3k_newsvalue.py) ----
NEWS_VALUES = {
    "Surprise": {"keywords": [
        "surprising","surprised","unexpected","unexpectedly",
        "counter-intuitive","counterintuitive","contrary to",
        "challenge","challenged","overturns","rethink",
        "for the first time","first time","never before",
        "unprecedented","remarkable","unusual","rare",
        "paradox","mystery","puzzle"]},
    "Bad News": {"keywords": [
        "risk","danger","dangerous","threat","threaten",
        "harm","harmful","toxic","death","mortality",
        "cancer","disease","epidemic","pandemic","crisis",
        "pollution","contamination","decline","loss",
        "extinction","collapse","damage","warning",
        "alarming","concern","worried"]},
    "Good News": {"keywords": [
        "breakthrough","cure","solution","solve","solved",
        "success","successful","promising","hope","hopeful",
        "improve","improved","improvement","benefit",
        "protect","protection","prevent","prevention",
        "advance","advancement","progress","discovery",
        "discover","discovered","innovation","innovative"]},
    "Magnitude": {"keywords": [
        "million","billion","thousand","percent",
        "global","worldwide","nationwide","national",
        "population","large-scale","massive","vast",
        "widespread","common","prevalent","majority",
        "significant","substantially","dramatically",
        "double","triple","half"]},
    "Relevance": {"keywords": [
        "diet","food","eat","drink","coffee","alcohol",
        "exercise","sleep","weight","obesity",
        "smoking","screen","phone","social media",
        "children","pregnancy","aging","elderly",
        "cost","price","afford","income","salary",
        "school","education","work","workplace",
        "commut","driving","travel"]},
    "Power Elite": {"keywords": [
        "harvard","oxford","cambridge","stanford","mit",
        "nasa","who","cdc","fda","nih",
        "lancet","nature","science","nejm",
        "world health","united nations",
        "government","federal","congress","parliament",
        "professor","leading","expert","authority"]},
    "Entertainment": {"keywords": [
        "funny","humor","amusing","quirky","weird",
        "bizarre","strange","curious","fascinating",
        "cute","adorable","pet","dog","cat",
        "dinosaur","shark","whale","dolphin",
        "sex","sexual","love","dating","attraction",
        "chocolate","beer","wine","pizza",
        "robot","alien","zombie"]},
    "Conflict": {"keywords": [
        "debate","debated","controversial","controversy",
        "conflict","conflicting","disagree","dispute",
        "oppose","opposition","critic","criticism",
        "skeptic","question","questioned","doubt",
        "refute","contradict","versus"]},
}

# ---- Stopwords (same as analysis_3g) ----
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


def tokenize(text: str):
    return [w for w in re.findall(r"\b[a-z]{3,}\b", text.lower())
            if w not in STOPWORDS]


def category_excludes(keywords: list):
    """Single-word keywords become tokenizer-level exclusions."""
    out = set()
    for kw in keywords:
        kw = kw.lower()
        if " " in kw or "-" in kw:
            continue  # multi-word phrases won't match tokens
        out.add(kw)
    return out


def top_k_df(token_sets: list, K: int, exclude: set) -> set:
    """Top-K by document frequency across a list of pre-tokenised sets."""
    cnt = Counter()
    for ts in token_sets:
        cnt.update(ts - exclude)
    return {w for w, _ in cnt.most_common(K)}


def top_k_lor(in_sets: list, out_sets: list, K: int, exclude: set,
              prior: float = 1.0, min_total: int = 5) -> set:
    """Top-K by log-odds ratio z-score: in-category vs out-of-category."""
    import math
    n_in = len(in_sets)
    n_out = len(out_sets)
    if n_in == 0 or n_out == 0:
        return set()
    cnt_in = Counter()
    cnt_out = Counter()
    for ts in in_sets:
        cnt_in.update(ts - exclude)
    for ts in out_sets:
        cnt_out.update(ts - exclude)

    scored = []
    vocab = set(cnt_in) | set(cnt_out)
    for w in vocab:
        fi = cnt_in.get(w, 0) + prior
        fo = cnt_out.get(w, 0) + prior
        total = cnt_in.get(w, 0) + cnt_out.get(w, 0)
        if total < min_total:
            continue
        oi = fi / (n_in - fi + prior) if (n_in - fi + prior) > 0 else 1e-9
        oo = fo / (n_out - fo + prior) if (n_out - fo + prior) > 0 else 1e-9
        lor = math.log2(oi / oo)
        var = 1.0 / fi + 1.0 / fo
        z = lor / math.sqrt(var)
        scored.append((w, z))
    scored.sort(key=lambda kv: -kv[1])
    return {w for w, _ in scored[:K]}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


def main() -> None:
    # ---- load triplet set (same filter as §5.4.1/§5.4.2) ----
    preds = json.loads(PRED_PATH.read_text())
    test = json.loads(TEST_PATH.read_text())
    doi_abs = {d["doi"]: d.get("abstract", "") for d in test}
    print(f"loading news_articles.json ...")
    news = json.loads(NEWS_PATH.read_text())
    doi_news = {a["doi"]: a for a in news if a.get("success")}
    print(f"news (success): {len(doi_news)}")

    triplets = []
    for p in preds:
        if p["true_label"] != 1 or p["predicted"] != 1:
            continue
        art = doi_news.get(p["doi"])
        if not art:
            continue
        news_text = art.get("text", "") or ""
        title = (p.get("title") or "").strip()
        if not (p["doi"].lower() in news_text.lower() or
                (title and title.lower() in news_text.lower())):
            continue
        abstract = doi_abs.get(p["doi"], "") or ""
        expl = (p.get("explanation") or "").strip()
        if not (abstract and expl and news_text):
            continue
        triplets.append({"expl": expl, "news": news_text[:3000]})
    n_tri = len(triplets)
    print(f"triplets: {n_tri}")

    # Pre-tokenise once
    expl_tokens = [set(tokenize(t["expl"])) for t in triplets]
    news_tokens = [set(tokenize(t["news"])) for t in triplets]
    expl_lower = [t["expl"].lower() for t in triplets]
    news_lower = [t["news"].lower() for t in triplets]

    rng = np.random.default_rng(RNG_SEED)

    # ---- per-category analysis ----
    results = {}
    for cat, info in NEWS_VALUES.items():
        kws = info["keywords"]
        exclude = category_excludes(kws)

        # Union category membership
        idx = [i for i in range(n_tri)
               if any(kw in expl_lower[i] for kw in kws)
               or any(kw in news_lower[i] for kw in kws)]
        n_cat = len(idx)
        if n_cat < 20:
            results[cat] = {"n": n_cat, "jaccard": None}
            continue

        # Indices in vs. out of category
        out_idx = [i for i in range(n_tri) if i not in set(idx)]
        in_expl = [expl_tokens[i] for i in idx]
        out_expl = [expl_tokens[i] for i in out_idx]
        in_news = [news_tokens[i] for i in idx]
        out_news = [news_tokens[i] for i in out_idx]

        # Observed top-K by log-odds z-score (in-category vs out-of-category)
        m_top = top_k_lor(in_expl, out_expl, K, exclude)
        n_top = top_k_lor(in_news, out_news, K, exclude)
        shared = m_top & n_top
        J_obs = jaccard(m_top, n_top)

        # Bootstrap CI: resample only the IN-category pool
        boots = np.empty(N_BOOT)
        for b in range(N_BOOT):
            sample = rng.choice(idx, size=n_cat, replace=True)
            in_e_b = [expl_tokens[i] for i in sample]
            in_n_b = [news_tokens[i] for i in sample]
            mb = top_k_lor(in_e_b, out_expl, K, exclude)
            nb = top_k_lor(in_n_b, out_news, K, exclude)
            boots[b] = jaccard(mb, nb)

        results[cat] = {
            "n": n_cat,
            "jaccard": float(J_obs),
            "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5)),
            "bootstrap": boots.astype(float).tolist(),
            "shared": sorted(shared),
            "model_top_25": sorted(m_top),
            "news_top_25": sorted(n_top),
        }
        print(f"  {cat:<16s} n={n_cat:4d}  J={J_obs:.3f}  "
              f"CI=[{results[cat]['ci_lo']:.3f}, {results[cat]['ci_hi']:.3f}]"
              f"  |shared|={len(shared)}")

    # ---- load §5.4.1 M/N ratios ----
    mn_data = json.loads(NEWSVALUE_JSON.read_text())
    mn_ratio = {x["signal"]: x["ratio"] for x in mn_data}

    # ---- figure β : raincloud per category, sorted by M/N ratio ------------
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#374151",
        "axes.labelcolor": "#1F2937",
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(12, 6.4))

    INK = "#111827"
    MUTED = "#6B7280"

    PALETTE = {
        "Surprise":      "#0072B2",
        "Bad News":      "#D55E00",
        "Good News":     "#009E73",
        "Magnitude":     "#CC79A7",
        "Relevance":     "#56B4E9",
        "Power Elite":   "#E69F00",
        "Entertainment": "#7B3FA5",
        "Conflict":      "#B22222",
    }

    # Sort categories by M/N ratio (ascending: left = worst aligned)
    cats_sorted = sorted(
        [c for c in NEWS_VALUES if results[c].get("jaccard") is not None],
        key=lambda c: mn_ratio.get(c, 0)
    )

    from scipy import stats as sps
    rng2 = np.random.default_rng(1)

    # Draw one raincloud per category at x-position i
    for i, cat in enumerate(cats_sorted):
        r = results[cat]
        boots = np.asarray(r["bootstrap"])
        obs = r["jaccard"]
        color = PALETTE[cat]

        # Half-violin on the LEFT of center line
        try:
            kde = sps.gaussian_kde(boots, bw_method=0.35)
        except np.linalg.LinAlgError:
            kde = None
        if kde is not None and boots.std() > 1e-6:
            ygrid = np.linspace(max(0, boots.min() - 0.02),
                                boots.max() + 0.02, 200)
            dens = kde(ygrid)
            dens = dens / dens.max() * 0.30  # half-width
            ax.fill_betweenx(ygrid, i, i - dens,
                             facecolor=color, alpha=0.35,
                             edgecolor=color, linewidth=0.6, zorder=2)

        # Box (IQR) slightly right of center
        q1, med, q3 = np.percentile(boots, [25, 50, 75])
        lo_w, hi_w = r["ci_lo"], r["ci_hi"]
        box_x = i + 0.04
        box_w = 0.10
        # whisker = 95% CI
        ax.plot([box_x, box_x], [lo_w, hi_w],
                color=color, linewidth=1.1, alpha=0.85, zorder=3)
        # IQR box (filled)
        ax.add_patch(plt.Rectangle(
            (box_x - box_w / 2, q1), box_w, q3 - q1,
            facecolor=color, edgecolor=color,
            alpha=0.85, linewidth=0.6, zorder=4,
        ))
        # median white line
        ax.plot([box_x - box_w / 2, box_x + box_w / 2],
                [med, med], color="white", linewidth=1.4, zorder=5,
                solid_capstyle="round")

        # Observed Jaccard marker (large white dot)
        ax.plot(box_x, obs, "o", color="white",
                markersize=6.5, markeredgecolor=color,
                markeredgewidth=1.4, zorder=6)

    # Global baseline reference
    ax.axhline(GLOBAL_JACCARD, color="#D97706",
               linestyle=":", linewidth=1.2, alpha=0.9, zorder=1)
    ax.text(len(cats_sorted) - 0.4, GLOBAL_JACCARD + 0.004,
            f"global baseline J = {GLOBAL_JACCARD:.3f} (§5.4.2)",
            fontsize=10, color="#B45309", style="italic",
            ha="right", va="bottom")

    # Y-axis of main plot
    ymax = max(r["ci_hi"] for r in results.values()
               if r.get("ci_hi") is not None) + 0.04
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.6, len(cats_sorted) - 0.3)
    ax.set_ylabel("Within-category top-25 vocabulary Jaccard",
                  fontsize=12, color=INK)
    ax.grid(axis="y", alpha=0.15, linestyle="--", linewidth=0.5, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    # ---- secondary y-axis: M/N ratio as a line ----
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    xs_cat = np.arange(len(cats_sorted))
    ratios = np.array([mn_ratio.get(c, np.nan) for c in cats_sorted])

    # neutral trend line connecting all categories
    ax2.plot(xs_cat, ratios, color="#9CA3AF",
             linewidth=1.6, linestyle="-",
             alpha=0.55, zorder=2)
    # category-coloured markers at each node
    for x, r, cat in zip(xs_cat, ratios, cats_sorted):
        col = PALETTE[cat]
        ax2.plot(x, r, marker="D", markersize=10,
                 color=col, markeredgecolor="white",
                 markeredgewidth=1.2, zorder=3)
        # label to the RIGHT of the diamond (avoid raincloud overlap)
        ax2.text(x + 0.16, r, f"{r:.2f}",
                 ha="left", va="center",
                 fontsize=10, color=col, fontweight="bold", zorder=4)

    # reference: equal-frequency line at M/N = 1
    ax2.axhline(1.0, color="#9CA3AF", linestyle="--",
                linewidth=0.8, alpha=0.5, zorder=1)
    ax2.text(len(cats_sorted) - 0.45, 1.005,
             "equal freq. (M/N = 1)",
             fontsize=9, color=MUTED, ha="right", va="bottom",
             style="italic")

    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("M/N mention-frequency ratio  (diamonds, secondary axis)",
                   fontsize=11, color=MUTED)
    ax2.tick_params(axis="y", colors=MUTED, labelsize=10)
    ax2.grid(False)

    # X tick labels on ax (main)
    ax.set_xticks(xs_cat)
    xticklabels = [f"{c}\n(n = {results[c]['n']:,})" for c in cats_sorted]
    ax.set_xticklabels(xticklabels, fontsize=10.5)
    for i, tick in enumerate(ax.get_xticklabels()):
        tick.set_color(PALETTE[cats_sorted[i]])
    ax.tick_params(axis='x', length=0)

    # Small annotation: sort order
    ax.text(-0.55, ymax - 0.003,
            "← lower M/N freq. alignment    |    "
            "higher M/N freq. alignment →",
            fontsize=9.5, color=MUTED, style="italic",
            ha="left", va="top")

    fig.subplots_adjust(top=0.95, bottom=0.14, left=0.07, right=0.93)
    out = PAPER_FIG_DIR / "fig_category_vocabulary.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"\nfig saved: {out}")

    # persist json
    outjson = ANALYSIS_DIR / "category_vocabulary.json"
    outjson.write_text(json.dumps({
        "triplets": n_tri,
        "K": K,
        "n_boot": N_BOOT,
        "global_jaccard_ref": GLOBAL_JACCARD,
        "per_category": results,
    }, indent=2, default=str))
    print(f"json saved: {outjson}")


if __name__ == "__main__":
    main()
