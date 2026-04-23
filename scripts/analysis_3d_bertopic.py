"""
Analysis 3d: BERTopic on model explanations and news articles.

Uses TF-IDF backend (no transformer download needed) + UMAP + HDBSCAN.

Usage:
    OPENBLAS_NUM_THREADS=8 python scripts/analysis_3d_bertopic.py
"""
import os, sys, json, re
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer

os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

ANALYSIS_DIR = Path("/mnt/nvme1/lcx/academic_social_impact/analysis")
NEWS_PATH = Path("/home/ubuntu/pilab_jiang/lcx/academic_new_policy/data/raw/news_text/news_articles.json")
PAPERS_PATH = Path("/home/ubuntu/pilab_jiang/lcx/academic_new_policy/data/raw/openalex/news_papers.json")


def run_bertopic(texts, name="", min_topic_size=100):
    """Run BERTopic with TF-IDF backend."""
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from umap import UMAP
    from hdbscan import HDBSCAN

    # Sentence embedding
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer("/mnt/nvme1/hf-model/all-MiniLM-L6-v2")
    embeddings = st_model.encode(texts, batch_size=256, show_progress_bar=True)

    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=min_topic_size, min_samples=10, prediction_data=True)

    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=5)
    ctfidf = ClassTfidfTransformer()

    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf,
        nr_topics="auto",
        verbose=True,
    )

    topics, probs = model.fit_transform(texts, embeddings=embeddings)

    # Print results
    topic_info = model.get_topic_info()
    print(f"\n{name}: {len(set(topics)) - 1} topics found (excluding outliers)")
    print(f"{'Topic':>6s} {'Count':>7s} {'Name':<60s}")
    print("-" * 75)
    for _, row in topic_info.iterrows():
        if row["Topic"] == -1:
            print(f"  {row['Topic']:>4d} {row['Count']:>7d}  [Outliers]")
        else:
            print(f"  {row['Topic']:>4d} {row['Count']:>7d}  {row['Name'][:60]}")

    return model, topics, embeddings


def main():
    # Load predictions
    with open(ANALYSIS_DIR / "test_predictions_with_explanations.json") as f:
        predictions = json.load(f)

    tp = [p for p in predictions if p["true_label"] == 1 and p["predicted"] == 1]
    tn = [p for p in predictions if p["true_label"] == 0 and p["predicted"] == 0]
    fp = [p for p in predictions if p["true_label"] == 0 and p["predicted"] == 1]
    fn = [p for p in predictions if p["true_label"] == 1 and p["predicted"] == 0]
    print(f"TP={len(tp)} TN={len(tn)} FP={len(fp)} FN={len(fn)}")

    # ============================================================
    # BERTopic on YES explanations
    # ============================================================
    yes_preds = tp + fp
    yes_texts = [p["explanation"] for p in yes_preds if p["explanation"]]
    print(f"\nYES explanations: {len(yes_texts)}")

    yes_model, yes_topics, _ = run_bertopic(yes_texts, "YES explanations", min_topic_size=80)

    # Per-field topic distribution
    print(f"\nPer-field dominant YES topics:")
    field_topics = defaultdict(list)
    for i, p in enumerate(yes_preds):
        if p["explanation"] and i < len(yes_topics):
            field_topics[p.get("field", "")].append(yes_topics[i])

    for field in sorted(field_topics.keys(), key=lambda x: -len(field_topics[x])):
        topics_in_field = [t for t in field_topics[field] if t != -1]
        if len(topics_in_field) < 30:
            continue
        dominant = Counter(topics_in_field).most_common(3)
        dom_str = ", ".join([f"T{t}({c})" for t, c in dominant])
        print(f"  {field:<45s} n={len(topics_in_field):>5d}  {dom_str}")

    # ============================================================
    # BERTopic on NO explanations
    # ============================================================
    no_preds = tn + fn
    no_texts = [p["explanation"] for p in no_preds if p["explanation"]]
    print(f"\nNO explanations: {len(no_texts)}")

    no_model, no_topics, _ = run_bertopic(no_texts, "NO explanations", min_topic_size=80)

    # ============================================================
    # BERTopic on NEWS articles
    # ============================================================
    with open(NEWS_PATH) as f:
        news = json.load(f)
    doi_news = {a["doi"]: a for a in news if a.get("success")}

    with open(PAPERS_PATH) as f:
        papers = json.load(f)
    doi_title = {p["doi"]: p.get("title", "") for p in papers}

    news_texts = []
    for p in tp:
        if p["doi"] in doi_news:
            article = doi_news[p["doi"]]
            text = article.get("text", "")
            title = doi_title.get(p["doi"], "")
            if p["doi"].lower() in text.lower() or (title and title.lower() in text.lower()):
                words = text.split()[:500]
                news_texts.append(" ".join(words))

    print(f"\nMatched news articles: {len(news_texts)}")
    if len(news_texts) > 500:
        news_model, news_topics, _ = run_bertopic(news_texts, "NEWS articles", min_topic_size=50)

    # Save
    results = {
        "yes_topics": [],
        "no_topics": [],
        "news_topics": [],
    }

    for tid in range(len(yes_model.get_topic_info()) - 1):
        topic_words = yes_model.get_topic(tid)
        if topic_words:
            results["yes_topics"].append({
                "id": tid,
                "words": [w for w, _ in topic_words[:8]],
                "count": int(yes_model.get_topic_info().loc[yes_model.get_topic_info()["Topic"] == tid, "Count"].values[0]) if tid in yes_model.get_topic_info()["Topic"].values else 0,
            })

    for tid in range(len(no_model.get_topic_info()) - 1):
        topic_words = no_model.get_topic(tid)
        if topic_words:
            results["no_topics"].append({
                "id": tid,
                "words": [w for w, _ in topic_words[:8]],
                "count": int(no_model.get_topic_info().loc[no_model.get_topic_info()["Topic"] == tid, "Count"].values[0]) if tid in no_model.get_topic_info()["Topic"].values else 0,
            })

    if len(news_texts) > 500:
        for tid in range(len(news_model.get_topic_info()) - 1):
            topic_words = news_model.get_topic(tid)
            if topic_words:
                results["news_topics"].append({
                    "id": tid,
                    "words": [w for w, _ in topic_words[:8]],
                    "count": int(news_model.get_topic_info().loc[news_model.get_topic_info()["Topic"] == tid, "Count"].values[0]) if tid in news_model.get_topic_info()["Topic"].values else 0,
                })

    out_path = ANALYSIS_DIR / "bertopic_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
