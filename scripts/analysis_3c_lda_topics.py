"""
Analysis 3c: LDA topic modeling on model explanations and news articles.

1. LDA on model explanations (Yes predictions) → model's signal topics
2. LDA on news articles → journalist's signal topics
3. Compare topic distributions

Usage:
    python scripts/analysis_3c_lda_topics.py
"""
import os, sys, json, re
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

ANALYSIS_DIR = Path("/mnt/nvme1/lcx/academic_social_impact/analysis")
NEWS_PATH = Path("/home/ubuntu/pilab_jiang/lcx/academic_new_policy/data/raw/news_text/news_articles.json")
PAPERS_PATH = Path("/home/ubuntu/pilab_jiang/lcx/academic_new_policy/data/raw/openalex/news_papers.json")


def run_lda(texts, n_topics=10, max_features=3000):
    """Run LDA and return topic-word distributions and document-topic distributions."""
    vectorizer = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=5,
        max_df=0.95,
        ngram_range=(1, 2),
    )
    dtm = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=30,
        learning_method="online",
        batch_size=256,
    )
    doc_topics = lda.fit_transform(dtm)

    # Extract top words per topic
    topics = []
    for i, topic_dist in enumerate(lda.components_):
        top_indices = topic_dist.argsort()[-10:][::-1]
        top_words = [feature_names[j] for j in top_indices]
        topics.append({
            "id": i,
            "top_words": top_words,
            "label": " / ".join(top_words[:4]),
        })

    return topics, doc_topics, lda, vectorizer


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
    # LDA on YES explanations (TP + FP)
    # ============================================================
    print(f"\n{'='*60}")
    print("LDA on YES explanations (why model predicts Yes)")
    print(f"{'='*60}")

    yes_explanations = [p["explanation"] for p in tp + fp if p["explanation"]]
    print(f"Documents: {len(yes_explanations)}")

    yes_topics, yes_doc_topics, yes_lda, yes_vec = run_lda(yes_explanations, n_topics=10)

    print("\nTopics (model's reasons for Yes):")
    for t in yes_topics:
        print(f"  Topic {t['id']}: {t['label']}")
        print(f"    Words: {', '.join(t['top_words'])}")

    # Topic distribution stats
    yes_dominant = np.argmax(yes_doc_topics, axis=1)
    yes_topic_counts = Counter(int(t) for t in yes_dominant)
    print(f"\nTopic frequency:")
    for tid, count in sorted(yes_topic_counts.items(), key=lambda x: -x[1]):
        pct = count / len(yes_explanations) * 100
        print(f"  Topic {tid} ({yes_topics[tid]['label']}): {count} ({pct:.1f}%)")

    # ============================================================
    # LDA on NO explanations (TN + FN)
    # ============================================================
    print(f"\n{'='*60}")
    print("LDA on NO explanations (why model predicts No)")
    print(f"{'='*60}")

    no_explanations = [p["explanation"] for p in tn + fn if p["explanation"]]
    print(f"Documents: {len(no_explanations)}")

    no_topics, no_doc_topics, no_lda, no_vec = run_lda(no_explanations, n_topics=8)

    print("\nTopics (model's reasons for No):")
    for t in no_topics:
        print(f"  Topic {t['id']}: {t['label']}")
        print(f"    Words: {', '.join(t['top_words'])}")

    no_dominant = np.argmax(no_doc_topics, axis=1)
    no_topic_counts = Counter(int(t) for t in no_dominant)
    print(f"\nTopic frequency:")
    for tid, count in sorted(no_topic_counts.items(), key=lambda x: -x[1]):
        pct = count / len(no_explanations) * 100
        print(f"  Topic {tid} ({no_topics[tid]['label']}): {count} ({pct:.1f}%)")

    # ============================================================
    # LDA on NEWS articles (matched with TP)
    # ============================================================
    print(f"\n{'='*60}")
    print("LDA on NEWS articles (what journalists actually wrote)")
    print(f"{'='*60}")

    with open(NEWS_PATH) as f:
        news = json.load(f)
    doi_news = {a["doi"]: a for a in news if a.get("success")}

    with open(PAPERS_PATH) as f:
        papers = json.load(f)
    doi_title = {p["doi"]: p.get("title", "") for p in papers}

    # Get matched news texts
    news_texts = []
    news_dois = []
    for p in tp:
        if p["doi"] in doi_news:
            article = doi_news[p["doi"]]
            text = article.get("text", "")
            title = doi_title.get(p["doi"], "")
            if p["doi"].lower() in text.lower() or (title and title.lower() in text.lower()):
                # Take first 500 words
                words = text.split()[:500]
                news_texts.append(" ".join(words))
                news_dois.append(p["doi"])

    print(f"Matched news articles: {len(news_texts)}")

    if news_texts:
        news_topics, news_doc_topics, news_lda, news_vec = run_lda(news_texts, n_topics=10)

        print("\nTopics (what journalists wrote about):")
        for t in news_topics:
            print(f"  Topic {t['id']}: {t['label']}")
            print(f"    Words: {', '.join(t['top_words'])}")

        news_dominant = np.argmax(news_doc_topics, axis=1)
        news_topic_counts = Counter(int(t) for t in news_dominant)
        print(f"\nTopic frequency:")
        for tid, count in sorted(news_topic_counts.items(), key=lambda x: -x[1]):
            pct = count / len(news_texts) * 100
            print(f"  Topic {tid} ({news_topics[tid]['label']}): {count} ({pct:.1f}%)")

    # ============================================================
    # Per-field topic analysis
    # ============================================================
    print(f"\n{'='*60}")
    print("Per-field dominant YES topic")
    print(f"{'='*60}")

    field_topics = defaultdict(list)
    for i, p in enumerate(tp + fp):
        if p["explanation"]:
            field_topics[p.get("field", "")].append(int(yes_dominant[i]))

    for field in sorted(field_topics.keys(), key=lambda x: -len(field_topics[x])):
        if len(field_topics[field]) < 30:
            continue
        top_topic = Counter(field_topics[field]).most_common(1)[0]
        total = len(field_topics[field])
        print(f"  {field:<45s} Topic {top_topic[0]} ({yes_topics[top_topic[0]]['label'][:30]}) {top_topic[1]/total:.0%}")

    # Save
    results = {
        "yes_topics": yes_topics,
        "yes_topic_counts": dict(yes_topic_counts),
        "no_topics": no_topics,
        "no_topic_counts": dict(no_topic_counts),
        "news_topics": news_topics if news_texts else [],
        "news_topic_counts": dict(news_topic_counts) if news_texts else {},
    }

    out_path = ANALYSIS_DIR / "lda_topics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
