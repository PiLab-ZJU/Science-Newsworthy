"""
Analysis 3b: Data-driven signal taxonomy via embedding + clustering.

1. Embed all model explanations using sentence-transformers
2. Cluster with K-Means
3. Auto-name clusters from representative texts
4. Compare model vs news signal distributions

Usage:
    python scripts/analysis_3b_signal_clustering.py --gpus 0
"""
import os, sys, json, re
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ANALYSIS_DIR = Path("/mnt/nvme1/lcx/academic_social_impact/analysis")
NEWS_PATH = Path("/home/ubuntu/pilab_jiang/lcx/academic_new_policy/data/raw/news_text/news_articles.json")
PAPERS_PATH = Path("/home/ubuntu/pilab_jiang/lcx/academic_new_policy/data/raw/openalex/news_papers.json")


def get_embeddings(texts, max_features=5000):
    from sklearn.feature_extraction.text import TfidfVectorizer
    print(f"  Using TF-IDF (max_features={max_features})...")
    vec = TfidfVectorizer(max_features=max_features, stop_words="english", ngram_range=(1, 2))
    embeddings = vec.fit_transform(texts).toarray()
    return embeddings


def name_cluster(texts, max_samples=20):
    """Auto-name a cluster by extracting common themes from representative texts."""
    # Extract key phrases
    word_freq = Counter()
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                 "this", "that", "it", "its", "of", "in", "to", "for", "and",
                 "or", "but", "not", "with", "on", "at", "by", "from", "as",
                 "will", "would", "can", "could", "may", "might", "has", "have",
                 "had", "do", "does", "did", "i", "we", "they", "he", "she",
                 "paper", "study", "research", "article", "likely", "because",
                 "think", "predict", "receive", "news", "coverage", "mainstream",
                 "media", "also", "which", "their", "these", "those", "such",
                 "about", "into", "than", "more", "very", "just", "being", "so",
                 "some", "other", "no", "if", "how", "what", "when", "where",
                 "there", "here", "all", "each", "both", "between", "through",
                 "however", "while", "although", "whether", "specific",
                 "particular", "general", "significant", "important", "new"}

    for t in texts[:max_samples]:
        words = re.findall(r'\b[a-z]{4,}\b', t.lower())
        for w in words:
            if w not in stopwords:
                word_freq[w] += 1

    # Get top keywords
    top_words = [w for w, c in word_freq.most_common(5)]
    return " / ".join(top_words) if top_words else "misc"


def main():
    # Load predictions with explanations
    with open(ANALYSIS_DIR / "test_predictions_with_explanations.json") as f:
        predictions = json.load(f)
    print(f"Loaded {len(predictions)} predictions")

    # Separate by prediction type
    tp = [p for p in predictions if p["true_label"] == 1 and p["predicted"] == 1]
    tn = [p for p in predictions if p["true_label"] == 0 and p["predicted"] == 0]
    fp = [p for p in predictions if p["true_label"] == 0 and p["predicted"] == 1]
    fn = [p for p in predictions if p["true_label"] == 1 and p["predicted"] == 0]
    print(f"TP={len(tp)} TN={len(tn)} FP={len(fp)} FN={len(fn)}")

    # Get all explanations
    all_explanations = [p["explanation"] for p in predictions if p["explanation"]]
    all_indices = [i for i, p in enumerate(predictions) if p["explanation"]]
    print(f"Explanations to embed: {len(all_explanations)}")

    # Embed
    print("\nGenerating embeddings...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    global tfidf_vec
    tfidf_vec = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    embeddings = tfidf_vec.fit_transform(all_explanations).toarray()
    print(f"Embedding shape: {embeddings.shape}")

    # Find optimal K
    print("\nFinding optimal K...")
    scores = {}
    for k in [8, 10, 12, 15, 20, 25]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, sample_size=5000)
        scores[k] = score
        print(f"  K={k}: silhouette={score:.4f}")

    best_k = max(scores, key=scores.get)
    print(f"\nBest K={best_k} (silhouette={scores[best_k]:.4f})")

    # Final clustering
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(embeddings)

    # Map back to predictions
    for idx, cluster in zip(all_indices, cluster_labels):
        predictions[idx]["cluster"] = int(cluster)

    # Name clusters
    print(f"\n{'='*60}")
    print("CLUSTER ANALYSIS")
    print(f"{'='*60}")

    cluster_texts = defaultdict(list)
    cluster_types = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
    cluster_fields = defaultdict(lambda: Counter())

    for p in predictions:
        if "cluster" not in p:
            continue
        c = p["cluster"]
        cluster_texts[c].append(p["explanation"])
        cluster_fields[c][p.get("field", "")] += 1

        if p["true_label"] == 1 and p["predicted"] == 1:
            cluster_types[c]["tp"] += 1
        elif p["true_label"] == 0 and p["predicted"] == 0:
            cluster_types[c]["tn"] += 1
        elif p["true_label"] == 0 and p["predicted"] == 1:
            cluster_types[c]["fp"] += 1
        else:
            cluster_types[c]["fn"] += 1

    cluster_info = []
    for c in sorted(cluster_texts.keys()):
        texts = cluster_texts[c]
        name = name_cluster(texts)
        types = cluster_types[c]
        total = sum(types.values())
        yes_rate = (types["tp"] + types["fp"]) / total if total > 0 else 0
        top_field = cluster_fields[c].most_common(1)[0][0] if cluster_fields[c] else ""

        info = {
            "cluster_id": c,
            "name": name,
            "size": total,
            "tp": types["tp"],
            "tn": types["tn"],
            "fp": types["fp"],
            "fn": types["fn"],
            "yes_rate": round(yes_rate, 3),
            "top_field": top_field,
            "sample_texts": texts[:3],
        }
        cluster_info.append(info)

        print(f"\nCluster {c}: \"{name}\" (n={total})")
        print(f"  TP={types['tp']} TN={types['tn']} FP={types['fp']} FN={types['fn']} | Yes rate={yes_rate:.1%}")
        print(f"  Top field: {top_field}")
        print(f"  Sample: {texts[0][:120]}...")

    # === News signal comparison ===
    print(f"\n{'='*60}")
    print("NEWS COMPARISON")
    print(f"{'='*60}")

    # Load news
    with open(NEWS_PATH) as f:
        news = json.load(f)
    doi_news = {a["doi"]: a.get("text", "") for a in news if a.get("success")}

    # Load paper titles for matching
    with open(PAPERS_PATH) as f:
        papers = json.load(f)
    doi_title = {p["doi"]: p.get("title", "") for p in papers}

    # Embed news texts for matched articles
    news_texts_for_embed = []
    news_dois = []
    for p in predictions:
        if p["true_label"] == 1 and p["doi"] in doi_news:
            text = doi_news[p["doi"]]
            title = doi_title.get(p["doi"], "")
            if p["doi"].lower() in text.lower() or (title and title.lower() in text.lower()):
                # Extract first few meaningful sentences
                sentences = re.split(r'(?<=[.!?])\s+', text)
                good = [s.strip() for s in sentences if 40 < len(s.strip()) < 400][:5]
                if good:
                    news_texts_for_embed.append(" ".join(good))
                    news_dois.append(p["doi"])

    print(f"News texts to embed: {len(news_texts_for_embed)}")

    if news_texts_for_embed:
        news_embeddings = tfidf_vec.transform(news_texts_for_embed).toarray()
        news_clusters = km.predict(news_embeddings)

        news_cluster_counts = Counter(int(c) for c in news_clusters)

        print(f"\n{'Cluster':<8s} {'Name':<40s} {'Model':>8s} {'News':>8s} {'Ratio':>8s}")
        print("-" * 74)
        for info in sorted(cluster_info, key=lambda x: -x["size"]):
            c = info["cluster_id"]
            model_count = info["tp"]
            news_count = news_cluster_counts.get(c, 0)
            ratio = model_count / max(news_count, 1)
            print(f"  {c:<6d} {info['name']:<40s} {model_count:>8d} {news_count:>8d} {ratio:>8.2f}")
            info["news_count"] = news_count
            info["ratio"] = round(ratio, 2)

    # Save
    out_path = ANALYSIS_DIR / "signal_clustering.json"
    # Remove sample_texts for cleaner output
    for info in cluster_info:
        info["sample_texts"] = info["sample_texts"][:2]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cluster_info, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")

    # Save embeddings for potential visualization
    np.save(str(ANALYSIS_DIR / "explanation_embeddings.npy"), embeddings)
    np.save(str(ANALYSIS_DIR / "explanation_clusters.npy"), cluster_labels)
    print(f"Embeddings saved to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
