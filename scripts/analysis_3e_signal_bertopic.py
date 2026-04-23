"""
Analysis 3e: BERTopic on extracted signal phrases from model explanations.

Pre-processes explanations to extract reasoning phrases (remove boilerplate),
then clusters to discover signal categories.

Usage:
    OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python scripts/analysis_3e_signal_bertopic.py
"""
import os, sys, json, re
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")

ANALYSIS_DIR = Path("/mnt/nvme1/lcx/academic_social_impact/analysis")
NEWS_PATH = Path("/home/ubuntu/pilab_jiang/lcx/academic_new_policy/data/raw/news_text/news_articles.json")
PAPERS_PATH = Path("/home/ubuntu/pilab_jiang/lcx/academic_new_policy/data/raw/openalex/news_papers.json")

# Boilerplate patterns to remove
BOILERPLATE = [
    r"^i (think|predict|believe) (this |that )?(paper|study|article) will (likely )?(receive|not receive|get) (mainstream )?media (news )?coverage because ",
    r"^i (think|predict|believe) (this |that )?(paper|study|article) will (likely )?(not )?receive (mainstream )?media (news )?coverage because ",
    r"^(this |the )?(paper|study|article) will (likely )?(receive|not receive) (mainstream )?(media )?(news )?coverage because ",
    r"^i (think|predict|believe) (this |that )?(paper|study) will (not )?(receive|get) ",
    r"^(this |the )?(paper|study) (is unlikely to|will not|will likely not) receive .* because ",
    r"^(yes|no)[,.]?\s*",
]


def extract_signal_phrase(explanation):
    """Remove boilerplate and extract the core reasoning."""
    text = explanation.strip()
    text_lower = text.lower()

    # Remove boilerplate prefixes
    for pattern in BOILERPLATE:
        text_lower_new = re.sub(pattern, "", text_lower, count=1)
        if text_lower_new != text_lower:
            # Apply same removal to original case
            removed_len = len(text_lower) - len(text_lower_new)
            text = text[removed_len:]
            break

    # Clean up
    text = text.strip().strip(",").strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    return text if len(text) > 20 else ""


def main():
    # Load predictions
    with open(ANALYSIS_DIR / "test_predictions_with_explanations.json") as f:
        predictions = json.load(f)

    tp = [p for p in predictions if p["true_label"] == 1 and p["predicted"] == 1]
    tn = [p for p in predictions if p["true_label"] == 0 and p["predicted"] == 0]
    fp = [p for p in predictions if p["true_label"] == 0 and p["predicted"] == 1]
    fn = [p for p in predictions if p["true_label"] == 1 and p["predicted"] == 0]
    print(f"TP={len(tp)} TN={len(tn)} FP={len(fp)} FN={len(fn)}")

    # Extract signal phrases
    yes_preds = tp + fp
    no_preds = tn + fn

    yes_phrases = []
    yes_indices = []
    for i, p in enumerate(yes_preds):
        phrase = extract_signal_phrase(p["explanation"])
        if phrase:
            yes_phrases.append(phrase)
            yes_indices.append(i)

    no_phrases = []
    no_indices = []
    for i, p in enumerate(no_preds):
        phrase = extract_signal_phrase(p["explanation"])
        if phrase:
            no_phrases.append(phrase)
            no_indices.append(i)

    print(f"\nYES signal phrases: {len(yes_phrases)}")
    print(f"NO signal phrases: {len(no_phrases)}")
    print(f"\nSample YES phrases:")
    for p in yes_phrases[:5]:
        print(f"  -> {p[:120]}")
    print(f"\nSample NO phrases:")
    for p in no_phrases[:5]:
        print(f"  -> {p[:120]}")

    # BERTopic on YES phrases
    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from bertopic.vectorizers import ClassTfidfTransformer

    st_model = SentenceTransformer("/mnt/nvme1/hf-model/all-MiniLM-L6-v2")

    def run_topic_model(texts, name, min_size=60):
        print(f"\n{'='*60}")
        print(f"BERTopic on {name}")
        print(f"{'='*60}")

        embeddings = st_model.encode(texts, batch_size=256, show_progress_bar=True)

        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=min_size, min_samples=10, prediction_data=True)
        vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=5)

        model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            nr_topics="auto",
            verbose=False,
        )
        topics, _ = model.fit_transform(texts, embeddings=embeddings)

        topic_info = model.get_topic_info()
        print(f"\n{len(set(topics)) - 1} signal categories found:")
        print(f"{'ID':>4s} {'Count':>6s} {'Signal Category':<60s}")
        print("-" * 72)
        for _, row in topic_info.iterrows():
            if row["Topic"] == -1:
                print(f" {row['Topic']:>3d} {row['Count']:>6d}  [Uncategorized]")
            else:
                print(f" {row['Topic']:>3d} {row['Count']:>6d}  {row['Name'][:60]}")

        return model, topics

    yes_model, yes_topics = run_topic_model(yes_phrases, "YES signals (why newsworthy)", min_size=60)
    no_model, no_topics = run_topic_model(no_phrases, "NO signals (why not newsworthy)", min_size=80)

    # Per-field analysis for YES
    print(f"\n{'='*60}")
    print("Per-field dominant YES signals")
    print(f"{'='*60}")

    field_signals = defaultdict(list)
    for idx, topic in zip(yes_indices, yes_topics):
        if topic != -1:
            field = yes_preds[idx].get("field", "")
            field_signals[field].append(topic)

    yes_topic_info = yes_model.get_topic_info()
    topic_names = {}
    for _, row in yes_topic_info.iterrows():
        if row["Topic"] != -1:
            topic_names[row["Topic"]] = row["Name"][:40]

    for field in sorted(field_signals.keys(), key=lambda x: -len(field_signals[x])):
        if len(field_signals[field]) < 30:
            continue
        top3 = Counter(field_signals[field]).most_common(3)
        parts = [f"T{t}:{topic_names.get(t,'')}({c})" for t, c in top3]
        print(f"  {field:<45s} {', '.join(parts)}")

    # News comparison
    print(f"\n{'='*60}")
    print("NEWS signal comparison")
    print(f"{'='*60}")

    with open(NEWS_PATH) as f:
        news = json.load(f)
    doi_news = {a["doi"]: a for a in news if a.get("success")}
    with open(PAPERS_PATH) as f:
        papers = json.load(f)
    doi_title = {p["doi"]: p.get("title", "") for p in papers}

    # Extract first sentences from matched news as "news signal phrases"
    news_phrases = []
    for p in tp:
        if p["doi"] in doi_news:
            article = doi_news[p["doi"]]
            text = article.get("text", "")
            title = doi_title.get(p["doi"], "")
            if p["doi"].lower() in text.lower() or (title and title.lower() in text.lower()):
                sentences = re.split(r'(?<=[.!?])\s+', text)
                good = [s.strip() for s in sentences
                        if 40 < len(s.strip()) < 400
                        and any(kw in s.lower() for kw in
                               ["study", "research", "found", "suggest", "show", "reveal", "discover"])]
                if good:
                    news_phrases.append(good[0])

    print(f"News signal phrases: {len(news_phrases)}")
    if len(news_phrases) > 200:
        news_model, news_topics = run_topic_model(news_phrases, "NEWS signals (what journalists highlighted)", min_size=30)

    # Save results
    results = {"yes_signals": [], "no_signals": [], "news_signals": []}

    for _, row in yes_model.get_topic_info().iterrows():
        if row["Topic"] != -1:
            words = yes_model.get_topic(row["Topic"])
            results["yes_signals"].append({
                "id": int(row["Topic"]),
                "name": row["Name"],
                "count": int(row["Count"]),
                "words": [w for w, _ in words[:8]] if words else [],
            })

    for _, row in no_model.get_topic_info().iterrows():
        if row["Topic"] != -1:
            words = no_model.get_topic(row["Topic"])
            results["no_signals"].append({
                "id": int(row["Topic"]),
                "name": row["Name"],
                "count": int(row["Count"]),
                "words": [w for w, _ in words[:8]] if words else [],
            })

    if len(news_phrases) > 200:
        for _, row in news_model.get_topic_info().iterrows():
            if row["Topic"] != -1:
                words = news_model.get_topic(row["Topic"])
                results["news_signals"].append({
                    "id": int(row["Topic"]),
                    "name": row["Name"],
                    "count": int(row["Count"]),
                    "words": [w for w, _ in words[:8]] if words else [],
                })

    out_path = ANALYSIS_DIR / "signal_bertopic.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
