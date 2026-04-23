"""
Analysis 3f: Discover meta-level signal categories using BERTopic.

Pre-processes explanations to remove domain-specific content words,
keeping only reasoning/signal vocabulary. Then clusters to discover
abstract signal categories (e.g., "Controversy", "Novelty", "Emotional Appeal").

Usage:
    OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python scripts/analysis_3f_meta_signals.py
"""
import os, sys, json, re
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")

ANALYSIS_DIR = Path("/mnt/nvme1/lcx/academic_social_impact/analysis")

# Domain-specific words to remove (keep only reasoning vocabulary)
DOMAIN_WORDS = {
    # Medical/Health
    "cancer", "tumor", "disease", "diabetes", "obesity", "heart", "brain",
    "neural", "neuroscience", "clinical", "patients", "drug", "therapy",
    "treatment", "vaccine", "infection", "virus", "bacteria", "dna", "gene",
    "protein", "cell", "cells", "molecular", "biomedical", "medical",
    "health", "mental", "cognitive", "psychiatric",
    # Environment/Earth
    "climate", "carbon", "ocean", "water", "ice", "species", "ecosystem",
    "pollution", "emission", "fossil", "geological", "marine", "coral",
    "biodiversity", "extinction", "warming", "temperature", "atmospheric",
    # Physics/Space
    "quantum", "particle", "atom", "photon", "laser", "magnetic",
    "gravitational", "star", "planet", "galaxy", "solar", "cosmic",
    "universe", "telescope", "spacecraft", "mars", "moon", "asteroid",
    # Technology
    "algorithm", "robot", "robotics", "semiconductor", "battery",
    "sensor", "wireless", "software", "hardware", "computing", "cyber",
    # Social/Psychology
    "psychology", "sociological", "ethnographic", "demographic",
    "cognitive", "behavioral", "linguistic",
    # Biology
    "plant", "insect", "animal", "bird", "fish", "mammal", "reptile",
    "primate", "whale", "dolphin", "dog", "cat", "pet",
    # Food/Agriculture
    "crop", "agricultural", "farming", "livestock", "nutrition",
    # Chemistry/Materials
    "chemical", "polymer", "catalyst", "nanoparticle", "graphene",
    # General academic
    "journal", "university", "professor", "researcher", "laboratory",
}


def clean_for_signal(text):
    """Remove domain words, keep signal/reasoning vocabulary."""
    words = text.lower().split()
    cleaned = []
    for w in words:
        w_clean = re.sub(r'[^a-z]', '', w)
        if w_clean and len(w_clean) > 2 and w_clean not in DOMAIN_WORDS:
            cleaned.append(w_clean)
    return " ".join(cleaned)


def main():
    with open(ANALYSIS_DIR / "test_predictions_with_explanations.json") as f:
        predictions = json.load(f)

    tp = [p for p in predictions if p["true_label"] == 1 and p["predicted"] == 1]
    tn = [p for p in predictions if p["true_label"] == 0 and p["predicted"] == 0]
    fp = [p for p in predictions if p["true_label"] == 0 and p["predicted"] == 1]
    fn = [p for p in predictions if p["true_label"] == 1 and p["predicted"] == 0]
    print(f"TP={len(tp)} TN={len(tn)} FP={len(fp)} FN={len(fn)}")

    # Clean explanations
    yes_preds = tp + fp
    no_preds = tn + fn

    yes_texts = [clean_for_signal(p["explanation"]) for p in yes_preds if p["explanation"]]
    no_texts = [clean_for_signal(p["explanation"]) for p in no_preds if p["explanation"]]
    yes_texts = [t for t in yes_texts if len(t) > 30]
    no_texts = [t for t in no_texts if len(t) > 30]

    print(f"YES cleaned phrases: {len(yes_texts)}")
    print(f"NO cleaned phrases: {len(no_texts)}")
    print(f"\nSample cleaned YES:")
    for t in yes_texts[:3]:
        print(f"  -> {t[:120]}")
    print(f"\nSample cleaned NO:")
    for t in no_texts[:3]:
        print(f"  -> {t[:120]}")

    # BERTopic
    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    st_model = SentenceTransformer("/mnt/nvme1/hf-model/all-MiniLM-L6-v2")

    def run_signals(texts, name, min_size=80):
        print(f"\n{'='*60}")
        print(f"{name}")
        print(f"{'='*60}")

        embeddings = st_model.encode(texts, batch_size=256, show_progress_bar=True)

        model = BERTopic(
            umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42),
            hdbscan_model=HDBSCAN(min_cluster_size=min_size, min_samples=10, prediction_data=True),
            vectorizer_model=CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=5),
            nr_topics="auto",
            verbose=False,
        )
        topics, _ = model.fit_transform(texts, embeddings=embeddings)

        topic_info = model.get_topic_info()
        print(f"\n{len(set(topics)) - 1} signal categories discovered:")
        print(f"{'Count':>6s}  {'Top Keywords':<70s}")
        print("-" * 78)
        for _, row in topic_info.iterrows():
            if row["Topic"] == -1:
                print(f"{row['Count']:>6d}  [Uncategorized]")
            else:
                words = model.get_topic(row["Topic"])
                kw = ", ".join([w for w, _ in words[:8]]) if words else ""
                print(f"{row['Count']:>6d}  {kw[:70]}")

        return model, topics

    yes_model, yes_topics = run_signals(yes_texts, "YES signals (why newsworthy)", min_size=60)
    no_model, no_topics = run_signals(no_texts, "NO signals (why not newsworthy)", min_size=80)

    # Save
    results = {"yes_signals": [], "no_signals": []}

    for _, row in yes_model.get_topic_info().iterrows():
        if row["Topic"] != -1:
            words = yes_model.get_topic(row["Topic"])
            results["yes_signals"].append({
                "count": int(row["Count"]),
                "keywords": [w for w, _ in words[:10]] if words else [],
            })

    for _, row in no_model.get_topic_info().iterrows():
        if row["Topic"] != -1:
            words = no_model.get_topic(row["Topic"])
            results["no_signals"].append({
                "count": int(row["Count"]),
                "keywords": [w for w, _ in words[:10]] if words else [],
            })

    out_path = ANALYSIS_DIR / "meta_signals.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
