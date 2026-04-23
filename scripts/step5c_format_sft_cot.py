"""
Step 5c: Create SFT data with CoT from verified news citing sentences.

Only uses papers where we can confirm the news article references the specific paper
(via DOI or title match). Extracts citing sentences as CoT reasoning.

Input: Title + Abstract
Output (positive with CoT): "Analysis: [citing sentences from news]\nPrediction: Yes"
Output (positive without match): skipped
Output (negative): "Prediction: No"

Usage:
    python scripts/step5c_format_sft_cot.py
"""
import os
import sys
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, SFT_DATA_DIR, RAW_DATA_DIR

INSTRUCTION = (
    "Based on the following academic paper's title and abstract, "
    "predict whether this paper will receive mainstream media news coverage. "
    "If yes, briefly explain why and predict the news angle. "
    "Answer with your analysis and prediction."
)

CITE_KEYWORDS = [
    "study", "research", "published", "journal", "found that",
    "according to", "paper", "findings", "researchers", "scientists",
    "evidence", "suggest", "conclude", "demonstrate", "report",
]


def match_paper_to_news(doi, title, text):
    """Check if news text actually references this specific paper."""
    text_lower = text.lower()

    # Method 1: DOI in text
    if doi.lower() in text_lower:
        return "doi"

    # Method 2: Full title match
    if title and len(title) > 20 and title.lower() in text_lower:
        return "title_full"

    # Method 3: Partial title match
    if title:
        words = [w for w in title.lower().split() if len(w) > 3][:5]
        if len(words) >= 3 and sum(1 for w in words if w in text_lower) >= 3:
            return "title_partial"

    return None


def extract_citing_sentences(text, max_sentences=3):
    """Extract sentences that discuss the research findings."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    citing = []

    for s in sentences:
        s = s.strip()
        if len(s) < 40 or len(s) > 400:
            continue

        s_lower = s.lower()
        # Skip boilerplate
        if any(skip in s_lower for skip in [
            'cookie', 'subscribe', 'sign up', 'newsletter', 'copyright',
            'click here', 'read more', 'advertisement', 'share this',
            'follow us', 'terms of use', 'privacy policy', 'all rights',
        ]):
            continue

        # Check if sentence discusses research
        if any(kw in s_lower for kw in CITE_KEYWORDS):
            citing.append(s)
            if len(citing) >= max_sentences:
                break

    return citing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_citing_sents", type=int, default=3)
    args = parser.parse_args()

    # Load news articles
    news_path = RAW_DATA_DIR / "news_text" / "news_articles.json"
    print(f"Loading news articles from {news_path}...")
    with open(news_path) as f:
        news_articles = json.load(f)
    print(f"  Total: {len(news_articles):,d}")

    # Load papers for title matching
    papers_path = RAW_DATA_DIR / "openalex" / "news_papers.json"
    with open(papers_path) as f:
        papers = json.load(f)
    doi_to_title = {p["doi"]: p.get("title", "") for p in papers}

    # Build DOI -> verified citing sentences
    doi_cot = {}
    match_stats = defaultdict(int)

    for article in news_articles:
        doi = article["doi"]
        text = article.get("text", "")
        title = doi_to_title.get(doi, "")

        match_type = match_paper_to_news(doi, title, text)
        if not match_type:
            match_stats["no_match"] += 1
            continue

        match_stats[match_type] += 1
        citing = extract_citing_sentences(text, args.max_citing_sents)
        if citing:
            doi_cot[doi] = " ".join(citing)

    print(f"\nMatch stats:")
    for k, v in sorted(match_stats.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:,d}")
    print(f"  DOIs with CoT: {len(doi_cot):,d}")

    # Build SFT data from processed splits
    output_dir = SFT_DATA_DIR / "combined_cot"
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        input_path = PROCESSED_DATA_DIR / "combined" / f"{split}.json"
        with open(input_path) as f:
            samples = json.load(f)

        sft_data = []
        pos_with_cot = 0
        pos_skip = 0
        neg_count = 0

        for s in samples:
            title = s.get("title") or ""
            abstract = s.get("abstract") or ""
            doi = s.get("doi", "")
            label = s["label"]

            input_text = f"Title: {title}\nAbstract: {abstract}"

            if label == 1:
                if doi in doi_cot:
                    cot = doi_cot[doi]
                    output = f"Analysis: {cot}\nPrediction: Yes"
                    pos_with_cot += 1
                else:
                    pos_skip += 1
                    continue
            else:
                output = "Analysis: This paper addresses a specialized topic with limited public relevance and no clear news angle.\nPrediction: No"
                neg_count += 1

            sft_data.append({
                "instruction": INSTRUCTION,
                "input": input_text,
                "output": output,
            })

        out_path = output_dir / f"{split}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sft_data, f, ensure_ascii=False, indent=2)
        print(f"\n  {split}: {len(sft_data)} samples (pos_cot={pos_with_cot}, neg={neg_count}, pos_skip={pos_skip})")

    # Show samples
    with open(output_dir / "train.json") as f:
        train = json.load(f)

    print(f"\n{'='*60}")
    print("Sample POSITIVE with CoT:")
    print("=" * 60)
    for s in train:
        if "Prediction: Yes" in s["output"]:
            print(f"Instruction: {s['instruction']}")
            print(f"Input: {s['input'][:200]}...")
            print(f"Output: {s['output'][:400]}...")
            break

    print(f"\n{'='*60}")
    print("Sample NEGATIVE:")
    print("=" * 60)
    for s in train:
        if "Prediction: No" in s["output"]:
            print(f"Input: {s['input'][:200]}...")
            print(f"Output: {s['output']}")
            break

    # Register dataset
    info_path = SFT_DATA_DIR / "dataset_info.json"
    with open(info_path) as f:
        info = json.load(f)
    info["media_combined_cot"] = {
        "file_name": "combined_cot/train.json",
        "formatting": "alpaca",
        "columns": {"prompt": "instruction", "query": "input", "response": "output"}
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nRegistered as: media_combined_cot")


if __name__ == "__main__":
    main()
