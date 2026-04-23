"""
Step 5b: Create SFT data using news articles.

- Only positive samples with matched news articles + all negative samples
- Output: structured format with news_title and prediction
- Prompt includes task description

Usage:
    python scripts/step5b_format_sft_with_news.py --field medicine
"""
import os
import sys
import json
import argparse
import re
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, SFT_DATA_DIR, RAW_DATA_DIR

INSTRUCTION = (
    "You are an expert in science communication and media analysis. "
    "Given an academic paper's title and abstract, determine whether this paper "
    "will receive mainstream media news coverage. If yes, predict the likely news headline. "
    "Respond in the following format:\n"
    "News Title: <predicted news headline or None>\n"
    "Prediction: <Yes or No>"
)


def extract_news_title(article: dict) -> str:
    """Extract news title from article data."""
    title = article.get("title", "")
    if title and len(title) > 10:
        return title.strip()

    # Fallback: extract from first sentence of text
    text = article.get("text", "")
    if text:
        first_line = text.split("\n")[0].strip()
        if 10 < len(first_line) < 200:
            return first_line
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", default="medicine")
    args = parser.parse_args()

    # Load news articles
    news_path = RAW_DATA_DIR / "news_text" / "news_articles.json"
    with open(news_path) as f:
        news_articles = json.load(f)

    # Build DOI -> news title mapping
    doi_news_title = {}
    for article in news_articles:
        if article.get("success"):
            doi = article["doi"]
            title = extract_news_title(article)
            if title:
                doi_news_title[doi] = title

    print(f"News articles with valid titles: {len(doi_news_title):,d}")

    # Load processed data
    field_dir = PROCESSED_DATA_DIR / args.field
    output_dir = SFT_DATA_DIR / f"{args.field}_news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        with open(field_dir / f"{split}.json") as f:
            samples = json.load(f)

        sft_data = []
        pos_with_news = 0
        pos_skip = 0
        neg_count = 0

        for s in samples:
            title = s.get("title") or ""
            abstract = s.get("abstract") or ""
            doi = s.get("doi", "")
            label = s["label"]

            input_text = f"Title: {title}\nAbstract: {abstract}"

            if label == 1:
                if doi in doi_news_title:
                    news_title = doi_news_title[doi]
                    output = f"News Title: {news_title}\nPrediction: Yes"
                    pos_with_news += 1
                else:
                    pos_skip += 1
                    continue  # Skip positive samples without news
            else:
                output = "News Title: None\nPrediction: No"
                neg_count += 1

            sft_data.append({
                "instruction": INSTRUCTION,
                "input": input_text,
                "output": output,
            })

        out_path = output_dir / f"{split}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sft_data, f, ensure_ascii=False, indent=2)
        print(f"  {split}: {len(sft_data)} samples (pos_with_news={pos_with_news}, neg={neg_count}, pos_skipped={pos_skip})")

    # Register in dataset_info
    info_path = SFT_DATA_DIR / "dataset_info.json"
    with open(info_path) as f:
        info = json.load(f)

    info[f"media_{args.field}_news"] = {
        "file_name": f"{args.field}_news/train.json",
        "formatting": "alpaca",
        "columns": {"prompt": "instruction", "query": "input", "response": "output"}
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    # Show samples
    with open(output_dir / "train.json") as f:
        train = json.load(f)

    print(f"\n--- Sample positive ---")
    for s in train:
        if "Prediction: Yes" in s["output"]:
            print(f"  Input:  {s['input'][:150]}...")
            print(f"  Output: {s['output']}")
            break

    print(f"\n--- Sample negative ---")
    for s in train:
        if "Prediction: No" in s["output"]:
            print(f"  Input:  {s['input'][:150]}...")
            print(f"  Output: {s['output']}")
            break


if __name__ == "__main__":
    main()
