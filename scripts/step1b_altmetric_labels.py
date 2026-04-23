"""
Step 1B: Query Altmetric API to get news/policy labels for papers.

Requires an Altmetric API key. Apply for free research access at:
https://www.altmetric.com/research-access/terms/

Usage:
    python scripts/step1b_altmetric_labels.py --api_key YOUR_KEY
    python scripts/step1b_altmetric_labels.py --api_key YOUR_KEY --field_name Medicine
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path
from tqdm import tqdm
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR, PRIMARY_FIELD_NAME

ALTMETRIC_API_URL = "https://api.altmetric.com/v1/doi"


def query_altmetric(doi: str, api_key: str) -> dict:
    """Query Altmetric API for a single DOI."""
    url = f"{ALTMETRIC_API_URL}/{doi}?key={api_key}"
    resp = requests.get(url, timeout=15)

    if resp.status_code == 404:
        # No Altmetric data for this DOI
        return {"found": False, "news_count": 0, "policy_count": 0}
    elif resp.status_code == 429:
        # Rate limited - wait and retry
        time.sleep(2)
        return query_altmetric(doi, api_key)

    resp.raise_for_status()
    data = resp.json()

    return {
        "found": True,
        "altmetric_id": data.get("altmetric_id", ""),
        "altmetric_score": data.get("score", 0),
        "news_count": data.get("cited_by_msm_count", 0),
        "blog_count": data.get("cited_by_feeds_count", 0),
        "policy_count": data.get("cited_by_policies_count", 0),
        "twitter_count": data.get("cited_by_tweeters_count", 0),
        "wikipedia_count": data.get("cited_by_wikipedia_count", 0),
        "mendeley_count": data.get("readers", {}).get("mendeley", 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True, help="Altmetric API key")
    parser.add_argument("--field_name", default=PRIMARY_FIELD_NAME)
    parser.add_argument("--batch_size", type=int, default=100, help="Checkpoint interval")
    parser.add_argument("--max_papers", type=int, default=None)
    args = parser.parse_args()

    field_dir = args.field_name.lower().replace(" ", "_")
    input_path = RAW_DATA_DIR / field_dir / "openalex_papers.json"
    output_path = RAW_DATA_DIR / field_dir / "papers_with_altmetric.json"

    with open(input_path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    if args.max_papers:
        papers = papers[:args.max_papers]

    # Resume from checkpoint
    processed = []
    processed_dois = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            processed = json.load(f)
        processed_dois = {p["doi"] for p in processed}
        remaining = [p for p in papers if p["doi"] not in processed_dois]
        print(f"Resuming: {len(processed)} done, {len(remaining)} remaining")
    else:
        remaining = papers

    print(f"Querying Altmetric API for {len(remaining)} papers...")
    print(f"Rate limit: ~1 request/sec, estimated time: ~{len(remaining)//60} minutes")

    for i, paper in enumerate(tqdm(remaining, desc="Altmetric")):
        try:
            altmetric = query_altmetric(paper["doi"], args.api_key)
            paper["altmetric"] = altmetric
            paper["news_count"] = altmetric["news_count"]
            paper["policy_count"] = altmetric["policy_count"]
            processed.append(paper)
        except Exception as e:
            print(f"\n  Error for {paper['doi']}: {e}")
            paper["altmetric"] = {"found": False, "news_count": 0, "policy_count": 0}
            paper["news_count"] = 0
            paper["policy_count"] = 0
            processed.append(paper)

        # Checkpoint
        if (i + 1) % args.batch_size == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(processed, f, ensure_ascii=False)
            tqdm.write(f"  Checkpoint: {len(processed)} papers saved")

        time.sleep(1.0)  # Rate limiting: 1 req/sec

    # Final save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    # Summary
    with_altmetric = sum(1 for p in processed if p.get("altmetric", {}).get("found", False))
    with_news = sum(1 for p in processed if p.get("news_count", 0) > 0)
    with_policy = sum(1 for p in processed if p.get("policy_count", 0) > 0)

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total papers queried:     {len(processed):,d}")
    print(f"  With Altmetric data:      {with_altmetric:,d} ({with_altmetric/len(processed)*100:.1f}%)")
    print(f"  With news coverage:       {with_news:,d} ({with_news/len(processed)*100:.1f}%)")
    print(f"  With policy citations:    {with_policy:,d} ({with_policy/len(processed)*100:.1f}%)")
    print(f"{'='*60}")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
