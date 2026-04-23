"""
Step 1: Fetch paper data from OpenAlex API.

Collects positive and negative samples for media and policy prediction tasks.

Usage:
    python scripts/step1_fetch_data.py
    python scripts/step1_fetch_data.py --field_id fields/23 --field_name "Environmental Science"
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
from config import (
    OPENALEX_BASE_URL, PUBLICATION_YEARS, PRIMARY_FIELD_ID,
    PRIMARY_FIELD_NAME, RAW_DATA_DIR,
    MEDIA_POSITIVE_COUNT, MEDIA_NEGATIVE_COUNT,
    POLICY_POSITIVE_COUNT, POLICY_NEGATIVE_COUNT,
)

API_KEY = os.environ.get("OPENALEX_API_KEY", "")


def reconstruct_abstract(inverted_index: dict) -> str:
    """Convert OpenAlex inverted index format to plain text abstract."""
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join([word for _, word in word_positions])


def fetch_works(filters: str, max_results: int = 5000) -> list:
    """
    Fetch papers from OpenAlex API with cursor-based pagination.
    See: https://developers.openalex.org/api-reference/works/filter-works
    """
    results = []
    cursor = "*"

    with tqdm(total=max_results, desc="Fetching") as pbar:
        while len(results) < max_results:
            params = {
                "filter": filters,
                "per_page": 200,
                "cursor": cursor,
                "select": (
                    "id,title,abstract_inverted_index,publication_date,"
                    "primary_topic,cited_by_count,"
                    "cited_by_news_count,cited_by_policy_count,"
                    "authorships"
                ),
            }
            if API_KEY:
                params["api_key"] = API_KEY

            resp = requests.get(OPENALEX_BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            batch = data.get("results", [])
            if not batch:
                break

            for work in batch:
                abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
                if abstract and len(abstract) > 100:
                    # Extract first author ID for author-based splitting
                    authorships = work.get("authorships", [])
                    first_author_id = ""
                    if authorships and authorships[0].get("author"):
                        first_author_id = authorships[0]["author"].get("id", "")

                    primary_topic = work.get("primary_topic") or {}
                    results.append({
                        "id": work["id"],
                        "title": work.get("title", ""),
                        "abstract": abstract,
                        "publication_date": work.get("publication_date", ""),
                        "field": primary_topic.get("field", {}).get("display_name", ""),
                        "subfield": primary_topic.get("subfield", {}).get("display_name", ""),
                        "topic": primary_topic.get("display_name", ""),
                        "cited_by_count": work.get("cited_by_count", 0),
                        "news_count": work.get("cited_by_news_count", 0),
                        "policy_count": work.get("cited_by_policy_count", 0),
                        "first_author_id": first_author_id,
                    })

            pbar.update(len(batch))
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break
            time.sleep(0.1)  # Rate limiting

    return results[:max_results]


def main():
    parser = argparse.ArgumentParser(description="Fetch data from OpenAlex")
    parser.add_argument("--field_id", default=PRIMARY_FIELD_ID)
    parser.add_argument("--field_name", default=PRIMARY_FIELD_NAME)
    parser.add_argument("--media_pos", type=int, default=MEDIA_POSITIVE_COUNT)
    parser.add_argument("--media_neg", type=int, default=MEDIA_NEGATIVE_COUNT)
    parser.add_argument("--policy_pos", type=int, default=POLICY_POSITIVE_COUNT)
    parser.add_argument("--policy_neg", type=int, default=POLICY_NEGATIVE_COUNT)
    args = parser.parse_args()

    output_dir = RAW_DATA_DIR / args.field_name.lower().replace(" ", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_filter = (
        f"publication_year:{PUBLICATION_YEARS},has_abstract:true,"
        f"primary_topic.field.id:{args.field_id}"
    )

    # --- Media task ---
    print(f"\n{'='*60}")
    print(f"Fetching MEDIA data for {args.field_name}")
    print(f"{'='*60}")

    print("\n[1/4] Fetching media positive samples (cited_by_news_count > 0)...")
    media_positive = fetch_works(
        f"{base_filter},cited_by_news_count:>0",
        max_results=args.media_pos,
    )
    print(f"  -> Got {len(media_positive)} samples")

    print("\n[2/4] Fetching media negative samples (cited_by_news_count = 0)...")
    media_negative = fetch_works(
        f"{base_filter},cited_by_news_count:0",
        max_results=args.media_neg,
    )
    print(f"  -> Got {len(media_negative)} samples")

    # --- Policy task ---
    print(f"\n{'='*60}")
    print(f"Fetching POLICY data for {args.field_name}")
    print(f"{'='*60}")

    print("\n[3/4] Fetching policy positive samples (cited_by_policy_count > 0)...")
    policy_positive = fetch_works(
        f"{base_filter},cited_by_policy_count:>0",
        max_results=args.policy_pos,
    )
    print(f"  -> Got {len(policy_positive)} samples")

    print("\n[4/4] Fetching policy negative samples (cited_by_policy_count = 0)...")
    policy_negative = fetch_works(
        f"{base_filter},cited_by_policy_count:0",
        max_results=args.policy_neg,
    )
    print(f"  -> Got {len(policy_negative)} samples")

    # --- Save ---
    media_path = output_dir / "media_data.json"
    with open(media_path, "w", encoding="utf-8") as f:
        json.dump({"positive": media_positive, "negative": media_negative}, f, ensure_ascii=False, indent=2)
    print(f"\nMedia data saved to {media_path}")

    policy_path = output_dir / "policy_data.json"
    with open(policy_path, "w", encoding="utf-8") as f:
        json.dump({"positive": policy_positive, "negative": policy_negative}, f, ensure_ascii=False, indent=2)
    print(f"Policy data saved to {policy_path}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Media:  {len(media_positive)} positive + {len(media_negative)} negative = {len(media_positive)+len(media_negative)}")
    print(f"  Policy: {len(policy_positive)} positive + {len(policy_negative)} negative = {len(policy_positive)+len(policy_negative)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
