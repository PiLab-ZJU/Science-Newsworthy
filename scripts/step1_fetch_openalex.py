"""
Step 1A: Fetch paper metadata (DOI, title, abstract) from OpenAlex.

This step collects papers from OpenAlex. Labels (news/policy) will be
added in step1b using the Altmetric API.

Usage:
    python scripts/step1_fetch_openalex.py
    python scripts/step1_fetch_openalex.py --field_id fields/23 --field_name "Environmental Science"
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
    PRIMARY_FIELD_NAME, RAW_DATA_DIR, OPENALEX_API_KEY,
)

API_KEY = os.environ.get("OPENALEX_API_KEY", OPENALEX_API_KEY)


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
    """Fetch papers from OpenAlex API with cursor-based pagination."""
    results = []
    cursor = "*"

    with tqdm(total=max_results, desc="Fetching") as pbar:
        while len(results) < max_results:
            params = {
                "filter": filters,
                "per_page": 200,
                "cursor": cursor,
                "select": (
                    "id,doi,title,abstract_inverted_index,publication_date,"
                    "primary_topic,cited_by_count,authorships,type"
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
                doi = work.get("doi", "")
                # Skip papers without abstract or DOI (need DOI for Altmetric lookup)
                if not abstract or len(abstract) < 100 or not doi:
                    continue

                # Clean DOI: remove https://doi.org/ prefix
                if doi.startswith("https://doi.org/"):
                    doi = doi[len("https://doi.org/"):]

                # Extract first author ID for author-based splitting
                authorships = work.get("authorships", [])
                first_author_id = ""
                if authorships and authorships[0].get("author"):
                    first_author_id = authorships[0]["author"].get("id", "")

                primary_topic = work.get("primary_topic") or {}
                results.append({
                    "id": work["id"],
                    "doi": doi,
                    "title": work.get("title", ""),
                    "abstract": abstract,
                    "publication_date": work.get("publication_date", ""),
                    "type": work.get("type", ""),
                    "field": primary_topic.get("field", {}).get("display_name", ""),
                    "subfield": primary_topic.get("subfield", {}).get("display_name", ""),
                    "topic": primary_topic.get("display_name", ""),
                    "cited_by_count": work.get("cited_by_count", 0),
                    "first_author_id": first_author_id,
                })

            pbar.update(len(batch))
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break
            time.sleep(0.1)

    return results[:max_results]


def main():
    parser = argparse.ArgumentParser(description="Fetch papers from OpenAlex")
    parser.add_argument("--field_id", default=PRIMARY_FIELD_ID)
    parser.add_argument("--field_name", default=PRIMARY_FIELD_NAME)
    parser.add_argument("--max_results", type=int, default=20000,
                        help="Total papers to fetch (will be filtered by Altmetric later)")
    args = parser.parse_args()

    output_dir = RAW_DATA_DIR / args.field_name.lower().replace(" ", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_filter = (
        f"publication_year:{PUBLICATION_YEARS},has_abstract:true,"
        f"primary_topic.field.id:{args.field_id},has_doi:true"
    )

    print(f"Fetching papers for {args.field_name} ({args.field_id})")
    print(f"Filter: {base_filter}")
    print(f"Target: {args.max_results} papers")

    papers = fetch_works(base_filter, max_results=args.max_results)

    # Save
    out_path = output_dir / "openalex_papers.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(papers)} papers to {out_path}")
    print(f"  With DOI: {sum(1 for p in papers if p['doi']):,d}")
    print(f"  Unique subfields: {len(set(p['subfield'] for p in papers))}")
    print(f"  Year range: {min(p['publication_date'][:4] for p in papers)} - {max(p['publication_date'][:4] for p in papers)}")


if __name__ == "__main__":
    main()
