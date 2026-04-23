"""
Step 2: Fetch paper metadata from OpenAlex for DOIs collected from CED.

Reads news_dois.json (or aggregates from monthly CED files if not yet aggregated),
then batch-queries OpenAlex to get title, abstract, field, etc.

Can run while CED is still collecting — just re-run later to catch new DOIs.

Usage:
    python scripts/step2_fetch_openalex.py
    python scripts/step2_fetch_openalex.py --batch_size 50
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path
from collections import defaultdict

import requests
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR, OPENALEX_API_KEY

API_KEY = os.environ.get("OPENALEX_API_KEY", OPENALEX_API_KEY)
OPENALEX_URL = "https://api.openalex.org/works"


def reconstruct_abstract(inverted_index: dict) -> str:
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join([w for _, w in word_positions])


def aggregate_dois_from_monthly(ced_dir: Path) -> dict:
    """Aggregate DOI counts from all available monthly newsfeed files."""
    doi_counts = defaultdict(int)
    files = sorted(ced_dir.glob("newsfeed_*.json"))
    print(f"Found {len(files)} monthly CED files")

    for fpath in files:
        with open(fpath) as f:
            events = json.load(f)
        for e in events:
            doi = e.get("obj_id", "").replace("https://doi.org/", "")
            if doi:
                doi_counts[doi] += 1

    print(f"Total unique DOIs from available months: {len(doi_counts):,d}")
    return dict(doi_counts)


def fetch_openalex_batch(dois: list) -> list:
    """Query OpenAlex for a batch of DOIs (max 50 per request)."""
    doi_filter = "|".join([f"https://doi.org/{d}" for d in dois])
    params = {
        "filter": f"doi:{doi_filter}",
        "per_page": 50,
        "select": ("id,doi,title,abstract_inverted_index,publication_date,"
                   "primary_topic,cited_by_count,authorships,type"),
    }
    if API_KEY:
        params["api_key"] = API_KEY

    resp = requests.get(OPENALEX_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for work in data.get("results", []):
        abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
        doi = (work.get("doi") or "").replace("https://doi.org/", "")
        if not abstract or len(abstract) < 100 or not doi:
            continue

        authorships = work.get("authorships", [])
        first_author_id = ""
        if authorships and authorships[0].get("author"):
            first_author_id = authorships[0]["author"].get("id", "")

        pt = work.get("primary_topic") or {}
        results.append({
            "doi": doi,
            "title": work.get("title", ""),
            "abstract": abstract,
            "publication_date": work.get("publication_date", ""),
            "type": work.get("type", ""),
            "field": pt.get("field", {}).get("display_name", ""),
            "subfield": pt.get("subfield", {}).get("display_name", ""),
            "topic": pt.get("display_name", ""),
            "cited_by_count": work.get("cited_by_count", 0),
            "first_author_id": first_author_id,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--checkpoint_interval", type=int, default=200,
                        help="Save checkpoint every N batches")
    args = parser.parse_args()

    ced_dir = RAW_DATA_DIR / "ced"
    output_dir = RAW_DATA_DIR / "openalex"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "news_papers.json"

    # Aggregate DOIs from available CED monthly files
    doi_counts = aggregate_dois_from_monthly(ced_dir)

    # Also save/update aggregated dois
    dois_path = ced_dir / "news_dois.json"
    with open(dois_path, "w") as f:
        json.dump(doi_counts, f)

    all_dois = list(doi_counts.keys())

    # Resume: load already fetched papers
    fetched_papers = []
    fetched_dois = set()
    if output_path.exists():
        with open(output_path) as f:
            fetched_papers = json.load(f)
        fetched_dois = {p["doi"] for p in fetched_papers}
        print(f"Already fetched: {len(fetched_papers)} papers")

    remaining_dois = [d for d in all_dois if d not in fetched_dois]
    print(f"Remaining DOIs to fetch: {len(remaining_dois):,d}")

    if not remaining_dois:
        print("Nothing to fetch. All DOIs already processed.")
        return

    # Batch query OpenAlex
    batches = [remaining_dois[i:i+args.batch_size]
               for i in range(0, len(remaining_dois), args.batch_size)]

    new_papers = []
    failed = 0

    for i, batch in enumerate(tqdm(batches, desc="OpenAlex")):
        try:
            papers = fetch_openalex_batch(batch)
            new_papers.extend(papers)
        except Exception as e:
            failed += 1
            if failed > 20:
                print(f"\nToo many failures ({failed}), stopping.")
                break
            tqdm.write(f"  Error: {e}")
            time.sleep(2)
            continue

        # Checkpoint
        if (i + 1) % args.checkpoint_interval == 0:
            all_papers = fetched_papers + new_papers
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_papers, f, ensure_ascii=False)
            tqdm.write(f"  Checkpoint: {len(all_papers)} papers total")

        time.sleep(0.1)  # Rate limiting

    # Final save
    all_papers = fetched_papers + new_papers
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_papers, f, ensure_ascii=False, indent=2)

    # Add news_count to papers
    for p in all_papers:
        p["news_count"] = doi_counts.get(p["doi"], 0)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_papers, f, ensure_ascii=False, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  DOIs queried: {len(all_dois):,d}")
    print(f"  Papers retrieved: {len(all_papers):,d}")
    print(f"  Hit rate: {len(all_papers)/len(all_dois)*100:.1f}%")

    # Field distribution
    from collections import Counter
    fields = Counter(p["field"] for p in all_papers if p["field"])
    print(f"\n  Top fields:")
    for field, count in fields.most_common(10):
        print(f"    {field:40s} {count:>6,d}")

    print(f"\n  Saved to {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
