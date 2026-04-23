"""
Step 3: Sample negative samples (no news mention) from OpenAlex.

Matches the field x year distribution of positive samples.
Excludes all DOIs that appear in CED news_dois.json.

Usage:
    python scripts/step3_sample_negatives.py
    python scripts/step3_sample_negatives.py --ratio 1.0 --year_start 2017 --year_end 2023
"""
import os
import sys
import json
import argparse
import time
import random
from pathlib import Path
from collections import Counter, defaultdict

import requests
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR, OPENALEX_API_KEY, RANDOM_SEED

API_KEY = os.environ.get("OPENALEX_API_KEY", OPENALEX_API_KEY)
OPENALEX_URL = "https://api.openalex.org/works"

# OpenAlex field name -> field ID mapping
FIELD_IDS = {
    "Medicine": "fields/27",
    "Social Sciences": "fields/33",
    "Environmental Science": "fields/23",
    "Biochemistry, Genetics and Molecular Biology": "fields/13",
    "Psychology": "fields/32",
    "Physics and Astronomy": "fields/31",
    "Engineering": "fields/22",
    "Neuroscience": "fields/28",
    "Agricultural and Biological Sciences": "fields/11",
    "Earth and Planetary Sciences": "fields/19",
    "Computer Science": "fields/17",
    "Materials Science": "fields/25",
    "Economics, Econometrics and Finance": "fields/20",
    "Health Professions": "fields/36",
    "Immunology and Microbiology": "fields/24",
    "Business, Management and Accounting": "fields/14",
    "Chemistry": "fields/16",
    "Arts and Humanities": "fields/12",
    "Decision Sciences": "fields/18",
    "Energy": "fields/21",
    "Mathematics": "fields/26",
    "Nursing": "fields/29",
    "Pharmacology, Toxicology and Pharmaceutics": "fields/30",
    "Dentistry": "fields/35",
    "Chemical Engineering": "fields/15",
    "Veterinary": "fields/34",
}


def reconstruct_abstract(inverted_index: dict) -> str:
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join([w for _, w in word_positions])


def fetch_negative_batch(field_id: str, year: int, exclude_dois: set,
                         needed: int, max_attempts: int = 5) -> list:
    """Fetch random papers from OpenAlex for a specific field+year, excluding positive DOIs."""
    results = []
    cursor = "*"
    attempts = 0

    while len(results) < needed and attempts < max_attempts:
        params = {
            "filter": (
                f"publication_year:{year},"
                f"has_abstract:true,"
                f"primary_topic.field.id:{field_id},"
                f"type:article,"
                f"has_doi:true"
            ),
            "per_page": 200,
            "cursor": cursor,
            "select": ("id,doi,title,abstract_inverted_index,publication_date,"
                       "primary_topic,cited_by_count,authorships,type"),
        }
        if API_KEY:
            params["api_key"] = API_KEY

        try:
            resp = requests.get(OPENALEX_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            attempts += 1
            time.sleep(2)
            continue

        batch = data.get("results", [])
        if not batch:
            break

        for work in batch:
            doi = (work.get("doi") or "").replace("https://doi.org/", "")
            if not doi or doi in exclude_dois:
                continue

            abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
            if not abstract or len(abstract) < 100:
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
                "news_count": 0,
            })

            if len(results) >= needed:
                break

        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            attempts += 1
            cursor = "*"

        time.sleep(0.1)

    return results[:needed]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Negative:positive ratio (default 1:1)")
    parser.add_argument("--year_start", type=int, default=2017)
    parser.add_argument("--year_end", type=int, default=2023)
    parser.add_argument("--min_field_count", type=int, default=50,
                        help="Skip fields with fewer positive samples than this")
    args = parser.parse_args()

    # Load positive papers
    pos_path = RAW_DATA_DIR / "openalex" / "news_papers.json"
    with open(pos_path) as f:
        all_papers = json.load(f)

    # Load all CED DOIs (to exclude from negatives)
    ced_dois_path = RAW_DATA_DIR / "ced" / "news_dois.json"
    with open(ced_dois_path) as f:
        ced_dois = set(json.load(f).keys())
    print(f"Total CED DOIs to exclude: {len(ced_dois):,d}")

    # Also exclude DOIs already in positive set
    pos_dois = {p["doi"] for p in all_papers}
    exclude_dois = ced_dois | pos_dois
    print(f"Total DOIs to exclude: {len(exclude_dois):,d}")

    # Filter positives to target year range
    years = set(str(y) for y in range(args.year_start, args.year_end + 1))
    positives = [p for p in all_papers
                 if p.get("publication_date", "")[:4] in years and p.get("field")]
    print(f"Positive papers ({args.year_start}-{args.year_end}): {len(positives):,d}")

    # Count field x year distribution
    field_year_counts = defaultdict(int)
    for p in positives:
        key = (p["field"], p["publication_date"][:4])
        field_year_counts[key] += 1

    # Group by field for summary
    field_totals = Counter(p["field"] for p in positives)
    print(f"\nPositive distribution by field:")
    for field, count in field_totals.most_common():
        if count >= args.min_field_count:
            print(f"  {field:<50s} {count:>6,d} -> neg target: {int(count * args.ratio):>6,d}")

    # Fetch negatives
    output_path = RAW_DATA_DIR / "openalex" / "negative_papers.json"

    # Resume
    neg_papers = []
    done_keys = set()
    if output_path.exists():
        with open(output_path) as f:
            neg_papers = json.load(f)
        for p in neg_papers:
            done_keys.add((p["field"], p["publication_date"][:4]))
        exclude_dois |= {p["doi"] for p in neg_papers}
        print(f"\nResuming: {len(neg_papers)} negatives already fetched")

    # Fetch by field x year
    tasks = []
    for (field, year), count in sorted(field_year_counts.items()):
        if field_totals[field] < args.min_field_count:
            continue
        if (field, year) in done_keys:
            continue
        field_id = FIELD_IDS.get(field)
        if not field_id:
            continue
        needed = int(count * args.ratio)
        if needed > 0:
            tasks.append((field, field_id, int(year), needed))

    print(f"\nFetching negatives for {len(tasks)} field-year combinations...")

    for field, field_id, year, needed in tqdm(tasks, desc="Sampling negatives"):
        batch = fetch_negative_batch(field_id, year, exclude_dois, needed)
        neg_papers.extend(batch)

        # Update exclude set
        for p in batch:
            exclude_dois.add(p["doi"])

        # Checkpoint every 10 tasks
        if len(neg_papers) % 5000 < needed:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(neg_papers, f, ensure_ascii=False)

    # Final save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(neg_papers, f, ensure_ascii=False, indent=2)

    # Summary
    neg_field_totals = Counter(p["field"] for p in neg_papers)
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total positive papers: {len(positives):,d}")
    print(f"  Total negative papers: {len(neg_papers):,d}")
    print(f"\n  {'Field':<50s} {'Pos':>6s} {'Neg':>6s}")
    print(f"  {'-'*64}")
    for field, pos_count in field_totals.most_common():
        if pos_count >= args.min_field_count:
            neg_count = neg_field_totals.get(field, 0)
            print(f"  {field:<50s} {pos_count:>6,d} {neg_count:>6,d}")
    print(f"\n  Saved to {output_path}")


if __name__ == "__main__":
    main()
