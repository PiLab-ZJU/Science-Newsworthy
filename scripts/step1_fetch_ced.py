"""
Step 1: Fetch Crossref Event Data (CED) for newsfeed and web sources.

Collects all events from 2017-2023, extracts DOIs and their mention counts.

Usage:
    python scripts/step1_fetch_ced.py
    python scripts/step1_fetch_ced.py --source newsfeed --year 2019
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import requests
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR

CED_BASE = "https://api.eventdata.crossref.org/v1/events"
EMAIL = "research@example.com"

# Policy domain filter (for web source)
POLICY_DOMAINS = [
    ".gov", ".gov.uk", ".gov.au", ".gov.ca", ".gov.cn",
    ".gc.ca", ".gob.mx", ".governo.it",
    "who.int", "un.org", "unicef.org", "undp.org",
    "oecd.org", "worldbank.org", "imf.org",
    "europa.eu", "ec.europa.eu",
    "cdc.gov", "nih.gov", "fda.gov", "epa.gov",
    "nice.org.uk", "nhs.uk", "ecdc.europa.eu",
    "parliament.uk", "congress.gov", "europarl.europa.eu",
    "rand.org", "brookings.edu", "nber.org",
    "chathamhouse.org", "cfr.org",
]


def is_policy_source(url: str) -> bool:
    url_lower = url.lower()
    return any(domain in url_lower for domain in POLICY_DOMAINS)


def fetch_ced_month(source: str, year: int, month: int, output_dir: Path) -> int:
    """Fetch all CED events for one source/month, save raw events."""
    from_date = f"{year}-{month:02d}-01"
    if month == 12:
        until_date = f"{year}-12-31"
    else:
        until_date = f"{year}-{month+1:02d}-01"

    checkpoint_path = output_dir / f"{source}_{year}_{month:02d}.json"
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            existing = json.load(f)
        print(f"  [SKIP] {source} {year}-{month:02d}: already have {len(existing)} events")
        return len(existing)

    all_events = []
    cursor = None
    retries = 0

    while True:
        params = {
            "mailto": EMAIL,
            "source": source,
            "from-occurred-date": from_date,
            "until-occurred-date": until_date,
            "rows": 1000,
        }
        if cursor:
            params["cursor"] = cursor

        try:
            resp = requests.get(CED_BASE, params=params, timeout=120)
            if resp.status_code != 200:
                print(f"    HTTP {resp.status_code}, retrying...")
                retries += 1
                if retries > 5:
                    print(f"    Too many retries, stopping at {len(all_events)} events")
                    break
                time.sleep(5)
                continue

            data = resp.json()
            events = data.get("message", {}).get("events", [])

            if not events:
                break

            # Extract only needed fields to save space
            for e in events:
                all_events.append({
                    "obj_id": e.get("obj_id", ""),
                    "subj_id": e.get("subj_id", ""),
                    "occurred_at": e.get("occurred_at", ""),
                    "source_id": e.get("source_id", ""),
                })

            cursor = data.get("message", {}).get("next-cursor")
            if not cursor:
                break

            retries = 0
            time.sleep(0.3)

            if len(all_events) % 5000 == 0:
                print(f"    ... {len(all_events)} events collected")

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as ex:
            print(f"    Connection error: {ex}, retrying...")
            retries += 1
            if retries > 5:
                break
            time.sleep(10)

    # Save checkpoint
    with open(checkpoint_path, "w") as f:
        json.dump(all_events, f)

    print(f"  {source} {year}-{month:02d}: {len(all_events)} events")
    return len(all_events)


def aggregate_doi_counts(output_dir: Path, source: str) -> dict:
    """Aggregate DOI mention counts from all monthly files for a source."""
    doi_counts = defaultdict(int)
    doi_sources = defaultdict(list)  # Track source URLs for policy filtering

    for fpath in sorted(output_dir.glob(f"{source}_*.json")):
        with open(fpath, "r") as f:
            events = json.load(f)
        for e in events:
            doi = e.get("obj_id", "").replace("https://doi.org/", "")
            if doi:
                doi_counts[doi] += 1
                if source == "web":
                    doi_sources[doi].append(e.get("subj_id", ""))

    return dict(doi_counts), dict(doi_sources)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["newsfeed", "web", "all"], default="all")
    parser.add_argument("--year", type=int, default=None, help="Fetch single year")
    parser.add_argument("--year_start", type=int, default=2017)
    parser.add_argument("--year_end", type=int, default=2023)
    args = parser.parse_args()

    output_dir = RAW_DATA_DIR / "ced"
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = ["newsfeed", "web"] if args.source == "all" else [args.source]
    years = [args.year] if args.year else list(range(args.year_start, args.year_end + 1))

    # Fetch events
    for source in sources:
        print(f"\n{'='*60}")
        print(f"Fetching CED source: {source}")
        print(f"{'='*60}")

        total = 0
        for year in years:
            print(f"\n  Year {year}:")
            for month in range(1, 13):
                count = fetch_ced_month(source, year, month, output_dir)
                total += count
        print(f"\n  Total {source} events: {total:,d}")

    # Aggregate DOI counts
    print(f"\n{'='*60}")
    print("Aggregating DOI counts...")
    print(f"{'='*60}")

    for source in sources:
        doi_counts, doi_sources = aggregate_doi_counts(output_dir, source)

        if source == "newsfeed":
            out_path = output_dir / "news_dois.json"
            with open(out_path, "w") as f:
                json.dump(doi_counts, f)
            print(f"\nNewsfeed: {len(doi_counts):,d} unique DOIs")
            print(f"  Top 5 by mention count:")
            for doi, count in sorted(doi_counts.items(), key=lambda x: -x[1])[:5]:
                print(f"    {doi[:50]:50s}  mentions={count}")
            print(f"  Saved to {out_path}")

        elif source == "web":
            # Filter for policy domains
            policy_doi_counts = defaultdict(int)
            for doi, urls in doi_sources.items():
                policy_mentions = sum(1 for u in urls if is_policy_source(u))
                if policy_mentions > 0:
                    policy_doi_counts[doi] = policy_mentions

            out_path_all = output_dir / "web_dois.json"
            out_path_policy = output_dir / "policy_dois.json"

            with open(out_path_all, "w") as f:
                json.dump(doi_counts, f)
            with open(out_path_policy, "w") as f:
                json.dump(dict(policy_doi_counts), f)

            print(f"\nWeb: {len(doi_counts):,d} unique DOIs total")
            print(f"  Policy (domain-filtered): {len(policy_doi_counts):,d} unique DOIs")
            print(f"  Saved to {out_path_all} and {out_path_policy}")

    print("\nDone! Next: run step2_fetch_openalex.py to get paper metadata.")


if __name__ == "__main__":
    main()
