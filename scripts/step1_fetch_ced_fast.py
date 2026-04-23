"""
Step 1: Fast CED fetcher — uses concurrent requests to speed up data collection.

Fetches newsfeed events from CED 2017-2023, saves monthly checkpoints.
Skips months already downloaded.

Usage:
    python scripts/step1_fetch_ced_fast.py
"""
import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR

CED_BASE = "https://api.eventdata.crossref.org/v1/events"
EMAIL = "research@example.com"

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
    return any(domain in url.lower() for domain in POLICY_DOMAINS)


def fetch_one_month(source: str, year: int, month: int, output_dir: Path) -> tuple:
    """Fetch all events for one month. Returns (year, month, count)."""
    from_date = f"{year}-{month:02d}-01"
    if month == 12:
        until_date = f"{year}-12-31"
    else:
        until_date = f"{year}-{month+1:02d}-01"

    checkpoint = output_dir / f"{source}_{year}_{month:02d}.json"
    if checkpoint.exists():
        with open(checkpoint) as f:
            data = json.load(f)
        return (year, month, len(data), True)

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
                retries += 1
                if retries > 5:
                    break
                time.sleep(3)
                continue

            data = resp.json()
            events = data.get("message", {}).get("events", [])
            if not events:
                break

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
            time.sleep(0.2)

        except Exception:
            retries += 1
            if retries > 5:
                break
            time.sleep(5)

    with open(checkpoint, "w") as f:
        json.dump(all_events, f)

    return (year, month, len(all_events), False)


def main():
    output_dir = RAW_DATA_DIR / "ced"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build task list
    tasks = []
    for year in range(2017, 2024):
        for month in range(1, 13):
            tasks.append(("newsfeed", year, month))

    print(f"Total months to fetch: {len(tasks)}")

    # Check how many already done
    done = sum(1 for s, y, m in tasks if (output_dir / f"{s}_{y}_{m:02d}.json").exists())
    print(f"Already done: {done}, remaining: {len(tasks) - done}")

    # Use 4 concurrent workers (be polite to CED API)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for source, year, month in tasks:
            f = executor.submit(fetch_one_month, source, year, month, output_dir)
            futures[f] = (year, month)

        pbar = tqdm(total=len(tasks), initial=done, desc="Fetching months")
        for future in as_completed(futures):
            year, month, count, was_cached = future.result()
            if not was_cached:
                pbar.write(f"  {year}-{month:02d}: {count} events")
            pbar.update(1)
        pbar.close()

    # Aggregate
    print(f"\n{'='*60}")
    print("Aggregating DOI counts...")
    print(f"{'='*60}")

    doi_counts = defaultdict(int)
    total_events = 0

    for fpath in sorted(output_dir.glob("newsfeed_*.json")):
        with open(fpath) as f:
            events = json.load(f)
        total_events += len(events)
        for e in events:
            doi = e.get("obj_id", "").replace("https://doi.org/", "")
            if doi:
                doi_counts[doi] += 1

    out_path = output_dir / "news_dois.json"
    with open(out_path, "w") as f:
        json.dump(dict(doi_counts), f)

    print(f"\nTotal newsfeed events: {total_events:,d}")
    print(f"Unique DOIs with news mentions: {len(doi_counts):,d}")
    print(f"\nTop 10 most mentioned DOIs:")
    for doi, count in sorted(doi_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {doi[:60]:60s}  mentions={count}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
