"""
Fetch news article text from CED event URLs.

Samples a batch of news URLs and tries to extract article text.

Usage:
    python scripts/fetch_news_text.py --sample 1000
"""
import os
import sys
import json
import argparse
import time
import random
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR, RANDOM_SEED

# Try newspaper3k or trafilatura for article extraction
try:
    from trafilatura import fetch_url, extract
    USE_TRAFILATURA = True
except ImportError:
    USE_TRAFILATURA = False

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AcademicResearchBot/1.0; +mailto:research@example.com)"
}


def extract_article(url: str, timeout: int = 15) -> dict:
    """Try to extract article title and text from a URL."""
    result = {"url": url, "success": False, "title": "", "text": "", "error": ""}

    try:
        if USE_TRAFILATURA:
            downloaded = fetch_url(url)
            if downloaded:
                text = extract(downloaded, include_comments=False, include_tables=False)
                if text and len(text) > 100:
                    result["text"] = text
                    result["success"] = True
                    # Try to get title
                    from trafilatura.metadata import extract_metadata
                    meta = extract_metadata(downloaded)
                    if meta and meta.title:
                        result["title"] = meta.title
        else:
            resp = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
            if resp.status_code == 200:
                # Basic extraction: just get text between <p> tags
                from html.parser import HTMLParser

                class ParagraphExtractor(HTMLParser):
                    def __init__(self):
                        super().__init__()
                        self.in_p = False
                        self.paragraphs = []
                        self.current = ""

                    def handle_starttag(self, tag, attrs):
                        if tag == "p":
                            self.in_p = True
                            self.current = ""

                    def handle_endtag(self, tag):
                        if tag == "p" and self.in_p:
                            self.in_p = False
                            if len(self.current.strip()) > 30:
                                self.paragraphs.append(self.current.strip())

                    def handle_data(self, data):
                        if self.in_p:
                            self.current += data

                parser = ParagraphExtractor()
                parser.feed(resp.text)
                text = "\n\n".join(parser.paragraphs)
                if len(text) > 100:
                    result["text"] = text[:5000]
                    result["success"] = True

    except Exception as e:
        result["error"] = str(e)[:200]

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    ced_dir = RAW_DATA_DIR / "ced"
    output_dir = RAW_DATA_DIR / "news_text"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "news_articles.json"

    # Collect all events with DOI -> URL mapping
    print("Loading CED events...")
    doi_urls = defaultdict(list)
    for fpath in sorted(ced_dir.glob("newsfeed_*.json")):
        with open(fpath) as f:
            events = json.load(f)
        for e in events:
            doi = e.get("obj_id", "").replace("https://doi.org/", "")
            url = e.get("subj_id", "")
            if doi and url and url.startswith("http"):
                doi_urls[doi].append(url)

    print(f"Total DOIs with news URLs: {len(doi_urls):,d}")
    total_urls = sum(len(v) for v in doi_urls.values())
    print(f"Total news URLs: {total_urls:,d}")

    # Sample unique URLs (one per DOI for efficiency)
    rng = random.Random(RANDOM_SEED)
    all_items = [(doi, rng.choice(urls)) for doi, urls in doi_urls.items()]
    rng.shuffle(all_items)
    sample = all_items[:args.sample]
    print(f"Sampling {len(sample)} URLs to fetch...")

    # Resume
    fetched = []
    fetched_urls = set()
    if output_path.exists():
        with open(output_path) as f:
            fetched = json.load(f)
        fetched_urls = {a["url"] for a in fetched}
        sample = [(doi, url) for doi, url in sample if url not in fetched_urls]
        print(f"Resuming: {len(fetched)} already done, {len(sample)} remaining")

    # Fetch with thread pool
    results = []
    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for doi, url in sample:
            f = executor.submit(extract_article, url)
            futures[f] = doi

        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching"):
            doi = futures[future]
            result = future.result()
            result["doi"] = doi

            if result["success"]:
                success += 1
                results.append(result)
            else:
                failed += 1

            # Checkpoint every 100
            if (success + failed) % 100 == 0:
                all_results = fetched + results
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False)

    # Final save
    all_results = fetched + results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Attempted: {success + failed}")
    print(f"  Success:   {success} ({success/(success+failed)*100:.1f}%)")
    print(f"  Failed:    {failed}")
    print(f"  Total articles saved: {len(all_results)}")
    print(f"  Avg text length: {sum(len(a['text']) for a in all_results) / max(len(all_results),1):.0f} chars")
    print(f"  Saved to {output_path}")


if __name__ == "__main__":
    main()
