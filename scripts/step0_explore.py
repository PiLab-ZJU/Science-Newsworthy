"""
Step 0: Data exploration — determine the best research fields using OpenAlex group_by.

Explores field-level and subfield-level distributions of paper counts,
helping decide the primary experiment field.

Note: OpenAlex does NOT have cited_by_news_count/cited_by_policy_count filters.
News/policy labels will come from Altmetric API in a later step.

Usage:
    python scripts/step0_explore.py
"""
import os
import sys
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENALEX_BASE_URL, PUBLICATION_YEARS, OPENALEX_API_KEY

API_KEY = os.environ.get("OPENALEX_API_KEY", OPENALEX_API_KEY)


def build_url(extra_filter: str, group_by: str) -> str:
    base_filter = f"publication_year:{PUBLICATION_YEARS},has_abstract:true"
    if extra_filter:
        base_filter += f",{extra_filter}"
    url = f"{OPENALEX_BASE_URL}?filter={base_filter}&group_by={group_by}"
    if API_KEY:
        url += f"&api_key={API_KEY}"
    return url


def explore_field_distribution():
    """Show paper count distribution by field."""
    print("=" * 60)
    print("Paper count distribution by field (2016-2020, with abstract)")
    print("=" * 60)

    url = build_url("", "primary_topic.field.id")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    print(f"\n{'Field':<50s} {'Count':>10s}")
    print("-" * 62)
    for item in sorted(data["group_by"], key=lambda x: x["count"], reverse=True)[:15]:
        print(f"  {item['key_display_name']:<48s} {item['count']:>10,d}")


def explore_subfield_distribution(field_id: str = "fields/27", field_name: str = "Medicine"):
    """Show paper count distribution by subfield within a specific field."""
    print(f"\n{'=' * 60}")
    print(f"Subfield distribution within {field_name} ({field_id})")
    print("=" * 60)

    extra = f"primary_topic.field.id:{field_id}"
    url = build_url(extra, "primary_topic.subfield.id")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    print(f"\n{'Subfield':<55s} {'Count':>10s}")
    print("-" * 67)
    for item in sorted(data["group_by"], key=lambda x: x["count"], reverse=True)[:20]:
        print(f"  {item['key_display_name']:<53s} {item['count']:>10,d}")


def explore_year_distribution(field_id: str = "fields/27", field_name: str = "Medicine"):
    """Show paper count distribution by publication year."""
    print(f"\n{'=' * 60}")
    print(f"Year distribution for {field_name}")
    print("=" * 60)

    extra = f"primary_topic.field.id:{field_id}"
    url = build_url(extra, "publication_year")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    print(f"\n{'Year':<10s} {'Count':>10s}")
    print("-" * 22)
    for item in sorted(data["group_by"], key=lambda x: x["key"]):
        print(f"  {item['key']:<8s} {item['count']:>10,d}")


def explore_high_citation_distribution(field_id: str = "fields/27", field_name: str = "Medicine"):
    """Show distribution of highly-cited papers (potential social impact proxy)."""
    print(f"\n{'=' * 60}")
    print(f"High-citation papers in {field_name} (proxy for visibility)")
    print("=" * 60)

    thresholds = [0, 10, 50, 100, 500]
    for threshold in thresholds:
        extra = f"primary_topic.field.id:{field_id},cited_by_count:>{threshold}"
        url = build_url(extra, "publication_year")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        total = sum(item["count"] for item in data["group_by"])
        print(f"  cited_by_count > {threshold:<5d}: {total:>10,d} papers")


if __name__ == "__main__":
    explore_field_distribution()

    # Explore main candidate fields
    for fid, fname in [
        ("fields/27", "Medicine"),
        ("fields/23", "Environmental Science"),
        ("fields/33", "Social Sciences"),
    ]:
        explore_subfield_distribution(fid, fname)
        explore_year_distribution(fid, fname)

    explore_high_citation_distribution()

    print(f"\n{'=' * 60}")
    print("RECOMMENDATION: Use Medicine (fields/27) as primary field.")
    print("News/policy labels will be obtained from Altmetric API.")
    print("=" * 60)
