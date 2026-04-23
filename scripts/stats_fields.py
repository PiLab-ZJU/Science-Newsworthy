"""Quick stats: papers by field with news mention counts."""
import json
from collections import Counter, defaultdict
from pathlib import Path

data_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "openalex" / "news_papers.json"

with open(data_path) as f:
    papers = json.load(f)

print(f"Total papers with news mentions: {len(papers):,d}")
print()

fields = Counter(p["field"] for p in papers if p["field"])
field_news = defaultdict(list)
field_citations = defaultdict(list)
for p in papers:
    if p["field"]:
        field_news[p["field"]].append(p.get("news_count", 0))
        field_citations[p["field"]].append(p.get("cited_by_count", 0))

col1 = "Field"
col2 = "Papers"
col3 = "Avg News"
col4 = "Avg Citations"
print(f"  {col1:<50s} {col2:>8s} {col3:>10s} {col4:>14s}")
print("  " + "-" * 84)
for field, count in fields.most_common(30):
    avg_news = sum(field_news[field]) / len(field_news[field])
    avg_cite = sum(field_citations[field]) / len(field_citations[field])
    print(f"  {field:<50s} {count:>8,d} {avg_news:>10.1f} {avg_cite:>14.1f}")

# Year distribution
print()
years = Counter(p.get("publication_date", "")[:4] for p in papers if p.get("publication_date"))
print("Year distribution:")
for year in sorted(years.keys()):
    if year >= "2010":
        print(f"  {year}: {years[year]:>8,d}")
