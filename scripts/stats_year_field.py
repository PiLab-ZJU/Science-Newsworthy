"""Stats: papers by year x field (2017-2023)."""
import json
from collections import defaultdict
from pathlib import Path

data_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "openalex" / "news_papers.json"

with open(data_path) as f:
    papers = json.load(f)

# Filter 2017-2023
years = range(2017, 2024)
field_year = defaultdict(lambda: defaultdict(int))
field_total = defaultdict(int)

for p in papers:
    year = p.get("publication_date", "")[:4]
    field = p.get("field", "")
    if year.isdigit() and int(year) in years and field:
        field_year[field][int(year)] += 1
        field_total[field] += 1

# Sort fields by total
sorted_fields = sorted(field_total.items(), key=lambda x: -x[1])

# Print table
header = f"{'Field':<45s}"
for y in years:
    header += f" {y:>6d}"
header += f" {'Total':>8s}"
print(header)
print("-" * (45 + 7 * 7 + 9))

grand_total_by_year = defaultdict(int)

for field, total in sorted_fields:
    row = f"  {field:<43s}"
    for y in years:
        count = field_year[field][y]
        row += f" {count:>6,d}"
        grand_total_by_year[y] += count
    row += f" {total:>8,d}"
    print(row)

# Total row
print("-" * (45 + 7 * 7 + 9))
row = f"  {'TOTAL':<43s}"
for y in years:
    row += f" {grand_total_by_year[y]:>6,d}"
row += f" {sum(grand_total_by_year.values()):>8,d}"
print(row)
