"""
Step 5: Convert processed data to LLaMA-Factory Alpaca SFT format.

Output format:
[
  {
    "instruction": "...",
    "input": "Title: ...\nAbstract: ...",
    "output": "Prediction: Yes/No"
  }
]

Usage:
    python scripts/step5_format_sft.py
    python scripts/step5_format_sft.py --field medicine
    python scripts/step5_format_sft.py --field combined
"""
import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, SFT_DATA_DIR, MEDIA_INSTRUCTION


def format_sample(sample: dict) -> dict:
    """Convert a single sample to Alpaca SFT format."""
    title = sample.get("title") or ""
    abstract = sample.get("abstract") or ""
    label = "Yes" if sample["label"] == 1 else "No"

    return {
        "instruction": MEDIA_INSTRUCTION,
        "input": f"Title: {title}\nAbstract: {abstract}",
        "output": label,
    }


def process_field(field_dir: Path, output_dir: Path, field_name: str):
    """Process one field's data into SFT format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        input_path = field_dir / f"{split}.json"
        if not input_path.exists():
            print(f"  [SKIP] {input_path} not found")
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        sft_data = [format_sample(s) for s in samples]

        out_path = output_dir / f"{split}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sft_data, f, ensure_ascii=False, indent=2)

        pos = sum(1 for s in samples if s["label"] == 1)
        print(f"  {split}: {len(sft_data)} samples (pos={pos}) -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", default="all",
                        help="Field name (directory) or 'all' or 'combined'")
    args = parser.parse_args()

    if args.field == "all":
        # Process combined + all per-field datasets
        fields = [d.name for d in PROCESSED_DATA_DIR.iterdir()
                  if d.is_dir() and (d / "train.json").exists()]
    else:
        fields = [args.field]

    for field in sorted(fields):
        field_dir = PROCESSED_DATA_DIR / field
        output_dir = SFT_DATA_DIR / field

        if not (field_dir / "train.json").exists():
            continue

        print(f"\n{'='*40} {field} {'='*40}")
        process_field(field_dir, output_dir, field)

    # Generate dataset_info.json for LLaMA-Factory
    print(f"\n{'='*60}")
    print("Generating dataset_info.json")
    print("=" * 60)

    dataset_info = {}
    for field_dir in sorted(SFT_DATA_DIR.iterdir()):
        if not field_dir.is_dir():
            continue
        train_path = field_dir / "train.json"
        if not train_path.exists():
            continue

        # Dataset name: field directory name
        name = f"media_{field_dir.name}"
        dataset_info[name] = {
            "file_name": str(train_path.relative_to(SFT_DATA_DIR)),
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            }
        }
        print(f"  Registered: {name} -> {train_path.relative_to(SFT_DATA_DIR)}")

    info_path = SFT_DATA_DIR / "dataset_info.json"
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)
    print(f"\n  Saved to {info_path}")

    print(f"\n{'='*60}")
    print("Done! To use with LLaMA-Factory:")
    print(f"  1. Copy/symlink {SFT_DATA_DIR} to LLaMA-Factory/data/")
    print(f"  2. Merge {info_path} into LLaMA-Factory/data/dataset_info.json")
    print(f"  3. Set dataset: media_combined (or media_medicine, etc.) in YAML config")
    print("=" * 60)


if __name__ == "__main__":
    main()
