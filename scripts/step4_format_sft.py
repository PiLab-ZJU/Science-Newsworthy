"""
Step 4: Convert processed data to LLaMA-Factory compatible SFT format (Alpaca style).

Output format:
[
  {
    "instruction": "...",
    "input": "Title: ...\nAbstract: ...",
    "output": "Analysis: ...\n\nPrediction: Yes/No"
  }
]

Usage:
    python scripts/step4_format_sft.py
"""
import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROCESSED_DATA_DIR, SFT_DATA_DIR, PRIMARY_FIELD_NAME,
    MEDIA_INSTRUCTION, POLICY_INSTRUCTION,
)


def format_sample(sample: dict, task: str) -> dict:
    """Convert a single sample to Alpaca SFT format."""
    instruction = MEDIA_INSTRUCTION if task == "media" else POLICY_INSTRUCTION
    input_text = f"Title: {sample['title']}\nAbstract: {sample['abstract']}"

    # Build output with CoT
    cot = sample.get("cot_explanation", "")
    label = "Yes" if sample["label"] == 1 else "No"

    if cot:
        # If CoT already ends with Prediction, use as-is
        if "Prediction:" in cot:
            output_text = cot
        else:
            output_text = f"Analysis: {cot}\n\nPrediction: {label}"
    else:
        # Fallback: no CoT, just prediction
        output_text = f"Prediction: {label}"

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
    }


def format_sample_no_cot(sample: dict, task: str) -> dict:
    """Convert sample without CoT (for ablation A7)."""
    instruction = MEDIA_INSTRUCTION if task == "media" else POLICY_INSTRUCTION
    input_text = f"Title: {sample['title']}\nAbstract: {sample['abstract']}"
    label = "Yes" if sample["label"] == 1 else "No"
    return {
        "instruction": instruction,
        "input": input_text,
        "output": f"Prediction: {label}",
    }


def process_split(input_path: Path, task: str, with_cot: bool = True) -> list:
    """Process one split file."""
    # Try CoT version first
    cot_path = input_path.parent / f"{input_path.stem}_cot.json"
    if with_cot and cot_path.exists():
        with open(cot_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        print(f"  Using CoT version: {cot_path}")
    elif input_path.exists():
        with open(input_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        print(f"  Using base version (no CoT): {input_path}")
    else:
        print(f"  [SKIP] {input_path} not found")
        return []

    format_fn = format_sample if with_cot else format_sample_no_cot
    return [format_fn(s, task) for s in samples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field_name", default=PRIMARY_FIELD_NAME)
    parser.add_argument("--no_cot", action="store_true", help="Generate without CoT (ablation A7)")
    args = parser.parse_args()

    field_dir = args.field_name.lower().replace(" ", "_")
    processed_dir = PROCESSED_DATA_DIR / field_dir
    output_dir = SFT_DATA_DIR / field_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_no_cot" if args.no_cot else ""

    for task in ["media", "policy"]:
        print(f"\n{'='*40} {task.upper()} {'='*40}")
        task_dir = processed_dir / task

        for split in ["train", "val", "test"]:
            input_path = task_dir / f"{split}.json"
            sft_data = process_split(input_path, task, with_cot=not args.no_cot)
            if sft_data:
                out_path = output_dir / f"{task}_{split}{suffix}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(sft_data, f, ensure_ascii=False, indent=2)
                print(f"  -> {out_path} ({len(sft_data)} samples)")

    # Create joint training data (media + policy combined)
    print(f"\n{'='*40} JOINT {'='*40}")
    for split in ["train", "val"]:
        joint = []
        for task in ["media", "policy"]:
            path = output_dir / f"{task}_{split}{suffix}.json"
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    joint.extend(json.load(f))
        if joint:
            import random
            random.seed(42)
            random.shuffle(joint)
            out_path = output_dir / f"joint_{split}{suffix}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(joint, f, ensure_ascii=False, indent=2)
            print(f"  -> {out_path} ({len(joint)} samples)")

    print("\nDone! SFT data ready for LLaMA-Factory.")
    print(f"Copy the train JSON files to LLaMA-Factory/data/ and register in dataset_info.json")


if __name__ == "__main__":
    main()
