"""
Ablation experiments (Section 6 of the research proposal).

A1: title-only vs abstract-only vs title+abstract
A2: training data efficiency (1K/3K/5K/7K)
A3: single-task vs multi-task (handled by comparing configs)
A4: text+metadata vs text-only
A7: CoT vs no-CoT SFT

Usage:
    python analysis/ablation.py --experiment A1 --task media
    python analysis/ablation.py --experiment A2 --task media
"""
import os
import sys
import json
import argparse
import random
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROCESSED_DATA_DIR, SFT_DATA_DIR, OUTPUTS_DIR,
    PRIMARY_FIELD_NAME, MEDIA_INSTRUCTION, POLICY_INSTRUCTION, RANDOM_SEED,
)


def ablation_a1_input_variants(task: str, field_name: str):
    """A1: Generate title-only, abstract-only, and title+abstract SFT data."""
    field_dir = field_name.lower().replace(" ", "_")
    task_dir = PROCESSED_DATA_DIR / field_dir / task
    output_dir = SFT_DATA_DIR / field_dir / "ablation_a1"
    output_dir.mkdir(parents=True, exist_ok=True)

    instruction = MEDIA_INSTRUCTION if task == "media" else POLICY_INSTRUCTION

    for split in ["train", "val"]:
        # Try CoT version first
        cot_path = task_dir / f"{split}_cot.json"
        base_path = task_dir / f"{split}.json"
        path = cot_path if cot_path.exists() else base_path

        with open(path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        for variant, input_fn in [
            ("title_only", lambda s: f"Title: {s['title']}"),
            ("abstract_only", lambda s: f"Abstract: {s['abstract']}"),
            ("title_abstract", lambda s: f"Title: {s['title']}\nAbstract: {s['abstract']}"),
        ]:
            sft_data = []
            for s in samples:
                label = "Yes" if s["label"] == 1 else "No"
                cot = s.get("cot_explanation", "")
                output = f"Analysis: {cot}\n\nPrediction: {label}" if cot else f"Prediction: {label}"
                sft_data.append({
                    "instruction": instruction,
                    "input": input_fn(s),
                    "output": output,
                })

            out_path = output_dir / f"{task}_{variant}_{split}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(sft_data, f, ensure_ascii=False, indent=2)
            print(f"  A1 {variant} {split}: {len(sft_data)} samples -> {out_path}")


def ablation_a2_data_efficiency(task: str, field_name: str):
    """A2: Generate subsets of different sizes for data efficiency curve."""
    field_dir = field_name.lower().replace(" ", "_")
    sft_dir = SFT_DATA_DIR / field_dir
    output_dir = SFT_DATA_DIR / field_dir / "ablation_a2"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = sft_dir / f"{task}_train.json"
    with open(train_path, "r", encoding="utf-8") as f:
        full_train = json.load(f)

    rng = random.Random(RANDOM_SEED)
    sizes = [1000, 3000, 5000, 7000]

    for size in sizes:
        if size > len(full_train):
            print(f"  A2 size={size}: SKIP (only {len(full_train)} available)")
            continue

        subset = rng.sample(full_train, size)
        out_path = output_dir / f"{task}_train_{size}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)
        print(f"  A2 size={size}: {len(subset)} samples -> {out_path}")


def ablation_a4_metadata(task: str, field_name: str):
    """A4: Add metadata (subfield, year) to input text."""
    field_dir = field_name.lower().replace(" ", "_")
    task_dir = PROCESSED_DATA_DIR / field_dir / task
    output_dir = SFT_DATA_DIR / field_dir / "ablation_a4"
    output_dir.mkdir(parents=True, exist_ok=True)

    instruction = MEDIA_INSTRUCTION if task == "media" else POLICY_INSTRUCTION

    for split in ["train", "val"]:
        cot_path = task_dir / f"{split}_cot.json"
        base_path = task_dir / f"{split}.json"
        path = cot_path if cot_path.exists() else base_path

        with open(path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        sft_data = []
        for s in samples:
            label = "Yes" if s["label"] == 1 else "No"
            cot = s.get("cot_explanation", "")
            output = f"Analysis: {cot}\n\nPrediction: {label}" if cot else f"Prediction: {label}"

            input_text = (
                f"Field: {s.get('field', 'N/A')}\n"
                f"Subfield: {s.get('subfield', 'N/A')}\n"
                f"Publication Year: {s.get('publication_date', 'N/A')[:4]}\n"
                f"Title: {s['title']}\n"
                f"Abstract: {s['abstract']}"
            )
            sft_data.append({
                "instruction": instruction,
                "input": input_text,
                "output": output,
            })

        out_path = output_dir / f"{task}_with_metadata_{split}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sft_data, f, ensure_ascii=False, indent=2)
        print(f"  A4 with_metadata {split}: {len(sft_data)} samples -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["A1", "A2", "A4", "all"], default="all")
    parser.add_argument("--task", choices=["media", "policy"], default="media")
    parser.add_argument("--field_name", default=PRIMARY_FIELD_NAME)
    args = parser.parse_args()

    experiments = {
        "A1": ablation_a1_input_variants,
        "A2": ablation_a2_data_efficiency,
        "A4": ablation_a4_metadata,
    }

    if args.experiment == "all":
        for name, fn in experiments.items():
            print(f"\n{'='*40} Ablation {name} {'='*40}")
            fn(args.task, args.field_name)
    else:
        print(f"\n{'='*40} Ablation {args.experiment} {'='*40}")
        experiments[args.experiment](args.task, args.field_name)

    print("\nDone! Register the ablation datasets in LLaMA-Factory dataset_info.json before training.")


if __name__ == "__main__":
    main()
