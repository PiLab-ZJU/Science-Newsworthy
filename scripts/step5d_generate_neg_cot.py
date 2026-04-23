"""
Step 5d: Generate diverse CoT for negative samples, then build balanced dataset.

Usage:
    python scripts/step5d_generate_neg_cot.py
"""
import os
import sys
import json
import random
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SFT_DATA_DIR, RANDOM_SEED

API_URL = "https://api.xty.app/v1/chat/completions"
API_KEY = "sk-YizsA7d2z2nA9vA2qt31T2xeqEDYcTGm8HumXSd5o3hEV3km"

SYSTEM_PROMPT = (
    "You are a media analyst. Given an academic paper's title and abstract, "
    "explain in 1-2 sentences why this paper is unlikely to receive mainstream news coverage. "
    "Be specific to the paper's content. Keep it under 50 words."
)

INSTRUCTION = (
    "Based on the following academic paper's title and abstract, "
    "predict whether this paper will receive mainstream media news coverage. "
    "If yes, briefly explain why and predict the news angle. "
    "Answer with your analysis and prediction."
)


def call_api(title: str, abstract: str) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "gpt-5.4-nano",
        "max_tokens": 80,
        "temperature": 0.7,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Title: {title}\nAbstract: {abstract[:500]}"},
        ],
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=data, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return ""


def generate_one(item):
    idx, sample = item
    inp = sample["input"]
    title, abstract = "", ""
    if "Title:" in inp and "Abstract:" in inp:
        parts = inp.split("Abstract:", 1)
        title = parts[0].replace("Title:", "").strip()
        abstract = parts[1].strip()
    cot = call_api(title, abstract)
    return idx, cot


def main():
    rng = random.Random(RANDOM_SEED)

    # Load CoT data
    cot_dir = SFT_DATA_DIR / "combined_cot"
    with open(cot_dir / "train.json") as f:
        train_data = json.load(f)

    pos = [s for s in train_data if "Prediction: Yes" in s["output"]]
    neg = [s for s in train_data if "Prediction: No" in s["output"]]
    print(f"Original: {len(pos)} pos, {len(neg)} neg")

    # Downsample negatives
    neg_sampled = rng.sample(neg, len(pos))
    print(f"Balanced: {len(pos)} pos, {len(neg_sampled)} neg")

    # Cache
    cache_path = cot_dir / "neg_cot_cache.json"
    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"Cached: {len(cache)}")

    # Find negatives needing CoT
    to_generate = [(i, s) for i, s in enumerate(neg_sampled) if str(i) not in cache]
    print(f"Need to generate: {len(to_generate)}")

    if to_generate:
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(generate_one, item): item[0] for item in to_generate}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
                idx, cot = future.result()
                if cot:
                    cache[str(idx)] = cot

                # Checkpoint every 1000
                if len(cache) % 1000 < 2:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(cache, f, ensure_ascii=False)

        # Final save cache
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        print(f"Generated: {len(cache)}")

    # Build final balanced dataset
    for i, s in enumerate(neg_sampled):
        cot = cache.get(str(i), "")
        if cot:
            s["output"] = f"Analysis: {cot}\nPrediction: No"
        s["instruction"] = INSTRUCTION

    # Update instruction for positives too
    for s in pos:
        s["instruction"] = INSTRUCTION

    final_train = pos + neg_sampled
    rng.shuffle(final_train)

    balanced_dir = SFT_DATA_DIR / "combined_cot_balanced"
    balanced_dir.mkdir(parents=True, exist_ok=True)
    with open(balanced_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(final_train, f, ensure_ascii=False, indent=2)

    # Balance val and test too
    for split in ["val", "test"]:
        with open(cot_dir / f"{split}.json") as f:
            data = json.load(f)
        sp = [s for s in data if "Prediction: Yes" in s["output"]]
        sn = [s for s in data if "Prediction: No" in s["output"]]
        sn_bal = rng.sample(sn, min(len(sp), len(sn)))
        balanced = sp + sn_bal
        rng.shuffle(balanced)
        with open(balanced_dir / f"{split}.json", "w", encoding="utf-8") as f:
            json.dump(balanced, f, ensure_ascii=False, indent=2)
        print(f"  {split}: {len(balanced)} ({len(sp)} pos + {len(sn_bal)} neg)")

    pos_f = sum(1 for s in final_train if "Prediction: Yes" in s["output"])
    print(f"  train: {len(final_train)} ({pos_f} pos + {len(final_train)-pos_f} neg)")

    # Register
    info_path = SFT_DATA_DIR / "dataset_info.json"
    with open(info_path) as f:
        info = json.load(f)
    info["media_combined_cot_balanced"] = {
        "file_name": "combined_cot_balanced/train.json",
        "formatting": "alpaca",
        "columns": {"prompt": "instruction", "query": "input", "response": "output"}
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    # Show samples
    print(f"\n--- Negative with GPT CoT ---")
    for s in final_train:
        if "Prediction: No" in s["output"] and "specialized topic" not in s["output"]:
            print(f"  {s['output']}")
            break

    print(f"\n--- Positive with news CoT ---")
    for s in final_train:
        if "Prediction: Yes" in s["output"]:
            print(f"  {s['output'][:300]}...")
            break

    print(f"\nDone! Registered as: media_combined_cot_balanced")


if __name__ == "__main__":
    main()
