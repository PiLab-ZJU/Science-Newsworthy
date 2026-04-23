"""
Baseline: SciBERT fine-tuning with classification head.

Usage (local model path, combined split):
    python baselines/scibert_baseline.py \
        --data_dir /root/pilab_jiang/cxlin/academic_new_policy/data/processed/combined \
        --model_name /root/pilab_jiang/hf-model/scibert_scivocab_uncased
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, OUTPUTS_DIR, PRIMARY_FIELD_NAME, RANDOM_SEED


class PaperDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        text = f"{s['title']} [SEP] {s['abstract']}"
        encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(s["label"], dtype=torch.long),
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "mcc": matthews_corrcoef(labels, preds),
        "auc_roc": roc_auc_score(labels, probs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="combined",
                        help="subdirectory under data/processed, or name used in output path")
    parser.add_argument("--data_dir", default=None,
                        help="full path to folder containing train.json/val.json/test.json")
    parser.add_argument("--field_name", default=PRIMARY_FIELD_NAME)
    parser.add_argument("--model_name", default="allenai/scibert_scivocab_uncased",
                        help="HuggingFace name OR local path to the SciBERT directory")
    parser.add_argument("--output_dir", default=None,
                        help="override checkpoint dir")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    if args.data_dir:
        task_dir = Path(args.data_dir)
        task_name = task_dir.name
    else:
        task_name = args.task
        task_dir = PROCESSED_DATA_DIR / task_name

    # Load data
    splits = {}
    for split in ["train", "val", "test"]:
        with open(task_dir / f"{split}.json", "r", encoding="utf-8") as f:
            splits[split] = json.load(f)

    print(f"SciBERT baseline: task_dir={task_dir}")
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    print(f"Model: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    train_dataset = PaperDataset(splits["train"], tokenizer, max_length=args.max_length)
    val_dataset = PaperDataset(splits["val"], tokenizer, max_length=args.max_length)
    test_dataset = PaperDataset(splits["test"], tokenizer, max_length=args.max_length)

    output_dir = Path(args.output_dir) if args.output_dir else (
        OUTPUTS_DIR / "baselines" / task_name / "scibert_finetuned"
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=RANDOM_SEED,
        logging_steps=50,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate on test set
    test_results = trainer.evaluate(test_dataset)
    print(f"\nTest results:")
    print(json.dumps(test_results, indent=2))

    # Save results
    results_path = output_dir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
