# Outputs Directory

Training outputs, prediction files, figures, and tables are **not included** in this repository.

## Expected layout

```
outputs/
├── media_lora_sft/                 # LoRA adapter for the media task
├── policy_lora_sft/                # LoRA adapter for the policy task
├── joint_lora_sft/                 # LoRA adapter for the joint task
├── predictions/
│   └── medicine/
│       ├── media_test_predictions.json
│       └── policy_test_predictions.json
├── figures/                        # PNG figures from analysis/*.py
└── tables/                         # CSV/XLSX result tables
```

## How to populate

- **LoRA adapters**: produced by LLaMA-Factory training (see `configs/*.yaml`).
  Base models are downloaded from Hugging Face the first time you run training.
- **Predictions**: produced by `evaluation/inference.py`.
- **Figures / tables**: produced by scripts in `analysis/`.

Adapters can also be downloaded separately (if released) and dropped into this
directory to skip training and go straight to evaluation.
