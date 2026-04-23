"""
Global configuration for the Academic Social Impact Prediction project.
"""
import os
from pathlib import Path

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Raw data
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SFT_DATA_DIR = DATA_DIR / "sft"

# ============================================================
# OpenAlex API
# ============================================================
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY", "Ufk8lRlb9U9kk3Au65QvLI")
OPENALEX_BASE_URL = "https://api.openalex.org/works"

# ============================================================
# Data collection parameters
# ============================================================
PUBLICATION_YEARS = "2016-2020"
PRIMARY_FIELD_ID = "fields/27"  # Medicine
PRIMARY_FIELD_NAME = "Medicine"

GENERALIZATION_FIELDS = {
    "fields/23": "Environmental Science",
    "fields/33": "Social Sciences",
}

MEDIA_POSITIVE_COUNT = 5000
MEDIA_NEGATIVE_COUNT = 5000
POLICY_POSITIVE_COUNT = 3000
POLICY_NEGATIVE_COUNT = 3000

# ============================================================
# Data split
# ============================================================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ============================================================
# SFT training
# ============================================================
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ALT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 8
CUTOFF_LEN = 1024

# ============================================================
# Prompts
# ============================================================
MEDIA_INSTRUCTION = (
    "Based on the following academic paper's title and abstract, "
    "predict whether this paper will receive mainstream media news coverage. "
    "Answer Yes or No."
)

POLICY_INSTRUCTION = (
    "You are an expert in science policy. Based on the following academic "
    "paper's title and abstract, predict whether this paper will be cited in "
    "public policy documents. First, analyze the key factors relevant to "
    "policy-making, then give your prediction (Yes/No)."
)

# ============================================================
# Evaluation
# ============================================================
METRICS = ["accuracy", "precision", "recall", "f1", "auc_roc", "mcc"]
