"""Quick test for the news-title SFT model."""
import json, torch, random, sys
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SFT_DATA_DIR

base_path = "/mnt/nvme1/hf-model/Meta-Llama-3.1-8B-Instruct"
adapter_path = "/mnt/nvme1/lcx/academic_social_impact/media_medicine_news_1k"

tokenizer = AutoTokenizer.from_pretrained(base_path)
model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

INSTRUCTION = (
    "You are an expert in science communication and media analysis. "
    "Given an academic paper's title and abstract, determine whether this paper "
    "will receive mainstream media news coverage. If yes, predict the likely news headline. "
    "Respond in the following format:\n"
    "News Title: <predicted news headline or None>\n"
    "Prediction: <Yes or No>"
)

with open(SFT_DATA_DIR / "medicine_news" / "test.json") as f:
    test = json.load(f)

random.seed(42)
subset = random.sample(test, min(300, len(test)))

labels = []
preds = []
samples_output = []

for s in tqdm(subset, desc="Testing"):
    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": s["input"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    resp = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    true_label = 1 if "Prediction: Yes" in s["output"] else 0
    resp_lower = resp.lower()
    if "prediction:" in resp_lower:
        after = resp_lower.split("prediction:")[-1].strip()
        pred_label = 1 if after.startswith("yes") else 0
    else:
        pred_label = 1 if "yes" in resp_lower else 0

    labels.append(true_label)
    preds.append(pred_label)

    if len(samples_output) < 8:
        samples_output.append({
            "true": true_label,
            "pred": pred_label,
            "response": resp[:300],
        })

print(f"\n{'='*60}")
print(f"Results (n={len(labels)}):")
print(f"  Accuracy: {accuracy_score(labels, preds):.4f}")
print(f"  F1:       {f1_score(labels, preds, zero_division=0):.4f}")
print(f"\n{classification_report(labels, preds, target_names=['No', 'Yes'])}")

print("Sample outputs:")
for r in samples_output:
    tag = "OK" if r["true"] == r["pred"] else "WRONG"
    print(f"  [{tag}] True={r['true']} Pred={r['pred']}")
    print(f"    {r['response']}")
    print()
