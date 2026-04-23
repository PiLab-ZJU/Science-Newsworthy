"""
Analysis 5: Model behavior analysis.

1. Abstract features vs model accuracy (length, readability, mentions, citations, year)
2. True Yes/No vs Predicted Yes/No vocabulary comparison
3. Overlap analysis: what the model learns correctly vs misses

Usage:
    python scripts/analysis_5_model_behavior.py
"""
import json, re, math, sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

ANALYSIS_DIR = Path("/mnt/nvme1/lcx/academic_social_impact/analysis")

STOPWORDS = {
    "the","a","an","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","can","shall",
    "i","we","they","he","she","it","you","me","him","her","this","that","these",
    "those","my","your","his","its","our","of","in","to","for","with","on","at",
    "by","from","as","into","about","between","through","during","before","after",
    "and","or","but","not","if","than","so","because","while","also","just","very",
    "more","most","only","even","still","such","each","both","all","any","some",
    "no","other","what","which","who","whom","how","when","where","why","there",
    "here","then","now","well","however","although","whether","since","until",
    "using","used","based","show","shown","found","results","method","approach",
}

def tokenize(text):
    return [w for w in re.findall(r'\b[a-z]{3,}\b', text.lower()) if w not in STOPWORDS]

def syllable_count(word):
    word = word.lower()
    count = len(re.findall(r'[aeiouy]+', word))
    return max(count, 1)

def flesch_kincaid(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if not sentences: return 0
    words = text.split()
    if not words: return 0
    n_sentences = len(sentences)
    n_words = len(words)
    n_syllables = sum(syllable_count(w) for w in words)
    score = 206.835 - 1.015*(n_words/n_sentences) - 84.6*(n_syllables/n_words)
    return round(score, 1)

def log_odds(ca, cb, na, nb, min_freq=10):
    results = {}
    for w in set(ca.keys()) | set(cb.keys()):
        fa = ca.get(w,0) + 1
        fb = cb.get(w,0) + 1
        if ca.get(w,0) + cb.get(w,0) < min_freq: continue
        lor = math.log2((fa/(na-fa+1)) / (fb/(nb-fb+1)))
        z = lor / math.sqrt(1/fa + 1/fb)
        results[w] = {"z": round(z,2), "freq_a": ca.get(w,0), "freq_b": cb.get(w,0)}
    return results

def get_doc_freq(texts):
    c = Counter()
    for t in texts:
        c.update(set(tokenize(t)))
    return c

def print_top(lor, name_a, name_b, n=20):
    s = sorted(lor.items(), key=lambda x: x[1]["z"], reverse=True)
    print(f"\n  Top {name_a} words:")
    print(f"  {'Word':<20s} {'Z':>7s} {name_a:>7s} {name_b:>7s}")
    print(f"  {'-'*40}")
    for w, info in s[:n]:
        print(f"  {w:<20s} {info['z']:>7.1f} {info['freq_a']:>7d} {info['freq_b']:>7d}")
    print(f"\n  Top {name_b} words:")
    for w, info in s[-n:]:
        print(f"  {w:<20s} {info['z']:>7.1f} {info['freq_a']:>7d} {info['freq_b']:>7d}")

def main():
    with open(ANALYSIS_DIR / "test_predictions_with_explanations.json") as f:
        preds = json.load(f)

    proc = Path("/home/ubuntu/pilab_jiang/lcx/academic_new_policy/data/processed/combined/test.json")
    with open(proc) as f:
        test = json.load(f)
    doi_data = {d["doi"]: d for d in test}

    # Attach abstract and metadata
    for p in preds:
        d = doi_data.get(p.get("doi",""), {})
        p["abstract"] = d.get("abstract", "")
        p["cited_by_count"] = d.get("cited_by_count", 0)
        p["news_count"] = d.get("news_count", 0)
        p["year"] = d.get("publication_date", "")[:4]
        p["abs_len"] = len(p["abstract"].split())
        p["title_len"] = len(p.get("title","").split())
        p["readability"] = flesch_kincaid(p["abstract"])

    tp = [p for p in preds if p["true_label"]==1 and p["predicted"]==1]
    tn = [p for p in preds if p["true_label"]==0 and p["predicted"]==0]
    fp = [p for p in preds if p["true_label"]==0 and p["predicted"]==1]
    fn = [p for p in preds if p["true_label"]==1 and p["predicted"]==0]

    # ============================================================
    # Part 1: Feature vs Accuracy
    # ============================================================
    print("="*60)
    print("PART 1: Feature vs Model Accuracy")
    print("="*60)

    # Abstract length bins
    print("\n  Abstract Length vs Accuracy:")
    print(f"  {'Length (words)':<20s} {'N':>6s} {'Acc':>7s} {'F1':>7s}")
    print(f"  {'-'*42}")
    bins = [(0,100,"<100"), (100,150,"100-150"), (150,200,"150-200"), (200,300,"200-300"), (300,9999,"300+")]
    for lo, hi, label in bins:
        subset = [p for p in preds if lo <= p["abs_len"] < hi]
        if len(subset) < 50: continue
        correct = sum(1 for p in subset if p["true_label"]==p["predicted"])
        labels = [p["true_label"] for p in subset]
        ps = [p["predicted"] for p in subset]
        tp_c = sum(1 for l,p in zip(labels,ps) if l==1 and p==1)
        fp_c = sum(1 for l,p in zip(labels,ps) if l==0 and p==1)
        fn_c = sum(1 for l,p in zip(labels,ps) if l==1 and p==0)
        prec = tp_c/max(tp_c+fp_c,1)
        rec = tp_c/max(tp_c+fn_c,1)
        f1 = 2*prec*rec/max(prec+rec,0.001)
        print(f"  {label:<20s} {len(subset):>6d} {correct/len(subset):>7.4f} {f1:>7.4f}")

    # Readability bins
    print("\n  Readability (Flesch-Kincaid) vs Accuracy:")
    print(f"  {'Readability':<20s} {'N':>6s} {'Acc':>7s} {'Label':<15s}")
    print(f"  {'-'*50}")
    rbins = [(-999,10,"Very hard (<10)"), (10,30,"Hard (10-30)"), (30,50,"Medium (30-50)"), (50,999,"Easy (50+)")]
    for lo, hi, label in rbins:
        subset = [p for p in preds if lo <= p["readability"] < hi]
        if len(subset) < 50: continue
        correct = sum(1 for p in subset if p["true_label"]==p["predicted"])
        print(f"  {label:<20s} {len(subset):>6d} {correct/len(subset):>7.4f}")

    # News mention count vs accuracy
    print("\n  News Mention Count vs Accuracy (positive samples only):")
    print(f"  {'Mentions':<15s} {'N':>6s} {'Recall':>8s}")
    print(f"  {'-'*32}")
    pos_preds = [p for p in preds if p["true_label"]==1]
    mbins = [(1,1,"1"), (2,2,"2"), (3,4,"3-4"), (5,9,"5-9"), (10,9999,"10+")]
    for lo, hi, label in mbins:
        subset = [p for p in pos_preds if lo <= p.get("news_count",0) <= hi]
        if len(subset) < 20: continue
        caught = sum(1 for p in subset if p["predicted"]==1)
        print(f"  {label:<15s} {len(subset):>6d} {caught/len(subset):>8.4f}")

    # Year vs accuracy
    print("\n  Year vs Accuracy:")
    print(f"  {'Year':<8s} {'N':>6s} {'Acc':>7s}")
    print(f"  {'-'*24}")
    for y in sorted(set(p["year"] for p in preds)):
        if not y: continue
        subset = [p for p in preds if p["year"]==y]
        if len(subset) < 50: continue
        correct = sum(1 for p in subset if p["true_label"]==p["predicted"])
        print(f"  {y:<8s} {len(subset):>6d} {correct/len(subset):>7.4f}")

    # Title length
    print("\n  Title Length vs Accuracy:")
    print(f"  {'Title (words)':<18s} {'N':>6s} {'Acc':>7s}")
    print(f"  {'-'*34}")
    tbins = [(0,8,"<8"), (8,12,"8-12"), (12,16,"12-16"), (16,20,"16-20"), (20,999,"20+")]
    for lo, hi, label in tbins:
        subset = [p for p in preds if lo <= p["title_len"] < hi]
        if len(subset) < 50: continue
        correct = sum(1 for p in subset if p["true_label"]==p["predicted"])
        print(f"  {label:<18s} {len(subset):>6d} {correct/len(subset):>7.4f}")

    # ============================================================
    # Part 2: True vs Predicted vocabulary
    # ============================================================
    print(f"\n{'='*60}")
    print("PART 2: True vs Predicted Vocabulary Comparison")
    print("="*60)

    # True signal: actual Yes vs actual No abstracts
    true_yes_abs = [p["abstract"] for p in preds if p["true_label"]==1 and p["abstract"]]
    true_no_abs = [p["abstract"] for p in preds if p["true_label"]==0 and p["abstract"]]
    true_lor = log_odds(get_doc_freq(true_yes_abs), get_doc_freq(true_no_abs), len(true_yes_abs), len(true_no_abs))

    print("\n  A. TRUE SIGNAL: Actual Yes vs Actual No abstracts")
    print_top(true_lor, "TrueYes", "TrueNo")

    # Model signal: predicted Yes vs predicted No abstracts
    pred_yes_abs = [p["abstract"] for p in preds if p["predicted"]==1 and p["abstract"]]
    pred_no_abs = [p["abstract"] for p in preds if p["predicted"]==0 and p["abstract"]]
    pred_lor = log_odds(get_doc_freq(pred_yes_abs), get_doc_freq(pred_no_abs), len(pred_yes_abs), len(pred_no_abs))

    print(f"\n  B. MODEL SIGNAL: Predicted Yes vs Predicted No abstracts")
    print_top(pred_lor, "PredYes", "PredNo")

    # ============================================================
    # Part 3: Overlap analysis
    # ============================================================
    print(f"\n{'='*60}")
    print("PART 3: True Signal vs Model Signal Overlap")
    print("="*60)

    true_yes_words = {w for w, info in true_lor.items() if info["z"] > 3}
    true_no_words = {w for w, info in true_lor.items() if info["z"] < -3}
    pred_yes_words = {w for w, info in pred_lor.items() if info["z"] > 3}
    pred_no_words = {w for w, info in pred_lor.items() if info["z"] < -3}

    # YES signal overlap
    shared_yes = true_yes_words & pred_yes_words
    true_only_yes = true_yes_words - pred_yes_words
    pred_only_yes = pred_yes_words - true_yes_words
    jaccard_yes = len(shared_yes) / len(true_yes_words | pred_yes_words) if (true_yes_words | pred_yes_words) else 0

    print(f"\n  YES signal words:")
    print(f"    True Yes signals (z>3):      {len(true_yes_words)}")
    print(f"    Predicted Yes signals (z>3):  {len(pred_yes_words)}")
    print(f"    Shared:                       {len(shared_yes)}")
    print(f"    Jaccard:                      {jaccard_yes:.4f}")

    shared_ranked = sorted(shared_yes, key=lambda w: true_lor.get(w,{}).get("z",0) + pred_lor.get(w,{}).get("z",0), reverse=True)
    print(f"\n    Shared YES signals (model learned correctly):")
    print(f"    {'Word':<20s} {'True Z':>8s} {'Pred Z':>8s}")
    print(f"    {'-'*38}")
    for w in shared_ranked[:20]:
        tz = true_lor.get(w,{}).get("z",0)
        pz = pred_lor.get(w,{}).get("z",0)
        print(f"    {w:<20s} {tz:>8.1f} {pz:>8.1f}")

    true_only_ranked = sorted(true_only_yes, key=lambda w: -true_lor.get(w,{}).get("z",0))
    print(f"\n    True-only YES signals (model misses, top 20):")
    print(f"    {', '.join(true_only_ranked[:20])}")

    pred_only_ranked = sorted(pred_only_yes, key=lambda w: -pred_lor.get(w,{}).get("z",0))
    print(f"\n    Pred-only YES signals (model over-relies on, top 20):")
    print(f"    {', '.join(pred_only_ranked[:20])}")

    # NO signal overlap
    shared_no = true_no_words & pred_no_words
    jaccard_no = len(shared_no) / len(true_no_words | pred_no_words) if (true_no_words | pred_no_words) else 0

    print(f"\n  NO signal words:")
    print(f"    True No signals:     {len(true_no_words)}")
    print(f"    Predicted No signals:{len(pred_no_words)}")
    print(f"    Shared:              {len(shared_no)}")
    print(f"    Jaccard:             {jaccard_no:.4f}")

    # Save
    results = {
        "true_yes_top30": [(w, true_lor[w]) for w in sorted(true_lor, key=lambda x: -true_lor[x]["z"])[:30]],
        "true_no_top30": [(w, true_lor[w]) for w in sorted(true_lor, key=lambda x: true_lor[x]["z"])[:30]],
        "pred_yes_top30": [(w, pred_lor[w]) for w in sorted(pred_lor, key=lambda x: -pred_lor[x]["z"])[:30]],
        "pred_no_top30": [(w, pred_lor[w]) for w in sorted(pred_lor, key=lambda x: pred_lor[x]["z"])[:30]],
        "shared_yes": shared_ranked[:50],
        "true_only_yes": true_only_ranked[:50],
        "pred_only_yes": pred_only_ranked[:50],
        "jaccard_yes": round(jaccard_yes, 4),
        "jaccard_no": round(jaccard_no, 4),
    }
    out = ANALYSIS_DIR / "model_behavior.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
