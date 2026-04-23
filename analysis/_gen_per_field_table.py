"""
Generate LaTeX fragments for the per-field table with an inline MCC bar
(forest-plot-style) as the 5th column.

Outputs:
  - rows printed to stdout ready to paste into \begin{tabular*}...
  - requires: a TikZ helper macro \mccbar{value}{lo}{hi} in preamble
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROJECT_ROOT

ANALYSIS_DIR = PROJECT_ROOT / "analysis"
N_BOOT = 1000
RNG_SEED = 42

SHORT = {
    "Biochemistry, Genetics and Molecular Biology": "Biochem., Gen.\\ \\& Mol.\\ Bio.",
    "Economics, Econometrics and Finance":         "Economics, Econ.\\ \\& Fin.",
    "Pharmacology, Toxicology and Pharmaceutics":  "Pharmacology, Tox.\\ \\& Ph.",
    "Business, Management and Accounting":         "Business, Mgmt.\\ \\& Acc.",
    "Agricultural and Biological Sciences":        "Agric.\\ and Bio.\\ Sci.",
    "Earth and Planetary Sciences":                "Earth and Planet.\\ Sci.",
    "Immunology and Microbiology":                 "Immunology \\& Microbiology",
    "Physics and Astronomy":                       "Physics and Astronomy",
    "Arts and Humanities":                         "Arts and Humanities",
}


def shorten(f: str) -> str:
    return SHORT.get(f, f)


def fmt_n(n: int) -> str:
    s = f"{n:,}"
    return s.replace(",", "{,}")


def main():
    preds = json.loads(
        (ANALYSIS_DIR / "test_predictions_with_explanations.json").read_text())

    by = defaultdict(lambda: {"y": [], "p": []})
    for r in preds:
        f = r.get("field", "")
        if not f:
            continue
        by[f]["y"].append(r["true_label"])
        by[f]["p"].append(r["predicted"])

    rng = np.random.default_rng(RNG_SEED)

    rows = []
    for f, d in by.items():
        y = np.array(d["y"]); p = np.array(d["p"])
        n = len(y)
        obs_f1 = float(f1_score(y, p, zero_division=0))
        obs_mcc = float(matthews_corrcoef(y, p))
        boot_f1 = np.empty(N_BOOT)
        for i in range(N_BOOT):
            idx = rng.integers(0, n, n)
            boot_f1[i] = f1_score(y[idx], p[idx], zero_division=0)
        rows.append({
            "field": f, "n": n, "f1": obs_f1, "mcc": obs_mcc,
            "lo": float(np.percentile(boot_f1, 2.5)),
            "hi": float(np.percentile(boot_f1, 97.5)),
        })

    # Sort by F1 desc (matches the existing table)
    rows.sort(key=lambda r: r["f1"], reverse=True)

    overall_mcc = float(matthews_corrcoef(
        [p["true_label"] for p in preds],
        [p["predicted"] for p in preds]))
    overall_f1 = float(f1_score(
        [p["true_label"] for p in preds],
        [p["predicted"] for p in preds], zero_division=0))

    print(f"% overall F1={overall_f1:.3f}  MCC={overall_mcc:.3f}  (n={len(rows)} fields)")
    print(f"% F1 bar range: [0.55, 0.90]; whisker = bootstrap 95% CI for F1.")
    print()

    # Split into two columns
    half = (len(rows) + 1) // 2
    left, right = rows[:half], rows[half:]

    print(r"% --- paste the rows below between \midrule ... \bottomrule ---")
    for i in range(half):
        l = left[i]
        r = right[i] if i < len(right) else None
        def fmt_row(r):
            if r is None:
                return "&  &  & "
            return (f"{shorten(r['field'])} & "
                    f"{fmt_n(r['n'])} & "
                    f"{r['f1']:.3f} & "
                    f"\\fbar{{{r['f1']:.3f}}}{{{r['lo']:.3f}}}{{{r['hi']:.3f}}}")
        print(f"{fmt_row(l)} & {fmt_row(r)} \\\\")

    # Dump raw CSV too (for reproducibility)
    csv_path = ANALYSIS_DIR / "per_field_mcc_ci.csv"
    csv_path.write_text(
        "field,n,f1,mcc,lo,hi\n" +
        "\n".join(f'"{r["field"]}",{r["n"]},{r["f1"]:.4f},'
                  f'{r["mcc"]:.4f},{r["lo"]:.4f},{r["hi"]:.4f}'
                  for r in rows))
    print()
    print(f"% wrote {csv_path}")


if __name__ == "__main__":
    main()
