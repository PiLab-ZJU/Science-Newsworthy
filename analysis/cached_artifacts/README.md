# Cached Analysis Artifacts

The `analysis/` scripts produce a number of intermediate artifacts that are **not committed** because they are large or regenerable:

| File | Source script | Purpose |
|------|---------------|---------|
| `test_predictions_with_explanations.json` | `scripts/analysis_1_generate_explanations.py` | Test-set predictions + CoT explanations |
| `explanation_embeddings.npy` (~900 MB) | `scripts/analysis_3b_signal_clustering.py` | Sentence-embedding matrix over explanations |
| `explanation_clusters.npy` | `scripts/analysis_3b_signal_clustering.py` | Cluster labels |
| `semantic_overlap_emb.npy` (~11 MB) | `analysis/fig_semantic_overlap.py` | Model/news vocabulary embeddings |
| `semantic_overlap_labels.npy` | same | Vocabulary labels |
| `signal_taxonomy.json` | `analysis/signal_taxonomy.py` | Mapped signals per instance |
| `signal_clustering.json` / `bertopic_results*.json` | `scripts/analysis_3d_bertopic*.py` | Topic-model outputs |
| `contrastive_signals.json` / `triplet_contrastive.json` | `scripts/analysis_3g_contrastive.py`, `scripts/analysis_3i_triplet.py` | Contrastive signal analyses |
| `*_signals.json`, `model_behavior.json`, `error_analysis.json`, etc. | `scripts/analysis_3*.py`, `analysis_4_*.py`, `analysis_5_*.py` | Per-stage analysis outputs |
| `per_field_metrics.json`, `per_field_mcc_ci.csv` | `scripts/analysis_2_per_field.py` | Per-field evaluation results |

To regenerate everything:

```bash
bash run_pipeline.sh analysis
```

Individual stages can be re-run by invoking the corresponding `scripts/analysis_*.py` file directly.
