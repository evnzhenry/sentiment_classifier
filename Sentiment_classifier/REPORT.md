Short Report: AfriSenti Multilingual Sentiment Analysis

This report summarizes EDA, PCA, and model comparisons. It supports both synthetic demo and real AfriSenti data. The real-data pipeline is implemented in `tools/run_real_afrisenti.py` and will populate `eda_outputs/` with figures and metrics.

How to generate real-data results

1) Ensure internet access and install the `datasets` library (or place AfriSenti CSVs locally in `data/`).
2) Run: `python tools\run_real_afrisenti.py`.
3) Outputs will be saved to `eda_outputs/`:
   - EDA plots: `real_eda_label_dist.png`, `real_eda_lang_dist.png`, `real_eda_char_len.png`, `real_eda_token_len.png`, `real_eda_heatmap_label_by_language.png`
   - PCA: `real_pca_tfidf_scatter.png`, `real_pca_evr.json`, `real_pca_loadings.json`
   - Models: `{tfidf_logreg|tfidf_nb|tfidf_svc}_metrics_real.json`, `{...}_confusion_real.png`, `model_f1_bar_real.png`, `model_comparison_summary_real.json`

References: Hugging Face datasets — HausaNLP/AfriSenti-Twitter, shmuhammad/AfriSenti.

Sections below provide analysis templates with example interpretations. Replace demo excerpts with real-data artifacts after running the pipeline.

## Demo run results (synthetic data)

Note: the following results come from a synthetic demo run (not the real AfriSenti dataset). They were generated to validate the pipeline and create example artifacts for the presentation. Replace with real results after running the notebook on the AfriSenti CSVs.

- Accuracy: 1.0
- Macro F1: 1.0

Per-class metrics (precision / recall / f1-score / support):

- negative: 1.0 / 1.0 / 1.0 / 30
- neutral : 1.0 / 1.0 / 1.0 / 31
- positive: 1.0 / 1.0 / 1.0 / 29

Artifacts produced by the demo (saved under `eda_outputs/`):

- `confusion_demo.png` — confusion matrix for the demo classifier
- `metrics_demo.json` — full metrics and classification report
- `examples_demo.json` — example inputs, predictions, and probabilities
- Multiple EDA plots: `heatmap_label_by_language.png`, `scatter_token_vs_char.png`, `hist_char_len.png`, `hist_token_len.png`, `pca_tfidf_scatter.png`, `correlation_heatmap.png`, and per-language label histograms.

To generate real results, place AfriSenti CSV(s) in `data/` (columns: `text`, `label`, `language`) and run the notebook `AfriSenti_sentiment_analysis.ipynb` end-to-end, then re-run `tools/generate_additional_plots.py` and `tools/generate_presentation.py` to update artifacts and the PPTX.
 
## Real-data analysis (sw/am/en) — to be filled after run

EDA highlights

- Provide statistical summaries: overall char/token length stats, per-label distributions, per-language label counts.
- Include and interpret plots:
  - `real_eda_label_dist.png` — label balance and any skew.
  - `real_eda_lang_dist.png` — language breakdown.
  - `real_eda_char_len.png`, `real_eda_token_len.png` — central tendency and spread; note outliers.
  - `real_eda_heatmap_label_by_language.png` — class-language interactions.

PCA results

- Explain variance (`real_pca_evr.json`): report EVR for PC1/PC2 and cumulative.
- Component loadings (`real_pca_loadings.json`): list top positive/negative tokens for PC1/PC2 and discuss sentiment axes.
- Include `real_pca_tfidf_scatter.png` with interpretation of cluster separability and any language drift.

Model comparison

- Tabulate metrics from `{tfidf_logreg|tfidf_nb|tfidf_svc}_metrics_real.json`: accuracy, macro/micro F1, per-class F1.
- Visuals:
  - Confusion matrices `{...}_confusion_real.png` per model.
  - Bar chart `model_f1_bar_real.png` contrasting per-class F1 across models.
  - (Optional) ROC curves using predicted probabilities; report macro ROC-AUC per model.

Critical remarks and conclusions

- Limitations: label noise, code-switching, slang/emojis; domain drift across languages; class imbalance.
- Improvements: multilingual transformers (e.g., XLM-R fine-tuning), better preprocessing (emoji/hashtag normalization), ablations on n-grams/feature caps and transformer hyperparameters, language-wise calibration.
- Key takeaways: polarity and neutrality axes dominate TF-IDF PCA; deep multilingual models typically improve cross-lingual generalization; combine aggregate metrics with per-language breakdown and error analysis for robust insights.