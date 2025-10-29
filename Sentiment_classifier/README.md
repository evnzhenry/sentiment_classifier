AfriSenti Multilingual Sentiment Analysis

This workspace contains a Jupyter notebook `AfriSenti_sentiment_analysis.ipynb` that performs EDA, preprocessing, modeling (XLM-R/AfriBERTa fine-tuning and an LSTM baseline), evaluation, ablations, and cross-lingual experiments on the AfriSenti Twitter dataset (Swahili, Amharic, English).

How to use

1. Create a Python 3.8+ environment and install dependencies:

   pip install -r requirements.txt

2. Place AfriSenti dataset files (CSV/JSON) in `data/` or use the Hugging Face `datasets` loader if you have network access.

3. Open `AfriSenti_sentiment_analysis.ipynb` in Jupyter/Colab and run cells sequentially. GPU is recommended for fine-tuning.

Files

- `AfriSenti_sentiment_analysis.ipynb`: Notebook scaffold for experiments.
- `requirements.txt`: Python package requirements.
- `REPORT.md`: Short report summarizing dataset, models, results, and insights (created after running experiments).

Notes

- The notebook contains placeholders for dataset paths and model checkpoints. If you want me to wire the notebook to download AfriSenti automatically from a specific URL or HF dataset id, tell me the exact dataset id or provide the files and I'll update the notebook to load them automatically.