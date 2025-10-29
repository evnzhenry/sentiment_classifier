"""Run a small demo: create synthetic AfriSenti-like data, run EDA, train a TF-IDF+LogReg classifier,
save plots and metrics to the project so the presentation can include them.
"""
import os
from pathlib import Path
import json
import random

ROOT = Path('c:/Users/BMC/Desktop/NLP')
DATA_DIR = ROOT / 'data'
OUT_DIR = ROOT / 'eda_outputs'
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

import sys
import pandas as pd
import numpy as np

# ensure src is importable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import simple_clean
from src.data_exploration import simple_eda
from src.eval_utils import compute_metrics, plot_confusion

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def make_synthetic(n_per_lang=200):
    # create synthetic tweets for three languages with three sentiment labels
    langs = ['sw', 'am', 'en']
    labels = ['negative', 'neutral', 'positive']
    rows = []
    for lang in langs:
        for i in range(n_per_lang):
            lbl = random.choice(labels)
            # create a simple synthetic text that signals sentiment
            if lbl == 'positive':
                text = f"{lang} good excellent love amazing {i}"
            elif lbl == 'negative':
                text = f"{lang} bad terrible hate awful {i}"
            else:
                text = f"{lang} okay fine average {i}"
            rows.append({'tweet_id': f'{lang}_{i}', 'text': text, 'language': lang, 'label': lbl})
    df = pd.DataFrame(rows)
    return df


def run():
    print('Creating synthetic dataset...')
    df = make_synthetic(n_per_lang=150)
    csv_path = DATA_DIR / 'synthetic_afrisenti.csv'
    df.to_csv(csv_path, index=False)
    print('Saved synthetic data to', csv_path)

    print('Running EDA...')
    eda_stats = simple_eda(df, text_col='text', lang_col='language', label_col='label', show_plots=True, save_prefix=str(OUT_DIR / 'eda'))
    print('EDA stats:', eda_stats)

    print('Preprocessing texts...')
    df['clean_text'] = df['text'].astype(str).apply(simple_clean)

    # train/test split
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    X_train = train_df['clean_text'].tolist()
    y_train = train_df['label'].tolist()
    X_test = test_df['clean_text'].tolist()
    y_test = test_df['label'].tolist()

    print('Training TF-IDF + LogisticRegression classifier...')
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1,2), max_features=5000), LogisticRegression(max_iter=200))
    pipe.fit(X_train, y_train)

    print('Evaluating...')
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)
    metrics = compute_metrics(y_test, preds, labels=['negative','neutral','positive'])
    print('Metrics:', metrics['accuracy'], metrics['macro_f1'])

    # save confusion matrix
    cm_path = OUT_DIR / 'confusion_demo.png'
    plot_confusion(y_test, preds, labels=['negative','neutral','positive'], save_path=str(cm_path))
    print('Saved confusion matrix to', cm_path)

    # save metrics
    metrics_path = OUT_DIR / 'metrics_demo.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({'accuracy': metrics['accuracy'], 'macro_f1': metrics['macro_f1'], 'report': metrics['report']}, f, indent=2)
    print('Saved metrics to', metrics_path)

    # save example predictions
    examples = []
    for text, pred, prob in zip(X_test[:10], preds[:10], probs[:10]):
        examples.append({'text': text, 'pred': pred, 'prob': prob.tolist()})
    examples_path = OUT_DIR / 'examples_demo.json'
    with open(examples_path, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2)
    print('Saved example predictions to', examples_path)

    print('Done. Generated EDA images and demo model outputs in', OUT_DIR)


if __name__ == '__main__':
    run()
