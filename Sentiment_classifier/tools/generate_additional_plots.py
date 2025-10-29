"""Generate additional EDA visualizations (heatmaps, scatter plots, histograms, PCA) and save to eda_outputs/"""
from pathlib import Path
import sys
ROOT = Path('c:/Users/BMC/Desktop/NLP')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

from src.preprocess import simple_clean

DATA_CSV = ROOT / 'data' / 'synthetic_afrisenti.csv'
OUT_DIR = ROOT / 'eda_outputs'
OUT_DIR.mkdir(exist_ok=True)


def load_data():
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_CSV}. Run run_demo.py first or provide data.")
    df = pd.read_csv(DATA_CSV)
    df['text'] = df['text'].astype(str)
    df['clean_text'] = df['text'].apply(simple_clean)
    df['char_len'] = df['clean_text'].apply(len)
    df['token_len'] = df['clean_text'].apply(lambda x: len(str(x).split()))
    return df


def heatmap_label_by_language(df):
    pivot = df.groupby(['language','label']).size().unstack(fill_value=0)
    plt.figure(figsize=(8,4))
    sns.heatmap(pivot, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Label counts per language')
    plt.ylabel('language')
    plt.xlabel('label')
    out = OUT_DIR / 'heatmap_label_by_language.png'
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def scatter_length(df):
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x='token_len', y='char_len', hue='label', style='language', alpha=0.8)
    plt.title('Token length vs Char length (colored by label)')
    out = OUT_DIR / 'scatter_token_vs_char.png'
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def histograms(df):
    out_files = []
    plt.figure(figsize=(8,4))
    sns.histplot(df['char_len'], bins=30)
    plt.title('Char length distribution')
    out1 = OUT_DIR / 'hist_char_len.png'
    plt.tight_layout()
    plt.savefig(out1)
    plt.close()
    out_files.append(out1)

    plt.figure(figsize=(8,4))
    sns.histplot(df['token_len'], bins=30)
    plt.title('Token length distribution')
    out2 = OUT_DIR / 'hist_token_len.png'
    plt.tight_layout()
    plt.savefig(out2)
    plt.close()
    out_files.append(out2)

    return out_files


def pca_tfidf_scatter(df, n_features=500, random_state=42):
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=n_features)
    X = vec.fit_transform(df['clean_text']).toarray()
    pca = PCA(n_components=2, random_state=random_state)
    X2 = pca.fit_transform(X)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X2[:,0], y=X2[:,1], hue=df['label'], style=df['language'], alpha=0.8)
    plt.title('PCA of TF-IDF features')
    out = OUT_DIR / 'pca_tfidf_scatter.png'
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def correlation_heatmap(df):
    num = df[['char_len','token_len']].copy()
    num['label_id'] = df['label'].astype('category').cat.codes
    corr = num.corr()
    plt.figure(figsize=(5,4))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation heatmap (lengths vs label)')
    out = OUT_DIR / 'correlation_heatmap.png'
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def per_language_histograms(df):
    out_files = []
    langs = df['language'].unique()
    for lang in langs:
        sub = df[df['language']==lang]
        plt.figure(figsize=(8,4))
        sns.countplot(x='label', data=sub, order=sorted(df['label'].unique()))
        plt.title(f'Label counts for {lang}')
        out = OUT_DIR / f'label_counts_{lang}.png'
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        out_files.append(out)
    return out_files


def main():
    df = load_data()
    files = []
    print('Generating heatmap of label by language...')
    files.append(heatmap_label_by_language(df))
    print('Generating scatter plot of token vs char length...')
    files.append(scatter_length(df))
    print('Generating histograms...')
    files.extend(histograms(df))
    print('Generating PCA TF-IDF scatter...')
    files.append(pca_tfidf_scatter(df))
    print('Generating correlation heatmap...')
    files.append(correlation_heatmap(df))
    print('Generating per-language label histograms...')
    files.extend(per_language_histograms(df))

    print('Saved plots:')
    for f in files:
        print(' -', f)


if __name__ == '__main__':
    main()
