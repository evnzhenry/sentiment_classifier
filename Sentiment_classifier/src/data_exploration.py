"""Data loading and initial exploration utilities for AfriSenti

Functions:
 - load_afrisenti: loads CSVs from data/ or fallback to Hugging Face datasets if `hf_id` provided
 - simple_eda: prints counts and plots language distribution, label balance, and text lengths
"""
import os
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_afrisenti(data_dir='data', hf_id=None):
    """Load AfriSenti dataset from local CSVs placed in data_dir.

    Expected columns: tweet_id, text, language (or lang), label
    If no local files found and hf_id provided, attempt to load via datasets.load_dataset (user must have network).
    """
    # first try local csv files
    if os.path.isdir(data_dir):
        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv') or f.endswith('.tsv')]
        if files:
            dfs = []
            for f in files:
                try:
                    df = pd.read_csv(f)
                except Exception:
                    df = pd.read_csv(f, sep='\t')
                dfs.append(df)
            df_all = pd.concat(dfs, ignore_index=True)
            # normalize column names
            cols = {c.lower(): c for c in df_all.columns}
            colmap = {}
            if 'text' in cols:
                colmap[cols['text']] = 'text'
            if 'tweet_id' in cols:
                colmap[cols['tweet_id']] = 'tweet_id'
            if 'lang' in cols:
                colmap[cols['lang']] = 'language'
            if 'language' in cols:
                colmap[cols['language']] = 'language'
            if 'label' in cols:
                colmap[cols['label']] = 'label'
            df_all = df_all.rename(columns=colmap)
            # basic cleaning
            if 'label' in df_all.columns:
                df_all = df_all[df_all['label'].notnull()]
            return df_all
    # fallback to HF datasets if requested
    if hf_id is not None:
        try:
            from datasets import load_dataset
            ds = load_dataset(hf_id)
            # convert to pandas
            part = list(ds.keys())[0]
            df = ds[part].to_pandas()
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from HF: {e}")
    raise FileNotFoundError(f"No local data found in {data_dir}. Place AfriSenti CSVs there or pass hf_id to load from HF.")


def simple_eda(df, text_col='text', lang_col='language', label_col='label', show_plots=True, save_prefix=None):
    """Run quick EDA: language distribution, label imbalance, text length histograms.
    Returns a dict with summary stats.
    """
    stats = {}
    stats['num_rows'] = len(df)
    stats['num_langs'] = df[lang_col].nunique() if lang_col in df.columns else None
    # language distribution
    if lang_col in df.columns:
        lang_counts = df[lang_col].value_counts()
        stats['lang_counts'] = lang_counts
        if show_plots:
            plt.figure(figsize=(8,4))
            sns.barplot(x=lang_counts.index, y=lang_counts.values)
            plt.title('Language distribution')
            plt.ylabel('count')
            plt.xlabel('language')
            plt.tight_layout()
            if save_prefix:
                plt.savefig(save_prefix + '_lang_dist.png')
            plt.show()
    # label distribution
    if label_col in df.columns:
        label_counts = df[label_col].value_counts()
        stats['label_counts'] = label_counts
        if show_plots:
            plt.figure(figsize=(6,4))
            sns.barplot(x=label_counts.index, y=label_counts.values)
            plt.title('Label distribution')
            plt.ylabel('count')
            plt.xlabel('label')
            plt.tight_layout()
            if save_prefix:
                plt.savefig(save_prefix + '_label_dist.png')
            plt.show()
    # text length distribution (chars and tokens approx)
    lengths = df[text_col].astype(str).apply(len)
    stats['char_length_mean'] = lengths.mean()
    stats['char_length_median'] = lengths.median()
    stats['char_length_std'] = lengths.std()
    if show_plots:
        plt.figure(figsize=(8,4))
        sns.histplot(lengths, bins=50)
        plt.title('Text length (chars) distribution')
        if save_prefix:
            plt.savefig(save_prefix + '_char_len_dist.png')
        plt.show()
    # token-length approx by whitespace split
    token_lens = df[text_col].astype(str).apply(lambda x: len(str(x).split()))
    stats['token_len_mean'] = token_lens.mean()
    stats['token_len_median'] = token_lens.median()
    if show_plots:
        plt.figure(figsize=(8,4))
        sns.histplot(token_lens, bins=50)
        plt.title('Token length (approx) distribution')
        if save_prefix:
            plt.savefig(save_prefix + '_token_len_dist.png')
        plt.show()
    return stats


if __name__ == '__main__':
    print('Run this module from notebook or import the functions to perform EDA')
