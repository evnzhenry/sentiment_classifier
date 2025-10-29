"""Load real AfriSenti data (via Hugging Face), run EDA, PCA, and train multiple baselines.
Saves plots and metrics under eda_outputs/ and updates REPORT.md if desired.

Usage:
    python tools\run_real_afrisenti.py

Dependencies:
    - datasets
    - scikit-learn, pandas, numpy, matplotlib, seaborn
    - project src modules (preprocess, data_exploration, eval_utils)
"""
from pathlib import Path
import sys
import os
import json
import warnings

ROOT = Path('c:/Users/BMC/Desktop/NLP').resolve()
OUT_DIR = ROOT / 'eda_outputs'
OUT_DIR.mkdir(exist_ok=True)
DATA_DIR = ROOT / 'data' / 'afrisenti'
DATA_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocess import simple_clean
from src.eval_utils import compute_metrics, plot_confusion
from src.data_exploration import load_afrisenti

warnings.filterwarnings('ignore')

def try_load_hf():
    """Attempt to load AfriSenti from Hugging Face datasets.
    Returns a pandas DataFrame with columns: text, label, language.
    """
    try:
        from datasets import load_dataset
    except Exception as e:
        print('datasets library not available:', e)
        return None

    candidates = [
        ('HausaNLP/AfriSenti-Twitter', None),
        ('shmuhammad/AfriSenti', None),
    ]
    for ds_id, config in candidates:
        try:
            print('Trying to load dataset:', ds_id, 'config:', config)
            if config:
                ds = load_dataset(ds_id, config)
            else:
                ds = load_dataset(ds_id)
            # Some datasets provide splits (train/dev/test) per language or overall
            # We'll concatenate available splits and normalize columns
            frames = []
            for split_name, split in ds.items():
                df = pd.DataFrame(split)
                frames.append(df)
            df_all = pd.concat(frames, ignore_index=True)
            # Normalize column names
            cols_lower = {c.lower(): c for c in df_all.columns}
            text_col = cols_lower.get('text')
            label_col = cols_lower.get('label')
            lang_col = cols_lower.get('language') or cols_lower.get('lang')
            # Some HF datasets store labels as int and class names in 'label' feature
            if label_col is None and 'label' in df_all.columns:
                label_col = 'label'
            if text_col is None:
                raise ValueError('Text column not found in HF dataset')
            # Map numeric labels to strings if needed
            if pd.api.types.is_integer_dtype(df_all[label_col]):
                # Try to read class names from features
                try:
                    class_names = ds[split_name].features['label'].names  # type: ignore
                except Exception:
                    class_names = ['negative','neutral','positive']
                df_all['label'] = df_all[label_col].apply(lambda i: class_names[int(i)])
            else:
                df_all['label'] = df_all[label_col].astype(str)
            df_all['text'] = df_all[text_col].astype(str)
            if lang_col:
                df_all['language'] = df_all[lang_col].astype(str)
            else:
                # Attempt to infer from dataset structure; default unknown
                df_all['language'] = df_all.get('lang', 'unknown')
            return df_all[['text','label','language']]
        except Exception as e:
            print('Failed to load', ds_id, 'error:', e)
    return None

def ensure_local_afrisenti_csvs(langs=('sw','am','en')):
    """Attempt to download AfriSenti CSV/TSV files from the Hugging Face Hub
    and assemble a unified DataFrame for the requested languages.

    Returns a pandas DataFrame with columns: text, label, language; or None.
    """
    repo_candidates = [
        ('afrisenti-semeval/afrisent-semeval-2023', 'dataset'),
        ('HausaNLP/AfriSenti-Twitter', 'dataset'),
    ]
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
    except Exception as e:
        print('huggingface_hub not available:', e)
        return None

    lang_patterns = {
        'sw': ['sw', 'swahili'],
        'am': ['am', 'amharic'],
        'en': ['en', 'english']
    }
    split_patterns = ['train', 'dev', 'valid']
    dfs = []

    for repo_id, repo_type in repo_candidates:
        try:
            files = list_repo_files(repo_id, repo_type=repo_type)
        except Exception as e:
            print('Failed to list files for', repo_id, 'error:', e)
            # Try snapshot_download as a fallback
            try:
                from huggingface_hub import snapshot_download
                local_repo_dir = snapshot_download(repo_id=repo_id, repo_type=repo_type)
                # scan local directory for csv/tsv
                files = []
                for root_dir, _, filenames in os.walk(local_repo_dir):
                    for fn in filenames:
                        if fn.lower().endswith('.csv') or fn.lower().endswith('.tsv'):
                            rel_path = os.path.relpath(os.path.join(root_dir, fn), local_repo_dir)
                            files.append(rel_path)
            except Exception as e2:
                print('snapshot_download failed for', repo_id, 'error:', e2)
                continue

        files_lower = [(f, f.lower()) for f in files]
        # Try to find per-language train/dev files
        found_any = False
        for lang, patterns in lang_patterns.items():
            for split in split_patterns:
                candidates = [
                    f for f, fl in files_lower
                    if (fl.endswith('.csv') or fl.endswith('.tsv'))
                    and any(('/'+p+'/' in fl) or (('/'+p+'_' in fl)) or (('_'+p+'_' in fl)) or (fl.startswith(p+'/')) for p in patterns)
                    and (split in fl)
                ]
                for path in candidates:
                    try:
                        local_path = hf_hub_download(repo_id=repo_id, filename=path, repo_type=repo_type)
                        # read
                        ext = os.path.splitext(local_path)[1].lower()
                        sep = '\t' if ext == '.tsv' else ','
                        try:
                            df = pd.read_csv(local_path, sep=sep)
                        except Exception:
                            # try without header if needed
                            df = pd.read_csv(local_path, sep=sep, header=0)
                        # normalize columns
                        cols = {c.lower(): c for c in df.columns}
                        text_col = cols.get('text') or cols.get('tweet') or cols.get('content')
                        label_col = cols.get('label') or cols.get('sentiment') or cols.get('category')
                        # Some dev sets may use 'labels' or 'sentiment_label'
                        if label_col is None:
                            label_col = cols.get('labels') or cols.get('sentiment_label')
                        if text_col is None or label_col is None:
                            # skip files that don't have both text and label
                            continue
                        df = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label'})
                        df['language'] = lang
                        # save a copy locally for reproducibility
                        target_name = f"{lang}_{split}{ext}"
                        target_path = DATA_DIR / target_name
                        try:
                            df.to_csv(target_path, index=False)
                        except Exception:
                            pass
                        dfs.append(df)
                        found_any = True
                    except Exception as e:
                        print('Failed to download/read', path, 'from', repo_id, 'error:', e)
        if found_any:
            break

    if not dfs:
        # As a last resort, attempt to assemble by reading all CSV/TSV and filtering by language columns
        # This covers repos where language is a column rather than a folder in path.
        try:
            from huggingface_hub import snapshot_download
            local_repo_dir = snapshot_download(repo_id='HausaNLP/AfriSenti-Twitter', repo_type='dataset')
            collected = []
            for root_dir, _, filenames in os.walk(local_repo_dir):
                for fn in filenames:
                    if fn.lower().endswith('.csv') or fn.lower().endswith('.tsv'):
                        p = os.path.join(root_dir, fn)
                        ext = os.path.splitext(p)[1].lower()
                        sep = '\t' if ext == '.tsv' else ','
                        try:
                            df = pd.read_csv(p, sep=sep)
                        except Exception:
                            continue
                        cols = {c.lower(): c for c in df.columns}
                        # identify columns
                        text_col = cols.get('text') or cols.get('tweet') or cols.get('content')
                        label_col = cols.get('label') or cols.get('sentiment') or cols.get('category') or cols.get('labels')
                        lang_col = cols.get('language') or cols.get('lang')
                        if text_col is None or label_col is None:
                            continue
                        df = df[[text_col, label_col] + ([lang_col] if lang_col else [])].rename(columns={text_col: 'text', label_col: 'label', (lang_col or 'language'): 'language'})
                        # If language column missing, try to infer from path fragments
                        if 'language' not in df.columns or df['language'].isnull().all():
                            lower_path = p.lower()
                            inferred = None
                            for code, pats in lang_patterns.items():
                                if any(('/'+pat+'/' in lower_path) or (lower_path.endswith('/'+pat)) for pat in pats):
                                    inferred = code
                                    break
                            df['language'] = inferred if inferred else 'unknown'
                        # filter by requested languages
                        df['language'] = df['language'].astype(str).str.lower()
                        df = df[df['language'].isin(list(lang_patterns.keys()))]
                        if not df.empty:
                            collected.append(df)
            if collected:
                dfs.extend(collected)
            else:
                print('No suitable CSV/TSV files found on HF Hub for requested languages.')
                return None
        except Exception as e:
            print('Final snapshot aggregating attempt failed:', e)
            return None
    df_all = pd.concat(dfs, ignore_index=True)
    # basic cleanup: drop NaNs, ensure labels are strings
    df_all = df_all.dropna(subset=['text', 'label'])
    df_all['text'] = df_all['text'].astype(str)
    df_all['label'] = df_all['label'].astype(str).str.lower()
    # normalize common label variants
    label_map = {
        'neg': 'negative', 'negative': 'negative', '-1': 'negative', '0': 'neutral', 'neu': 'neutral', 'neutral': 'neutral', 'pos': 'positive', 'positive': 'positive', '1': 'positive'
    }
    df_all['label'] = df_all['label'].map(lambda x: label_map.get(x, x))
    # keep only three classes
    df_all = df_all[df_all['label'].isin(['negative','neutral','positive'])]
    return df_all


def load_github_map_and_download(map_path: str, langs=('sw','am','en')):
    """Download train/dev CSV/TSV for specified languages using a JSON mapping of raw URLs.

    JSON format example:
    {
      "sw": {"train": "https://.../sw/train.tsv", "dev": "https://.../sw/dev.tsv"},
      "am": {"train": "https://.../am/train.tsv", "dev": "https://.../am/dev.tsv"},
      "en": {"train": "https://.../en/train.tsv", "dev": "https://.../en/dev.tsv"}
    }
    """
    import json as _json
    import requests

    try:
        with open(map_path, 'r', encoding='utf-8') as f:
            url_map = _json.load(f)
    except Exception as e:
        print('Failed to read github_map JSON:', e)
        return None

    dfs = []
    for lang in langs:
        entry = url_map.get(lang)
        if not entry:
            print(f'No URLs provided for language: {lang}')
            continue
        for split in ['train', 'dev', 'valid']:
            url = entry.get(split)
            if not url:
                continue
            try:
                print('Downloading', lang, split, 'from', url)
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                # Determine separator by extension
                ext = os.path.splitext(url)[1].lower()
                sep = '\t' if ext == '.tsv' else ','
                from io import StringIO
                df = pd.read_csv(StringIO(r.text), sep=sep)
                # normalize columns
                cols = {c.lower(): c for c in df.columns}
                text_col = cols.get('text') or cols.get('tweet') or cols.get('content')
                label_col = cols.get('label') or cols.get('sentiment') or cols.get('category') or cols.get('labels')
                if text_col is None or label_col is None:
                    print('Missing required columns in downloaded file; skipping', lang, split)
                    continue
                df = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label'})
                df['language'] = lang
                dfs.append(df)
            except Exception as e:
                print('Failed to download/parse', lang, split, 'from', url, 'error:', e)

    if not dfs:
        print('No dataframes assembled from GitHub raw URLs.')
        return None
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.dropna(subset=['text', 'label'])
    df_all['text'] = df_all['text'].astype(str)
    df_all['label'] = df_all['label'].astype(str).str.lower()
    label_map = {
        'neg': 'negative', 'negative': 'negative', '-1': 'negative',
        'neu': 'neutral', 'neutral': 'neutral', '0': 'neutral',
        'pos': 'positive', 'positive': 'positive', '1': 'positive'
    }
    df_all['label'] = df_all['label'].map(lambda x: label_map.get(x, x))
    df_all = df_all[df_all['label'].isin(['negative','neutral','positive'])]
    return df_all


def download_and_scan_github_repo_zip(repo_spec: str, langs=('sw','am','en')):
    """Download a GitHub repo ZIP and scan for train/dev TSV/CSV files.

    repo_spec format: "owner/repo@branch" or "owner/repo" (defaults to main)
    We download the ZIP, extract, traverse all files, and assemble a DataFrame
    restricted to requested languages and splits (train/dev/valid/val).
    """
    import requests
    import zipfile
    import tempfile
    import shutil

    if '@' in repo_spec:
        owner_repo, branch = repo_spec.split('@', 1)
    else:
        owner_repo, branch = repo_spec, 'main'
    owner_repo = owner_repo.strip('/')
    # Prefer codeload which is optimized for ZIP downloads
    zip_urls = [
        f'https://codeload.github.com/{owner_repo}/zip/refs/heads/{branch}',
        f'https://github.com/{owner_repo}/archive/refs/heads/{branch}.zip'
    ]
    tmpdir = tempfile.mkdtemp(prefix='afrisenti_zip_')
    zip_path = os.path.join(tmpdir, 'repo.zip')
    last_err = None
    for zip_url in zip_urls:
        print('Downloading GitHub ZIP from', zip_url)
        try:
            with requests.get(zip_url, stream=True, timeout=(10, 60)) as r:
                r.raise_for_status()
                with open(zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024 * 64):
                        if chunk:  # filter out keep-alive chunks
                            f.write(chunk)
            last_err = None
            break
        except Exception as e:
            print('ZIP download attempt failed:', e)
            last_err = e

    if last_err is not None:
        print('Failed to download GitHub ZIP:', last_err)
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(tmpdir)
    except Exception as e:
        print('Failed to extract ZIP:', e)
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None

    # repo root inside zip is a single top-level folder
    # Scan for candidate files
    lang_set = {l.lower() for l in langs}
    splits = {'train', 'dev', 'valid', 'val'}

    dfs = []
    for root, dirs, files in os.walk(tmpdir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in ('.tsv', '.csv'):
                continue
            full_path = os.path.join(root, fname)
            path_lower = full_path.lower()
            # only include train/dev/valid files based on path keywords
            if not any(s in path_lower for s in splits):
                continue
            # try to detect language from path fragments
            path_parts = re.split(r'[\\/\-_\.]+', path_lower)
            lang_in_path = None
            for lp in path_parts:
                if lp in lang_set:
                    lang_in_path = lp
                    break
            sep = '\t' if ext == '.tsv' else ','
            try:
                df = pd.read_csv(full_path, sep=sep)
            except Exception:
                # try common encoding fallback
                try:
                    df = pd.read_csv(full_path, sep=sep, encoding='utf-8', errors='ignore')
                except Exception as e:
                    print('Failed to read file', full_path, 'error:', e)
                    continue

            cols = {c.lower(): c for c in df.columns}
            text_col = cols.get('text') or cols.get('tweet') or cols.get('content')
            label_col = cols.get('label') or cols.get('sentiment') or cols.get('category') or cols.get('labels')
            lang_col = cols.get('language') or cols.get('lang')
            if text_col is None or label_col is None:
                # no required columns; skip
                continue

            df = df[[text_col, label_col] + ([lang_col] if lang_col else [])].rename(
                columns={text_col: 'text', label_col: 'label', (lang_col or 'language'): 'language'}
            ) if lang_col else df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label'})

            if 'language' in df.columns:
                df['language'] = df['language'].astype(str).str.lower()
                df = df[df['language'].isin(lang_set)]
            else:
                if lang_in_path is None:
                    # cannot assign language reliably; skip to avoid mixing other languages
                    continue
                df['language'] = lang_in_path

            if df.empty:
                continue
            dfs.append(df)

    if not dfs:
        print('No suitable train/dev files found in the GitHub ZIP scan.')
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.dropna(subset=['text', 'label'])
    df_all['text'] = df_all['text'].astype(str)
    df_all['label'] = df_all['label'].astype(str).str.lower()

    label_map = {
        'neg': 'negative', 'negative': 'negative', '-1': 'negative',
        'neu': 'neutral', 'neutral': 'neutral', '0': 'neutral',
        'pos': 'positive', 'positive': 'positive', '1': 'positive'
    }
    df_all['label'] = df_all['label'].map(lambda x: label_map.get(x, x))
    df_all = df_all[df_all['label'].isin(['negative','neutral','positive'])]

    # cleanup temp dir
    shutil.rmtree(tmpdir, ignore_errors=True)
    return df_all

def filter_languages(df, langs=('sw','am','en')):
    # Accept both code and full names
    lang_map = {
        'swahili': 'sw', 'sw': 'sw',
        'amharic': 'am', 'am': 'am',
        'english': 'en', 'en': 'en',
    }
    df_norm = df.copy()
    df_norm['language'] = df_norm['language'].astype(str).str.lower().map(lambda x: lang_map.get(x, x))
    return df_norm[df_norm['language'].isin(langs)].reset_index(drop=True)

def eda_plots(df, prefix='real_eda'):
    files = []
    # Label distribution
    plt.figure(figsize=(6,4))
    counts = df['label'].value_counts()
    sns.barplot(x=counts.index, y=counts.values)
    plt.title('Label distribution (real)')
    out = OUT_DIR / f'{prefix}_label_dist.png'
    plt.tight_layout(); plt.savefig(out); plt.close(); files.append(out)
    # Language distribution
    plt.figure(figsize=(6,4))
    counts = df['language'].value_counts()
    sns.barplot(x=counts.index, y=counts.values)
    plt.title('Language distribution (real)')
    out = OUT_DIR / f'{prefix}_lang_dist.png'
    plt.tight_layout(); plt.savefig(out); plt.close(); files.append(out)
    # Char length hist
    df['char_len'] = df['text'].astype(str).str.len()
    plt.figure(figsize=(8,4))
    sns.histplot(df['char_len'], bins=50)
    plt.title('Text length (chars) – real')
    out = OUT_DIR / f'{prefix}_char_len.png'
    plt.tight_layout(); plt.savefig(out); plt.close(); files.append(out)
    # Token length hist
    df['token_len'] = df['text'].astype(str).str.split().apply(len)
    plt.figure(figsize=(8,4))
    sns.histplot(df['token_len'], bins=50)
    plt.title('Token length (approx) – real')
    out = OUT_DIR / f'{prefix}_token_len.png'
    plt.tight_layout(); plt.savefig(out); plt.close(); files.append(out)
    # Heatmap label by language
    plt.figure(figsize=(6,5))
    pivot = df.groupby(['language','label']).size().unstack(fill_value=0)
    sns.heatmap(pivot, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Label counts by language (real)')
    out = OUT_DIR / f'{prefix}_heatmap_label_by_language.png'
    plt.tight_layout(); plt.savefig(out); plt.close(); files.append(out)
    return files

def pca_tfidf_scatter(df, prefix='real_pca_tfidf_scatter'):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import PCA
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=10000)
    X = vec.fit_transform(df['text'].astype(str))
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X.toarray())
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X2[:,0], y=X2[:,1], hue=df['label'], style=df['language'], alpha=0.6)
    plt.title('PCA of TF-IDF features (real)')
    out = OUT_DIR / f'{prefix}.png'
    plt.tight_layout(); plt.savefig(out); plt.close()
    # Save explained variance
    evr = pca.explained_variance_ratio_.tolist()
    with open(OUT_DIR / 'real_pca_evr.json', 'w', encoding='utf-8') as f:
        json.dump({'explained_variance_ratio': evr, 'cumulative_2': float(sum(evr[:2]))}, f, indent=2)
    # Save top loadings
    comps = pca.components_
    feats = vec.get_feature_names_out()
    loadings = {}
    for i in range(2):
        comp = comps[i]
        idx_sorted = np.argsort(comp)
        top_pos = [feats[j] for j in idx_sorted[-10:][::-1]]
        top_neg = [feats[j] for j in idx_sorted[:10]]
        loadings[f'PC{i+1}'] = {'top_positive': top_pos, 'top_negative': top_neg}
    with open(OUT_DIR / 'real_pca_loadings.json', 'w', encoding='utf-8') as f:
        json.dump(loadings, f, indent=2)
    return out

def train_and_compare(df):
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score

    df = df.copy()
    df['clean_text'] = df['text'].astype(str).apply(simple_clean)
    X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, stratify=df['label'], random_state=42)

    models = {
        'tfidf_logreg': make_pipeline(TfidfVectorizer(ngram_range=(1,2), max_features=20000), LogisticRegression(max_iter=300)),
        'tfidf_nb': make_pipeline(TfidfVectorizer(ngram_range=(1,2), max_features=20000), MultinomialNB()),
        'tfidf_svc': make_pipeline(TfidfVectorizer(ngram_range=(1,2), max_features=20000), CalibratedClassifierCV(LinearSVC(), cv=3)),
    }

    labels = sorted(df['label'].unique())
    results = {}
    for name, model in models.items():
        print('Training', name)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        # probabilities for ROC (NB and LogReg have predict_proba; Calibrated SVC too)
        try:
            probs = model.predict_proba(X_test)
        except Exception:
            probs = None
        metrics = compute_metrics(y_test, preds, labels=labels)
        results[name] = metrics
        # save metrics
        with open(OUT_DIR / f'{name}_metrics_real.json', 'w', encoding='utf-8') as f:
            json.dump({'accuracy': metrics['accuracy'], 'macro_f1': metrics['macro_f1'], 'micro_f1': metrics['micro_f1'], 'report': metrics['report']}, f, indent=2)
        # confusion
        cm_path = OUT_DIR / f'{name}_confusion_real.png'
        plot_confusion(y_test, preds, labels=labels, save_path=str(cm_path))
        # ROC
        if probs is not None:
            try:
                # map labels to indices in sorted list
                label_to_idx = {l:i for i,l in enumerate(labels)}
                y_idx = np.array([label_to_idx[l] for l in y_test])
                from sklearn.preprocessing import label_binarize
                y_true_bin = label_binarize(y_idx, classes=list(range(len(labels))))
                auc = roc_auc_score(y_true_bin, probs, average='macro', multi_class='ovr')
                results[name]['macro_roc_auc'] = auc
            except Exception:
                results[name]['macro_roc_auc'] = None
    # bar chart of per-class F1 by model
    reports = {m: r['report'] for m,r in results.items()}
    per_class = sorted(labels)
    data = []
    for m in models.keys():
        for c in per_class:
            f1 = reports[m].get(c, {}).get('f1-score', np.nan)
            data.append({'model': m, 'label': c, 'f1': f1})
    df_f1 = pd.DataFrame(data)
    plt.figure(figsize=(8,5))
    sns.barplot(x='label', y='f1', hue='model', data=df_f1)
    plt.ylim(0,1)
    plt.title('Per-class F1 by model (real)')
    plt.tight_layout()
    bar_path = OUT_DIR / 'model_f1_bar_real.png'
    plt.savefig(bar_path); plt.close()
    # save summary
    with open(OUT_DIR / 'model_comparison_summary_real.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run real AfriSenti EDA/PCA/baselines')
    parser.add_argument('--data_dir', type=str, default=str(DATA_DIR), help='Local directory containing AfriSenti CSV/TSV files')
    parser.add_argument('--langs', type=str, default='sw,am,en', help='Comma-separated language codes to include (e.g., sw,am,en)')
    parser.add_argument('--github_map', type=str, default=None, help='Path to JSON mapping of GitHub raw URLs for train/dev per language')
    parser.add_argument('--github_repo_zip', type=str, default=None, help='GitHub repo spec owner/repo@branch to download and scan (defaults branch=main)')
    args = parser.parse_args()

    langs_tuple = tuple([x.strip() for x in args.langs.split(',') if x.strip()]) or ('sw','am','en')

    df = try_load_hf()
    if df is None:
        print('HF load failed. Attempting to download CSV/TSV from HF Hub...')
        df = ensure_local_afrisenti_csvs(langs=langs_tuple)
        if df is None:
            print('Hub download failed or files unavailable. Trying local data_dir:', args.data_dir)
            try:
                df_local = load_afrisenti(data_dir=args.data_dir)
                df = df_local
            except Exception as e:
                print('Could not load local AfriSenti CSVs from', args.data_dir, 'error:', e)
                if args.github_map:
                    print('Attempting GitHub raw URLs using map:', args.github_map)
                    df_map = load_github_map_and_download(args.github_map, langs=langs_tuple)
                    if df_map is None:
                        print('GitHub map download failed.')
                        # try ZIP repo if provided
                        if args.github_repo_zip:
                            print('Attempting GitHub ZIP repo scan:', args.github_repo_zip)
                            df_zip = download_and_scan_github_repo_zip(args.github_repo_zip, langs=langs_tuple)
                            if df_zip is None:
                                print('GitHub ZIP scan failed. Place AfriSenti CSVs locally and re-run.')
                                return
                            df = df_zip
                        else:
                            print('Provide --github_repo_zip or place CSVs locally, then re-run.')
                            return
                    df = df_map
                else:
                    if args.github_repo_zip:
                        print('Attempting GitHub ZIP repo scan:', args.github_repo_zip)
                        df_zip = download_and_scan_github_repo_zip(args.github_repo_zip, langs=langs_tuple)
                        if df_zip is None:
                            print('GitHub ZIP scan failed. Place AfriSenti CSVs locally and re-run.')
                            return
                        df = df_zip
                    else:
                        print('Place AfriSenti CSVs in', args.data_dir, 'or provide --github_map / --github_repo_zip, then re-run.')
                        return
    df = filter_languages(df, langs=langs_tuple)
    print('Loaded real AfriSenti rows (sw/am/en):', len(df))
    print('Language breakdown:', df['language'].value_counts().to_dict())
    print('Label breakdown:', df['label'].value_counts().to_dict())
    # EDA
    eda_files = eda_plots(df)
    print('Saved EDA plots:', [str(p) for p in eda_files])
    # PCA
    pca_path = pca_tfidf_scatter(df)
    print('Saved PCA scatter:', pca_path)
    # Models
    results = train_and_compare(df)
    print('Model comparison summary saved to eda_outputs/model_comparison_summary_real.json')

if __name__ == '__main__':
    main()