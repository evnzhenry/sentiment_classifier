"""Cross-lingual experiment helpers: split by language, run train/test across languages."""
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def split_by_language(df, lang_col='language'):
    groups = {}
    for lang, g in df.groupby(lang_col):
        groups[lang] = g.reset_index(drop=True)
    return groups


def prepare_train_test_for_lang(groups, train_langs, test_langs, stratify_col='label', test_size=0.2, random_state=42):
    train_dfs = [groups[l] for l in train_langs if l in groups]
    test_dfs = [groups[l] for l in test_langs if l in groups]
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    # further split train into train/val
    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df[stratify_col], random_state=random_state)
    return train_df, val_df, test_df


if __name__ == '__main__':
    print('Cross-lingual helpers ready')
