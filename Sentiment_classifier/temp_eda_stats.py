import pandas as pd
from pathlib import Path
import numpy as np
ROOT = Path('c:/Users/BMC/Desktop/NLP')
df = pd.read_csv(ROOT / 'data' / 'synthetic_afrisenti.csv')
print('Rows:', len(df))
print('Languages:', df['language'].value_counts().to_dict())
print('Labels:', df['label'].value_counts().to_dict())
# compute lengths
texts = df['text'].astype(str)
df['char_len'] = texts.str.len()
df['token_len'] = texts.str.split().apply(len)
# overall stats
def stats(s):
    return {'min': int(s.min()), 'max': int(s.max()), 'mean': float(np.round(s.mean(),2)), 'median': int(np.round(s.median())), 'std': float(np.round(s.std(),2))}
print('Char length stats:', stats(df['char_len']))
print('Token length stats:', stats(df['token_len']))
# per-label stats
by_label = df.groupby('label').agg({'char_len':['mean','median','min','max'], 'token_len':['mean','median','min','max']})
print(by_label)
# per-language label distribution
print('Per-language label counts:')
print(df.groupby(['language','label']).size().unstack(fill_value=0))
