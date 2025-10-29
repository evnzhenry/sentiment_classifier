import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
ROOT = Path('c:/Users/BMC/Desktop/NLP')
df = pd.read_csv(ROOT / 'data' / 'synthetic_afrisenti.csv')
X_train, X_test, y_train, y_test = train_test_split(df['text'].astype(str), df['label'], test_size=0.2, stratify=df['label'], random_state=42)
pipe = make_pipeline(TfidfVectorizer(ngram_range=(1,2), max_features=5000), LogisticRegression(max_iter=200))
pipe.fit(X_train, y_train)
probs = pipe.predict_proba(X_test)
# Map labels to indices in established order
labels = ['negative','neutral','positive']
label_to_idx = {l:i for i,l in enumerate(labels)}
y_true_idx = np.array([label_to_idx[l] for l in y_test])
y_true_bin = label_binarize(y_true_idx, classes=list(range(len(labels))))
auc = roc_auc_score(y_true_bin, probs, average='macro', multi_class='ovr')
print('Macro ROC-AUC:', round(auc,4))
