import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from pathlib import Path
ROOT = Path('c:/Users/BMC/Desktop/NLP')
df = pd.read_csv(ROOT / 'data' / 'synthetic_afrisenti.csv')
vec = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X = vec.fit_transform(df['text'].astype(str))
pca = PCA(n_components=10)
X2 = pca.fit_transform(X.toarray())
print('Explained variance ratio (first 5):', pca.explained_variance_ratio_[:5].round(4))
print('Cumulative variance (first 5):', pca.explained_variance_ratio_[:5].cumsum().round(4))
feature_names = vec.get_feature_names_out()
for i in range(2):
    comp = pca.components_[i]
    idx_sorted = comp.argsort()
    top_pos = [feature_names[j] for j in idx_sorted[-10:][::-1]]
    top_neg = [feature_names[j] for j in idx_sorted[:10]]
    print(f'PC{i+1} top positive loadings:', top_pos)
    print(f'PC{i+1} top negative loadings:', top_neg)
