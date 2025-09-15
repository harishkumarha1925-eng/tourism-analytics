# src/recommenders.py
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

def user_item_matrix(df):
    piv = df.pivot_table(index="UserId", columns="AttractionId", values="Rating", aggfunc="mean").fillna(0)
    return piv

def simple_svd_recommender(df, user_id, n_components=50, top_k=10):
    piv = user_item_matrix(df)
    if user_id not in piv.index:
        return []
    n_comp = min(n_components, max(2, piv.shape[1]-1))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    latent = svd.fit_transform(piv)
    user_idx = list(piv.index).index(user_id)
    user_vec = latent[user_idx]
    sims = np.dot(latent, user_vec) / (np.linalg.norm(latent, axis=1) * (np.linalg.norm(user_vec)+1e-9))
    sim_series = pd.Series(sims, index=piv.index)
    weighted = (sim_series.values.reshape(-1,1) * piv.values).sum(axis=0) / (sim_series.sum()+1e-9)
    item_scores = pd.Series(weighted, index=piv.columns).sort_values(ascending=False)
    return item_scores.head(top_k).index.tolist()

def content_knn_recommend(df, item_id, item_feature_cols, top_k=10):
    items = df[[ "AttractionId"] + item_feature_cols].drop_duplicates("AttractionId").set_index("AttractionId").fillna(0)
    if item_id not in items.index:
        return []
    knn = NearestNeighbors(n_neighbors=min(top_k+1, len(items)), metric="cosine").fit(items.values)
    idx = list(items.index).index(item_id)
    dists, idxs = knn.kneighbors([items.values[idx]])
    recs = [items.index[i] for i in idxs[0] if items.index[i] != item_id][:top_k]
    return recs
