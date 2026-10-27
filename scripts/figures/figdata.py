"""Data access for the cluster figures (Figures 4-6, 8, 9), from the deposit.

The original plotting scripts read the archived consensus runs and recomputed the
read-out from the checkpoint. Here the same quantities come from the deposit:

* partitions: clusterings_showcased/, identical to the archived runs;
* features for the transition networks: softmax(logits) / (1 - gamma), the scaled
  successor representation;
* UMAP: of the pre-softmax logits (n_neighbors=15, min_dist=0.1, euclidean,
  random_state=42). UMAP depends on row order. The original embedded the words in
  the order of its KMeans assignment table at K = 3 (NVA) or K = 13 (extended),
  sorted by cluster and then by distance to the centroid in PCA space. That order
  is rebuilt here, which reproduces the archived coordinates exactly. One
  embedding per vocabulary serves every K, attached to the words, so a word keeps
  its position across figures.
"""
import os

import numpy as np
import pandas as pd
from scipy.special import softmax

DEPOSIT = os.environ.get("SRWC_DEPOSIT", "")
OUT = os.environ.get("SRWC_FIGURES", os.path.abspath("figures_out"))
VOCAB = {"nva": "nva", "full": "ext", "ext": "ext"}
NAME = {"nva": "nva", "ext": "extended"}


ORDER_K = {"nva": 3, "ext": 13}


def logits(head, setting):
    return np.load(os.path.join(DEPOSIT, "sr_vectors", f"sr_head_g{head[1:]}_{VOCAB[setting]}.npy"))


def tokens(head, setting):
    return pd.read_csv(os.path.join(DEPOSIT, "sr_vectors", f"tokens_g{head[1:]}_{VOCAB[setting]}.csv"),
                       keep_default_na=False)


def _table_order(head, setting, labels):
    """The original assignment-table order: by cluster, then by distance to the cluster
    mean in the PCA space of the clustering."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
    from sklearn.metrics.pairwise import euclidean_distances
    from srwc.cluster import reduce_pca
    P = reduce_pca(logits(head, setting))[0].astype(np.float64)
    clusters = sorted(set(labels))
    centroids = np.array([P[labels == c].mean(axis=0) for c in clusters])
    d = euclidean_distances(P, centroids)[np.arange(len(labels)), [clusters.index(c) for c in labels]]
    # In two-word clusters both words lie exactly as far from the mean, and rounding noise
    # would order them differently from run to run; ties fall back to word-table order.
    # (Leaves the K = 3 and K = 13 orders, and hence the UMAPs, unchanged.)
    return np.lexsort((np.round(d, 9), labels))


def _partition(head, setting, method, k):
    tok = tokens(head, setting)
    part = pd.read_csv(os.path.join(DEPOSIT, "clusterings_showcased",
                                    f"{NAME[VOCAB[setting]]}_g{head[1:]}_{method}.csv"), keep_default_na=False)
    return tok.assign(cluster=part[part.k == k].set_index("token_id").loc[tok.token_id, "cluster"].to_numpy())


def assignments(head, setting, method, k):
    """One partition with the columns of the original assignments.csv. KMeans tables come
    in the original's row order (which decides plotting order and colour assignment);
    other methods in word-table order, to be matched by token_id."""
    df = _partition(head, setting, method, k)
    if method == "kmeans":
        df = df.iloc[_table_order(head, setting, df.cluster.to_numpy())].reset_index(drop=True)
    return df


def _rows(head, setting, token_ids):
    if token_ids is None:
        return slice(None)
    where = {t: i for i, t in enumerate(tokens(head, setting).token_id)}
    return np.array([where[t] for t in token_ids])


def features(head, setting, token_ids=None):
    """Scaled successor representation softmax(z) / (1 - gamma), rows aligned to `token_ids`."""
    z = logits(head, setting)
    X = (softmax(z.astype(np.float64), axis=1) / (1 - float(head[1:]))).astype(np.float32)
    return X[_rows(head, setting, token_ids)]


def umap_coords(head, setting, token_ids=None):
    """2-D UMAP coordinates, rows aligned to `token_ids` (default: word-table order).
    Cached under $SRWC_FIGURES/cache."""
    path = os.path.join(OUT, "cache", f"umap_{head}_{VOCAB[setting]}.npy")
    if os.path.exists(path):
        return np.load(path)[_rows(head, setting, token_ids)]
    import umap
    k = ORDER_K[VOCAB[setting]]
    order = _table_order(head, setting, _partition(head, setting, "kmeans", k).cluster.to_numpy())
    emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42).fit_transform(
        logits(head, setting)[order])
    coords = np.empty_like(emb)
    coords[order] = emb
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, coords)
    return coords[_rows(head, setting, token_ids)]
