"""Consensus clustering (Methods; Supplementary Appendix E.3).

The head's logits are reduced by PCA to the fewest components holding 99.99% of the
variance. 240 base partitions follow, by KMeans or a Bayesian Gaussian mixture,
cycling through the 24 prime base resolutions 3..97 with trial seeds 42 + 1337 i.
The co-association matrix of the ensemble is clustered by average linkage, and the
resulting hierarchy is cut at each requested K, so every K comes from the same tree.

The result is deterministic only within one numerical environment: given the input
and seed, it also depends on the CPU, the BLAS and OpenMP libraries and their
thread counts. Co-association values are multiples of 1/240, so average linkage meets
near-ties, and a single base partition that flips between two near-equal solutions
because of floating-point rounding can reorder merges. In another environment the
procedure therefore gives different but equivalent partitions, much as a different
consensus seed does (Supplementary Appendix F.3). ``scripts/check_regeneration.py``
measures that difference against the deposit.

PCA and the base partitions run with a fixed number of threads (``THREADS``, 8, or
the ``SRWC_THREADS`` environment variable; 0 means library defaults), so repeated
runs on one machine give identical results. The deposit was computed on a compute
cluster, and no thread setting is guaranteed to reproduce it elsewhere.
"""
import os
from contextlib import nullcontext

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
from threadpoolctl import threadpool_limits

BASE_K = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
          53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
N_TRIALS = 240
PCA_VARIANCE = 0.9999
SEED = 42
KMEANS_N_INIT = 10
BGM_KW = dict(covariance_type="full", n_init=3, max_iter=300)
THREADS = int(os.environ.get("SRWC_THREADS", 8))   # 0: library defaults


def _limits():
    return threadpool_limits(limits=THREADS) if THREADS else nullcontext()


def rescale(X, scale="logit"):
    """Feature scales of Supplementary Appendix F.5, computed from the logits."""
    if scale == "logit":
        return X
    # float64 arithmetic, stored back to float32 as in the reported feature-scale run
    Z = X.astype(np.float64)
    if scale == "clr":
        return (Z - Z.mean(axis=1, keepdims=True)).astype(np.float32)
    m = Z.max(axis=1, keepdims=True)
    logpsi = Z - (m + np.log(np.exp(Z - m).sum(axis=1, keepdims=True)))
    return (logpsi if scale == "logpsi" else np.exp(logpsi)).astype(np.float32)


def reduce_pca(X, variance=PCA_VARIANCE):
    with _limits():
        full = PCA(n_components=min(X.shape)).fit(X)
        cum = np.cumsum(full.explained_variance_ratio_)
        n = int(np.argmax(cum >= variance) + 1)
        n = min(max(n, min(3, *X.shape)), len(cum))
        return PCA(n_components=n).fit_transform(X), n


def base_partitions(P, method, n_trials=N_TRIALS, seed=SEED):
    labels = []
    base_seed = seed
    for i in range(n_trials):
        seed = (base_seed + i * 1337) % 2**32
        k = BASE_K[i % len(BASE_K)]
        if method == "kmeans":
            est = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=seed)
        else:
            est = BayesianGaussianMixture(n_components=k, random_state=seed,
                                          weight_concentration_prior=None, **BGM_KW)
        with _limits():
            labels.append(est.fit_predict(P))
    return labels


def consensus_linkage(all_labels):
    n = len(all_labels[0])
    C = np.zeros((n, n))
    for lab in all_labels:
        valid = lab != -1
        C += (lab[:, None] == lab[None, :]) & valid[:, None] & valid[None, :]
    C /= len(all_labels)
    D = 1.0 - C
    np.fill_diagonal(D, 0)
    D = (D + D.T) / 2.0
    return linkage(squareform(D), method="average")


def cut(Z, k):
    return fcluster(Z, k, criterion="maxclust") - 1


def cluster(X, method="kmeans", scale="logit", seed=SEED):
    """Consensus linkage for one feature matrix; cut it with `cut(Z, k)`.

    `seed` is the consensus seed: 42 in every reported analysis, and 0, 7 and 123
    for the re-seeding check of Supplementary Appendix F.3."""
    P, _ = reduce_pca(rescale(X, scale))
    np.random.seed(SEED)
    return consensus_linkage(base_partitions(P.astype(np.float64), method, seed=seed)), P
