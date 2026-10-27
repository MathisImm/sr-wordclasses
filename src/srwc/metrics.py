"""Agreement with the reference tags (Supplementary Appendix E.4) and transition weights (E.5)."""
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def agreement(labels, pos) -> dict:
    """Mean_Purity: unweighted mean over clusters of the dominant tag's share.
    Purity_Standard: the same weighted by cluster size. NMI and ARI as in sklearn."""
    labels, pos = np.asarray(labels), np.asarray(pos)
    purities, sizes = [], []
    for c in np.unique(labels):
        members = pos[labels == c]
        purities.append(np.unique(members, return_counts=True)[1].max() / len(members))
        sizes.append(len(members))
    return {"Mean_Purity": float(np.mean(purities)),
            "Purity_Standard": sum(p * s for p, s in zip(purities, sizes)) / sum(sizes),
            "NMI": normalized_mutual_info_score(pos, labels),
            "ARI": adjusted_rand_score(pos, labels)}


def transitions(logits, labels, token_ids, gamma) -> tuple[np.ndarray, list]:
    """Row-normalised share of each cluster's pooled successor mass falling on each cluster,
    with targets restricted to the clustered words. The (1 - gamma) rescaling and the
    source cluster's size are constant within a row and cancel; they are kept for fidelity."""
    sr = softmax(np.asarray(logits, dtype=np.float32), axis=1) / (1 - gamma)
    sr = sr[:, np.asarray(token_ids)]
    labels = np.asarray(labels)
    clusters = sorted(c for c in np.unique(labels) if c != -1)
    M = np.zeros((len(clusters), len(clusters)))
    for i, a in enumerate(clusters):
        src = labels == a
        for j, b in enumerate(clusters):
            M[i, j] = sr[np.ix_(src, labels == b)].sum() / src.sum()
    rows = M.sum(axis=1, keepdims=True)
    rows[rows == 0] = 1.0
    return M / rows, clusters


def majority_tag(pos: pd.Series, labels, cluster) -> str:
    return pos[np.asarray(labels) == cluster].value_counts().idxmax()


def kscan(Z, P, truth, token_ids, ks, reference=None, showcase=()):
    """Score one consensus hierarchy at every K, as in the deposit's cell_*.json files.

    `reference` maps K to the reported partition (aligned to `token_ids`) for an
    agreement column; at `showcase` resolutions the labels and per-cluster purities
    are kept as well.
    """
    from sklearn.metrics import silhouette_score
    from .cluster import cut
    truth = np.asarray(truth)
    scan = []
    for k in ks:
        lab = cut(Z, k)
        per = [float(np.unique(truth[lab == c], return_counts=True)[1].max() / (lab == c).sum())
               for c in np.unique(lab)]
        rec = dict(target_k=int(k), k=int(len(np.unique(lab))), meanPurity=float(np.mean(per)),
                   purity_min=min(per), purity_max=max(per),
                   NMI=float(normalized_mutual_info_score(truth, lab)),
                   ARI=float(adjusted_rand_score(truth, lab)),
                   silhouette=float(silhouette_score(P.astype(np.float64), lab)) if len(per) > 1 else float("nan"))
        if reference is not None and k in reference:
            rec["ARI_vs_SR_g0.2"] = float(adjusted_rand_score(reference[k], lab))
        if k in showcase:
            rec.update(per_cluster_purity=per, cluster_sizes=np.bincount(lab).tolist(), labels=lab.tolist())
        scan.append(rec)
    return scan
