#!/usr/bin/env python3
"""Token listings of Supplementary Appendices A-D (KMeans, gamma = 0.2).

Within each cluster the words are ordered by Euclidean distance to the cluster mean
in the PCA space the clustering ran in. Words at the same distance -- in particular
both words of a two-word cluster, which are equidistant from their mean by
construction -- keep their order in the word table (by frequency). In the original
run such ties were broken by floating-point rounding, so the supplement may list
those pairs the other way round.

    python scripts/06_appendix_listings.py --out listings.tex
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

from srwc.cluster import reduce_pca

CASES = [("nva", "nva", 3), ("nva", "nva", 30), ("extended", "ext", 13), ("extended", "ext", 120)]


def escape(s):
    return (str(s).replace("\\", "\\textbackslash ").replace("_", "\\_").replace("$", "\\$")
            .replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")
            .replace("{", "\\{").replace("}", "\\}"))


def distances(labels, P):
    labels = np.asarray(labels)
    clusters = sorted(set(labels))
    centroids = np.array([P[labels == c].mean(axis=0) for c in clusters])
    return euclidean_distances(P, centroids)[np.arange(len(labels)), [clusters.index(c) for c in labels]]


def listing(tokens, labels, P):
    labels = np.asarray(labels)
    clusters = sorted(set(labels))
    d = distances(labels, P)
    out = []
    for c in clusters:
        idx = np.where(labels == c)[0]
        idx = idx[np.argsort(np.round(d[idx], 9), kind="stable")]
        out.append(f"\\textbf{{Cluster {c}}}: " + ", ".join(escape(tokens[i]) for i in idx))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sr-dir", default=None, help="default: $SRWC_DEPOSIT/sr_vectors")
    ap.add_argument("--clusterings", default=None, help="default: $SRWC_DEPOSIT/clusterings_showcased")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dep = os.environ.get("SRWC_DEPOSIT", "")
    sr_dir = a.sr_dir or os.path.join(dep, "sr_vectors")
    cl_dir = a.clusterings or os.path.join(dep, "clusterings_showcased")
    blocks = []
    for name, stem, k in CASES:
        tok = pd.read_csv(os.path.join(sr_dir, f"tokens_g0.2_{stem}.csv"), keep_default_na=False)
        P, _ = reduce_pca(np.load(os.path.join(sr_dir, f"sr_head_g0.2_{stem}.npy")))
        part = pd.read_csv(os.path.join(cl_dir, f"{name}_g0.2_kmeans.csv"), keep_default_na=False)
        labels = part[part.k == k].set_index("token_id").loc[tok.token_id, "cluster"].to_numpy()
        blocks.append(f"% {name} K={k}\n" + "\n".join(listing(tok.token.tolist(), labels, P.astype(np.float64))))
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocks) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
