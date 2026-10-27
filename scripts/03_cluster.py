#!/usr/bin/env python3
"""Consensus clusterings for one head, algorithm and vocabulary.

Writes the full every-integer-K scan (K = 1..100 for NVA, 1..200 extended) in the
deposit's long format (k, token_id, token, pos, cluster), the linkage, and the PCA
coordinates used for the Appendix A-D listings.

    python scripts/03_cluster.py --vocab nva --gamma 0.2 --method kmeans --out results/
"""
import argparse
import os

import numpy as np
import pandas as pd

from srwc import cluster

MAX_K = {"nva": 100, "ext": 200}
NAME = {"nva": "nva", "ext": "extended"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocab", choices=["nva", "ext"], required=True)
    ap.add_argument("--gamma", choices=["0.2", "0.5", "0.8"], required=True)
    ap.add_argument("--method", choices=["kmeans", "bgm"], required=True)
    ap.add_argument("--sr-dir", default=None, help="folder with sr_head_*.npy and tokens_*.csv "
                    "(default: $SRWC_DEPOSIT/sr_vectors)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sr_dir = a.sr_dir or os.path.join(os.environ["SRWC_DEPOSIT"], "sr_vectors")
    X = np.load(os.path.join(sr_dir, f"sr_head_g{a.gamma}_{a.vocab}.npy"))
    tok = pd.read_csv(os.path.join(sr_dir, f"tokens_g{a.gamma}_{a.vocab}.csv"), keep_default_na=False)
    Z, P = cluster.cluster(X, a.method)

    stem = f"{NAME[a.vocab]}_g{a.gamma}_{a.method}"
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, f"{stem}_linkage.npy"), Z)
    np.save(os.path.join(a.out, f"{stem}_pca.npy"), P)
    rows = [pd.DataFrame({"k": k, "token_id": tok.token_id, "token": tok.token, "pos": tok.pos,
                          "cluster": cluster.cut(Z, k)}) for k in range(1, MAX_K[a.vocab] + 1)]
    pd.concat(rows).to_csv(os.path.join(a.out, f"{stem}.csv"), index=False)
    print(f"wrote {stem}")


if __name__ == "__main__":
    main()
