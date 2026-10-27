#!/usr/bin/env python3
"""Inter-cluster transition matrices behind Figure 5c-d (NVA, K = 3) and Figure 9 (extended, K = 120).

    python scripts/05_transitions.py --out transitions/
"""
import argparse
import os

import numpy as np
import pandas as pd

from srwc.metrics import majority_tag, transitions

CASES = [("nva", "nva", 3), ("extended", "ext", 120)]
GAMMA, METHOD = 0.2, "kmeans"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sr-dir", default=None, help="default: $SRWC_DEPOSIT/sr_vectors")
    ap.add_argument("--clusterings", default=None, help="default: $SRWC_DEPOSIT/clusterings_showcased")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dep = os.environ.get("SRWC_DEPOSIT", "")
    sr_dir = a.sr_dir or os.path.join(dep, "sr_vectors")
    cl_dir = a.clusterings or os.path.join(dep, "clusterings_showcased")
    os.makedirs(a.out, exist_ok=True)
    for name, stem, k in CASES:
        tok = pd.read_csv(os.path.join(sr_dir, f"tokens_g{GAMMA}_{stem}.csv"), keep_default_na=False)
        X = np.load(os.path.join(sr_dir, f"sr_head_g{GAMMA}_{stem}.npy"))
        part = pd.read_csv(os.path.join(cl_dir, f"{name}_g{GAMMA}_{METHOD}.csv"), keep_default_na=False)
        part = part[part.k == k].set_index("token_id").loc[tok.token_id].reset_index()
        M, clusters = transitions(X, part.cluster, tok.token_id, GAMMA)
        out = pd.DataFrame(M, columns=[f"C{c}" for c in clusters])
        out.insert(0, "n_tokens", [int((part.cluster == c).sum()) for c in clusters])
        out.insert(0, "majority_pos", [majority_tag(part.pos, part.cluster, c) for c in clusters])
        out.insert(0, "source_cluster", [f"C{c}" for c in clusters])
        path = os.path.join(a.out, f"{name}_k{k}_gamma{GAMMA}_{METHOD}.csv")
        out.to_csv(path, index=False)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
