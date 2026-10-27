#!/usr/bin/env python3
"""Robustness grid of Supplementary Appendix F (F.2-F.4).

Re-runs the consensus clustering while varying only the head (gamma 0.2/0.5/0.8),
the vocabulary (NVA/extended), the algorithm (KMeans/BGM), the feature columns
(all; without <unk>; without the 17 unused <unk_POS> slots; without all 20 reserved
IDs) and the consensus seed (42, plus 0, 7 and 123 for gamma 0.2 KMeans with all
columns). Output rows follow the deposit's robustness/robustness_grid.csv.

    python scripts/appendix_F.py --out grid.csv [--methods kmeans] [--cells g0.2:nva:kmeans:a_all:42 ...]
    python scripts/appendix_F.py --out grid.csv --verify   # compare with the deposit

A full run is dominated by BGM (about a day); cells already in --out are skipped.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from srwc import cluster

KS = {"nva": [3, 6, 9, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100],
      "ext": [13, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 150, 160, 180, 200]}
ALL = np.arange(20000)
COLSETS = {"a_all": ALL, "b_nounk": np.setdiff1d(ALL, [0]),
           "c_nopos": np.setdiff1d(ALL, np.arange(3, 20)), "d_none": np.setdiff1d(ALL, np.arange(0, 20))}
PUB = {"nva": "nva", "ext": "extended"}
COLUMNS = ["head", "setting", "method", "colset", "seed", "target_k", "k", "meanPurity", "NMI", "ARI",
           "ARI_vs_published"]


def jobs(methods):
    out = [(h, st, m, cs, 42) for h in ("g0.2", "g0.5", "g0.8") for st in ("ext", "nva")
           for m in ("kmeans", "bgm") for cs in COLSETS]
    out += [("g0.2", st, "kmeans", "a_all", sd) for st in ("ext", "nva") for sd in (0, 7, 123)]
    return [j for j in out if j[2] in methods]


def run_cell(head, st, method, colset, seed):
    dep = os.environ["SRWC_DEPOSIT"]
    g = head[1:]
    tok = pd.read_csv(os.path.join(dep, "sr_vectors", f"tokens_g{g}_{st}.csv"), keep_default_na=False)
    X = np.load(os.path.join(dep, "sr_vectors", f"sr_head_g{g}_{st}.npy"))[:, COLSETS[colset]]
    pub = pd.read_csv(os.path.join(dep, "clusterings_showcased", f"{PUB[st]}_g{g}_{method}.csv"),
                      keep_default_na=False)
    ref = {int(k): p.set_index("token_id").loc[tok.token_id, "cluster"].to_numpy()
           for k, p in pub.groupby("k") if int(k) in KS[st]}
    Z, _ = cluster.cluster(X, method, seed=seed)
    pos = tok.pos.to_numpy()
    rows = []
    for target in KS[st]:
        lab = cluster.cut(Z, target)
        k = len(np.unique(lab))      # one fewer than requested when merge heights tie
        pur = np.mean([np.unique(pos[lab == c], return_counts=True)[1].max() / (lab == c).sum()
                       for c in np.unique(lab)])
        rows.append(dict(head=head, setting=st, method=method, colset=colset, seed=seed, target_k=target, k=k,
                         meanPurity=float(pur), NMI=normalized_mutual_info_score(pos, lab),
                         ARI=adjusted_rand_score(pos, lab),
                         ARI_vs_published=adjusted_rand_score(ref[k], lab) if k in ref else np.nan))
    return rows


def verify(path):
    ours = pd.read_csv(path)
    ref = pd.read_csv(os.path.join(os.environ["SRWC_DEPOSIT"], "robustness", "robustness_grid.csv"))
    key = ["head", "setting", "method", "colset", "seed", "target_k"]
    m = ours.merge(ref, on=key, suffixes=("", "_ref"))
    bad = 0
    for cell, g in m.groupby(key[:5]):
        diff = max(np.nanmax(np.abs(g[c] - g[f"{c}_ref"])) if g[c].notna().any() else 0.0
                   for c in ("meanPurity", "NMI", "ARI", "ARI_vs_published"))
        ok = diff < 1e-12 and (g.k == g.k_ref).all()
        bad += not ok
        print(f"{'ok  ' if ok else 'DIFF'} {' '.join(map(str, cell)):32s} max metric diff {diff:.3g}")
    print(f"{len(m.groupby(key[:5]))} cells compared, {bad} differ")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", default="kmeans,bgm")
    ap.add_argument("--cells", nargs="*", help="head:setting:method:colset:seed, e.g. g0.2:nva:kmeans:b_nounk:42")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        sys.exit(1 if verify(a.out) else 0)
    todo = ([(h, s, m, c, int(sd)) for h, s, m, c, sd in (x.split(":") for x in a.cells)] if a.cells
            else jobs(a.methods.split(",")))
    done = pd.read_csv(a.out) if os.path.exists(a.out) else pd.DataFrame(columns=COLUMNS)
    have = set(map(tuple, done[COLUMNS[:5]].astype(str).values))
    for job in todo:
        if tuple(map(str, job)) in have:
            continue
        done = pd.concat([done, pd.DataFrame(run_cell(*job))], ignore_index=True)
        done.to_csv(a.out, index=False)
        print("cell", *job, flush=True)


if __name__ == "__main__":
    main()
