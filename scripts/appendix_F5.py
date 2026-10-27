#!/usr/bin/env python3
"""Feature scale (Supplementary Appendix F.5).

The clustered vectors are the heads' pre-softmax outputs z. Since psi = softmax(z),
z = log psi + c(w) with one constant per word. The same pipeline is run on:
  logit     z itself, the reported condition;
  logpsi    z - logsumexp(z) = log psi;
  psi       the successor distribution;
  clr       z - mean(z), the centred log-ratio of psi, in which c(w) cancels;
  srscaled  psi / (1 - gamma), at gamma = 0.2 on NVA only, as a check.

variance   How variance spreads over the vocabulary axis per scale
           (variance_by_scale.csv). Computed from the deposit's vectors, exact.
grid       One consensus clustering per scale, head and vocabulary, scored at every K
           (cell_<scale>_g<gamma>_<setting>_kmeans.json). This step regenerates.
           Recomputed transforms are additionally sensitive to precision (srwc/cluster.py).
summarise  Collects cells into feature_scale.csv (exact from the deposit's cells).

    python scripts/appendix_F5.py variance  --out F5/
    python scripts/appendix_F5.py grid      --out F5/
    python scripts/appendix_F5.py summarise --cells $SRWC_DEPOSIT/feature_scale --out F5/
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

from srwc import cluster
from srwc.metrics import kscan

GAMMAS, SETTINGS, SCALES = ("0.2", "0.5", "0.8"), ("nva", "ext"), ("logit", "logpsi", "psi", "clr")
KSCAN = {"nva": range(2, 101), "ext": range(2, 201)}
SHOWCASE = {"nva": [3, 30], "ext": [13, 120]}
PUB = {"nva": "nva", "ext": "extended"}
SCALE_ORDER = ["logit", "clr", "logpsi", "psi", "srscaled"]
CELL_RE = re.compile(r"cell_(?P<scale>[a-z]+)_g(?P<gamma>[\d.]+)_(?P<setting>nva|ext)_kmeans\.json$")


def _sr(name):
    return os.path.join(os.environ["SRWC_DEPOSIT"], "sr_vectors", name)


def _logsumexp(Z):
    m = Z.max(axis=1, keepdims=True)
    return m[:, 0] + np.log(np.exp(Z - m).sum(axis=1))


def cmd_variance(a):
    rows = []
    for g in GAMMAS:
        for st in SETTINGS:
            Z = np.load(_sr(f"sr_head_g{g}_{st}.npy")).astype(np.float64)
            c = _logsumexp(Z)
            L = Z - c[:, None]
            for scale, X in (("logit", Z), ("logpsi", L), ("psi", np.exp(L)), ("clr", Z - Z.mean(axis=1, keepdims=True))):
                v = np.sort(X.var(axis=0))[::-1]
                tot = v.sum()
                rows.append(dict(scale=scale, gamma=float(g), setting=st, total_variance=float(tot),
                                 top10_share=float(v[:10].sum() / tot), top100_share=float(v[:100].sum() / tot),
                                 dims_for_90pct=int((np.cumsum(v) / tot < 0.90).sum()) + 1,
                                 c_sd=float(c.std()), c_min=float(c.min()), c_max=float(c.max())))
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "variance_by_scale.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def cmd_grid(a):
    os.makedirs(a.out, exist_ok=True)
    jobs = [(s, g, st) for g in GAMMAS for st in SETTINGS for s in SCALES] + [("srscaled", "0.2", "nva")]
    for scale, g, st in jobs:
        path = os.path.join(a.out, f"cell_{scale}_g{g}_{st}_kmeans.json")
        if os.path.exists(path):
            continue
        tok = pd.read_csv(_sr(f"tokens_g{g}_{st}.csv"), keep_default_na=False)
        Z = np.load(_sr(f"sr_head_g{g}_{st}.npy"))
        if scale == "srscaled":
            X = (np.exp(Z.astype(np.float64) - _logsumexp(Z.astype(np.float64))[:, None]) / (1 - float(g))).astype(np.float32)
        else:
            X = cluster.rescale(Z, scale)
        ids, truth = tok.token_id.to_numpy(), tok.pos.to_numpy()
        pub = pd.read_csv(os.path.join(os.environ["SRWC_DEPOSIT"], "clusterings_fullscan", f"{PUB[st]}_g{g}_kmeans.csv"),
                          keep_default_na=False)
        ref = {int(k): p.set_index("token_id").loc[ids, "cluster"].to_numpy() for k, p in pub.groupby("k")}
        Zl, P = cluster.cluster(X, "kmeans")
        scan = kscan(Zl, P, truth, ids, KSCAN[st], ref, SHOWCASE[st])
        with open(path, "w") as f:
            json.dump(dict(variant=f"{scale}_g{g}", setting=st, method="kmeans", seed=cluster.SEED,
                           n_tokens=len(tok), pca_components=int(P.shape[1]), kscan=scan), f)
        print(f"{scale} g{g} {st} done", flush=True)


def cmd_summarise(a):
    rows = []
    for path in sorted(glob.glob(os.path.join(a.cells, "cell_*.json"))):
        m = CELL_RE.search(os.path.basename(path))
        if not m:
            continue
        d = json.load(open(path))
        rows += [dict(scale=m["scale"], gamma=float(m["gamma"]), setting=m["setting"], target_k=r["target_k"],
                      k=r["k"], ARI=r["ARI"], NMI=r["NMI"], meanPurity=r["meanPurity"],
                      purity_min=r["purity_min"], purity_max=r["purity_max"],
                      ARI_vs_reported=r.get("ARI_vs_SR_g0.2"), pca_components=d["pca_components"])
                 for r in d["kscan"]]
    df = pd.DataFrame(rows)
    df["scale"] = pd.Categorical(df["scale"], SCALE_ORDER, ordered=True)
    df = df.sort_values(["setting", "gamma", "scale", "target_k"])
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "feature_scale.csv"), index=False)
    floor = {"nva": 3, "ext": 13}
    for st in SETTINGS:
        sub = df[(df.setting == st) & (df.target_k >= floor[st])]
        print(f"\n{st}: mean over cuts from K={floor[st]}")
        print(sub.groupby(["scale", "gamma"], observed=True)[["meanPurity", "NMI", "ARI"]].mean().round(4).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["variance", "grid", "summarise"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", help="summarise: folder with cell_*.json")
    a = ap.parse_args()
    globals()[f"cmd_{a.stage}"](a)


if __name__ == "__main__":
    main()
