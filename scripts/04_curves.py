#!/usr/bin/env python3
"""Clustering-quality scans over gamma and K (Figures 7 and 10), in the deposit's format.

    python scripts/04_curves.py --clusterings $SRWC_DEPOSIT/clusterings_fullscan --out curves.csv
"""
import argparse
import glob
import os
import re

import pandas as pd

from srwc.metrics import agreement

SETTING = {"nva": "nva", "extended": "ex-PUNCT_X_INTJ_SPACE_SYM"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clusterings", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for path in sorted(glob.glob(os.path.join(a.clusterings, "*_g*_*.csv"))):
        name = os.path.basename(path)
        vocab, gamma, method = re.match(r"(nva|extended)_g([\d.]+)_(kmeans|bgm)\.csv", name).groups()
        df = pd.read_csv(path, keep_default_na=False)
        for k, part in df.groupby("k", sort=True):
            rows.append({**agreement(part.cluster, part.pos), "Gamma": float(gamma), "K": k,
                         "Method": method, "Setting": SETTING[vocab],
                         "Source": f"clusterings_fullscan/{name}"})
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
