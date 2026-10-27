#!/usr/bin/env python3
"""Compare regenerated consensus clusterings with the deposited ones.

Consensus partitions are deterministic only within one numerical environment (CPU,
BLAS, thread count, library versions). On another machine, regenerating them gives
different but equivalent partitions. The deposit was produced on a compute cluster,
so a regeneration elsewhere is not expected to match it exactly.

What the paper's claims rest on is agreement with the reference tags, and that
should stay put. For each scan and showcased resolution this script reports:

* ARI against the deposited partition. This is for information: re-seeding the
  aggregation alone moves it to 0.74-0.95 (Supplementary Appendix F.3, Table S2).
* The change in agreement with the tags (ARI, NMI, mean purity). This is checked:
  the ARI change must stay within TOLERANCE, just above the largest re-seeding
  shift in Table S2 (0.015).

    python scripts/check_regeneration.py results/regen     # exits non-zero if a check fails
"""
import glob
import os
import re
import sys

import pandas as pd
from sklearn.metrics import adjusted_rand_score

from srwc.metrics import agreement

SHOWCASED = {"nva": [3, 30], "extended": [13, 120]}
TOLERANCE = 0.02
# Table S2: re-seeding the consensus aggregation (KMeans, gamma = 0.2), ARI vs the reported partition
RESEED = {("nva", 3): (0.889, 0.945), ("nva", 30): (0.744, 0.772),
          ("extended", 13): (0.867, 0.902), ("extended", 120): (0.840, 0.871)}


def main(folder):
    dep = os.path.join(os.environ["SRWC_DEPOSIT"], "clusterings_fullscan")
    failed = 0
    print(f"{'scan':24s} {'K':>4s} {'ARI vs deposit':>15s} {'re-seed range':>14s} "
          f"{'dARI(tags)':>11s} {'dNMI':>7s} {'dPurity':>8s}")
    for path in sorted(glob.glob(os.path.join(folder, "*_g*_*.csv"))):
        name = os.path.basename(path)[:-4]
        vocab = re.match(r"(nva|extended)_", name).group(1)
        ours = pd.read_csv(path, keep_default_na=False)
        ref = pd.read_csv(os.path.join(dep, f"{name}.csv"), keep_default_na=False)
        for k in SHOWCASED[vocab]:
            o = ours[ours.k == k].set_index("token_id")
            r = ref[ref.k == k].set_index("token_id").loc[o.index]
            a, b = agreement(o.cluster, o.pos), agreement(r.cluster, r.pos)
            d_ari = a["ARI"] - b["ARI"]
            ok = abs(d_ari) <= TOLERANCE
            failed += not ok
            lo, hi = RESEED[vocab, k]
            print(f"{name:24s} {k:4d} {adjusted_rand_score(r.cluster, o.cluster):15.3f} "
                  f"{f'{lo:.3f}-{hi:.3f}':>14s} {d_ari:+11.4f} {a['NMI'] - b['NMI']:+7.4f} "
                  f"{a['Mean_Purity'] - b['Mean_Purity']:+8.4f}{'' if ok else '  FAIL'}")
    print(f"\n{'all checks passed' if not failed else f'{failed} checks outside tolerance'} "
          f"(|dARI vs tags| <= {TOLERANCE})")
    return failed


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1]) else 0)
