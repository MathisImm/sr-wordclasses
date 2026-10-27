#!/usr/bin/env python3
"""Tables and claims of Supplementary Appendix F (F.2-F.4) from the robustness grid.

Reads a grid in the format of the deposit's robustness/robustness_grid.csv, by default
that file, or one written by scripts/appendix_F.py. Means use every evaluated cluster
count: 14 for NVA, 16 for the extended vocabulary. Rows are aligned on the requested
count (``target_k``), because a cut can return one cluster fewer when merge heights tie.

    python scripts/appendix_F_tables.py [--grid grid.csv]
"""
import argparse
import os

import numpy as np
import pandas as pd

COLSETS = ["a_all", "b_nounk", "c_nopos", "d_none"]
HEADS = ["g0.2", "g0.5", "g0.8"]
METRICS = ["meanPurity", "NMI", "ARI"]
LABEL = {"a_all": "all dimensions", "b_nounk": "-<unk>",
         "c_nopos": "-unused", "d_none": "-all 20"}
REQUESTED = {
    "nva": [3, 6, 9, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "ext": [13, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 150, 160, 180, 200],
}


def load(path):
    d = pd.read_csv(path)
    d = d.drop_duplicates(subset=["head", "setting", "method", "colset", "seed", "target_k"])
    return d.assign(req=d.target_k)


def table_f2(r):
    print("=" * 72)
    print("Table F.2 -- removal of the unknown dimensions, gamma = 0.2")
    print("=" * 72)
    for st, meth, K in [("nva", "kmeans", 3), ("nva", "bgm", 3),
                        ("ext", "kmeans", 13), ("ext", "bgm", 13)]:
        sub = r[(r["head"] == "g0.2") & (r.setting == st) & (r.method == meth)
                & (r.k == K) & (r.seed == 42)].set_index("colset")
        print(f"  {st}, K={K}, consensus {meth}")
        for m, name in [("meanPurity", "purity"), ("NMI", "NMI"), ("ARI", "ARI")]:
            print(f"    {name:7s} " + " & ".join(f"{sub.loc[c, m]:.4f}" for c in COLSETS))


def claims_f2(r):
    print()
    print("Claim: largest |dNMI| from removing the unknown dimensions")
    rows = []
    for (h, st, meth), g in r[r.seed == 42].groupby(["head", "setting", "method"]):
        p = g.pivot_table(index="req", columns="colset", values="NMI")
        if "a_all" not in p:
            continue
        delta = p[["b_nounk", "c_nopos", "d_none"]].sub(p["a_all"], axis=0).abs()
        rows.append(dict(head=h, setting=st, method=meth, max_abs=delta.max().max()))
    f = pd.DataFrame(rows).sort_values("max_abs", ascending=False)
    print(f.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"  -> largest {f.max_abs.iloc[0]:.4f}"
          f" ({f.iloc[0].setting}, {f.iloc[0]["head"]}, {f.iloc[0].method});"
          f" all {len(f)-1} others below {f.max_abs.iloc[1]:.4f}")
    deltas = []
    for (_, _, _), g in r[r.seed == 42].groupby(["head", "setting", "method"]):
        p = g.pivot_table(index="req", columns="colset", values="NMI")
        if "a_all" in p:
            deltas.append(p[["b_nounk", "c_nopos", "d_none"]].sub(p["a_all"], axis=0).values.ravel())
    a = np.concatenate(deltas)
    a = a[~np.isnan(a)]
    print(f"  -> direction: {(a > 0).sum()} upward / {(a < 0).sum()} downward of {len(a)}")


def table_f3(r):
    print()
    print("=" * 72)
    print("Table F.3 -- ARI against the archived reference partition (gamma=0.2, KMeans)")
    print("=" * 72)
    g = r[(r["head"] == "g0.2") & (r.method == "kmeans")]
    for st in ["nva", "ext"]:
        rem = g[(g.setting == st) & (g.seed == 42) & (g.colset != "a_all")].ARI_vs_published.dropna()
        sd = g[(g.setting == st) & (g.seed != 42) & (g.colset == "a_all")].ARI_vs_published.dropna()
        print(f"  {st}")
        print(f"    removing unknown dimensions  n={len(rem):2d}  min {rem.min():.3f}"
              f"  median {rem.median():.3f}  max {rem.max():.3f}")
        print(f"    changing only the seed       n={len(sd):2d}  min {sd.min():.3f}"
              f"  median {sd.median():.3f}  max {sd.max():.3f}")
    print()
    print("  Four-seed spread of the aggregate measures (all dimensions):")
    for st, K in [("nva", 3), ("ext", 13)]:
        s = g[(g.setting == st) & (g.colset == "a_all") & (g.k == K)]
        print(f"    {st} K={K}: purity {s.meanPurity.min():.4f}-{s.meanPurity.max():.4f}"
              f"   NMI {s.NMI.min():.4f}-{s.NMI.max():.4f}"
              f"   (seed 42 = {s[s.seed == 42].meanPurity.iloc[0]:.4f} /"
              f" {s[s.seed == 42].NMI.iloc[0]:.4f})")


def table_f4(r):
    print()
    print("=" * 72)
    print("Table F.4 -- mean purity, NMI and ARI by head and feature condition")
    print(f"  basis: every evaluated cluster count -- NVA {REQUESTED['nva']}")
    print(f"                                          ext {REQUESTED['ext']}")
    print("=" * 72)
    r42 = r[r.seed == 42]
    for meth in ["kmeans", "bgm"]:
        print(f"  consensus {meth}")
        for st in ["nva", "ext"]:
            for cs in COLSETS:
                v = [r42[(r42["head"] == h) & (r42.setting == st) & (r42.method == meth)
                         & (r42.colset == cs)][m].mean() for m in METRICS for h in HEADS]
                flag = "" if all(v[i] > v[i + 1] > v[i + 2] for i in (0, 3, 6)) else "   <-- NOT MONOTONE"
                print(f"    {st:4s} {LABEL[cs]:15s} " + " & ".join(f"{x:.4f}" for x in v) + flag)


def claims_f4(r):
    print()
    print("Claim: gamma = 0.2 > 0.5 > 0.8")
    r42 = r[r.seed == 42]
    total = bad = 0
    for metric in ["meanPurity", "NMI", "ARI"]:
        for st in ["nva", "ext"]:
            for meth in ["kmeans", "bgm"]:
                for cs in COLSETS:
                    v = [r42[(r42["head"] == h) & (r42.setting == st) & (r42.method == meth)
                             & (r42.colset == cs)][metric].mean()
                         for h in HEADS]
                    total += 1
                    if not v[0] > v[1] > v[2]:
                        bad += 1
                        print(f"  VIOLATION {metric} {st} {meth} {cs}")
    print(f"  -> {total - bad}/{total} mean-level comparisons hold")
    for metric in ["NMI", "ARI", "meanPurity"]:
        for st in ["nva", "ext"]:
            ok = n = 0
            rates = []
            for meth in ["kmeans", "bgm"]:
                for cs in COLSETS:
                    p = r42[(r42.setting == st) & (r42.method == meth)
                            & (r42.colset == cs)].pivot_table(
                                index="req", columns="head", values=metric).dropna()
                    good = int(((p["g0.2"] > p["g0.5"]) & (p["g0.5"] > p["g0.8"])).sum())
                    ok += good
                    n += len(p)
                    rates.append(good / len(p))
            print(f"  per-cluster-count {metric:10s} {st:4s}: {ok}/{n} = {ok/n*100:.1f}%"
                  f"   per-cell range {min(rates)*100:.0f}-{max(rates)*100:.0f}%")


def claims_f4_detail(r):
    """The F.4 statements that go beyond the ordering itself."""
    print()
    print("Detail: spread across feature conditions vs. across seeds")
    r42 = r[r.seed == 42]
    for metric in METRICS:
        spreads = []
        for st in ["nva", "ext"]:
            for meth in ["kmeans", "bgm"]:
                for h in HEADS:
                    v = [r42[(r42["head"] == h) & (r42.setting == st) & (r42.method == meth)
                             & (r42.colset == cs)][metric].mean() for cs in COLSETS]
                    spreads.append(max(v) - min(v))
        # seeds were run only on the baseline cell: kmeans, all dimensions, gamma = 0.2
        seed_spreads = []
        for st in ["nva", "ext"]:
            m = r[(r.method == "kmeans") & (r.colset == "a_all") & (r["head"] == "g0.2")
                  & (r.setting == st)].groupby("seed")[metric].mean()
            seed_spreads.append(m.max() - m.min())
        drops = []
        for st in ["nva", "ext"]:
            for meth in ["kmeans", "bgm"]:
                for cs in COLSETS:
                    sub = r42[(r42.setting == st) & (r42.method == meth) & (r42.colset == cs)]
                    drops.append(sub[sub["head"] == "g0.2"][metric].mean()
                                 - sub[sub["head"] == "g0.8"][metric].mean())
        print(f"  {metric:10s} feature spread <= {max(spreads):.4f}"
              f"   seed spread <= {max(seed_spreads):.4f}"
              f"   gamma drop {min(drops):.4f}-{max(drops):.4f}")

    print()
    print("Detail: where mean purity inverts")
    rows = []
    for st in ["nva", "ext"]:
        for meth in ["kmeans", "bgm"]:
            for cs in COLSETS:
                p = r42[(r42.setting == st) & (r42.method == meth) & (r42.colset == cs)].pivot_table(
                    index="req", columns="head", values="meanPurity").dropna()
                for k, row in p.iterrows():
                    if row["g0.2"] > row["g0.5"] > row["g0.8"]:
                        continue
                    if row["g0.2"] <= row["g0.5"]:
                        pair, margin = "0.2<=0.5", row["g0.5"] - row["g0.2"]
                    else:
                        pair, margin = "0.5<=0.8", row["g0.8"] - row["g0.5"]
                    rows.append((st, meth, cs, int(k), pair, margin))
    inv = pd.DataFrame(rows, columns=["setting", "method", "colset", "k", "pair", "margin"])
    print(f"  {len(inv)} inversions, settings {sorted(inv.setting.unique())}, "
          f"pairs {sorted(inv.pair.unique())}")
    print(f"  cluster counts {sorted(inv.k.unique())}")
    print(f"  margin median {inv.margin.median():.4f}  max {inv.margin.max():.4f}")
    print("  per condition: " + ", ".join(
        f"{meth}/{LABEL[cs]}={int(((inv.method == meth) & (inv.colset == cs)).sum())}"
        for meth in ["kmeans", "bgm"] for cs in COLSETS))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default=None, help="default: $SRWC_DEPOSIT/robustness/robustness_grid.csv")
    a = ap.parse_args()
    r = load(a.grid or os.path.join(os.environ["SRWC_DEPOSIT"], "robustness", "robustness_grid.csv"))
    print(f"loaded {len(r)} rows "
          f"({r[r.seed == 42].groupby(['head', 'setting', 'method', 'colset']).ngroups} conditions)\n")
    table_f2(r)
    claims_f2(r)
    table_f3(r)
    table_f4(r)
    claims_f4(r)
    claims_f4_detail(r)


if __name__ == "__main__":
    main()
