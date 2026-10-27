#!/usr/bin/env python3
"""Re-training stability (Supplementary Appendix F.3, Table S2).

Two further models were trained identically to the reported one except for the seed
(2 and 3; checkpoints in the deposit's seed_replicates/).

cluster    Extracts a checkpoint's SR read-out for the evaluated words (as
           scripts/02_extract_sr.py) and runs the consensus clustering (seed 42) at
           the robustness grid's cluster counts. Writes labels_<tag>.npz and
           metrics_<tag>.csv in the deposit's format. This step regenerates, so on
           another machine it gives equivalent, not identical, partitions
           (srwc/cluster.py).

summarise  Adjusted Rand index between runs at every cluster count:
           published_check (the reported model re-clustered) against the archived
           partition, which isolates the consensus run, and each replicate against
           both and against each other. Writes pairwise_agreement.csv. From the
           deposit's labels it reproduces the deposit's table exactly.

    python scripts/appendix_F3.py cluster --checkpoint $SRWC_DEPOSIT/seed_replicates/model_checkpoint_seed2.pt --tag seed2 --out F3/
    python scripts/appendix_F3.py summarise --labels $SRWC_DEPOSIT/seed_replicates --out F3/
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from srwc import cluster, lexicon, select, vocab
from srwc.model import load_model

KS = {"nva": [3, 6, 9, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100],
      "ext": [13, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 150, 160, 180, 200]}
HEADS = ["g0.2", "g0.5", "g0.8"]
SPEC = {"nva": select.NVA, "ext": select.EXTENDED}
PUB = {"nva": "nva", "ext": "extended"}
TAGS = ["published_check", "seed2", "seed3"]


def archived(head, st, token_ids):
    path = os.path.join(os.environ["SRWC_DEPOSIT"], "clusterings_showcased", f"{PUB[st]}_g{head[1:]}_kmeans.csv")
    df = pd.read_csv(path, keep_default_na=False)
    return {int(k): p.set_index("token_id").loc[token_ids, "cluster"].to_numpy()
            for k, p in df.groupby("k") if int(k) in KS[st]}


def purity(lab, pos):
    return float(np.mean([np.unique(pos[lab == c], return_counts=True)[1].max() / (lab == c).sum()
                          for c in np.unique(lab)]))


def cmd_cluster(a):
    freqs = vocab.load_frequencies()
    table = select.word_table(vocab.build_vocab(freqs), freqs, lexicon.load())
    model = load_model(a.checkpoint)
    labels, rows = {}, []
    for st in a.settings.split(","):
        words = select.select(table, **SPEC[st])
        labels[f"tokens|{st}"] = words.token_id.to_numpy()
        pos = words.pos.astype(str).to_numpy()
        for head in a.heads.split(","):
            with torch.no_grad():
                X = model(torch.tensor(words.token_id.values), float(head[1:])).numpy().astype(np.float32)
            Z, _ = cluster.cluster(X, "kmeans")
            ref = archived(head, st, words.token_id.to_numpy())
            for target in KS[st]:
                lab = cluster.cut(Z, target)
                k = len(np.unique(lab))
                labels[f"{head}|{st}|{k}"] = lab
                rows.append(dict(tag=a.tag, head=head, setting=st, k=k, meanPurity=purity(lab, pos),
                                 NMI=normalized_mutual_info_score(pos, lab), ARI=adjusted_rand_score(pos, lab),
                                 ARI_vs_published=adjusted_rand_score(ref[k], lab) if k in ref else np.nan))
            print(f"{a.tag} {head} {st} done", flush=True)
    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, f"labels_{a.tag}.npz"), **labels)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"metrics_{a.tag}.csv"), index=False)


def cmd_summarise(a):
    L = {t: dict(np.load(os.path.join(a.labels, f"labels_{t}.npz")))
         for t in TAGS if os.path.exists(os.path.join(a.labels, f"labels_{t}.npz"))}
    rows = []
    for st in ("nva", "ext"):
        toks = [L[t][f"tokens|{st}"] for t in L if f"tokens|{st}" in L[t]]
        if not toks:
            continue
        assert all(np.array_equal(t, toks[0]) for t in toks), f"token order differs across runs ({st})"
        for head in HEADS:
            arch = archived(head, st, toks[0])
            for k in KS[st]:
                have = {t: L[t][f"{head}|{st}|{k}"] for t in L if f"{head}|{st}|{k}" in L[t]}
                pairs = []
                if "published_check" in have and k in arch:
                    pairs.append(("published_check vs archived", arch[k], have["published_check"]))
                for s in ("seed2", "seed3"):
                    if s in have and "published_check" in have:
                        pairs.append((f"{s} vs published_check", have["published_check"], have[s]))
                    if s in have and k in arch:
                        pairs.append((f"{s} vs archived", arch[k], have[s]))
                if "seed2" in have and "seed3" in have:
                    pairs.append(("seed2 vs seed3", have["seed2"], have["seed3"]))
                rows += [dict(setting=st, head=head, k=k, contrast=n, ARI=adjusted_rand_score(x, y))
                         for n, x, y in pairs]
    d = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    d.to_csv(os.path.join(a.out, "pairwise_agreement.csv"), index=False)
    for st in ("nva", "ext"):
        s = d[(d.setting == st) & (d["head"] == "g0.2")]
        print(f"\n{st}, gamma = 0.2, over {s.k.nunique()} cluster counts")
        print(s.groupby("contrast").ARI.agg(["count", "min", "median", "max"]).to_string(float_format="{:.3f}".format))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cluster")
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--tag", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--heads", default=",".join(HEADS))
    c.add_argument("--settings", default="nva,ext")
    s = sub.add_parser("summarise")
    s.add_argument("--labels", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    (cmd_cluster if a.cmd == "cluster" else cmd_summarise)(a)


if __name__ == "__main__":
    main()
