#!/usr/bin/env python3
"""Count-based controls of Supplementary Appendix G.

    python scripts/appendix_G.py counts  --out G/ [--ids cache/train_ids.npy] [--tokens single-pass]
    python scripts/appendix_G.py vectors --out G/          # dense analytic SR: needs ~10 GB RAM
    python scripts/appendix_G.py grid    --out G/ [--variants ...] [--settings nva,ext]
    python scripts/appendix_G.py seeds   --out G/                      # Table G.1: four consensus seeds
    python scripts/appendix_G.py annotate --out G/   # add the metadata to cells written before it existed
    python scripts/appendix_G.py verify  --out G/ --reference DIR   # compare with earlier outputs

`counts` uses the model's training token stream by default (`--tokens training`): the
encoded training split from --ids, or, if that file does not exist yet, a fresh
encoding (several CPU hours over --workers processes) saved there. `--tokens
single-pass` reproduces the originally reported controls, whose tokenisation differs
slightly from the model's (see `srwc/baselines.py`).

The evaluated words are those of $SRWC_DEPOSIT/sr_vectors/tokens_g0.2_{nva,ext}.csv.
`grid` writes one cell_<variant>_<setting>_kmeans.json per feature, in the deposit's
baselines/ format: a consensus clustering scored at every K, plus agreement with the
reported gamma = 0.2 SR partition from $SRWC_DEPOSIT/clusterings_fullscan.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

from srwc import baselines, cluster, vocab
from srwc.metrics import kscan

KSCAN = {"nva": range(2, 101), "ext": range(2, 201)}
SHOWCASE = {"nva": [3, 30], "ext": [13, 120]}
VARIANTS = ["bigram", "bigram_log", "ppmi", "sr_g0.2", "sr_g0.2_log",
            "sr_g0.5", "sr_g0.5_log", "sr_g0.8", "sr_g0.8_log"]
PUB = {"nva": "nva", "ext": "extended"}


def tokens(setting):
    return pd.read_csv(os.path.join(os.environ["SRWC_DEPOSIT"], "sr_vectors", f"tokens_g0.2_{setting}.csv"),
                       keep_default_na=False)


def training_ids(a, v):
    if a.ids and os.path.exists(a.ids):
        return np.load(a.ids, mmap_mode="r")
    from srwc import corpus
    ids = np.asarray(corpus.encode_split("train", v, freqs=vocab.load_frequencies(), max_rows=a.max_rows,
                                         workers=a.workers)[0], dtype=np.int64)
    if a.ids and a.max_rows is None:
        os.makedirs(os.path.dirname(a.ids) or ".", exist_ok=True)
        np.save(a.ids, ids)
    return ids


def cmd_counts(a):
    v = vocab.build_vocab(vocab.load_frequencies())
    sel = np.concatenate([tokens(st).token_id.to_numpy() for st in ("nva", "ext")])
    if a.tokens == "training":
        c = baselines.count_ids(v, sel, training_ids(a, v))
    else:
        from datasets import load_dataset
        rows = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")["text"]
        if a.max_rows:
            rows = rows[:a.max_rows]
        c = baselines.count(v, sel, rows, workers=a.workers)
    with open(os.path.join(a.out, "counts_info.json"), "w") as f:
        json.dump({"tokens": a.tokens, "max_rows": a.max_rows}, f)
    sp.save_npz(os.path.join(a.out, "bigram_counts.npz"), c["C"])
    sp.save_npz(os.path.join(a.out, "window_counts_selected.npz"), c["W"])
    np.savez_compressed(os.path.join(a.out, "marginals.npz"), unigram=c["unigram"], w_rowsum=c["w_rowsum"],
                        w_colsum=c["w_colsum"], selected=c["selected"])


def cmd_vectors(a):
    marg = np.load(os.path.join(a.out, "marginals.npz"))
    counts = dict(C=sp.load_npz(os.path.join(a.out, "bigram_counts.npz")),
                  W=sp.load_npz(os.path.join(a.out, "window_counts_selected.npz")),
                  w_rowsum=marg["w_rowsum"], w_colsum=marg["w_colsum"])
    ids = {st: tokens(st).token_id.to_numpy() for st in ("nva", "ext")}
    for (variant, st), X in baselines.vectors(counts, ids).items():
        np.save(os.path.join(a.out, f"X_{variant}_{st}.npy"), X)


def metadata(a, X, truth):
    """PCA diagnostics, ensemble settings and token stream, as in the deposit's cell files."""
    from sklearn.decomposition import PCA
    with cluster._limits():
        ratio = PCA(n_components=min(X.shape)).fit(X).explained_variance_ratio_
    n = int(np.argmax(np.cumsum(ratio) >= cluster.PCA_VARIANCE) + 1)
    info = os.path.join(a.out, "counts_info.json")
    stream = json.load(open(info))["tokens"] if os.path.exists(info) else "unknown"
    return dict(token_stream=stream, feature_dim=int(X.shape[1]), pca_components=n,
                pca_variance_target=cluster.PCA_VARIANCE, pca_variance_actual=float(np.cumsum(ratio)[n - 1]),
                pca_explained_variance_ratio_top20=[float(r) for r in ratio[:20]],
                n_trials=cluster.N_TRIALS, base_resolutions=list(cluster.BASE_K),
                consensus_method="hierarchical", metric="euclidean",
                pos_counts={str(k): int(v) for k, v in zip(*np.unique(truth, return_counts=True))})


def cmd_annotate(a):
    for name in sorted(os.listdir(a.out)):
        if not (name.startswith("cell_") and name.endswith(".json")):
            continue
        path = os.path.join(a.out, name)
        cell = json.load(open(path))
        X = np.load(os.path.join(a.out, f"X_{cell['variant']}_{cell['setting']}.npy"))
        meta = metadata(a, X, np.asarray(cell["pos"]))
        assert meta["pca_components"] == cell["pca_components"], name
        head = {k: cell[k] for k in ("variant", "setting", "method", "seed", "n_tokens")}
        with open(path, "w") as f:
            json.dump({**head, **meta, **{k: cell[k] for k in ("token_ids", "pos", "kscan")}}, f)
        print("annotated", name)


def cmd_grid(a):
    for setting in a.settings.split(","):
        df = tokens(setting)
        ids, truth = df.token_id.to_numpy(), df.pos.to_numpy()
        pub = pd.read_csv(os.path.join(os.environ["SRWC_DEPOSIT"], "clusterings_fullscan",
                                       f"{PUB[setting]}_g0.2_kmeans.csv"), keep_default_na=False)
        pub = {int(k): g.set_index("token_id").loc[ids, "cluster"].to_numpy() for k, g in pub.groupby("k")}
        for variant in a.variants.split(","):
            X = np.load(os.path.join(a.out, f"X_{variant}_{setting}.npy"))
            Z, P = cluster.cluster(X, "kmeans")
            scan = kscan(Z, P, truth, ids, KSCAN[setting], pub, SHOWCASE[setting])
            out = dict(variant=variant, setting=setting, method="kmeans", seed=cluster.SEED,
                       n_tokens=len(df), **metadata(a, X, truth), token_ids=ids.tolist(),
                       pos=truth.tolist(), kscan=scan)
            with open(os.path.join(a.out, f"cell_{variant}_{setting}_kmeans.json"), "w") as f:
                json.dump(out, f)
            print(f"{variant} {setting}: ARI at showcase "
                  f"{[round(r['ARI'], 3) for r in scan if r['target_k'] in SHOWCASE[setting]]}", flush=True)


# Table G.1 and the text of Appendix G report the mean and range over these consensus
# seeds (42 as everywhere, 0, 7 and 123 as in the re-seeding check of Appendix F.3).
SEEDS = [42, 0, 7, 123]
SEED_CELLS = ([("sr", "0.2", st) for st in ("nva", "ext")] + [("sr", g, "ext") for g in ("0.5", "0.8")]
              + [(v, None, st) for v in ("bigram_log", "sr_g0.2_log", "ppmi") for st in ("nva", "ext")]
              + [(v, None, "ext") for v in ("sr_g0.5_log", "sr_g0.8_log")]
              + [(v, None, "nva") for v in ("sr_g0.2", "sr_g0.8")])


def cmd_seeds(a):
    """Showcase scores of the trained SR heads and the count features under each consensus seed."""
    from srwc.metrics import agreement
    path = os.path.join(a.out, "seeds.csv")
    rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
    done = {(r["features"], r["setting"], r["seed"]) for r in rows}
    for seed in SEEDS:
        for variant, gamma, setting in SEED_CELLS:
            name = f"trained_sr_g{gamma}" if variant == "sr" else variant
            if (name, setting, seed) in done:
                continue
            if variant == "sr":
                sr_dir = os.path.join(os.environ["SRWC_DEPOSIT"], "sr_vectors")
                X = np.load(os.path.join(sr_dir, f"sr_head_g{gamma}_{setting}.npy"))
                pos = pd.read_csv(os.path.join(sr_dir, f"tokens_g{gamma}_{setting}.csv"), keep_default_na=False).pos
            else:
                X, pos = np.load(os.path.join(a.out, f"X_{variant}_{setting}.npy")), tokens(setting).pos
            Z, _ = cluster.cluster(X, "kmeans", seed=seed)
            for k in SHOWCASE[setting]:
                labels = cluster.cut(Z, k)
                m = agreement(labels, pos.to_numpy())
                pur = pd.Series(pos.to_numpy()).groupby(labels).agg(lambda s: s.value_counts().iloc[0] / len(s))
                rows.append(dict(features=name, setting=setting, seed=seed, k=k, ARI=m["ARI"], NMI=m["NMI"],
                                 mean_purity=m["Mean_Purity"], min_purity=float(pur.min())))
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"{name} {setting} seed {seed}", flush=True)
    df = pd.DataFrame(rows)
    summary = df.groupby(["setting", "k", "features"]).agg(
        ARI_mean=("ARI", "mean"), ARI_min=("ARI", "min"), ARI_max=("ARI", "max"),
        purity_mean=("mean_purity", "mean"), purity_min=("mean_purity", "min"), purity_max=("mean_purity", "max"),
        least_pure_cluster_mean=("min_purity", "mean"))
    summary.to_csv(os.path.join(a.out, "seeds_summary.csv"))
    print(summary.round(3).to_string())


def cmd_verify(a):
    """Compare counts, vectors and cells in --out with those in --reference (exact where possible)."""
    bad = 0
    for name in ("bigram_counts.npz", "window_counts_selected.npz"):
        p, r = os.path.join(a.out, name), os.path.join(a.reference, name)
        if os.path.exists(p) and os.path.exists(r):
            ok = (sp.load_npz(p) != sp.load_npz(r)).nnz == 0
            bad += not ok
            print(f"{'ok  ' if ok else 'DIFF'} {name}")
    for name in sorted(os.listdir(a.out)):
        p, r = os.path.join(a.out, name), os.path.join(a.reference, name)
        if not os.path.exists(r):
            continue
        if name.startswith("X_") and name.endswith(".npy"):
            x, y = np.load(p), np.load(r)
            ok = x.shape == y.shape and np.array_equal(x, y)
            bad += not ok
            print(f"{'ok  ' if ok else 'DIFF'} {name}" + ("" if ok else f"  max|diff|={np.abs(x - y).max():.3g}"))
        elif name.startswith("cell_"):
            x, y = json.load(open(p)), json.load(open(r))
            ys = {e["target_k"]: e for e in y["kscan"]}
            worst = max(abs(e[m] - ys[e["target_k"]][m]) for e in x["kscan"]
                        for m in ("ARI", "NMI", "meanPurity", "ARI_vs_SR_g0.2") if m in e)
            ok = worst < 1e-12
            bad += not ok
            print(f"{'ok  ' if ok else 'DIFF'} {name}  max metric diff over K = {worst:.3g}")
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["counts", "vectors", "grid", "seeds", "annotate", "verify"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", help="verify: folder with earlier outputs")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--settings", default="nva,ext")
    ap.add_argument("--tokens", choices=["training", "single-pass"], default="training")
    ap.add_argument("--ids", help="counts: encoded training split (.npy), written if missing")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-rows", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    globals()[f"cmd_{a.stage}"](a)


if __name__ == "__main__":
    main()
