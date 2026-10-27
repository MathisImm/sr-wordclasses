#!/usr/bin/env python3
"""Train the successor-representation network on WikiText-103 (Table 1 settings).

    python scripts/01_train.py --out runs/sr --cache cache/

Preprocessing tags the whole training split with spaCy (several CPU hours); --cache
keeps the encoded splits for later runs. Training itself took 15 h 11 min on one NVIDIA A40
for the reported model. Checkpoints are written per epoch to --out/epoch_XXX/.
"""
import argparse
import os

import numpy as np

from srwc import corpus, train, vocab


def encoded(split, v, cache, freqs=None, max_rows=None, workers=1):
    path = os.path.join(cache, f"{split}_ids.npy") if cache and max_rows is None else None
    if path and os.path.exists(path):
        return np.load(path)
    ids = np.asarray(corpus.encode_split(split, v, freqs=freqs, max_rows=max_rows, workers=workers)[0], dtype=np.int64)
    if path:
        os.makedirs(cache, exist_ok=True)
        np.save(path, ids)
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None, help="folder for the encoded splits")
    ap.add_argument("--seed", type=int, default=None,
                    help="the reported model was trained unseeded; the replicates used seeds 2 and 3")
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=1, help="processes for the spaCy preprocessing")
    ap.add_argument("--max-rows", type=int, default=None, help="smoke test: use only the first rows of each split")
    ap.add_argument("--epochs", type=int, default=train.CONFIG["epochs"])
    ap.add_argument("--batch-size", type=int, default=train.CONFIG["batch_size"])
    ap.add_argument("--warmup", type=int, default=train.CONFIG["warmup"])
    a = ap.parse_args()
    freqs = vocab.load_frequencies()
    v = vocab.build_vocab(freqs)
    train_ids = encoded("train", v, a.cache, freqs, a.max_rows, a.workers)
    val_ids = encoded("validation", v, a.cache, max_rows=a.max_rows, workers=a.workers)
    print(f"{len(train_ids):,} training and {len(val_ids):,} validation samples")
    _, history = train.train(train_ids, val_ids, a.out, seed=a.seed, device=a.device,
                             epochs=a.epochs, batch_size=a.batch_size, warmup=a.warmup)
    print({k: [round(x, 4) for x in v] for k, v in history.items() if k.endswith("loss")})


if __name__ == "__main__":
    main()
