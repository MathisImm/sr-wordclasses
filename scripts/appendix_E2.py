#!/usr/bin/env python3
"""Composition of the lambda-return target (Supplementary Appendix E.2).

1. Observed-token share. Over an unbounded horizon, the observed features carry
   (1 - gamma) / (1 - gamma * lambda) of the target mass, and the bootstrap carries
   the rest: 98% at gamma = 0.2 and 71% at gamma = 0.8 (lambda = 0.9).

2. Padding share. Lines shorter than 80 tokens are padded, and padding enters the
   target through the observed feature. Training masks a loss site by its *source*
   token only (pad and <eos> sources are excluded), so the last real token of a line
   is a loss site, and its target falls mostly on padding. The share is computed
   with `srwc.targets.lambda_return` itself on one padded line per length, weighted
   by how many lines have that length. Every target sums to 1 (closed-loop bootstrap
   at the window end). The bootstrap's own mass on <pad>, 0.05% at gamma = 0.2
   (Appendix F.1), is left out.

The line-length histogram comes from the spaCy tokeniser in a single pass (fast).
With --ids, it comes from token IDs encoded exactly as for training
(`scripts/01_train.py --cache`).

    python scripts/appendix_E2.py [--ids cache/train_ids.npy]
"""
import argparse

import numpy as np
import torch

from srwc.targets import lambda_return, source_mask
from srwc.vocab import PAD

T, LAMBDA, GAMMAS = 80, 0.9, (0.2, 0.5, 0.8)
WORD = 20                                  # first ordinary vocabulary ID


def histogram_from_ids(ids):
    lengths = (np.asarray(ids) != PAD).sum(1)
    return np.bincount(lengths, minlength=T + 1)


def histogram_from_corpus():
    import spacy
    from datasets import load_dataset
    rows = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")["text"]
    tokenizer = spacy.load("en_core_web_sm").tokenizer
    counts = np.zeros(T + 1, dtype=np.int64)
    for doc in tokenizer.pipe((r for r in rows if r.strip()), batch_size=1000):
        counts[min(len(doc), T)] += 1
    return counts


def padding_share(counts, gamma, lam=LAMBDA):
    ids = torch.full((T, T), PAD, dtype=torch.long)
    for L in range(1, T + 1):
        ids[L - 1, :L] = WORD               # any ordinary word (not <pad> or <eos>)
    logits = torch.zeros(T, T, WORD + 1)
    logits[..., PAD] = -1e9                 # bootstrap places no mass on <pad>
    G = lambda_return(ids, logits, gamma, lam)
    sites = source_mask(ids)[:, :-1]
    pad = (G[..., PAD] * sites).sum(1).numpy()
    total = (G.sum(-1) * sites).sum(1).numpy()
    w = np.asarray(counts[1:], dtype=float)
    return float((w * pad).sum() / (w * total).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", help="encoded training split (.npy) for the exact length histogram")
    a = ap.parse_args()
    for g in GAMMAS:
        print(f"gamma={g}: observed-token share of the target = {(1 - g) / (1 - g * LAMBDA):.1%}")
    counts = histogram_from_ids(np.load(a.ids)) if a.ids else histogram_from_corpus()
    print(f"\n{int(counts[1:].sum()):,} lines, {counts[T] / counts[1:].sum():.1%} at the {T}-token cap")
    for g in GAMMAS:
        print(f"gamma={g}: target mass on <pad> = {padding_share(counts, g):.2%}")


if __name__ == "__main__":
    main()
