#!/usr/bin/env python3
"""Successor-representation read-out for the evaluated vocabularies.

For each head and each vocabulary (NVA, extended) writes the word table
tokens_g{gamma}_{nva|ext}.csv and the heads' pre-softmax outputs
sr_head_g{gamma}_{nva|ext}.npy, in the format of the deposit's sr_vectors/.

    python scripts/02_extract_sr.py --checkpoint $SRWC_DEPOSIT/model/model_checkpoint.pt --out sr_vectors/
"""
import argparse
import os

import numpy as np
import torch

from srwc import lexicon, select, vocab
from srwc.model import GAMMAS, load_model

VOCABS = {"nva": select.NVA, "ext": select.EXTENDED}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    freqs = vocab.load_frequencies()
    table = select.word_table(vocab.build_vocab(freqs), freqs, lexicon.load())
    model = load_model(a.checkpoint)
    os.makedirs(a.out, exist_ok=True)
    for name, spec in VOCABS.items():
        words = select.select(table, **spec)
        ids = torch.tensor(words.token_id.values)
        for g in GAMMAS:
            with torch.no_grad():
                logits = model(ids, g).numpy().astype(np.float32)
            words.to_csv(os.path.join(a.out, f"tokens_g{g}_{name}.csv"), index=False)
            np.save(os.path.join(a.out, f"sr_head_g{g}_{name}.npy"), logits)
        print(f"{name}: {len(words)} words")


if __name__ == "__main__":
    main()
