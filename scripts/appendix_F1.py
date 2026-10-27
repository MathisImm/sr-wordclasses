#!/usr/bin/env python3
"""The twenty reserved vocabulary dimensions (Supplementary Appendix F.1).

IDs 0-19 are <unk>, <pad>, <eos> and seventeen <unk_POS> slots that training never
emitted (see srwc/corpus.py). They remain dimensions of the predicted successor
distribution. For the gamma = 0.2 head on the NVA words this prints:
  * predicted softmax mass, variance share and variance rank of each reserved dimension;
  * how well part of speech is decoded from <unk> alone and from all twenty, against
    equally many ordinary word dimensions (5-fold logistic regression).

    python scripts/appendix_F1.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

SPECIAL = {0: "<unk>", 1: "<pad>", 2: "<eos>"}
SEED = 0


def main():
    sr = os.path.join(os.environ["SRWC_DEPOSIT"], "sr_vectors")
    X = np.load(os.path.join(sr, "sr_head_g0.2_nva.npy"))
    df = pd.read_csv(os.path.join(sr, "tokens_g0.2_nva.csv"), keep_default_na=False)

    e = np.exp(X - X.max(axis=1, keepdims=True))
    prob = (e / e.sum(axis=1, keepdims=True)).mean(axis=0)
    var = X.var(axis=0)
    rank = var.argsort().argsort()
    var_share = var / var.sum()
    print(f"{'dim':>5} {'name':14s} {'softmax mass':>13} {'var share':>11} {'var rank':>9}")
    for i in range(20):
        print(f"{i:5d} {SPECIAL.get(i, f'<unk_POS #{i}>'):14s} {prob[i]:13.6f} "
              f"{var_share[i] * 100:10.6f}% {20000 - rank[i]:9d}")
    unused = np.arange(3, 20)
    print(f"\n17 unused slots: mass {prob[unused].sum():.2e}, variance share {var_share[unused].sum() * 100:.6f}%")
    print(f"all 20 reserved: mass {prob[:20].sum():.6f}, variance share {var_share[:20].sum() * 100:.6f}%")

    y = df.pos.values
    def acc(cols):
        return cross_val_score(LogisticRegression(max_iter=3000), X[:, cols], y, cv=5).mean()
    rng = np.random.default_rng(SEED)
    ordinary = np.arange(20, X.shape[1])
    single = [acc([j]) for j in rng.choice(ordinary, 40, replace=False)]
    twenty = [acc(rng.choice(ordinary, 20, replace=False)) for _ in range(20)]
    print(f"\nmajority-class rate                 {pd.Series(y).value_counts(normalize=True).max():.3f}")
    print(f"POS from <unk> alone                {acc([0]):.3f}")
    print(f"POS from one random word dimension  {np.mean(single):.3f} (mean of 40)")
    print(f"POS from all 20 reserved dimensions {acc(list(range(20))):.3f}")
    print(f"POS from 20 random word dimensions  {np.mean(twenty):.3f} "
          f"(mean of 20 draws, range {min(twenty):.3f}-{max(twenty):.3f})")


if __name__ == "__main__":
    main()
