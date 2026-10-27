"""Listings against the supplement's own text ($SRWC_SUPPLEMENT: the supplement's .tex source).

Each cluster must list the same words, and the supplement's order must be the order
of increasing distance to the cluster mean. Words whose distances agree to rounding
precision are ties, whose printed order the original run left to floating-point noise.
"""
import importlib.util
import os
import re

import numpy as np
import pandas as pd
import pytest

from conftest import needs_deposit
from srwc import paths
from srwc.cluster import reduce_pca

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location("listings", os.path.join(ROOT, "scripts", "06_appendix_listings.py"))
listings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(listings)


@needs_deposit
@pytest.mark.skipif(not os.environ.get("SRWC_SUPPLEMENT"), reason="set SRWC_SUPPLEMENT to the supplement .tex")
def test_listings_agree_with_supplement():
    text = open(os.environ["SRWC_SUPPLEMENT"], encoding="utf-8").read()
    text = text[:text.index("\\section{Methods Details}")]
    printed = re.findall(r"^\\textbf\{Cluster (\d+)\}: (.*)$", text, re.M)
    assert len(printed) == sum(k for *_, k in listings.CASES)
    i, ties = 0, 0
    for name, stem, k in listings.CASES:
        tok = pd.read_csv(paths.deposit("sr_vectors", f"tokens_g0.2_{stem}.csv"), keep_default_na=False)
        P, _ = reduce_pca(np.load(paths.deposit("sr_vectors", f"sr_head_g0.2_{stem}.npy")))
        part = pd.read_csv(paths.deposit("clusterings_showcased", f"{name}_g0.2_kmeans.csv"), keep_default_na=False)
        labels = part[part.k == k].set_index("token_id").loc[tok.token_id, "cluster"].to_numpy()
        d = dict(zip(tok.token, listings.distances(labels, P.astype(np.float64))))
        ours = listings.listing(tok.token.tolist(), labels, P.astype(np.float64))
        for c in sorted(set(labels)):
            cid, words = printed[i]
            assert int(cid) == c
            words = [w.replace("\\_", "_") for w in words.strip().split(", ")]
            assert sorted(words) == sorted(tok.token[labels == c])
            gaps = [d[b] - d[a] for a, b in zip(words, words[1:])]
            assert all(g >= -1e-9 * max(d[a], 1e-12) for g, a in zip(gaps, words)), (name, k, c)
            ties += [w.replace("\\_", "_") for w in ours[c].split(": ", 1)[1].split(", ")] != words
            i += 1
    assert ties <= 20, ties   # only two-word clusters differ, by tie order
