"""The evaluated vocabularies (Methods, Table 2).

Every vocabulary word with a training-split count gets a reference tag (`lexicon.label`)
and a frequency quintile, computed over the whole vocabulary. Within each tag the 200
most frequent words are kept, ties broken by token ID, and the union is re-sorted by
frequency. The pandas operations are those of the original code, so the order is too,
including among words with equal counts:

* NVA: NOUN, VERB and ADJ, 600 words.
* Extended: every tag except PUNCT, X, INTJ, SPACE and SYM, 1,377 words over 13 tags.
"""
import pandas as pd

from . import lexicon as lex

NVA = dict(include=["NOUN", "VERB", "ADJ"])
EXTENDED = dict(exclude=["PUNCT", "X", "INTJ", "SPACE", "SYM"])
PER_TAG = 200
MIN_COUNT = 10


def word_table(vocab: dict, freqs: dict, lexicon: dict) -> pd.DataFrame:
    counts = {vocab[w]: c for w, c in freqs.items() if w in vocab}
    ids = sorted(tid for tid, c in counts.items() if c >= MIN_COUNT)
    inverse = {i: w for w, i in vocab.items()}
    df = pd.DataFrame({"token_id": ids, "token": [inverse[i] for i in ids],
                       "freq": [counts[i] for i in ids]})
    df["pos"] = pd.Categorical([lex.label(t, lexicon) for t in df.token])
    df["frequency_bin"] = pd.qcut(df["freq"], q=5, duplicates="drop")
    return df


def select(df: pd.DataFrame, include=None, exclude=None, per_tag: int = PER_TAG) -> pd.DataFrame:
    mask = df["freq"] >= MIN_COUNT
    if include:
        mask &= df["pos"].isin(include)
    if exclude:
        mask &= ~df["pos"].isin(exclude)
    kept = df[mask].copy()
    picked = []
    for tag in sorted(kept["pos"].unique()):
        picked.extend(kept[kept["pos"] == tag]["freq"].nlargest(per_tag).index)
    order = kept.loc[picked].sort_values("freq", ascending=False).index
    return kept.loc[order].reset_index(drop=True)
