"""Count-based controls of Supplementary Appendix G.

Counts are taken over the WikiText-103 training split, one row per line, lowercased,
out-of-vocabulary words mapped to ``<unk>`` and each row cut to its first 80 tokens.
Pairs never cross a row boundary. Two token streams are supported:

* ``count_ids``: the token IDs the model was trained on (``corpus.encode_split``),
  including the ``<unk>`` that opens every line and the abbreviation handling of
  the two-pass preprocessing.
* ``count``: a single pass of the spaCy ``en_core_web_sm`` tokeniser over every row
  (after first keeping only the first 160 whitespace chunks). The originally
  reported controls used this stream. It differs from the model's in the leading
  ``<unk>`` of every line and in a few abbreviations ("U.S.", "St."); on the first
  20,000 rows that is the first token of every line plus 1.8% of lines beyond it.

Features, one dense row over the 20,000-word vocabulary per evaluated word:

* ``bigram``: row-normalised P(w'|w); ``bigram_log``: log(P + 1e-9).
* ``ppmi``: positive PMI over a symmetric +/-5 window.
* ``sr_g{gamma}``: the closed-form arrival SR, sum_k gamma^k P^(k+1), truncated
  once gamma^K < 1e-10.
* ``sr_g0.2_log``: log of the float64 series + 1e-9. ``sr_g0.5_log`` and
  ``sr_g0.8_log`` were added later as log(x + 1e-9) of the stored float32 vectors;
  both paths are kept as run.
"""
import multiprocessing as mp
import re

import numpy as np
import scipy.sparse as sp

MAX_LENGTH = 80
WINDOW = 5
CHUNK_PREFIX = 2 * MAX_LENGTH
LOG_EPS = 1e-9
NEUMANN_TOL = 1e-10

# ---------------------------------------------------------------- counting

_G = {}


def _char_prefix(text, n_chunks):
    c = 0
    for m in re.finditer(r"\S+", text):
        c += 1
        if c == n_chunks:
            return text[:m.end()]
    return text


def _init_worker(vocab, max_length):
    import spacy
    _G.update(tok=spacy.load("en_core_web_sm").tokenizer, vocab=vocab,
              unk=vocab.get("<unk>", 0), max_length=max_length)


def _tokenise_shard(texts):
    tok, vocab, unk, max_length = _G["tok"], _G["vocab"], _G["unk"], _G["max_length"]
    flat, lens = [], []
    for text in texts:
        if not text or not text.strip():
            lens.append(0)
            continue
        ids = [vocab.get(t.text.lower(), unk) for t in tok(_char_prefix(text, CHUNK_PREFIX))]
        ids = ids[:max_length]
        flat.extend(ids)
        lens.append(len(ids))
    return np.asarray(flat, dtype=np.uint16), np.asarray(lens, dtype=np.uint8)


def _pairs_within_rows(ids, lens, offset):
    n = ids.shape[0]
    if n <= offset:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    starts = np.concatenate(([0], np.cumsum(lens)[:-1]))
    row_of = np.repeat(np.arange(lens.shape[0]), lens)
    within = np.arange(n, dtype=np.int64) - starts[row_of]
    room = np.repeat(lens.astype(np.int64), lens) - within
    ok = room > offset
    ok[n - offset:] = False
    idx = np.nonzero(ok)[0]
    return ids[idx].astype(np.int64), ids[idx + offset].astype(np.int64)


class _PairSink:
    """Accumulates pair keys i*V+j, compacting periodically to bound memory."""

    def __init__(self, V, row_filter=None, flush_at=40_000_000):
        self.V, self.row_filter, self.flush_at = V, row_filter, flush_at
        self.buf, self.buf_n = [], 0
        self.keys = np.empty(0, dtype=np.int64)
        self.cnt = np.empty(0, dtype=np.int64)

    def add(self, src, dst):
        if self.row_filter is not None:
            keep = self.row_filter[src]
            src, dst = src[keep], dst[keep]
        if src.size:
            self.buf.append(src * self.V + dst)
            self.buf_n += src.size
        if self.buf_n >= self.flush_at:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        new = np.concatenate(self.buf)
        self.buf, self.buf_n = [], 0
        keys = np.concatenate([self.keys, new])
        cnt = np.concatenate([self.cnt, np.ones(new.size, dtype=np.int64)])
        self.keys, inv = np.unique(keys, return_inverse=True)
        self.cnt = np.bincount(inv, weights=cnt.astype(np.float64), minlength=self.keys.size).astype(np.int64)

    def finish(self):
        self.flush()
        rows, cols = np.divmod(self.keys, self.V)
        return sp.csr_matrix((self.cnt, (rows, cols)), shape=(self.V, self.V))


def _accumulate(V, selected_ids, shards):
    sel = np.zeros(V, dtype=bool)
    sel[sorted(set(int(i) for i in selected_ids))] = True
    bigram, window = _PairSink(V), _PairSink(V, row_filter=sel)
    unigram, w_row, w_col = (np.zeros(V, dtype=np.int64) for _ in range(3))
    for ids, lens in shards:
        ids = ids.astype(np.int64, copy=False)
        unigram += np.bincount(ids, minlength=V)
        for off in range(1, WINDOW + 1):
            src, dst = _pairs_within_rows(ids, lens, off)
            if off == 1:
                bigram.add(src, dst)
            for a, b in ((src, dst), (dst, src)):
                w_row += np.bincount(a, minlength=V)
                w_col += np.bincount(b, minlength=V)
            window.add(src, dst)
            window.add(dst, src)
    return dict(C=bigram.finish(), W=window.finish(), unigram=unigram,
                w_rowsum=w_row, w_colsum=w_col, selected=np.nonzero(sel)[0])


def count(vocab: dict, selected_ids, rows: list[str], workers: int = 16, shard_rows: int = 20000):
    """Bigram counts C, window counts W (selected centre rows only) and the marginals,
    from a single tokeniser pass over the raw rows."""
    shards = (rows[lo:lo + shard_rows] for lo in range(0, len(rows), shard_rows))
    with mp.get_context("fork").Pool(workers, initializer=_init_worker, initargs=(vocab, MAX_LENGTH)) as pool:
        return _accumulate(len(vocab), selected_ids, pool.imap(_tokenise_shard, shards, chunksize=1))


def count_ids(vocab: dict, selected_ids, ids: np.ndarray, shard_rows: int = 20000):
    """As ``count``, from padded training rows (n_lines x 80 token IDs, <pad> at the end)."""
    from .vocab import PAD

    def shards():
        for lo in range(0, len(ids), shard_rows):
            block = np.asarray(ids[lo:lo + shard_rows])
            real = block != PAD
            yield block[real], real.sum(1).astype(np.int64)

    return _accumulate(len(vocab), selected_ids, shards())


# ---------------------------------------------------------------- features

def _dense_rows(M, ids):
    return np.asarray(M[ids].todense(), dtype=np.float64)


def vectors(counts: dict, ids: dict, gammas=(0.2, 0.5, 0.8)):
    """{(variant, subset): float32 matrix}; `ids` maps 'nva'/'ext' to token-ID arrays."""
    C, W = counts["C"].tocsr(), counts["W"].tocsr()
    V = C.shape[0]
    out = {}
    rowsum = np.asarray(C.sum(1)).ravel()
    for st, rows in ids.items():
        denom = rowsum[rows].copy()
        denom[denom == 0] = 1.0
        P = _dense_rows(C, rows) / denom[:, None]
        out["bigram", st] = P.astype(np.float32)
        out["bigram_log", st] = np.log(P + LOG_EPS).astype(np.float32)

    w_row, w_col = counts["w_rowsum"].astype(np.float64), counts["w_colsum"].astype(np.float64)
    total = float(w_col.sum())
    for st, rows in ids.items():
        joint = _dense_rows(W, rows)
        rm = w_row[rows].copy()
        rm[rm == 0] = 1.0
        cm = w_col.copy()
        cm[cm == 0] = 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log((joint * total) / (rm[:, None] * cm[None, :]))
        pmi[~np.isfinite(pmi)] = 0.0
        pmi[joint == 0] = 0.0
        out["ppmi", st] = np.maximum(pmi, 0.0).astype(np.float32)

    rs = rowsum.copy()
    rs[rs == 0] = 1.0
    Pdense = np.zeros((V, V), dtype=np.float32)
    for lo in range(0, V, 2000):
        hi = min(lo + 2000, V)
        Pdense[lo:hi] = np.asarray(C[lo:hi].todense(), dtype=np.float32) / rs[lo:hi, None].astype(np.float32)
    ext = ids["ext"]
    where = {int(t): i for i, t in enumerate(ext)}
    for gamma in gammas:
        K = int(np.ceil(np.log(NEUMANN_TOL) / np.log(gamma)))
        cur = Pdense[ext].copy()
        acc = cur.copy()
        for k in range(1, K + 1):
            cur = cur @ Pdense
            acc += np.float32(gamma ** k) * cur
        for st, rows in ids.items():
            X = acc[np.array([where[int(t)] for t in rows])].astype(np.float64)
            out[f"sr_g{gamma}", st] = X.astype(np.float32)
            if abs(gamma - 0.2) < 1e-9:
                out[f"sr_g{gamma}_log", st] = np.log(X + LOG_EPS).astype(np.float32)
            else:   # derived afterwards from the stored float32 vectors
                out[f"sr_g{gamma}_log", st] = np.log(out[f"sr_g{gamma}", st] + LOG_EPS)
    return out
