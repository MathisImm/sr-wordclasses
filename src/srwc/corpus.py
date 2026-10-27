"""WikiText-103 as the model saw it.

Every non-blank line of ``wikitext-103-raw-v1`` is one training sample. Each line
passes through two stages, exactly as in the original run:

1. Preparation (``_prepare_batch``): spaCy ``en_core_web_sm`` (tokeniser, tagger, attribute ruler)
   tokenises and lowercases the line. Tokens outside the split's 19,997 most frequent
   words become ``<unk_POS>``, and the tokens are joined by single spaces.
2. ``encode``: the ``en_core_web_sm`` tokeniser runs again on that string, with
   ``<unk...>`` vocabulary entries protected from splitting. The tokens are lowercased
   and mapped to IDs (out-of-vocabulary words map to ``<unk>``). The sequence is cut
   to its first 80 tokens and padded with ``<pad>``.

Because stage 2 lowercases before the lookup, every ``<unk_POS>`` string becomes
``<unk>``. In effect a word is kept if it is among the 19,980 vocabulary words and is
``<unk>`` otherwise, which is why IDs 3-19 never occur (Supplementary Appendix F.1).
The two stages are nonetheless kept as they were: re-tokenising the joined string
decides the final token sequence.

The "split's 19,997 most frequent words" are counted on the split itself. The
validation split has fewer distinct words than that, so stage 1 replaces nothing there.
"""
from collections import Counter

import spacy
from datasets import load_dataset

from .vocab import PAD

SEQ_LENGTH = 80
BATCH = 1000  # lines per spaCy call, as in the original preprocessing


def load_lines(split: str, max_rows: int | None = None) -> list[str]:
    """Non-blank lines of a WikiText-103 split, in corpus order."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    if max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))
    return [t for t in ds["text"] if t.strip()]


def count_tokens(lines: list[str]) -> Counter:
    """Lowercased token counts with the en_core_web_sm tokeniser (the frequency cache)."""
    tokenizer = spacy.load("en_core_web_sm").tokenizer
    counts = Counter()
    for doc in tokenizer.pipe(lines, batch_size=BATCH):
        counts.update(t.text.lower() for t in doc)
    return counts


def top_words(freqs: dict, vocab_size: int = 20000) -> set[str]:
    ranked = sorted(freqs.items(), key=lambda kv: kv[1], reverse=True)
    return {w for w, _ in ranked[:vocab_size - 3]}


def _prepare_batch(nlp, lines, known):
    out = []
    for doc in nlp.pipe(lines):
        toks = []
        for t in doc:
            w = t.text.lower()
            toks.append(w if w in known else (f"<unk_{t.pos_}>" if t.pos_ else "<unk>"))
        out.append(" ".join(toks))
    return out


def _stage1_nlp():
    return spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer", "textcat"])


class Encoder:
    def __init__(self, vocab: dict, seq_length: int = SEQ_LENGTH):
        self.vocab = vocab
        self.seq_length = seq_length
        self.tokenizer = spacy.load("en_core_web_sm").tokenizer
        for tok in vocab:
            if tok.startswith("<unk") and tok.endswith(">"):
                self.tokenizer.add_special_case(tok, [{"ORTH": tok}])

    def __call__(self, text: str) -> tuple[list[int], list[int]]:
        ids = [self.vocab.get(t.text.lower(), 0) for t in self.tokenizer(text)][:self.seq_length]
        mask = [1] * len(ids) + [0] * (self.seq_length - len(ids))
        return ids + [PAD] * (self.seq_length - len(ids)), mask


_W = {}


def _init_worker(known, vocab):
    _W.update(nlp=_stage1_nlp(), known=known, encoder=Encoder(vocab))


def _encode_batch(lines):
    return [_W["encoder"](t) for t in _prepare_batch(_W["nlp"], lines, _W["known"])]


def encode_split(split: str, vocab: dict, freqs: dict | None = None,
                 max_rows: int | None = None, workers: int = 1) -> tuple[list[list[int]], list[list[int]]]:
    """Token IDs and attention masks for every non-blank line of a split.

    `freqs` are the split's own token counts. They default to a fresh count of the
    split, which reproduces the shipped training-split cache. With `workers` > 1 the
    1,000-line batches are spread over processes; each batch is tagged exactly as in
    the sequential run, so the result is identical.
    """
    lines = load_lines(split, max_rows)
    if freqs is None:
        freqs = count_tokens(load_lines(split))
    known = top_words(freqs, len(vocab))
    batches = [lines[i:i + BATCH] for i in range(0, len(lines), BATCH)]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers, initializer=_init_worker, initargs=(known, vocab)) as pool:
            encoded = [e for batch in pool.imap(_encode_batch, batches, chunksize=4) for e in batch]
    else:
        _init_worker(known, vocab)
        encoded = [e for batch in batches for e in _encode_batch(batch)]
    return [e[0] for e in encoded], [e[1] for e in encoded]
