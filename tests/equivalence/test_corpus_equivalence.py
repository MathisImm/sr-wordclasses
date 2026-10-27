"""corpus.py against the original two-stage preprocessing, on real WikiText-103 lines.

Runs the original prepare_pos_dataset (stage 1) and SpacyTokenizer (stage 2) on the
first rows of the training split and on the whole validation split, and requires
identical token IDs and masks. Output files of the original code are redirected to
a temporary directory.
"""
import os

import pytest

from conftest import needs_original
from original import import_original, prepare_pos_files, read_jsonl
from srwc import corpus, vocab

TRAIN_ROWS = int(os.environ.get("SRWC_EQUIV_ROWS", "20000"))


@needs_original
@pytest.mark.parametrize("split,max_rows", [("train", TRAIN_ROWS), ("validation", None)])
def test_encoding_matches_original(tmp_path, monkeypatch, split, max_rows):
    jsonl, freqs_o = prepare_pos_files(tmp_path, monkeypatch, split, max_rows)
    import_original()
    from training.spacy_tokenizer import SpacyTokenizer

    v = vocab.build_vocab(vocab.load_frequencies())
    enc = SpacyTokenizer(v, max_length=80, lowercase=True)(
        read_jsonl(jsonl), truncation=True, max_length=80, padding="max_length", return_tensors=None)

    freqs = vocab.load_frequencies() if split == "train" else None
    ids, mask = corpus.encode_split(split, v, freqs=freqs, max_rows=max_rows)
    if freqs_o is not None:
        assert list(corpus.count_tokens(corpus.load_lines(split)).items()) == list(freqs_o.items())
    assert ids == enc["input_ids"]
    assert mask == enc["attention_mask"]
