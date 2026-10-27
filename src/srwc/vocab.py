"""The 20,000-token vocabulary.

Built from the lowercased token counts of the WikiText-103 training split
(``resources/wikitext-103_train_freqs_lowercase.json``, spaCy ``en_core_web_sm``
tokeniser). IDs 0-2 are ``<unk>``, ``<pad>`` and ``<eos>``. IDs 3-19 are reserved
``<unk_POS>`` entries from an earlier preprocessing design: the trained model never
saw them (Supplementary Appendix F.1), but they are kept so that token IDs match
the published checkpoint. The remaining 19,980 IDs are the most frequent words.
"""
import json
from collections import Counter

from . import paths

SPECIALS = ["<unk>", "<pad>", "<eos>"]
RESERVED_POS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
                "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]
UNK, PAD, EOS = 0, 1, 2


def load_frequencies(path=paths.FREQS) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_vocab(freqs: dict, size: int = 20000) -> dict:
    vocab = {tok: i for i, tok in enumerate(SPECIALS)}
    for tag in RESERVED_POS:
        vocab[f"<unk_{tag}>"] = len(vocab)
    for word, _ in Counter(freqs).most_common(size - len(vocab)):
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab
