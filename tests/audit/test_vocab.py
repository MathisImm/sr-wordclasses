import json

from conftest import needs_deposit
from srwc import paths, vocab


@needs_deposit
def test_vocab_equals_deposited_vocab():
    with open(paths.deposit("model", "spacy_vocab.json"), encoding="utf-8") as f:
        deposited = json.load(f)
    built = vocab.build_vocab(vocab.load_frequencies())
    assert list(built.items()) == list(deposited.items())
