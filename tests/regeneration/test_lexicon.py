"""The shipped POS lexicon, rebuilt from the corpus with spaCy en_core_web_lg.

Identical on the machine it was checked on. spaCy's tagger is a neural network run
on the CPU, so on other hardware a word whose two most frequent tags are nearly
tied could in principle come out differently.
"""
from srwc import lexicon


def test_lexicon_rebuilds_from_corpus():
    assert list(lexicon.build().items()) == list(lexicon.load().items())
