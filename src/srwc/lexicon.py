"""Reference part-of-speech labels (post-hoc evaluation only; never a model or clustering input).

`build` reproduces ``resources/pos_lexicon_wikitext-103_lowercase.json``: the
majority Universal POS tag that spaCy ``en_core_web_lg`` assigns to each lowercased
word over the first 2,000 rows of the WikiText-103 training split. The text is
tagged in its original case and lowercased afterwards, punctuation and whitespace
tokens are skipped, and ties go to the tag seen first. It covers 12,686 word types.

`label` tags a word type as the analysis did. Strings consisting only of
punctuation are PUNCT; otherwise the lexicon's tag is used. A word missing from the
lexicon falls back to an NLTK unigram tagger trained on the Penn Treebank (universal
tagset, mapped to UD names), and to X if that tagger does not know the word either.
Of the evaluated words, this fallback tags 18 of the 1,377 in the extended
vocabulary and none of the 600 NVA words.
"""
import json
import string
from collections import Counter
from functools import lru_cache

from . import paths

LEXICON_ROWS = 2000
_UD = {"CONJ": "CCONJ", ".": "PUNCT", "PRT": "PART"}


def build(max_rows: int = LEXICON_ROWS) -> dict:
    import spacy
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    texts = [t for t in ds.select(range(max_rows))["text"] if t.strip()]
    nlp = spacy.load("en_core_web_lg")
    nlp.max_length = 2_000_000
    counts: dict[str, Counter] = {}
    for doc in nlp.pipe(texts, disable=["ner", "lemmatizer"]):
        for tok in doc:
            if tok.is_punct or tok.is_space:
                continue
            counts.setdefault(tok.text.lower(), Counter())[tok.pos_] += 1
    return {w: c.most_common(1)[0][0] for w, c in counts.items()}


def load(path=paths.LEXICON) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _nltk_tagger():
    import nltk
    for pkg, where in (("treebank", "corpora/treebank"), ("universal_tagset", "taggers/universal_tagset")):
        try:
            nltk.data.find(where)
        except LookupError:
            nltk.download(pkg, quiet=True)
    from nltk.corpus import treebank
    return nltk.UnigramTagger(treebank.tagged_sents(tagset="universal"))


def label(word: str, lexicon: dict) -> str:
    w = word.lstrip("▁Ġ")
    if w.startswith("##"):
        w = w[2:]
    if not w or not w.strip():
        return "SPACE"
    if all(c in string.punctuation for c in w):
        return "PUNCT"
    if lexicon.get(w.lower()):
        return lexicon[w.lower()]
    tag = _nltk_tagger().tag([w.lower()])[0][1]
    return _UD.get(tag, tag) if tag is not None else "X"
