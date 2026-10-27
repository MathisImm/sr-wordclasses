"""Helpers that drive the original research code ($SRWC_ORIGINAL) for the equivalence tests."""
import json
import os
import sys


def import_original():
    root = os.environ["SRWC_ORIGINAL"]
    for p in (root, os.path.join(root, "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)


def prepare_pos_files(tmp_path, monkeypatch, split, max_rows):
    """Run the original stage-1 preprocessing; return (path of its jsonl, the split's counts or None)."""
    import_original()
    import prepare_pos_dataset as ppd
    from vocab_analysis.compute_frequencies import compute_word_frequencies
    from vocab_analysis.vocab_utils import save_frequencies

    freqs = None
    with monkeypatch.context() as m:
        if split == "train":
            real = ppd.get_cache_path("wikitext-103", "train")
            m.setattr(ppd, "get_cache_path", lambda d, s: real)
        else:   # the original computed this cache on the cluster; recompute it
            freqs = compute_word_frequencies("wikitext-103", split, lowercase=True)
            cache = tmp_path / f"wikitext-103_{split}_freqs.json"
            save_frequencies(freqs, str(cache).replace(".json", "_lowercase.json"))
            m.setattr(ppd, "get_cache_path", lambda d, s: str(cache))
        # The original writes under the user's home; send it to tmp_path instead (the
        # training test sets HOME=tmp_path, so the original trainer finds it there).
        real_expand = os.path.expanduser
        m.setattr(ppd.os.path, "expanduser",
                  lambda p: str(tmp_path) + p[1:] if p.startswith("~/") else real_expand(p))
        ppd.prepare_pos_dataset("wikitext-103", split=split, max_samples=max_rows,
                                lowercase=True, vocab_size=20000)
    out, = tmp_path.rglob(f"{split}_lowercase_v20000.jsonl")
    return out, freqs


def read_jsonl(path):
    return [json.loads(line)["text"] for line in open(path, encoding="utf-8")]
