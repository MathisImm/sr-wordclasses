import pandas as pd
import pytest

from conftest import needs_deposit
from srwc import lexicon, paths, select, vocab


@pytest.fixture(scope="module")
def table():
    freqs = vocab.load_frequencies()
    return select.word_table(vocab.build_vocab(freqs), freqs, lexicon.load())


@needs_deposit
@pytest.mark.parametrize("name,spec", [("nva", select.NVA), ("ext", select.EXTENDED)])
@pytest.mark.parametrize("gamma", ["0.2", "0.5", "0.8"])
def test_selection_equals_deposited_tokens(table, name, spec, gamma):
    deposited = pd.read_csv(paths.deposit("sr_vectors", f"tokens_g{gamma}_{name}.csv"),
                            keep_default_na=False, dtype={"token": str})
    ours = select.select(table, **spec)
    assert len(ours) == len(deposited)
    assert ours.token_id.tolist() == deposited.token_id.tolist()
    assert ours.token.tolist() == deposited.token.tolist()
    assert ours.freq.tolist() == deposited.freq.tolist()
    assert ours.pos.astype(str).tolist() == deposited.pos.tolist()
    assert ours.frequency_bin.astype(str).tolist() == deposited.frequency_bin.tolist()
