import numpy as np
import pandas as pd
import pytest
import torch

from conftest import needs_deposit
from srwc import paths
from srwc.model import GAMMAS, SRNet, load_model

CHECKPOINTS = ["model/model_checkpoint.pt",
               "seed_replicates/model_checkpoint_seed2.pt",
               "seed_replicates/model_checkpoint_seed3.pt"]


def test_parameter_count():
    assert sum(p.numel() for p in SRNet().parameters()) == 56_283_744


@needs_deposit
@pytest.mark.parametrize("checkpoint", CHECKPOINTS)
def test_checkpoints_load_strictly(checkpoint):
    load_model(paths.deposit(checkpoint))


@needs_deposit
@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("vocab", ["nva", "ext"])
def test_recomputes_deposited_sr_vectors(gamma, vocab):
    model = load_model(paths.deposit(CHECKPOINTS[0]))
    tokens = pd.read_csv(paths.deposit("sr_vectors", f"tokens_g{gamma}_{vocab}.csv"),
                         keep_default_na=False)
    deposited = np.load(paths.deposit("sr_vectors", f"sr_head_g{gamma}_{vocab}.npy"))
    with torch.no_grad():
        logits = model(torch.tensor(tokens.token_id.values), gamma).numpy()
    np.testing.assert_allclose(logits, deposited, rtol=0, atol=1e-4)
