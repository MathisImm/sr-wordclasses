"""targets.py against the original successor_representation_loss_td_lambda_correct (kl_div)."""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

from conftest import needs_original
from srwc import targets
from srwc.vocab import EOS, PAD


@needs_original
@pytest.mark.parametrize("gamma", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_loss_and_gradient_match_original(gamma, seed):
    sys.path.insert(0, os.environ["SRWC_ORIGINAL"])
    from training.SR import successor_representation_loss_td_lambda_correct as original

    g = torch.Generator().manual_seed(seed)
    batch, T, vocab = 4, 16, 60
    ids = torch.randint(3, vocab, (batch, T), generator=g)
    ids[0, 10:] = PAD                     # a short sample, padded
    ids[1, 5] = EOS
    ids[2, 7] = 0                         # <unk> stays in the loss
    logits = torch.randn(batch, T, vocab, generator=g, requires_grad=True)
    ema = torch.randn(batch, T, vocab, generator=g)

    mask = targets.source_mask(ids)
    vocab_mask = torch.ones(vocab); vocab_mask[[PAD, EOS]] = 0.0
    ref = original(F.one_hot(ids, vocab).float(), logits, gamma, 0.9, sequence_mask=mask,
                   vocab_mask=vocab_mask, target_ids=ids, psi_target=ema, loss_type="kl_div")
    ref_grad, = torch.autograd.grad(ref, logits)

    ours = targets.head_loss(ids, logits, ema, gamma, 0.9)
    our_grad, = torch.autograd.grad(ours, logits)

    assert torch.equal(ours.detach(), ref.detach())
    assert torch.equal(our_grad, ref_grad)
