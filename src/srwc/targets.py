"""TD(lambda) targets and the KL objective (Methods, Supplementary Appendix E.2).

Arrival-reward lambda-return over one sample of T positions, computed backwards:

    G_{T-2} = (1-gamma) phi_{T-1} + gamma psi_{T-1}
    G_t     = (1-gamma) phi_{t+1} + gamma [(1-lambda) psi_{t+1} + lambda G_{t+1}]

phi is the one-hot of the observed token, and psi is the EMA target network's
softmax. The return runs through padded positions: padding is never a trace
boundary. Padding and <eos> are only excluded as source positions of the loss
(<unk> is not). The arithmetic follows the original implementation operation for
operation.
"""
import torch
import torch.nn.functional as F

from .vocab import EOS, PAD

IGNORED_SOURCES = (PAD, EOS)


def lambda_return(token_ids, target_logits, gamma, lam):
    """Targets G_t for t = 0..T-2, shape [batch, T-1, vocab]."""
    vocab = target_logits.shape[-1]
    phi = F.one_hot(token_ids, num_classes=vocab).float() * (1 - gamma)
    psi = F.softmax(target_logits, dim=-1)
    batch, T, _ = phi.shape
    G = torch.zeros(batch, T - 1, vocab, device=phi.device, dtype=phi.dtype)
    G[:, -1, :] = phi[:, -1, :] + gamma * psi[:, -1, :].detach()
    for t in range(T - 3, -1, -1):
        G[:, t, :] = phi[:, t + 1, :] + gamma * ((1 - lam) * psi[:, t + 1, :].detach() + lam * G[:, t + 1, :])
    return G


def source_mask(token_ids):
    mask = torch.ones(token_ids.shape, dtype=torch.float32, device=token_ids.device)
    for tok in IGNORED_SOURCES:
        mask = mask * (token_ids != tok).float()
    return mask


def kl_loss(logits, targets, mask):
    """Mean over valid source positions of KL(G_t || softmax(psi_t)), summed over the vocabulary."""
    log_pred = F.log_softmax(logits[:, :-1, :], dim=-1)
    per_position = F.kl_div(log_pred, torch.clamp(targets.detach(), min=0.0), reduction="none").sum(dim=-1)
    m = mask[:, :-1]
    return (per_position * m).sum() / m.sum().clamp(min=1.0)


def head_loss(token_ids, logits, target_logits, gamma, lam):
    return kl_loss(logits, lambda_return(token_ids, target_logits, gamma, lam), source_mask(token_ids))
