"""The network of Supplementary Appendix E.1.

Each residual block computes x' = x + W2 GELU(W1 LayerNorm(x) + b1) + b2. An
embedding table of 20000 x 512 feeds a shared trunk of eight blocks, followed by one
head per discount factor: seven blocks and a linear projection to the vocabulary.
There is no context: every position is processed independently, so a word type's
successor representation is a function of its token ID alone.

Modules are created in the same order as in the original code, so a seeded
initialisation is identical, and the state-dict keys match the published checkpoints.
"""
import torch
import torch.nn as nn

GAMMAS = (0.2, 0.5, 0.8)


def head_key(gamma: float) -> str:
    """'g0_20' for gamma = 0.2, the key used in the checkpoints' state dicts."""
    return f"g{gamma:.2f}".replace(".", "_")


class ResidualBlock(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)

    def forward(self, x):
        return x + self.fc2(nn.functional.gelu(self.fc1(self.norm(x))))


class SRNet(nn.Module):
    def __init__(self, vocab=20000, d=512, trunk_blocks=8, head_blocks=7, gammas=GAMMAS):
        super().__init__()
        self.gammas = tuple(gammas)
        self.embedding = nn.Embedding(vocab, d)
        self.layers = nn.Sequential(*[ResidualBlock(d) for _ in range(trunk_blocks)])
        self.sr_heads = nn.ModuleDict({
            head_key(g): nn.Sequential(*[ResidualBlock(d) for _ in range(head_blocks)],
                                       nn.Linear(d, vocab))
            for g in self.gammas
        })

    def forward(self, token_ids, gamma=None):
        """Pre-softmax logits: one tensor for `gamma`, or a dict over all heads."""
        h = self.layers(self.embedding(token_ids))
        if gamma is not None:
            return self.sr_heads[head_key(gamma)](h)
        return {g: self.sr_heads[head_key(g)](h) for g in self.gammas}


def load_model(checkpoint, device="cpu") -> SRNet:
    state = torch.load(checkpoint, map_location=device, weights_only=False)["model_state_dict"]
    model = SRNet().to(device)
    model.load_state_dict(state, strict=True)
    return model.eval()
