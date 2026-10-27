"""Training loop of the reported model (Table 1).

AdamW (lr 1e-4, weight decay 1e-5); cosine schedule with 1,000 warm-up steps that
decays to zero (the run's saved scheduler state ends at lr = 0); gradient-norm
clipping at 1.0, after any NaN/Inf gradient entries are zeroed; an EMA target network
(decay 0.999, updated after every optimiser step) supplies the bootstrap; the loss is
the mean of the three heads' KL losses. Batches of 160 samples are drawn by a
shuffling DataLoader, 10 epochs. Validation at the end of each epoch bootstraps from
the EMA network too.

The reported model was trained without a fixed seed. `seed` makes a run
reproducible; it is applied before the network is initialised, as in the original
code, so a seeded run here follows the original code step for step.
"""
import copy
import json
import math
import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import SRNet
from .targets import head_loss

CONFIG = dict(gammas=(0.2, 0.5, 0.8), lam=0.9, epochs=10, batch_size=160, lr=1e-4,
              weight_decay=1e-5, warmup=1000, clip=1.0, ema_decay=0.999)


def cosine_with_warmup(warmup, total):
    """transformers.get_cosine_schedule_with_warmup, which has no floor."""
    def factor(step):
        if step < warmup:
            return float(step) / float(max(1, warmup))
        progress = float(step - warmup) / float(max(1, total - warmup))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 0.5 * 2.0 * progress)))
    return factor


def _loss(model, ema, ids, gammas, lam):
    logits = model(ids)
    with torch.no_grad():
        ema_logits = ema(ids)
    per_head = [head_loss(ids, logits[g], ema_logits[g], g, lam) for g in gammas]
    total = torch.tensor(0.0, device=ids.device)
    for h in per_head:
        total = total + h * 1.0
    return total / len(gammas), per_head


def _save(out_dir, model, optimizer, scheduler, epoch, step, avg_loss, history):
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "batch_idx": 0,
                "global_step": step, "avg_loss": avg_loss}, os.path.join(out_dir, "model_checkpoint.pt"))
    torch.save({"optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch, "global_step": step}, os.path.join(out_dir, "training_state.pt"))
    with open(os.path.join(out_dir, "loss_history.json"), "w") as f:
        json.dump(history, f, indent=2)


def train(train_ids, val_ids, out_dir, seed=None, device=None, **overrides):
    cfg = {**CONFIG, **overrides}
    gammas, lam = cfg["gammas"], cfg["lam"]
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    model = SRNet(gammas=gammas).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loader = DataLoader(TensorDataset(torch.as_tensor(train_ids)), batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.as_tensor(val_ids)), batch_size=cfg["batch_size"], shuffle=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, cosine_with_warmup(cfg["warmup"], len(loader) * cfg["epochs"]))
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad = False

    history = {"epochs": [], "train_sr_loss": [], "val_sr_loss": [],
               **{f"{s}_sr_loss_g{g}": [] for s in ("train", "val") for g in gammas}}
    step = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        total, heads, n = 0.0, [0.0] * len(gammas), 0
        for (ids,) in loader:
            ids = ids.to(device)
            loss, per_head = _loss(model, ema, ids, gammas, lam)
            total += sum(h.item() * 1.0 for h in per_head) / len(gammas)
            heads = [a + h.item() for a, h in zip(heads, per_head)]
            n += 1
            loss.backward()
            for p in model.parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    p.grad[torch.isnan(p.grad)] = 0.0
                    p.grad[torch.isinf(p.grad)] = 0.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["clip"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.data.mul_(cfg["ema_decay"]).add_(pm.data, alpha=1 - cfg["ema_decay"])

        model.eval()
        v_total, v_heads, vn = 0.0, [0.0] * len(gammas), 0
        with torch.no_grad():
            for (ids,) in val_loader:
                loss, per_head = _loss(model, ema, ids.to(device), gammas, lam)
                v_total += loss.item()
                v_heads = [a + h.item() for a, h in zip(v_heads, per_head)]
                vn += 1

        history["epochs"].append(epoch + 1)
        history["train_sr_loss"].append(total / n)
        history["val_sr_loss"].append(v_total / max(1, vn))
        for g, a, b in zip(gammas, heads, v_heads):
            history[f"train_sr_loss_g{g}"].append(a / n)
            history[f"val_sr_loss_g{g}"].append(b / max(1, vn))
        _save(os.path.join(out_dir, f"epoch_{epoch + 1:03d}"), model, optimizer, scheduler,
              epoch + 1, step, total / n, history)
    return model, history
