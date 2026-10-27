"""train.py against the original training script, run end to end on CPU.

Both are given the same seed and the same short corpus (the original's own
preprocessing of the first WikiText-103 rows), with the reported configuration
except for batch size, epochs and warm-up, which are shrunk so the run is short.
The original's periodic SR diagnostics (--eval_sr_interval) are switched off: they
read the model without changing it. Requires identical per-epoch losses and final
weights (to float tolerance).

One side effect has to be replayed: the original's validation also computes a
"masked_mse" diagnostic, which draws torch.rand_like(...) of shape [batch, T-1, vocab]
once per head and validation batch. That consumes the global random stream and so
changes the next epoch's shuffle. It does not affect training otherwise (and the
reported model was trained unseeded), so the clean trainer does not do it; the test
consumes the same random numbers instead.
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np
import torch

from conftest import needs_original
from original import prepare_pos_files, read_jsonl
from srwc import corpus, train, vocab

ROWS = {"train": 400, "validation": 150}
SMALL = dict(batch_size=16, epochs=2, warmup=10)


@needs_original
def test_training_matches_original(tmp_path, monkeypatch):
    files = {s: prepare_pos_files(tmp_path, monkeypatch, s, n)[0] for s, n in ROWS.items()}
    env = {**os.environ, "HOME": str(tmp_path), "CUDA_VISIBLE_DEVICES": "",
           "HF_HOME": os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))}
    cmd = [sys.executable, "-m", "training.train", "--model_type", "resnet", "--dataset", "wikitext-103",
           "--pos_tagged", "--no_mask_unk", "--hidden_size", "512", "--num_layers", "8",
           "--vocab_size", "20000", "--learning_rate", "1e-4", "--weight_decay", "1e-5",
           "--batch_size", str(SMALL["batch_size"]), "--accumulation_steps", "1",
           "--num_epochs", str(SMALL["epochs"]), "--seq_length", "80", "--lr_scheduler", "cosine",
           "--lr_min", "1e-6", "--lr_warmup_steps", str(SMALL["warmup"]), "--save_steps", "20000",
           "--gamma", "0.7", "--lambda_param", "0.9", "--gammas", "0.2,0.5,0.8", "--ema_decay", "0.999",
           "--sum_loss_type", "relative_mse", "--sr_loss_type", "kl_div", "--eval_sr_interval", "0",
           "--grad_clip_norm", "1.0", "--disable_tqdm", "--check_phi_grad", "--block_type", "modern",
           "--expansion_rate", "1", "--head_num_layers", "8", "--resnet_head", "--embedding_dim", "512",
           "--seed", "0", "--ckpt_dir", str(tmp_path / "ckpt")]
    log = subprocess.run(cmd, cwd=os.environ["SRWC_ORIGINAL"], env=env, capture_output=True, text=True)
    (tmp_path / "original.log").write_text(log.stdout + log.stderr)
    assert log.returncode == 0, log.stderr[-3000:]
    final = glob.glob(str(tmp_path / "ckpt" / "*" / f"epoch_{SMALL['epochs']:03d}" / "model_checkpoint.pt"))
    assert len(final) == 1, final
    ref_state = torch.load(final[0], map_location="cpu", weights_only=False)["model_state_dict"]
    ref_hist = json.load(open(os.path.join(os.path.dirname(final[0]), "loss_history.json")))

    real_loss = train._loss

    def replay_validation_rng(model, ema, ids, gammas, lam):
        out = real_loss(model, ema, ids, gammas, lam)
        if not model.training:
            for _ in gammas:
                torch.rand_like(torch.empty(ids.shape[0], ids.shape[1] - 1, 20000))
        return out
    monkeypatch.setattr(train, "_loss", replay_validation_rng)

    v = vocab.build_vocab(vocab.load_frequencies())
    enc = corpus.Encoder(v)
    ids = {s: [enc(t)[0] for t in read_jsonl(f)] for s, f in files.items()}
    model, hist = train.train(ids["train"], ids["validation"], tmp_path / "clean", seed=0, device="cpu", **SMALL)

    for key in ["train_sr_loss", "val_sr_loss"] + [f"{s}_sr_loss_g{g}" for s in ("train", "val") for g in (0.2, 0.5, 0.8)]:
        np.testing.assert_allclose(hist[key], ref_hist[key], rtol=1e-6, atol=0, err_msg=key)
    state = model.state_dict()
    assert list(state) == list(ref_state)
    worst = max((state[k] - ref_state[k]).abs().max().item() for k in state)
    assert worst < 1e-6, worst
