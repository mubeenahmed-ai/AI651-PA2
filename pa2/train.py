"""Training protocol of the handout, resumable across Colab disconnects.

Protocol: Adam(lr=1e-3), no schedule, batch 32, 30 epochs, final-epoch weights.
One epoch = one pass over the 8,000 training videos in shuffled order with one
randomly sampled window per video. Loss = squared error averaged over pixels
and the *available* target frames of each video, then over the batch.

Resume logic
------------
* ``<run_dir>/ckpt.pt`` holds model + optimiser + progress; it is written
  atomically at the end of every epoch and every ``ckpt_minutes`` inside an epoch.
* The sampling plan of an epoch depends only on (seed, epoch), so an epoch that
  was interrupted is regenerated and continued from the next unfinished batch.
* ``<run_dir>/final.pt`` + ``log.json`` with ``"done": true`` mark a finished run;
  calling ``train_run`` again simply loads it.
* Training time per epoch only counts time spent in training batches (data
  preparation, forward, backward, step); checkpoint writes and the optional
  validation monitoring are excluded. Partial epochs are summed across sessions.
"""
import os
import time

import torch

from . import config
from .data import apply_group_per_sample, epoch_plan, eval_windows, make_windows, to_unit
from .models import build_arch, n_params
from .utils import atomic_json_save, atomic_torch_save, hardware_string, load_json, sync


def masked_video_mse(pred, Y, mask):
    """Per-video MSE over pixels and available target frames: (B,)."""
    se = (pred - Y).pow(2).mean(dim=(2, 3, 4))            # (B, 5) per-horizon MSE
    m = mask.to(se.dtype)
    return (se * m).sum(1) / m.sum(1).clamp_min(1.0)


@torch.no_grad()
def quick_val_e5(model, val_videos, device, batch_size=128):
    """Validation E(5) at 1x, t=1 (training monitor only)."""
    model.eval()
    X, Y, _ = eval_windows(val_videos, k=1)
    tot = 0.0
    for i in range(0, len(X), batch_size):
        p = model(X[i:i + batch_size].to(device))
        tot += (p - Y[i:i + batch_size].to(device)).pow(2).mean(dim=(1, 2, 3, 4)).sum().item()
    model.train()
    return tot / len(X)


def train_run(run_dir, arch, speed_aug, sym_aug, seed, train_videos, device,
              val_videos=None, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
              lr=config.LR, ckpt_minutes=5.0, val_every=5, log=print, max_batches=None):
    """Train one model with one seed (or resume / load it). Returns (model, log_dict).

    train_videos: uint8 (N, 20, 1, 64, 64); putting it on the GPU makes batching fast.
    max_batches: only for smoke tests (stops each epoch early).
    """
    os.makedirs(run_dir, exist_ok=True)
    tag = f"{os.path.basename(os.path.dirname(run_dir))}/{os.path.basename(run_dir)}"
    ckpt_path = os.path.join(run_dir, "ckpt.pt")
    final_path = os.path.join(run_dir, "final.pt")
    log_path = os.path.join(run_dir, "log.json")

    torch.manual_seed(seed)                          # controls initialisation
    model = build_arch(arch).to(device)

    info = load_json(log_path)
    if info and info.get("done") and info.get("epochs") == epochs and os.path.exists(final_path):
        model.load_state_dict(torch.load(final_path, map_location=device))
        log(f"[{tag}] already trained "
            f"({info['epochs']} epochs, {info['mean_epoch_time_s']:.1f} s/epoch on {info['hardware'][0]})")
        return model, info

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    state = dict(epoch=0, batch=0, epoch_time=0.0, epoch_loss=0.0, epoch_count=0,
                 history=[], hardware=[])
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        log(f"[{tag}] resuming at epoch {state['epoch'] + 1}, "
            f"batch {state['batch']}")
    hw = hardware_string(device)
    if hw not in state["hardware"]:
        if state["hardware"]:
            log(f"WARNING: hardware changed ({state['hardware'][-1]} -> {hw}); "
                "per-epoch timings will mix devices.")
        state["hardware"].append(hw)

    def save_ckpt():
        atomic_torch_save({"model": model.state_dict(), "opt": opt.state_dict(),
                           "state": state, "arch": arch, "seed": seed,
                           "speed_aug": speed_aug, "sym_aug": sym_aug}, ckpt_path)

    n = len(train_videos)
    n_batches = (n + batch_size - 1) // batch_size
    if max_batches:
        n_batches = min(n_batches, max_batches)
    model.train()
    last_save = time.time()

    while state["epoch"] < epochs:
        ep = state["epoch"]
        perm, k, t, gid = epoch_plan(seed, ep, n, speed_aug, sym_aug)
        b = state["batch"]
        while b < n_batches:
            sl = slice(b * batch_size, min((b + 1) * batch_size, n))
            sync(device)
            t0 = time.perf_counter()
            vids = train_videos[perm[sl].to(train_videos.device)].to(device, non_blocking=True)
            X, Y, mask = make_windows(vids, t[sl], k[sl])
            X, Y = to_unit(X), to_unit(Y)
            if sym_aug:                                   # same g for input and target
                g = gid[sl].to(device)
                X, Y = apply_group_per_sample(X, g), apply_group_per_sample(Y, g)
            pred = model(X)
            loss = masked_video_mse(pred, Y, mask).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_v = loss.item()                          # also synchronises
            sync(device)
            state["epoch_time"] += time.perf_counter() - t0
            state["epoch_loss"] += loss_v * (sl.stop - sl.start)
            state["epoch_count"] += sl.stop - sl.start
            b += 1
            state["batch"] = b
            if time.time() - last_save > ckpt_minutes * 60 and b < n_batches:
                save_ckpt()
                last_save = time.time()

        rec = dict(epoch=ep + 1, train_loss=state["epoch_loss"] / max(state["epoch_count"], 1),
                   time_s=state["epoch_time"])
        if val_videos is not None and val_every and ((ep + 1) % val_every == 0 or ep + 1 == epochs):
            rec["val_E5_1x"] = quick_val_e5(model, val_videos, device)
        state["history"].append(rec)
        msg = f"[{tag}] epoch {ep + 1:2d}/{epochs}  loss {rec['train_loss']:.5f}  " \
              f"{rec['time_s']:.1f} s"
        if "val_E5_1x" in rec:
            msg += f"  val E5 {rec['val_E5_1x']:.5f}"
        log(msg)
        state.update(epoch=ep + 1, batch=0, epoch_time=0.0, epoch_loss=0.0, epoch_count=0)
        save_ckpt()
        last_save = time.time()

    atomic_torch_save(model.state_dict(), final_path)
    times = [h["time_s"] for h in state["history"]]
    info = dict(done=True, arch=arch if isinstance(arch, str) else dict(arch), seed=seed,
                speed_aug=speed_aug, sym_aug=sym_aug, epochs=epochs, batch_size=batch_size, lr=lr,
                n_params=n_params(model), mean_epoch_time_s=sum(times) / len(times),
                hardware=state["hardware"], history=state["history"],
                timing_note="training-batch time only; checkpoint I/O and validation excluded")
    atomic_json_save(info, log_path)
    log(f"[{tag}] done: {info['mean_epoch_time_s']:.1f} s/epoch")
    return model, info
