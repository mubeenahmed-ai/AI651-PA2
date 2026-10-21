"""Metrics of eq. (5)-(6), cached per (model, seed, stage) on Drive.

Stages are kept separate on purpose, so the baseline notebook can produce
Table 2 without running any mirror / rotation / speed experiment before your
Q2.5 prediction is committed:
    "std"    1x, t=1: per-video per-horizon MSE           -> Table 2, E(5), E(2)@1x
    "tr"     1x, t=1: per-video C_M, E_M, C_R, E_R          -> Tables 3, 5, 6
    "speed"  2x and 3x, t=1: per-video per-horizon MSE    -> Tables 4, 5, 6
    "time"   inference time on the 1,000 original test videos
"""
import os
import time

import numpy as np
import torch

from . import config
from .data import eval_windows, mirror, rotate
from .utils import atomic_json_save, atomic_npz_save, hardware_string, load_json, sync

TRANSFORMS = {"M": mirror, "R": lambda x: rotate(x, 1)}


@torch.no_grad()
def predict(model, X, device, batch_size=config.BATCH_SIZE):
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        out.append(model(X[i:i + batch_size].to(device)).float().cpu())
    return torch.cat(out)


def horizon_mse(P, Y, mask=None):
    """(N, 5) MSE over the 64x64 pixels of each horizon; NaN where the target is missing."""
    m = (P - Y).pow(2).mean(dim=(2, 3, 4)).double().numpy()
    if mask is not None:
        m = np.where(mask.numpy(), m, np.nan)
    return m


def video_mse(P, Q):
    """(N,) MSE over pixels and all five horizons."""
    return (P - Q).pow(2).mean(dim=(1, 2, 3, 4)).double().numpy()


def _cache(paths, key, seed, stage):
    tag = f"{key}_s{seed}" if seed is not None else f"{key}"
    d = os.path.join(paths["metrics"], tag)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{stage}.npz")


def stage_std(model, test_videos, device):
    X, Y, mask = eval_windows(test_videos, k=1)
    return {"mse_1x": horizon_mse(predict(model, X, device), Y, mask)}


def stage_tr(model, test_videos, device):
    X, Y, _ = eval_windows(test_videos, k=1)
    P = predict(model, X, device)
    out = {}
    for name, T in TRANSFORMS.items():
        PT = predict(model, T(X), device)                  # f(T(X))
        out[f"C_{name}"] = video_mse(PT, T(P))             # vs T(f(X))
        out[f"E_{name}"] = video_mse(PT, T(Y))             # vs T(Y)
    return out


def stage_speed(model, test_videos, device):
    out = {}
    for k in (2, 3):
        X, Y, mask = eval_windows(test_videos, k=k)
        out[f"mse_{k}x"] = horizon_mse(predict(model, X, device), Y, mask)
    return out


@torch.no_grad()
def time_inference(model, test_videos, device, batch_size=config.BATCH_SIZE, warmup=True):
    """Wall-clock time to predict all original (1x, t=1) test videos at batch 32,
    under eval() and no_grad(), counted from when each batch is on the device.
    One untimed warm-up batch excludes one-off CUDA initialisation."""
    model.eval()
    X, _, _ = eval_windows(test_videos, k=1)
    if warmup:
        model(X[:batch_size].to(device))
        sync(device)
    total = 0.0
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(device)
        sync(device)
        t0 = time.perf_counter()
        model(xb)
        sync(device)
        total += time.perf_counter() - t0
    return total


STAGES = {"std": stage_std, "tr": stage_tr, "speed": stage_speed}


def get_metrics(paths, key, seed, stage, model_fn, test_videos, device, recompute=False):
    """Load cached metrics or compute them. model_fn() builds the model lazily."""
    path = _cache(paths, key, seed, stage)
    if stage == "time":
        jpath = path.replace(".npz", ".json")
        rec = load_json(jpath)
        if rec is None or recompute:
            secs = time_inference(model_fn(), test_videos, device)
            rec = {"inference_s": secs, "hardware": hardware_string(device)}
            atomic_json_save(rec, jpath)
        return rec
    if os.path.exists(path) and not recompute:
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    res = STAGES[stage](model_fn(), test_videos, device)
    atomic_npz_save(path, **res)
    return res


# ---------------------------------------------------------------------------
# Summaries (per seed)
# ---------------------------------------------------------------------------

def summarise(std=None, tr=None, speed=None):
    s = {}
    if std is not None:
        m = std["mse_1x"]
        for h in range(5):
            s[f"MSE{h + 1}"] = float(np.mean(m[:, h]))
        s["E5_1x"] = float(np.mean(np.mean(m, axis=1)))
        s["E2_1x"] = float(np.mean(np.mean(m[:, :2], axis=1)))
    if tr is not None:
        for k in ("C_M", "E_M", "C_R", "E_R"):
            s[k] = float(np.mean(tr[k]))
    if speed is not None:
        for kx in ("2x", "3x"):
            m = speed[f"mse_{kx}"]
            s[f"E2_{kx}"] = float(np.mean(np.mean(m[:, :2], axis=1)))
            if not np.isnan(m).any():
                s[f"E5_{kx}"] = float(np.mean(np.mean(m, axis=1)))
    return s
