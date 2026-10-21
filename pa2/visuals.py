"""Figures 2-6 and the animated GIFs.

Video ids are reported both as test index (0-999) and dataset index (9000 + i).
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import config
from .data import eval_windows, overlap_pixels, window_frame_numbers
from .evaluation import TRANSFORMS, predict

TEST0 = config.SPLITS["test"][0]


def vid_label(i):
    return f"test #{i} (dataset idx {TEST0 + i})"


def _img(ax, frame, title=None):
    ax.imshow(np.asarray(frame).squeeze(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=7)


def _blank(ax):
    ax.axis("off")


# ---------------------------------------------------------------------------
# Selecting example videos
# ---------------------------------------------------------------------------

def overlap_table(test_videos):
    return np.stack([overlap_pixels(test_videos[i]) for i in range(len(test_videos))])  # (N, 20)


def pick_fig2_videos(test_videos, n_clean=2, tol=3):
    """Two videos without overlap in frames 1-10 plus the one with the largest
    overlap in target frames 6-10 (at least one overlapping video is required)."""
    ov = overlap_table(test_videos)
    clean = [i for i in range(len(ov)) if ov[i, :10].max() <= tol][:n_clean]
    over = int(np.argmax(ov[:, 5:10].max(axis=1)))
    return clean + [over]


def pick_fig5_video(test_videos, tol=3):
    """First test video whose digits do not overlap in any displayed true frame
    (inputs + first two targets at 1x, 2x and 3x, t = 1)."""
    frames = set()
    for k in (1, 2, 3):
        inp, tgt = window_frame_numbers(1, k)
        frames.update(inp + tgt[:2])
    idx = sorted(f - 1 for f in frames)
    ov = overlap_table(test_videos)
    for i in range(len(ov)):
        if ov[i, idx].max() <= tol:
            return i
    raise RuntimeError("no overlap-free video found; pick one manually")


# ---------------------------------------------------------------------------
# Figure 2: input / prediction / ground truth
# ---------------------------------------------------------------------------

def figure2(model, test_videos, ids, device, out_path, model_name="model A (seed 0)"):
    X, Y, _ = eval_windows(test_videos[ids], k=1)
    P = predict(model, X, device)
    fig, axes = plt.subplots(3 * len(ids), 5, figsize=(5 * 1.15, 3 * len(ids) * 1.25))
    for r, i in enumerate(ids):
        for c in range(5):
            _img(axes[3 * r, c], X[r, c], f"in: frame {c + 1}")
            _img(axes[3 * r + 1, c], P[r, c], f"pred: frame {c + 6}")
            _img(axes[3 * r + 2, c], Y[r, c], f"true: frame {c + 6}")
        axes[3 * r, 0].set_ylabel(vid_label(i), fontsize=7)
    fig.suptitle(f"Figure 2 – {model_name}: input, prediction, ground truth (1×, t = 1)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Figure 3 / 6(a): four rows of ten frames per video, plus GIF
# ---------------------------------------------------------------------------

def consistency_rows(model, test_videos, i, T_name, device):
    T = TRANSFORMS[T_name]
    X, Y, _ = eval_windows(test_videos[i:i + 1], k=1)
    fX = predict(model, X, device)
    fTX = predict(model, T(X), device)
    TX = T(X)
    rows = [
        (f"X | f(X)", X[0], fX[0]),
        (f"{T_name}(X) | {T_name}(Y)", TX[0], T(Y)[0]),
        (f"{T_name}(X) | f({T_name}(X))", TX[0], fTX[0]),
        (f"{T_name}(X) | {T_name}(f(X))", TX[0], T(fX)[0]),
    ]
    return rows


def figure_consistency_panel(model, test_videos, i, T_name, device, out_png, out_gif, title):
    rows = consistency_rows(model, test_videos, i, T_name, device)
    fig, axes = plt.subplots(4, 10, figsize=(10 * 0.95, 4 * 1.05))
    for r, (lab, inp, fut) in enumerate(rows):
        for c in range(5):
            _img(axes[r, c], inp[c], f"in {c + 1}" if r == 0 else None)
            _img(axes[r, 5 + c], fut[c], f"h = {c + 1}" if r == 0 else None)
        axes[r, 0].set_ylabel(f"({r + 1}) {lab}", fontsize=7, rotation=0, ha="right", va="center")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    save_rows_gif(rows, out_gif, title)
    return fig


def save_rows_gif(rows, out_gif, title, ms_per_frame=700):
    """Five GIF frames (one per forecast horizon); each keeps the same four rows:
    left = last frame of the row's input, right = the row's frame at horizon h."""
    from PIL import Image
    frames = []
    for h in range(5):
        fig, axes = plt.subplots(4, 2, figsize=(4.2, 7.6))
        for r, (lab, inp, fut) in enumerate(rows):
            _img(axes[r, 0], inp[-1], "last input" if r == 0 else None)
            _img(axes[r, 1], fut[h], f"horizon h = {h + 1}" if r == 0 else None)
            axes[r, 0].set_ylabel(f"({r + 1}) {lab}", fontsize=7)
        fig.suptitle(f"{title}\nh = {h + 1}", fontsize=7)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        frames.append(Image.fromarray(buf))
    frames[0].save(out_gif, save_all=True, append_images=frames[1:], duration=ms_per_frame, loop=0)
    return out_gif


def top_videos(per_video, n=2):
    return [int(i) for i in np.argsort(-per_video)[:n]]


# ---------------------------------------------------------------------------
# Figure 4: histograms of per-video C_M and C_R
# ---------------------------------------------------------------------------

def figure4(tr, marked, out_path, model_name="model A (seed 0)"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, T in zip(axes, ("M", "R")):
        v = tr[f"C_{T}"]
        ax.hist(v, bins=50, color="#4C72B0", edgecolor="white")
        for rank, i in enumerate(marked[T]):
            ax.axvline(v[i], color="crimson", ls="--", lw=1.2)
            ax.annotate(f"#{i}", (v[i], ax.get_ylim()[1] * (0.9 - 0.1 * rank)), color="crimson",
                        fontsize=7, ha="right", rotation=90)
        ax.set(xlabel=f"per-video C_{T}", ylabel="number of test videos",
               title=f"C_{T} ({'mirror' if T == 'M' else 'rotation'}), {model_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Figure 5 / 6(b): one video at 1x, 2x, 3x
# ---------------------------------------------------------------------------

def figure_speeds(model, test_videos, i, device, out_path, title):
    fig, axes = plt.subplots(3, 9, figsize=(9 * 0.95, 3 * 1.25))
    for r, k in enumerate((1, 2, 3)):
        X, Y, mask = eval_windows(test_videos[i:i + 1], k=k)
        P = predict(model, X, device)
        inp, tgt = window_frame_numbers(1, k)
        for c in range(5):
            _img(axes[r, c], X[0, c], f"in: f{inp[c]}")
        for h in range(2):
            _img(axes[r, 5 + h], Y[0, h], f"true: f{tgt[h]}")
            _img(axes[r, 7 + h], P[0, h], f"pred h={h + 1}")
        axes[r, 0].set_ylabel(f"{k}×", fontsize=9, rotation=0, ha="right", va="center")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig
