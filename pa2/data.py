"""Moving MNIST loading, windows (eq. 1-2), symmetry group G, sampling plans.

Conventions
-----------
* Videos are kept as uint8 tensors of shape (N, 20, 1, 64, 64) and converted
  with ``to_unit`` (float32 / 255) right before use, which is exactly the
  "convert to float32 and divide by 255" step of the handout.
* Frame numbers are 1-based (1..20) as in the handout; tensor indices are 0-based.
* The same window / transformation functions are used by the Table 1 checks and
  by every experiment.
"""
import numpy as np
import torch

from . import config

# ----------------------------------------------------------------------------
# Loading and splitting
# ----------------------------------------------------------------------------

def load_moving_mnist(data_root, download=True):
    """All 10,000 videos in dataset-index order, uint8, shape (10000, 20, 1, 64, 64)."""
    from torchvision.datasets import MovingMNIST

    ds = MovingMNIST(root=data_root, split=None, download=download)
    data = getattr(ds, "data", None)
    if data is None:  # very old / very new torchvision without .data
        data = torch.stack([ds[i] for i in range(len(ds))])
    if data.dim() == 4:  # (N, 20, 64, 64) -> add channel axis
        data = data.unsqueeze(2)
    data = data.contiguous()
    expected = (config.N_VIDEOS, config.N_FRAMES) + config.FRAME_SHAPE
    assert tuple(data.shape) == expected, f"unexpected shape {tuple(data.shape)}"
    assert data.dtype == torch.uint8, data.dtype
    return data


def split_videos(data):
    """Manual split by dataset index: 0-7999 train, 8000-8999 val, 9000-9999 test."""
    return {name: data[a:b] for name, (a, b) in config.SPLITS.items()}


def to_unit(x):
    """uint8 -> float32 in [0, 1]. Floats are returned unchanged."""
    if x.dtype == torch.uint8:
        return x.float().div_(255.0)
    return x


def to_uint8(x):
    """float in [0, 1] -> uint8 (multiply by 255, rounded)."""
    return (x * 255.0).round_().clamp_(0, 255).to(torch.uint8)


# ----------------------------------------------------------------------------
# Windows (eq. 1 and 2)
# ----------------------------------------------------------------------------

def window_frame_numbers(t, k, n_in=config.N_IN, n_out=config.N_OUT, n_frames=config.N_FRAMES):
    """1-based frame numbers (inputs, existing targets) of the window starting at t, speed k."""
    inp = [t + k * i for i in range(n_in)]
    tgt = [t + k * (n_in + i) for i in range(n_out)]
    assert inp[-1] <= n_frames, "input frames must exist"
    return inp, [f for f in tgt if f <= n_frames]


def allowed_starts(k, speed_augmented):
    """Allowed start frames t (1-based) for speed k under the two training rules."""
    if not speed_augmented:  # models A and B: 1x only, all five targets exist
        assert k == 1
        return list(range(1, 12))
    # all five inputs (t + 4k <= 20) and at least two targets (t + 6k <= 20)
    return list(range(1, config.N_FRAMES - 6 * k + 1))


assert allowed_starts(1, True) == list(range(1, 15))
assert allowed_starts(2, True) == list(range(1, 9))
assert allowed_starts(3, True) == [1, 2]


def make_windows(videos, t, k, n_in=config.N_IN, n_out=config.N_OUT):
    """Cut windows out of a batch of videos.

    videos : (B, 20, C, H, W) tensor of any dtype
    t, k   : int or LongTensor of shape (B,) -- start frame (1-based) and speed
    returns X (B, n_in, C, H, W), Y (B, n_out, C, H, W), mask (B, n_out) bool.
    Target frames that do not exist (frame number > 20) are zero-filled and
    marked False in ``mask`` so they are left out of every loss and metric.
    """
    B, T = videos.shape[:2]
    dev = videos.device
    t = torch.as_tensor(t, device=dev, dtype=torch.long).expand(B) if not torch.is_tensor(t) or t.dim() == 0 \
        else t.to(dev, torch.long)
    k = torch.as_tensor(k, device=dev, dtype=torch.long).expand(B) if not torch.is_tensor(k) or k.dim() == 0 \
        else k.to(dev, torch.long)
    offs = torch.arange(n_in + n_out, device=dev)
    frame_no = t[:, None] + k[:, None] * offs[None, :]           # 1-based, (B, 10)
    exists = frame_no <= T
    if not bool(exists[:, :n_in].all()):
        raise ValueError("some input frames do not exist for the requested (t, k)")
    idx = (frame_no - 1).clamp(max=T - 1)
    frames = videos[torch.arange(B, device=dev)[:, None], idx]    # (B, 10, C, H, W)
    X, Y = frames[:, :n_in], frames[:, n_in:]
    mask = exists[:, n_in:]
    Y = torch.where(mask[:, :, None, None, None], Y, torch.zeros_like(Y))
    return X, Y, mask


def eval_windows(videos, k, t=1):
    """Float windows for evaluation (default t = 1) on CPU."""
    X, Y, mask = make_windows(videos, t, k)
    return to_unit(X), to_unit(Y), mask


# ----------------------------------------------------------------------------
# Symmetries of the square
# ----------------------------------------------------------------------------

def mirror(x):
    """(Mx)[i, j] = x[i, 63 - j]  (flip columns, i.e. the last axis)."""
    return torch.flip(x, dims=(-1,))


def rotate(x, a=1):
    """R^a: rotate a * 90 degrees anticlockwise, torch.rot90(x, k=a, dims=(-2, -1))."""
    a = a % 4
    return x if a == 0 else torch.rot90(x, k=a, dims=(-2, -1))


# G = { R^a, M R^a : a = 0..3 } with (M R^a)(x) = M(R^a(x)). Element = (m, a).
GROUP = [(0, a) for a in range(4)] + [(1, a) for a in range(4)]
GROUP_NAMES = ["I", "R", "R2", "R3", "M", "MR", "MR2", "MR3"]


def apply_g(x, g):
    m, a = g
    y = rotate(x, a)
    return mirror(y) if m else y


def invert_g(x, g):
    """g^{-1}(x): R^a has inverse R^{-a}; every M R^a is its own inverse."""
    m, a = g
    if m:
        return apply_g(x, g)
    return rotate(x, -a)


def apply_group_per_sample(x, gid):
    """Apply GROUP[gid[b]] to sample b of x (B, ...). gid: LongTensor (B,)."""
    if not bool((gid != 0).any()):
        return x
    out = x.clone()
    for i, g in enumerate(GROUP):
        if i == 0:
            continue
        sel = gid == i
        if bool(sel.any()):
            out[sel] = apply_g(x[sel], g)
    return out


# ----------------------------------------------------------------------------
# Per-epoch sampling plan (deterministic in (seed, epoch) -> exact resume)
# ----------------------------------------------------------------------------

def epoch_plan(seed, epoch, n_videos, speed_aug, sym_aug):
    """Shuffle order, speed k, start t and symmetry id for every training video.

    Everything is drawn from a generator seeded with (seed, epoch), so the
    seed controls shuffling, window sampling and augmentation sampling, and an
    interrupted epoch can be regenerated exactly when a session resumes.
    """
    g = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(epoch) * 7919 + 17)
    perm = torch.randperm(n_videos, generator=g)
    if speed_aug:
        k = torch.randint(1, 4, (n_videos,), generator=g)          # uniform on {1,2,3}
        t_max = config.N_FRAMES - 6 * k                            # 14, 8, 2
        u = torch.rand(n_videos, generator=g)
        t = 1 + torch.floor(u * t_max).long()                      # uniform on {1..t_max}
    else:
        k = torch.ones(n_videos, dtype=torch.long)
        t = torch.randint(1, 12, (n_videos,), generator=g)        # uniform on {1..11}
    gid = torch.randint(0, len(GROUP), (n_videos,), generator=g) if sym_aug \
        else torch.zeros(n_videos, dtype=torch.long)
    return perm, k, t, gid


# ----------------------------------------------------------------------------
# Overlap detection (used only to pick example videos for the figures)
# ----------------------------------------------------------------------------

def overlap_pixels(video, thr=0.2):
    """Per-frame overlap estimate for one video (20, 1, 64, 64).

    Digits translate rigidly and are combined with a pixel-wise max, so the
    number of bright pixels is constant except when the digits overlap. The
    drop below the per-video maximum is used as an overlap score per frame.
    """
    v = to_unit(video)
    counts = (v > thr).flatten(1).sum(1).float()
    return (counts.max() - counts).cpu().numpy()   # (20,), 0 = no overlap
