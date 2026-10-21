"""Table 1 (implementation checks) and Figure 1 (optical-flow statistics).

Both call the same transformation / window functions as the experiments.
"""
import os

import numpy as np
import torch

from .data import GROUP, invert_g, make_windows, mirror, rotate, to_uint8, to_unit
from .utils import save_table


def _maxdiff(a, b):
    return float((a - b).abs().max())


def _bright(p):
    idx = (p == p.max()).nonzero()
    assert len(idx) == 1, "expected a single bright pixel"
    f, c, i, j = idx[0].tolist()
    return (f, i, j)


def _window_frames(k):
    v = torch.arange(1, 21, dtype=torch.float32).view(20, 1, 1, 1).expand(20, 1, 64, 64).clone()
    X, Y, mask = make_windows(v[None], t=1, k=k)
    inp = [int(round(X[0, i, 0, 0, 0].item())) for i in range(X.shape[1])]
    tgt = [int(round(Y[0, i, 0, 0, 0].item())) for i in range(Y.shape[1]) if mask[0, i]]
    return inp, tgt


def _fmt_frames(fr):
    inp, tgt = fr
    def f(xs):
        if xs == list(range(xs[0], xs[0] + len(xs))) and len(xs) > 2:
            return f"{xs[0]}–{xs[-1]}"
        return ", ".join(map(str, xs))
    return f"{f(inp)} / {f(tgt)}"


def table1(paths=None):
    torch.manual_seed(0)
    x = torch.rand(20, 1, 64, 64)
    p = torch.zeros(20, 1, 64, 64)
    p[7, 0, 10, 20] = 1

    R_inv = lambda z: invert_g(z, GROUP[1])            # R^{-1}
    R4 = rotate(rotate(rotate(rotate(x))))
    MRM = mirror(rotate(mirror(x)))
    rows_raw = [
        ("(a)", "max|M(M(x)) − x|", "0", mirror(mirror(x)), x),
        ("(b)", "max|R^4(x) − x|", "0", R4, x),
        ("(c)", "max|MRM(x) − R^−1(x)|", "0", MRM, R_inv(x)),
    ]
    rows, ok = [], True
    for tag, check, exp, lhs, rhs in rows_raw:
        d, eq = _maxdiff(lhs, rhs), torch.equal(lhs, rhs)
        ok &= (d == 0.0 and eq)
        rows.append([tag, check, exp, f"{d:g} (torch.equal: {eq})"])

    dM, dR = _bright(mirror(p)), _bright(rotate(p))
    ok &= dM == (7, 10, 43) and dR == (7, 43, 10)
    rows.append(["(d)", "bright pixel of M(p) (frame, row, column)", "(7, 10, 43)", str(dM)])
    rows.append(["(e)", "bright pixel of R(p) (frame, row, column)", "(7, 43, 10)", str(dR)])

    expected = {1: ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]),
                2: ([1, 3, 5, 7, 9], [11, 13, 15, 17, 19]),
                3: ([1, 4, 7, 10, 13], [16, 19])}
    exp_txt = {1: "1–5 / 6–10", 2: "1, 3, 5, 7, 9 / 11, 13, 15, 17, 19", 3: "1, 4, 7, 10, 13 / 16, 19"}
    for tag, k in (("(f)", 1), ("(g)", 2), ("(h)", 3)):
        got = _window_frames(k)
        ok &= tuple(got) == expected[k]
        rows.append([tag, f"{k}× input / target frames", exp_txt[k], _fmt_frames(got)])

    header = ["Check", "", "Expected", "Yours"]
    md = save_table(paths, "table1_checks", header, rows) if paths else None
    return rows, ok, md


# ---------------------------------------------------------------------------
# Figure 1: Farneback optical flow on the first 50 training videos
# ---------------------------------------------------------------------------

def flow_statistics(train_videos, n_videos=50, thr=0.2, min_mag=0.5):
    import cv2
    angles, speeds = [], []
    for v in range(n_videos):
        x = to_unit(train_videos[v])[:, 0]              # (20, 64, 64) float in [0, 1]
        u8 = to_uint8(x).numpy()
        xf = x.numpy()
        for s in range(19):                             # pairs (x_s, x_{s+1})
            flow = cv2.calcOpticalFlowFarneback(u8[s], u8[s + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            dx, dy = flow[..., 0], flow[..., 1]
            mag = np.sqrt(dx ** 2 + dy ** 2)
            keep = (xf[s] > thr) & (xf[s + 1] > thr) & (mag > min_mag)
            angles.append(np.degrees(np.arctan2(-dy[keep], dx[keep])) % 360.0)
            speeds.append(mag[keep])
    return np.concatenate(angles), np.concatenate(speeds)


def figure1(paths, angles, speeds):
    import matplotlib.pyplot as plt
    n = len(speeds)
    med = float(np.median(speeds))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    w = np.full(n, 1.0 / n)
    ax[0].hist(angles, bins=np.arange(0, 361, 20), weights=w, color="#4C72B0", edgecolor="white")
    ax[0].axhline(1 / 18, color="k", ls="--", lw=1, label="uniform (1/18)")
    ax[0].set(xlabel="flow direction θ = atan2(−dy, dx) [deg]", ylabel="fraction of kept pixels",
              title="(a) direction", xticks=np.arange(0, 361, 60), xlim=(0, 360))
    ax[0].legend(loc="lower right")
    ax[1].hist(speeds, bins=60, weights=w, color="#DD8452", edgecolor="white")
    ax[1].axvline(med, color="k", lw=1.5, label=f"median = {med:.3g} px/frame")
    ax[1].set(xlabel="flow speed [pixels per frame]", ylabel="fraction of kept pixels", title="(b) speed")
    ax[1].legend()
    fig.tight_layout()
    out = os.path.join(paths["figures"], "fig1_flow.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig, dict(n_kept=n, median_speed=med, path=out)
