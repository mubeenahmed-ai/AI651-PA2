"""Forecasters for Question 2.

ConvLSTMForecaster implements exactly the equations of the handout (Shi et al.
2015 without peepholes): one convolution per layer produces all four gates
[i, f, o, g] from the concatenation [u_tau, h_{tau-1}]. Model A is
ConvLSTMForecaster(**config.ARCHS["A"]); model D variants change only patch
size / kernel size / dilation / width.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config
from .data import GROUP, apply_g, invert_g


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, kernel=3, dilation=1):
        super().__init__()
        assert kernel % 2 == 1, "use odd kernels so the grid size is preserved"
        self.hid = hid_ch
        pad = dilation * (kernel - 1) // 2
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel_size=kernel,
                              stride=1, padding=pad, dilation=dilation, bias=True)

    def forward(self, u, state):
        h, c = state
        i, f, o, g = torch.split(self.conv(torch.cat([u, h], dim=1)), self.hid, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)      # eq. (4)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTMForecaster(nn.Module):
    """pixel_unshuffle -> stacked ConvLSTM -> 1x1 conv -> pixel_shuffle -> sigmoid.

    Rollout: 9 steps; steps 1-5 read the input frames, steps 6-9 read the
    network's own previous prediction (no teacher forcing, gradients flow
    through the fed-back predictions). Outputs of steps 5-9 are returned.
    """

    def __init__(self, patch=4, hidden=(64, 64), kernels=(3, 3), dilations=(1, 1),
                 n_in=config.N_IN, n_out=config.N_OUT):
        super().__init__()
        assert len(hidden) == len(kernels) == len(dilations)
        self.patch, self.hidden = patch, tuple(hidden)
        self.n_in, self.n_out = n_in, n_out
        cells, prev = [], patch * patch
        for h, k, d in zip(hidden, kernels, dilations):
            cells.append(ConvLSTMCell(prev, h, k, d))
            prev = h
        self.cells = nn.ModuleList(cells)
        self.out = nn.Conv2d(prev, patch * patch, kernel_size=1, bias=True)

    def step(self, frame, states):
        u = F.pixel_unshuffle(frame, self.patch)
        new_states = []
        for cell, st in zip(self.cells, states):
            h, c = cell(u, st)
            new_states.append((h, c))
            u = h
        x_hat = torch.sigmoid(F.pixel_shuffle(self.out(u), self.patch))
        return x_hat, new_states

    def forward(self, X):
        B, T, C, H, W = X.shape
        assert T == self.n_in
        gh, gw = H // self.patch, W // self.patch
        states = [(X.new_zeros(B, h, gh, gw), X.new_zeros(B, h, gh, gw)) for h in self.hidden]
        preds, x_hat = [], None
        for tau in range(1, self.n_in + self.n_out):          # tau = 1..9
            u = X[:, tau - 1] if tau <= self.n_in else x_hat
            x_hat, states = self.step(u, states)
            if tau >= self.n_in:                               # outputs of steps 5..9
                preds.append(x_hat)
        return torch.stack(preds, dim=1)                       # (B, 5, 1, 64, 64)


class Blank(nn.Module):
    """All-zero frames."""
    n_out = config.N_OUT

    def forward(self, X):
        return torch.zeros_like(X[:, :1]).expand(-1, self.n_out, -1, -1, -1).contiguous()


class CopyLast(nn.Module):
    """Repeat the last input frame."""
    n_out = config.N_OUT

    def forward(self, X):
        return X[:, -1:].expand(-1, self.n_out, -1, -1, -1).contiguous()


class Symmetrised(nn.Module):
    """f_bar(X) = 1/8 sum_g g^{-1}( f( g(X) ) )   (eq. 7). No retraining."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, X):
        acc = None
        for g in GROUP:
            y = invert_g(self.base(apply_g(X, g)), g)
            acc = y if acc is None else acc + y
        return acc / len(GROUP)


def build_arch(arch):
    cfg = config.ARCHS[arch] if isinstance(arch, str) else dict(arch)
    return ConvLSTMForecaster(**cfg)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_arch_budget(arch, is_model_d):
    """Assert the parameter constraints of the handout. Returns the count."""
    n = n_params(build_arch(arch))
    if arch == "A":
        assert n == config.MODEL_A_PARAMS, f"model A should have {config.MODEL_A_PARAMS}, got {n}"
    if is_model_d:
        assert n <= config.PARAM_LIMIT_D, f"{arch}: {n} params > limit {config.PARAM_LIMIT_D}"
    return n
