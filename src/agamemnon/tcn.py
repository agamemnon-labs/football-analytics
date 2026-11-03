"""Numpy inference for the temporal convolutional emission model.

The network is trained on external games only (scripts/train_tcn.py, PyTorch, GPU). Inference
needs nothing but numpy: weights live in a small JSON file next to this module, the arithmetic is
float64, and outputs are rounded, so the submission does not depend on PyTorch, on a GPU or on the
last bits of a BLAS kernel.

Architecture (mirrored exactly by the training script): a pointwise input layer, residual blocks
of [dilated 3-tap convolution, ReLU, pointwise convolution] with dilations doubling up to the
receptive field, a ReLU and a pointwise output layer giving logits for H, A, D. Each period is run
on its own with zero padding at its edges.
"""

import json
import os

import numpy as np

from . import deadball, decoder

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "tcn.json")


def inputs(Xt, Xd):
    """Network input channels: both design matrices without their bias columns."""
    return np.concatenate([Xt[:, 1:], Xd[:, 1:]], axis=1)


N_INPUTS = len(decoder.TEAM_COLUMNS) - 1 + len(deadball.DEAD_COLUMNS) - 1


def load(path=WEIGHTS):
    with open(path) as handle:
        raw = json.load(handle)
    out = {k: np.asarray(v, dtype=np.float64) for k, v in raw["params"].items()}
    out["dilations"] = [int(d) for d in raw["dilations"]]
    return out


def _dilated(x, W, b, d):
    """3-tap convolution with dilation d and zero padding. x (C, T), W (O, C, 3)."""
    T = x.shape[1]
    y = W[:, :, 1] @ x
    if d < T:
        y[:, d:] += W[:, :, 0] @ x[:, :-d]
        y[:, :-d] += W[:, :, 2] @ x[:, d:]
    return y + b[:, None]


def forward(net, x):
    """Logits (T, 3) for one contiguous period of inputs x (T, C)."""
    x = ((x - net["mean"]) / net["std"]).T.astype(np.float64)
    h = net["inp.w"][:, :, 0] @ x + net["inp.b"][:, None]
    for i, d in enumerate(net["dilations"]):
        y = np.maximum(_dilated(h, net["b{}.conv.w".format(i)], net["b{}.conv.b".format(i)], d),
                       0.0)
        h = h + net["b{}.mix.w".format(i)][:, :, 0] @ y + net["b{}.mix.b".format(i)][:, None]
    h = np.maximum(h, 0.0)
    return (net["out.w"][:, :, 0] @ h + net["out.b"][:, None]).T


def log_probs(net, X, periods):
    """Per-frame log P(H), log P(A), log P(D), rounded; zeros outside the periods."""
    out = np.zeros((len(X), 3), dtype=np.float64)
    for start, stop in periods:
        z = forward(net, X[start:stop])
        z = z - z.max(axis=1, keepdims=True)
        out[start:stop] = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return np.round(out, 5)
