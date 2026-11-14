"""The committed network and the decoder must give the same labels on every machine.

The submission is reproducible only if BLAS kernels, thread counts and maths libraries that differ
in the last bit still lead to the same file. This runs the real weights and the locked decoder on
generated inputs as long as a match and compares the labels with a digest recorded once. CI runs it
on Linux, Windows and macOS.
"""

import hashlib

import numpy as np

from agamemnon import decoder, priors, tcn

FRAMES = 150000
PERIODS = [(0, 75000), (75000, 150000)]
EXPECTED = "8f914fc79b016943b2c7482940d23796d2381dcedf2acfd139643dd19595f475"


def generated_inputs(net):
    """Feature-like inputs built from integers only, so they are bit-identical everywhere."""
    rng = np.random.default_rng(2027)
    channels = len(net["mean"])
    # Slow drift held for a second at a time, plus per-frame jitter.
    slow = rng.integers(-2000, 2001, size=(FRAMES // 25, channels)).repeat(25, axis=0)
    fast = rng.integers(-300, 301, size=(FRAMES, channels))
    z = (slow + fast) / 1000.0
    touches = rng.integers(0, 400, size=(2, FRAMES)) == 0
    return net["mean"] + net["std"] * z, touches[0], touches[1] & ~touches[0]


def test_labels_do_not_depend_on_the_machine():
    net = tcn.load()
    x, touch_home, touch_away = generated_inputs(net)
    logp = tcn.log_probs(net, x, PERIODS)
    labels = decoder.decode_emissions(decoder.network_emissions(logp, priors), touch_home,
                                      touch_away, PERIODS, [1000, 76000], 25.0, priors)
    # All three states and plenty of hand-overs, or the digest would prove very little.
    counts = np.bincount(labels, minlength=3)
    assert counts.min() > FRAMES // 50
    assert np.count_nonzero(np.diff(labels)) > 200
    assert hashlib.sha256(labels.astype(np.int8).tobytes()).hexdigest() == EXPECTED
