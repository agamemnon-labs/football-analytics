import numpy as np

from agamemnon.external.degrade import degrade
from agamemnon.schema import A, H, Match


def synthetic(n=6000, slots=8, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 25.0
    ball = np.stack([52 + 30 * np.sin(t / 7), 34 + 20 * np.cos(t / 5)], 1).astype(np.float32)
    players = (ball[:, None, :] + rng.normal(0, 8, (1, slots, 2))).astype(np.float32)
    team = np.array([H, A] * (slots // 2), dtype=np.int8)
    return Match("syn", 25.0, (105.0, 68.0), ball, players, team, [(0, n // 2), (n // 2, n)],
                 [0, n // 2], labels=np.zeros(n, np.int8)).validate()


def test_deterministic():
    m = synthetic()
    a, b = degrade(m, "mid"), degrade(m, "mid")
    assert np.array_equal(a.ball, b.ball, equal_nan=True)
    assert np.array_equal(a.players, b.players, equal_nan=True)


def test_ball_loss_reaches_target():
    m = synthetic()
    for level, target in (("low", 0.2), ("high", 0.5)):
        missing = np.isnan(degrade(m, level).ball[:, 0]).mean()
        assert target - 0.02 <= missing <= target + 0.12


def test_labels_and_periods_untouched():
    m = synthetic()
    d = degrade(m, "severe")
    assert d.periods == m.periods and np.array_equal(d.labels, m.labels)
    assert d.players.shape[1] > m.players.shape[1]
