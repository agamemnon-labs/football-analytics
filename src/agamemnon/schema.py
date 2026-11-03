"""The internal match format shared by the competition loader and the external converters.

Everything downstream of a loader depends on `Match` and nothing else, so the same feature,
dead-ball and decoder code runs unchanged on the competition match and on every benchmark game.

Conventions:
  * coordinates are metres with the origin at a pitch corner: x in [0, length], y in [0, width];
  * a missing observation is NaN, for the ball and for players alike;
  * labels use the scorer's codes H=0, A=1, D=2, X=3 (see evaluation/possession_metrics.py);
  * frame spans are half-open [start, stop).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

H, A, D, X = 0, 1, 2, 3
LETTERS = np.array(["H", "A", "D", "X"])

SET_PIECES = "SET PIECES"
GOALS = "GOALS"


@dataclass
class Tag:
    """One Smart Tagging style interval. `team` is H, A or None (the file's N/A)."""

    code: str
    start: int
    stop: int
    team: Optional[int] = None
    kind: Optional[str] = None


@dataclass
class Match:
    name: str
    fps: float
    pitch: Tuple[float, float]
    ball: np.ndarray                       # (n, 2) float32
    players: np.ndarray                    # (n, slots, 2) float32
    team: np.ndarray                       # (slots,) int8, H or A per slot
    periods: List[Tuple[int, int]]         # graded span of each half, in play order
    kickoffs: List[int] = field(default_factory=list)  # first live frame of each half
    tags: List[Tag] = field(default_factory=list)
    labels: Optional[np.ndarray] = None    # (n,) int8 ground truth, external data only

    @property
    def n_frames(self):
        return len(self.ball)

    def validate(self):
        n = self.n_frames
        assert self.ball.shape == (n, 2), "ball must be (n, 2)"
        assert self.players.ndim == 3 and self.players.shape[0] == n \
            and self.players.shape[2] == 2, "players must be (n, slots, 2)"
        assert self.team.shape == (self.players.shape[1],), "one team code per slot"
        assert set(np.unique(self.team)).issubset({H, A}), "slot teams must be H or A"
        for start, stop in self.periods:
            assert 0 <= start < stop <= n, "period outside the match"
        assert all(a[1] <= b[0] for a, b in zip(self.periods, self.periods[1:])), \
            "periods must be ordered and disjoint"
        if self.labels is not None:
            assert self.labels.shape == (n,), "one label per frame"
        return self

    def graded_mask(self):
        mask = np.zeros(self.n_frames, dtype=bool)
        for start, stop in self.periods:
            mask[start:stop] = True
        return mask

    def team_slots(self, side):
        return np.flatnonzero(self.team == side)
