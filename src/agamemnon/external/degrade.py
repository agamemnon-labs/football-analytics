"""Make clean external tracking look like uncorrected automatic tracking.

The competition ATD loses the ball in blocks, jitters, drops players for a while, swaps slot
identities and occasionally puts a player in the wrong team's slot. None of these effects is
measured on the competition match (that would be fitting to it); each is drawn from a range wide
enough to bracket what automatic tracking plausibly does, and the benchmark is run at several
ball-loss levels so a method that only works at one of them is visible.

All randomness comes from one numpy Generator seeded by (match name, level), so a degraded copy
is identical on every run and every machine.
"""

import zlib
from dataclasses import dataclass, replace

import numpy as np

from ..schema import A, H

LEVELS = {"clean": 0.0, "low": 0.20, "mid": 0.35, "high": 0.50, "severe": 0.60}


@dataclass(frozen=True)
class Profile:
    ball_missing: float            # target fraction of frames without a ball
    gap_median_s: float = 1.2      # gap lengths are log-normal around this
    gap_sigma: float = 1.0
    crowd_weight: float = 0.5      # share of the loss placed where players crowd the ball
    ball_noise_m: tuple = (0.15, 0.45)
    ball_outlier_rate: float = 0.002   # per-frame start rate of short false detections
    player_noise_m: tuple = (0.10, 0.30)
    player_dropout: tuple = (0.03, 0.12)
    swap_every_s: tuple = (20.0, 60.0)
    mislabel_rate: float = 0.0005  # per-frame start rate of a slot drifting to the wrong team
    spare_slots: int = 2


def _rng(name, level):
    return np.random.default_rng(zlib.crc32("{}|{}".format(name, level).encode()))


def _gap_mask(rng, n, target, median, sigma, weight, crowd):
    """Boolean mask of frames to hide, built from log-normal blocks until `target` is reached."""
    hidden = np.zeros(n, dtype=bool)
    if target <= 0:
        return hidden
    # Start points are drawn partly uniformly and partly in crowded frames, since occlusion is
    # what makes automatic trackers lose the ball.
    crowd_idx = np.flatnonzero(crowd)
    want = int(min(target, 0.98) * n)
    mean_len = max(1.0, median * 25 * np.exp(sigma ** 2 / 2))
    for _ in range(60):
        missing = int(hidden.sum())
        if missing >= want:
            break
        # Overlaps make each batch fall short of its nominal coverage, so draw in halving batches
        # and stop at the first one that reaches the target.
        k = max(1, int(0.5 * (want - missing) / mean_len) + 1)
        lengths = np.maximum(1, rng.lognormal(np.log(median * 25), sigma, k)).astype(int)
        use_crowd = rng.random(k) < weight
        starts = rng.integers(0, n, k)
        if len(crowd_idx):
            starts[use_crowd] = crowd_idx[rng.integers(0, len(crowd_idx), int(use_crowd.sum()))]
        edges = np.zeros(n + 1, dtype=np.int32)
        np.add.at(edges, starts, 1)
        np.add.at(edges, np.minimum(starts + lengths, n), -1)
        hidden |= np.cumsum(edges[:-1]) > 0
    return hidden


def _smooth_noise(rng, shape, sigma, corr_frames=5):
    """Temporally correlated Gaussian noise (AR(1)) with stationary std `sigma`."""
    alpha = np.exp(-1.0 / corr_frames)
    white = rng.normal(0.0, sigma * np.sqrt(1 - alpha ** 2), shape).astype(np.float32)
    out = np.empty_like(white)
    out[0] = rng.normal(0.0, sigma, shape[1:])
    for t in range(1, shape[0]):
        out[t] = alpha * out[t - 1] + white[t]
    return out


def degrade(match, level, profile=None):
    """A degraded copy of `match`. `level` is a key of LEVELS or a float ball-loss fraction."""
    if isinstance(level, str):
        target = LEVELS[level]
        tag = level
    else:
        target = float(level)
        tag = "{:.2f}".format(target)
    if tag == "clean" or (profile is None and target == 0.0):
        return match
    profile = profile or Profile(ball_missing=target)
    rng = _rng(match.name, tag)
    n, slots = match.n_frames, match.players.shape[1]
    length, width = match.pitch

    ball = match.ball.copy()
    players = match.players.copy()

    # Ball noise and short false detections.
    sigma_b = rng.uniform(*profile.ball_noise_m)
    ball += rng.normal(0.0, sigma_b, ball.shape).astype(np.float32)
    starts = np.flatnonzero(rng.random(n) < profile.ball_outlier_rate)
    for s in starts:
        k = int(rng.integers(1, 6))
        ball[s:s + k] += rng.normal(0.0, 6.0, 2).astype(np.float32)

    # Ball loss: count frames already missing towards the target.
    seen = ~np.isnan(ball[:, 0])
    already = 1.0 - seen.mean()
    if target > already:
        dist = np.linalg.norm(players - ball[:, None, :], axis=2)
        crowd = np.sum(dist < 3.0, axis=1) >= 3
        extra = (target - already) / max(seen.mean(), 1e-6)
        hide = _gap_mask(rng, n, extra, profile.gap_median_s, profile.gap_sigma,
                         profile.crowd_weight, crowd & seen)
        ball[hide] = np.nan

    # Player jitter and dropouts.
    sigma_p = rng.uniform(*profile.player_noise_m)
    players += _smooth_noise(rng, players.shape, sigma_p)
    drop = rng.uniform(*profile.player_dropout)
    for slot in range(slots):
        hide = _gap_mask(rng, n, drop, 2.0, 1.0, 0.0, np.zeros(n, bool))
        players[hide, slot] = np.nan

    # Identity swaps: at random times two slots of the same team exchange their tracks from then on.
    team = match.team.copy()
    every = rng.uniform(*profile.swap_every_s) * match.fps
    t = int(rng.exponential(every))
    while t < n:
        side = H if rng.random() < 0.5 else A
        cand = np.flatnonzero(team == side)
        if len(cand) >= 2:
            a, b = rng.choice(cand, 2, replace=False)
            players[t:, [a, b]] = players[t:, [b, a]]
        t += int(rng.exponential(every)) + 1

    # Wrong-team fragments: a player's track briefly appears in a spare slot of the other team.
    spare = np.full((n, 2 * profile.spare_slots, 2), np.nan, dtype=np.float32)
    spare_team = np.array([H] * profile.spare_slots + [A] * profile.spare_slots, dtype=np.int8)
    for s in np.flatnonzero(rng.random(n) < profile.mislabel_rate):
        src = int(rng.integers(0, slots))
        k = int(rng.integers(12, 125))
        dst_choices = np.flatnonzero(spare_team != team[src])
        dst = int(rng.choice(dst_choices))
        spare[s:s + k, dst] = players[s:s + k, src]
        players[s:s + k, src] = np.nan

    players = np.concatenate([players, spare], axis=1)
    team = np.concatenate([team, spare_team])
    np.clip(players[..., 0], -5.0, length + 5.0, out=players[..., 0])
    np.clip(players[..., 1], -5.0, width + 5.0, out=players[..., 1])

    out = replace(match, name="{}@{}".format(match.name, tag), ball=ball.astype(np.float32),
                  players=players.astype(np.float32), team=team)
    return out.validate()
