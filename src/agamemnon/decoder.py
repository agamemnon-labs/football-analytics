"""Three-state (H, A, D) decoding of per-frame evidence with Viterbi.

Per-frame evidence comes from two small logistic models: one for "home rather than away" on live
frames, one for "dead rather than alive". Their log-odds are tempered (consecutive frames are far
from independent) and used as emission scores; switching state costs a fixed penalty, which is what
stops a possession from flickering. All weights are fitted on external games only.
"""

import numpy as np

from . import deadball
from .schema import A, D, H

TEAM_COLUMNS = ("bias", "dist", "close", "touch", "next_touch", "spread", "ball_field",
                "centroid_field", "tag")

def _decayed_last(events_home, events_away, fps, memory_s):
    """+1/-1 for the side that touched last, fading with time since that touch."""
    idx = np.arange(len(events_home))
    ev = np.where(events_home > 0, 1, np.where(events_away > 0, -1, 0))
    t_last = np.maximum.accumulate(np.where(ev != 0, idx, -1))
    sign = np.where(t_last >= 0, ev[np.maximum(t_last, 0)], 0).astype(np.float32)
    last_t = np.where(t_last >= 0, t_last, -10 ** 9)
    return sign * np.exp(-(idx - last_t) / (memory_s * fps)).astype(np.float32)


def team_design(F, match, prm, tags=None):
    have = F["have"] > 0
    dh = np.nan_to_num(F["d_home"], nan=np.inf)
    da = np.nan_to_num(F["d_away"], nan=np.inf)
    with np.errstate(invalid="ignore"):
        dist = np.clip((da - dh) / prm.TEAM_DIST_SCALE, -prm.TEAM_DIST_CLIP, prm.TEAM_DIST_CLIP)
    dist = np.where(have & np.isfinite(dist), dist, 0.0)
    close = ((dh < prm.CLOSE_RADIUS).astype(np.float32) - (da < prm.CLOSE_RADIUS)) * have
    touch = _decayed_last(F["touch_home"], F["touch_away"], match.fps, prm.TOUCH_MEMORY_S)
    # The same, looking forward: a pass in flight is going to whoever touches it next.
    next_touch = _decayed_last(F["touch_home"][::-1], F["touch_away"][::-1], match.fps,
                               prm.TOUCH_MEMORY_S)[::-1]
    spread = np.nan_to_num(np.clip((F["spread_home"] - F["spread_away"]) / prm.SPREAD_SCALE,
                                   -3, 3))
    tag_team = tags[0] if tags is not None else np.zeros(match.n_frames, np.float32)
    return np.column_stack([np.ones(match.n_frames), dist, close, touch, next_touch, spread,
                            F["ball_field"], F["centroid_field"], tag_team]).astype(np.float32)


def logit(X, coef):
    """X @ coef, accumulated column by column in float64 and rounded.

    A BLAS dot product may sum in a CPU-dependent order; element-wise accumulation does not, and
    the rounding keeps last-bit noise from tipping a Viterbi tie, so the output file is identical
    on every machine.
    """
    acc = np.zeros(len(X), dtype=np.float64)
    for j, c in enumerate(coef):
        acc += X[:, j].astype(np.float64) * float(c)
    return np.round(acc, 6)


def viterbi(log_e, costs, segments, forced_dead=None):
    """Most likely state path. log_e (n, 3) emission scores, costs[i][j] transition penalty.

    Decoded independently inside each [start, stop) segment; outside every segment the output is D.
    """
    n = len(log_e)
    out = np.full(n, D, dtype=np.int8)
    c = [[float(costs[i][j]) for j in range(3)] for i in range(3)]
    for start, stop in segments:
        e = log_e[start:stop].astype(np.float64)
        if forced_dead is not None:
            fd = forced_dead[start:stop]
            e[fd, 0] = e[fd, 1] = -1e9
        m = stop - start
        back = np.empty((m, 3), dtype=np.int8)
        s0, s1, s2 = e[0, 0], e[0, 1], e[0, 2]
        back[0] = (0, 1, 2)
        rows = e.tolist()
        c00, c01, c02 = c[0]
        c10, c11, c12 = c[1]
        c20, c21, c22 = c[2]
        bk = [None] * m
        for t in range(1, m):
            r = rows[t]
            a0, a1, a2 = s0 - c00, s1 - c10, s2 - c20
            if a0 >= a1 and a0 >= a2:
                n0, b0 = a0, 0
            elif a1 >= a2:
                n0, b0 = a1, 1
            else:
                n0, b0 = a2, 2
            a0, a1, a2 = s0 - c01, s1 - c11, s2 - c21
            if a1 >= a0 and a1 >= a2:
                n1, b1 = a1, 1
            elif a0 >= a2:
                n1, b1 = a0, 0
            else:
                n1, b1 = a2, 2
            a0, a1, a2 = s0 - c02, s1 - c12, s2 - c22
            if a2 >= a0 and a2 >= a1:
                n2, b2 = a2, 2
            elif a0 >= a1:
                n2, b2 = a0, 0
            else:
                n2, b2 = a1, 1
            s0, s1, s2 = n0 + r[0], n1 + r[1], n2 + r[2]
            bk[t] = (b0, b1, b2)
        state = int(np.argmax([s0, s1, s2]))
        path = np.empty(m, dtype=np.int8)
        for t in range(m - 1, 0, -1):
            path[t] = state
            state = bk[t][state]
        path[0] = state
        out[start:stop] = path
    return out


def snap_handovers(labels, touch_home, touch_away, window, segments):
    """Move each H<->A hand-over to the taker's first touch within +/- window frames.

    The decoder's switch point lags or leads the real one by however long the evidence takes to
    outweigh the switch penalty; the first touch by the team that ends up with the ball is the
    better estimate of when control changed.
    """
    out = labels.copy()
    if window <= 0:
        return out
    for start, stop in segments:
        seg = out[start:stop]
        change = np.flatnonzero((seg[1:] != seg[:-1]) & (seg[1:] != D) & (seg[:-1] != D)) + 1
        for c in change:
            new = seg[c]
            old = seg[c - 1]
            touches = touch_home if new == H else touch_away
            lo = max(1, c - window)
            hi = min(len(seg) - 1, c + window)
            # Stay inside the old and new spells so snapping never crosses another boundary.
            while lo < c and seg[lo - 1] != old:
                lo += 1
            while hi > c and seg[hi] != new:
                hi -= 1
            cand = np.flatnonzero(touches[start + lo:start + hi + 1] > 0)
            if not len(cand):
                continue
            t = lo + int(cand[0])
            if t < c:
                seg[t:c] = new
            elif t > c:
                seg[c:t] = old
    return out


def decode_design(Xt, Xd, touch_home, touch_away, periods, kickoffs, fps, prm):
    """Decode from precomputed design matrices; the benchmark caches these per game."""
    team = logit(Xt, prm.TEAM_COEF)
    dead = logit(Xd, prm.DEAD_COEF)
    prior = np.log(prm.DEAD_PRIOR / (1 - prm.DEAD_PRIOR))
    t = prm.TEAM_TEMPER * team
    d = prm.DEAD_TEMPER * (dead - prior)
    log_e = np.column_stack([t / 2, -t / 2, d]).astype(np.float32)
    costs = [[0.0, prm.SWITCH_COST, prm.END_COST],
             [prm.SWITCH_COST, 0.0, prm.END_COST],
             [prm.START_COST, prm.START_COST, 0.0]]
    forced = np.zeros(len(Xt), dtype=bool)
    for (p0, _), k in zip(periods, kickoffs or [p[0] for p in periods]):
        forced[p0:k] = True
    labels = viterbi(log_e, costs, periods, forced)
    return snap_handovers(labels, touch_home, touch_away, int(prm.SNAP_WINDOW_S * fps), periods)


def decode(F, match, prm, tags=None):
    """H/A/D per frame inside match.periods; D elsewhere (labels.py turns that into X)."""
    tags = deadball.tag_features(match, prm) if tags is None else tags
    return decode_design(team_design(F, match, prm, tags), deadball.dead_design(F, match, prm, tags),
                         F["touch_home"], F["touch_away"], match.periods, match.kickoffs,
                         match.fps, prm)
