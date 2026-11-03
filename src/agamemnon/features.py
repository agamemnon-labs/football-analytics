"""Per-frame features from a `Match`, identical for the competition game and external games.

Nothing here looks at labels. Every threshold is a physical or geometric constant passed in from
priors.py, so the same numbers apply to every source.
"""

import warnings

import numpy as np

from .schema import A, H

NAN = np.float32(np.nan)


def _runs(mask):
    """[start, stop) spans where mask is True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def clean_ball(ball, fps, max_speed):
    """Drop ball samples that imply an impossible speed to both neighbours (false detections)."""
    ball = ball.copy()
    step = max_speed / fps
    for _ in range(2):
        prev = np.vstack([ball[:1] * NAN, ball[:-1]])
        nxt = np.vstack([ball[1:], ball[:1] * NAN])
        jump_prev = np.linalg.norm(ball - prev, axis=1)
        jump_next = np.linalg.norm(ball - nxt, axis=1)
        bad = (jump_prev > step * 2) & (jump_next > step * 2)
        ball[bad] = NAN
    return ball


def interpolate(values, max_gap, periods):
    """Linear fill of NaN gaps no longer than `max_gap` frames, never across a period edge.

    Returns (filled, age): age is frames since the ball was last actually seen (0 when seen,
    large inside unfilled gaps), which downstream code uses to fade evidence.
    """
    out = values.copy()
    n = len(values)
    seen = ~np.isnan(values[:, 0])
    age = np.full(n, 10 ** 6, dtype=np.int32)
    for p_start, p_stop in periods:
        s = seen[p_start:p_stop]
        idx = np.flatnonzero(s)
        if not len(idx):
            continue
        local = np.arange(p_stop - p_start)
        last = np.maximum.accumulate(np.where(s, local, -1))
        a = np.where(last >= 0, local - last, 10 ** 6)
        age[p_start:p_stop] = a
        starts, stops = _runs(~s)
        for g0, g1 in zip(starts, stops):
            if g0 == 0 or g1 == len(s) or g1 - g0 > max_gap:
                continue
            left = values[p_start + g0 - 1]
            right = values[p_start + g1]
            w = (np.arange(1, g1 - g0 + 1) / (g1 - g0 + 1))[:, None].astype(np.float32)
            out[p_start + g0:p_start + g1] = left * (1 - w) + right * w
    return out, age


def smooth(values, window):
    """Centred moving average along axis 0 that ignores NaN; NaN where nothing is observed."""
    if window <= 1:
        return values.copy()
    valid = ~np.isnan(values)
    filled = np.where(valid, values, 0.0).astype(np.float64)
    pad = window // 2

    def conv(a):
        cs = np.cumsum(np.concatenate([np.zeros((1,) + a.shape[1:]), a]), axis=0)
        hi = np.minimum(np.arange(len(a)) + pad + 1, len(a))
        lo = np.maximum(np.arange(len(a)) - pad, 0)
        return cs[hi] - cs[lo]

    total = conv(filled)
    count = conv(valid.astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / count
    out[count == 0] = np.nan
    return out.astype(np.float32)


def velocity(positions, fps, half_window):
    """Central-difference velocity (m/s) over +/- half_window frames."""
    k = half_window
    v = np.full_like(positions, np.nan)
    v[k:-k] = (positions[2 * k:] - positions[:-2 * k]) * (fps / (2.0 * k))
    return v


def rolling_mean(x, window):
    return smooth(x[:, None], window)[:, 0]


def compute(match, prm):
    """Dict of per-frame float32 arrays. `prm` is the priors namespace (see priors.py)."""
    fps = match.fps
    n = match.n_frames
    length, width = match.pitch

    raw = clean_ball(match.ball, fps, prm.BALL_MAX_SPEED)
    seen = ~np.isnan(raw[:, 0])
    ball, age = interpolate(raw, int(prm.BALL_INTERP_MAX_S * fps), match.periods)
    have = ~np.isnan(ball[:, 0])

    ball_s = smooth(ball, prm.BALL_SMOOTH_FRAMES)
    bv = velocity(ball_s, fps, prm.VEL_HALF_WINDOW)
    ball_speed = np.linalg.norm(bv, axis=1)

    players = match.players
    rel = players - ball[:, None, :]
    dist = np.sqrt(np.sum(rel ** 2, axis=2))
    home = match.team == H
    away = match.team == A
    with np.errstate(invalid="ignore"):
        d_home = np.where(np.all(np.isnan(dist[:, home]), axis=1), np.inf,
                          np.nanmin(np.where(np.isnan(dist[:, home]), np.inf, dist[:, home]), axis=1))
        d_away = np.where(np.all(np.isnan(dist[:, away]), axis=1), np.inf,
                          np.nanmin(np.where(np.isnan(dist[:, away]), np.inf, dist[:, away]), axis=1))
    d_home = np.where(have, d_home, np.nan).astype(np.float32)
    d_away = np.where(have, d_away, np.nan).astype(np.float32)

    # Collective movement: median player speed is robust to the spikes slot swaps produce.
    p_s = smooth(players.reshape(n, -1), prm.PLAYER_SMOOTH_FRAMES).reshape(players.shape)
    pv = velocity(p_s, fps, prm.VEL_HALF_WINDOW)
    p_speed = np.linalg.norm(pv, axis=2)
    p_speed[p_speed > prm.PLAYER_MAX_SPEED] = np.nan
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med_speed = np.nanmedian(p_speed, axis=1)
        q75_speed = np.nanpercentile(p_speed, 75, axis=1)
    med_speed_1s = rolling_mean(med_speed, int(fps))
    med_speed_3s = rolling_mean(med_speed, int(3 * fps))
    q75_speed_3s = rolling_mean(q75_speed, int(3 * fps))

    # Team shape: the side in possession spreads out, the side without it compacts.
    def spread(mask):
        pts = players[:, mask, :]
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sx = np.nanstd(pts[..., 0], axis=1)
            sy = np.nanstd(pts[..., 1], axis=1)
        return rolling_mean(np.sqrt(sx ** 2 + sy ** 2), int(fps))

    spread_home = spread(home)
    spread_away = spread(away)

    # Kinematic touches: an abrupt change of ball velocity with a player close by.
    k = prm.TOUCH_HALF_WINDOW
    dv = np.full(n, np.nan, dtype=np.float32)
    v_before = np.full_like(bv, np.nan)
    v_after = np.full_like(bv, np.nan)
    v_before[k:] = (ball_s[k:] - ball_s[:-k]) * (fps / k)
    v_after[:-k] = (ball_s[k:] - ball_s[:-k]) * (fps / k)
    dv[:] = np.linalg.norm(v_after - v_before, axis=1)
    seen_around = seen.copy()
    for shift in range(1, k + 1):
        seen_around[shift:] &= seen[:-shift]
        seen_around[:-shift] &= seen[shift:]
    is_touch = seen_around & (dv > prm.TOUCH_DV)
    # Keep one frame per touch: the local maximum of dv.
    peak = np.zeros(n, dtype=bool)
    starts, stops = _runs(is_touch)
    dv0 = np.nan_to_num(dv)
    for s, e in zip(starts, stops):
        peak[s + int(np.argmax(dv0[s:e]))] = True
    near_home = np.nan_to_num(d_home, nan=np.inf) < prm.TOUCH_RADIUS
    near_away = np.nan_to_num(d_away, nan=np.inf) < prm.TOUCH_RADIUS
    closer_home = np.nan_to_num(d_home, nan=np.inf) <= np.nan_to_num(d_away, nan=np.inf)
    touch_home = peak & near_home & (closer_home | ~near_away)
    touch_away = peak & near_away & (~closer_home | ~near_home)

    # Out of bounds: the ball seen beyond a line, and how long ago that last happened.
    m = prm.OOB_MARGIN
    bx, by = raw[:, 0], raw[:, 1]
    with np.errstate(invalid="ignore"):
        oob = seen & ((bx < -m) | (bx > length + m) | (by < -m) | (by > width + m))
        far_oob = seen & ((bx < -prm.FAR_OOB) | (bx > length + prm.FAR_OOB)
                          | (by < -prm.FAR_OOB) | (by > width + prm.FAR_OOB))
    # Last seen position, carried through gaps: a ball that vanished outside the lines is dead.
    last_idx = np.maximum.accumulate(np.where(seen, np.arange(n), -1))
    last_oob = np.where(last_idx >= 0, oob[np.maximum(last_idx, 0)], False)
    last_x = np.where(last_idx >= 0, bx[np.maximum(last_idx, 0)], np.nan)
    last_y = np.where(last_idx >= 0, by[np.maximum(last_idx, 0)], np.nan)
    edge_dist = np.minimum.reduce([last_x, length - last_x, last_y, width - last_y])

    ball_still = rolling_mean(np.where(seen, (ball_speed < prm.BALL_STILL_SPEED).astype(np.float32),
                                       np.nan), int(fps))

    # Attacking direction per period from where each team stands on average: +1 when home attacks
    # towards x = length. No labels involved, and teams change ends only at half time.
    attack_home = np.zeros(n, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for p0, p1 in match.periods:
            hx = np.nanmean(players[p0:p1][:, home, 0])
            ax = np.nanmean(players[p0:p1][:, away, 0])
            attack_home[p0:p1] = 1.0 if hx < ax else -1.0
        centroid_x = np.nanmean(players[..., 0], axis=1)
        # Share of players in their own half; close to 1 only while lining up for a kick-off.
        own = np.where(home[None, :], (players[..., 0] - length / 2) * attack_home[:, None] < 0,
                       (players[..., 0] - length / 2) * attack_home[:, None] > 0)
        present = ~np.isnan(players[..., 0])
        own_half = np.where(present.sum(1) > 0, (own & present).sum(1) / np.maximum(present.sum(1), 1),
                            np.nan)
    centre_dist = np.hypot(ball[:, 0] - length / 2, ball[:, 1] - width / 2)
    ball_field = np.where(have, (ball[:, 0] - length / 2) / (length / 2), 0.0) * attack_home
    centroid_field = np.nan_to_num((centroid_x - length / 2) / (length / 2)) * attack_home

    return {
        "attack_home": attack_home,
        "own_half": rolling_mean(own_half.astype(np.float32), int(fps)),
        "centre_dist": centre_dist.astype(np.float32),
        "ball_field": ball_field.astype(np.float32),
        "centroid_field": rolling_mean(centroid_field.astype(np.float32), int(fps)),
        "seen": seen.astype(np.float32),
        "have": have.astype(np.float32),
        "age_s": (np.minimum(age, 10 ** 5) / fps).astype(np.float32),
        "ball_x": ball[:, 0], "ball_y": ball[:, 1],
        "ball_speed": ball_speed.astype(np.float32),
        "ball_still": ball_still,
        "d_home": d_home, "d_away": d_away,
        "med_speed": med_speed.astype(np.float32),
        "med_speed_1s": med_speed_1s, "med_speed_3s": med_speed_3s,
        "q75_speed_3s": q75_speed_3s,
        "spread_home": spread_home, "spread_away": spread_away,
        "touch_home": touch_home.astype(np.float32),
        "touch_away": touch_away.astype(np.float32),
        "oob": oob.astype(np.float32), "far_oob": far_oob.astype(np.float32),
        "last_oob": last_oob.astype(np.float32),
        "edge_dist": edge_dist.astype(np.float32),
        "n_players": np.sum(~np.isnan(players[..., 0]), axis=1).astype(np.float32),
    }
