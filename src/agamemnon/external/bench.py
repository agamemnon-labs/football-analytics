"""External benchmark: run the pipeline on degraded external games and score with the official metric.

Scoring imports evaluation/possession_metrics.py unchanged, so a number here means exactly what it
would mean on the competition board.
"""

import os
import sys
import zlib
from dataclasses import replace

import numpy as np

from .. import deadball, decoder, features
from ..schema import A, D, GOALS, H, SET_PIECES, X, Tag
from . import CACHE_DIR, idsse, skillcorner
from .degrade import degrade
from .mapping import apply_control_rule

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

LEVELS = ("low", "mid", "high", "severe")
ATTACK = ("BUILD UP", "PROGRESSION", "CREATING CHANCES", "BALL IN FINAL THIRD")
DEFEND = ("DEFENDING IN MIDDLE THIRD", "DEFENDING IN DEFENSIVE THIRD")


def all_games():
    """(dataset, id) of every benchmark game that is present on disk."""
    games = []
    for mid in idsse.MATCH_IDS:
        try:
            idsse._file(os.path.join(idsse.EXTERNAL_DIR, "idsse"), "04_03_positions_raw_observed",
                        mid)
            games.append(("idsse", mid))
        except FileNotFoundError:
            pass
    try:
        games += [("skillcorner", mid) for mid in skillcorner.match_ids()]
    except FileNotFoundError:
        pass
    return games


def load_game(dataset, mid):
    match = idsse.load(mid) if dataset == "idsse" else skillcorner.load(mid)
    labels = apply_control_rule(match.labels, match.periods, match.fps)
    return replace(match, labels=labels)


def ground_truth(match):
    """Labels as the scorer sees them: X outside the graded periods."""
    gt = np.full(match.n_frames, X, dtype=np.int8)
    mask = match.graded_mask()
    gt[mask] = match.labels[mask]
    return gt


def synthetic_tags(match, seed_name):
    """Smart-Tagging-like intervals generated from a game's own labels, with realistic flaws.

    The intervals have the shape of the real file, which is fixed-length windows rather than
    events: a set piece opens a few seconds before the restart and lasts 35 s (5 s for a
    kick-off), a goal interval runs from the build-up until after the kick-off that follows,
    and a phase is a 10 to 30 s window that starts near a possession and often outlives it.
    On top of that restarts are often missed, throw-ins and kick-offs after goals carry no
    team, some teams are wrong, and some phases are false positives.
    """
    rng = np.random.default_rng(zlib.crc32(("tags|" + seed_name).encode()))
    fps = match.fps
    length, width = match.pitch
    lab = match.labels
    tags = []
    centre = np.array([length / 2, width / 2])

    def seconds(lo, hi):
        return int(rng.uniform(lo, hi) * fps)

    for p_start, p_stop in match.periods:
        seg = lab[p_start:p_stop]
        spells = pm.segments(seg, [(0, len(seg))])
        for i, (a, b, k) in enumerate(spells):
            a, b = a + p_start, b + p_start
            if k == D:
                continue
            other = A if k == H else H
            prev = spells[i - 1][2] if i > 0 else D
            if prev == D:
                bx, by = match.ball[a] if not np.isnan(match.ball[a, 0]) else (np.nan, np.nan)
                at_centre = np.hypot(bx - centre[0], by - centre[1]) < 3.0
                throw_in = min(by, width - by) < 2.0 if np.isfinite(by) else False
                kind = ("Kick Off (Start)" if a == p_start or i <= 1 else
                        "Kick Off (After Goal)" if at_centre else
                        "Throw-in" if throw_in else "Free Kick")
                kick_off = kind.startswith("Kick Off")
                if kind == "Kick Off (After Goal)" and i >= 2 and rng.random() < 0.9:
                    goal = spells[i - 2][1] + p_start
                    tags.append(Tag(GOALS, goal - seconds(20, 90), a + seconds(20, 100), other,
                                    None))
                if rng.random() < (0.9 if kick_off else 0.45):
                    team = k
                    if kind == "Throw-in" or kind == "Kick Off (After Goal)":
                        team = None
                    elif rng.random() < 0.08:
                        team = None
                    elif rng.random() < 0.07:
                        team = other
                    start = a - seconds(2.0, 6.0)
                    tags.append(Tag(SET_PIECES, start, start + int((5 if kick_off else 35) * fps),
                                    team, kind))
            # A phase is tagged once for the side attacking and once for the side defending,
            # over the same window; the window does not know where the possession ends.
            if b - a >= 3 * fps and rng.random() < 0.75:
                lo = a + seconds(-3, 3)
                hi = lo + (int(30 * fps) if rng.random() < 0.3 else seconds(10, 30))
                attacking = other if rng.random() < 0.1 else k
                defending = A if attacking == H else H
                tags.append(Tag(ATTACK[rng.integers(len(ATTACK))], lo, hi, attacking, None))
                tags.append(Tag(DEFEND[rng.integers(len(DEFEND))], lo, hi, defending, None))
                # Ball-in-final-third and players-in-the-box windows come without a partner
                # and start later in the attack.
                if rng.random() < 0.5:
                    lo = a + seconds(0, 10)
                    tags.append(Tag(ATTACK[rng.integers(len(ATTACK))], lo, lo + seconds(15, 40),
                                    attacking, None))
        # False positives: about one every three minutes.
        for _ in range(rng.poisson((p_stop - p_start) / (fps * 180))):
            s = int(rng.integers(p_start, p_stop))
            tags.append(Tag(ATTACK[rng.integers(len(ATTACK))], s, s + seconds(10, 30),
                            H if rng.random() < 0.5 else A, None))
    return sorted(tags, key=lambda t: t.start)


def prepare(dataset, mid, level, prm, tags=True):
    """(degraded match, features, tag features, ground truth) for one benchmark cell."""
    match = load_game(dataset, mid)
    gt = ground_truth(match)
    if tags:
        match = replace(match, tags=synthetic_tags(match, match.name))
    noisy = degrade(match, level)
    noisy = replace(noisy, labels=None)
    F = features.compute(noisy, prm)
    T = deadball.tag_features(noisy, prm)
    return noisy, F, T, gt


def score(gt, pred):
    result = pm.evaluate(gt, pred)
    return {
        "spell_f1": result["spell_f1_mean"],
        "macro_f1": result["macro_f1"],
        "n_pred": result["n_pred_spells"],
        "n_gt": result["n_gt_spells"],
        "dur_pred": result["mean_spell_duration_s"],
        "dur_gt": result["gt_mean_spell_duration_s"],
        "alive_dead_f1": result["alive_dead_f1"],
        "team_acc": result["team_acc_both_alive"],
    }


def cell_cache_path(dataset, mid, level, tags):
    return os.path.join(CACHE_DIR, "cells", "{}-{}@{}{}.npz".format(
        dataset, mid, level, "" if tags else "-notags"))


def build_cell(dataset, mid, level, prm, force=False):
    """Compute and store the design matrices of one (game, level) cell; returns its path."""
    path = cell_cache_path(dataset, mid, level, True)
    if os.path.isfile(path) and not force:
        return path
    match, F, T, gt = prepare(dataset, mid, level, prm)
    Xt = decoder.team_design(F, match, prm, T)
    Xd = deadball.dead_design(F, match, prm, T)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, Xt=Xt, Xd=Xd, gt=gt,
                        touch_home=F["touch_home"].astype(np.int8),
                        touch_away=F["touch_away"].astype(np.int8),
                        periods=np.array(match.periods), kickoffs=np.array(match.kickoffs),
                        fps=np.array(match.fps))
    return path


class Cell:
    """Cached design matrices of one benchmark (game, level)."""

    def __init__(self, dataset, mid, level):
        self.dataset, self.mid, self.level = dataset, mid, level
        with np.load(cell_cache_path(dataset, mid, level, True)) as data:
            self.Xt = data["Xt"]
            self.Xd = data["Xd"]
            self.gt = data["gt"]
            self.touch_home = data["touch_home"]
            self.touch_away = data["touch_away"]
            self.periods = [tuple(int(v) for v in p) for p in data["periods"]]
            self.kickoffs = [int(v) for v in data["kickoffs"]]
            self.fps = float(data["fps"])

    @property
    def game(self):
        return "{}-{}".format(self.dataset, self.mid)

    def designs(self, tags=True):
        if tags:
            return self.Xt, self.Xd
        Xt = self.Xt.copy()
        Xd = self.Xd.copy()
        Xt[:, decoder.TEAM_COLUMNS.index("tag")] = 0
        for name in ("setpiece", "post_goal"):
            Xd[:, deadball.DEAD_COLUMNS.index(name)] = 0
        return Xt, Xd

    def decode(self, prm, tags=True, logp=None):
        Xt, Xd = self.designs(tags)
        return decoder.decode_design(Xt, Xd, self.touch_home, self.touch_away, self.periods,
                                     self.kickoffs, self.fps, prm, logp)

    def oof_logp(self, tags=True):
        """Out-of-fold network log-probabilities written by scripts/train_tcn.py, if present."""
        path = os.path.join(CACHE_DIR, os.environ.get("AGAMEMNON_OOF", "tcn_oof"), "{}-{}@{}.npz".format(
            self.dataset, self.mid, self.level))
        if not os.path.isfile(path):
            return None
        with np.load(path) as data:
            return data["logp" if tags else "logp_notags"]

    def score(self, prm, tags=True):
        logp = self.oof_logp(tags) if getattr(prm, "EMISSION", "logistic") == "network" else None
        return score(self.gt, self.decode(prm, tags, logp))
