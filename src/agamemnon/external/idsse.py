"""IDSSE (DFL Bundesliga 2022/23 positions + events, CC BY 4.0) to `Match`.

The DFL position feed carries BallStatus (0 dead, 1 alive) and BallPossession (1 home, 2 away) on
every ball frame. That is the same kind of signal the competition ground truth is derived from --
corrected tracking that marks the ball dead or names the owning team -- which is why this dataset
is the primary benchmark. See docs/POSSESSION_RULES.md for how the two definitions line up.

Parsed with lxml.iterparse one FrameSet at a time, so a 400 MB file never sits in memory as a tree.
"""

import glob
import os

import numpy as np
from lxml import etree

from ..schema import A, D, H, Match
from . import EXTERNAL_DIR, cache

MATCH_IDS = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
FPS = 25.0
SECTIONS = ("firstHalf", "secondHalf")


def _file(root, kind, match_id):
    hits = glob.glob(os.path.join(root, "DFL_{}_*DFL-MAT-{}.xml".format(kind, match_id)))
    if len(hits) != 1:
        raise FileNotFoundError("IDSSE {} file for {} not found under {}".format(kind, match_id,
                                                                                root))
    return hits[0]


def _teams(info_path):
    root = etree.parse(info_path).getroot()
    general = root.find("MatchInformation").find("General")
    home_id = general.get("HomeTeamId")
    away_id = general.get("GuestTeamId") or general.get("AwayTeamId")
    env = root.find("MatchInformation").find("Environment")
    pitch = (float(env.get("PitchX")), float(env.get("PitchY")))
    persons = {}
    for team in root.iter("Team"):
        side = H if team.get("TeamId") == home_id else A if team.get("TeamId") == away_id else None
        if side is None:
            continue
        for player in team.iter("Player"):
            persons[player.get("PersonId")] = side
    return persons, pitch


def convert(match_id, root=None):
    root = root or os.path.join(EXTERNAL_DIR, "idsse")
    persons, pitch = _teams(_file(root, "02_01_matchinformation", match_id))
    roster = sorted(persons, key=lambda p: (persons[p] != H, p))
    slot_of = {p: i for i, p in enumerate(roster)}
    team = np.array([persons[p] for p in roster], dtype=np.int8)

    ball = {}
    status = {}
    owner = {}
    tracks = {s: [] for s in SECTIONS}
    for _, fs in etree.iterparse(_file(root, "04_03_positions_raw_observed", match_id),
                                 tag="FrameSet"):
        section = fs.get("GameSection")
        if section not in SECTIONS:
            fs.clear()
            continue
        frames = fs.findall("Frame")
        n = np.fromiter((int(f.get("N")) for f in frames), dtype=np.int64, count=len(frames))
        x = np.fromiter((float(f.get("X")) for f in frames), dtype=np.float32, count=len(frames))
        y = np.fromiter((float(f.get("Y")) for f in frames), dtype=np.float32, count=len(frames))
        if fs.get("TeamId", "").lower() == "ball":
            ball[section] = (n, x, y)
            status[section] = np.fromiter((int(float(f.get("BallStatus"))) for f in frames),
                                          dtype=np.int8, count=len(frames))
            owner[section] = np.fromiter((int(float(f.get("BallPossession") or 0))
                                          for f in frames), dtype=np.int8, count=len(frames))
        else:
            slot = slot_of.get(fs.get("PersonId"))
            if slot is not None:
                tracks[section].append((slot, n, x, y))
        fs.clear()
        while fs.getprevious() is not None:
            del fs.getparent()[0]

    half_length, half_width = pitch[0] / 2, pitch[1] / 2
    out_ball, out_players, out_labels, periods, kickoffs = [], [], [], [], []
    offset = 0
    for section in SECTIONS:
        n, x, y = ball[section]
        first = int(n[0])
        length = int(n[-1]) - first + 1
        b = np.full((length, 2), np.nan, dtype=np.float32)
        b[n - first] = np.stack([x + half_length, y + half_width], axis=1)
        p = np.full((length, len(roster), 2), np.nan, dtype=np.float32)
        for slot, pn, px, py in tracks[section]:
            keep = (pn >= first) & (pn < first + length)
            p[pn[keep] - first, slot] = np.stack([px[keep] + half_length,
                                                  py[keep] + half_width], axis=1)
        lab = np.full(length, D, dtype=np.int8)
        alive = status[section] == 1
        own = owner[section]
        idx = n - first
        lab[idx[alive & (own == 1)]] = H
        lab[idx[alive & (own == 2)]] = A
        out_ball.append(b)
        out_players.append(p)
        out_labels.append(lab)
        periods.append((offset, offset + length))
        live = np.flatnonzero(lab != D)
        kickoffs.append(offset + int(live[0]) if len(live) else offset)
        offset += length

    match = Match(
        name="idsse-{}".format(match_id), fps=FPS, pitch=pitch,
        ball=np.concatenate(out_ball), players=np.concatenate(out_players), team=team,
        periods=periods, kickoffs=kickoffs, labels=np.concatenate(out_labels),
    )
    return match.validate()


def load(match_id, root=None, refresh=False):
    name = "idsse-{}".format(match_id)
    match = None if refresh else cache.load(name)
    if match is None:
        match = convert(match_id, root)
        cache.save(match)
    return match
