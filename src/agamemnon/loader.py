"""Load the competition match into the internal `Match` format.

The loader never prints or raises with file content. Every check is on an aggregate (a count, a
fraction, a column shape), and the expected values come from DATA.md, so a wrong assumption about
the bundle fails loudly here instead of quietly corrupting every frame downstream.
"""

import io
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .schema import A, H, Match, Tag

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import games  # noqa: E402

GAME = "BAR-ATH"

# Published in DATA.md. Tolerances are loose enough to survive rounding in the document and tight
# enough to catch a wrong column or a shifted frame index.
EXPECTED_FRAMES = 199319
EXPECTED_SLOTS = 46
EXPECTED_BALL_SEEN = 113488
EXPECTED_BALL_SEEN_GRADED = 94214
BALL_SEEN_TOLERANCE = 0.002
EXPECTED_MEAN_TRACKED_GRADED = (19.5, 21.5)
EXPECTED_TAG_CODES = {
    "PLAYERS IN THE BOX": 97, "BALL IN FINAL THIRD": 93, "BALL IN THE BOX": 72,
    "PROGRESSION": 66, "DEFENDING IN MIDDLE THIRD": 66, "SET PIECES": 59,
    "CREATING CHANCES": 30, "DEFENDING IN DEFENSIVE THIRD": 30, "LONG BALL": 29,
    "ATTACKING TRANSITION": 24, "DEFENSIVE TRANSITION": 24, "BUILD UP": 22,
    "DEFENDING IN ATTACKING THIRD": 22, "GOALS": 4,
}
EXPECTED_TAG_TEAMS = {"Barcelona": 314, "Atletic Club": 288, "N/A": 36}
EXPECTED_TAG_TYPES = {"Throw-in": 26, "Corner Kick": 12, "Goal Kick": 10, "Free Kick": 5,
                      "Kick Off (After Goal)": 4, "Kick Off (Start)": 2}
# Normalised coordinates legitimately leave [0, 1] a little (ball over the touchline, projection
# error), but anything far outside means the channels are not what we think they are.
COORD_HARD_BOUNDS = (-1.0, 2.0)
PLAYER_INSIDE_MIN_FRACTION = 0.99
# Second half is graded from 110808 but kicks off at 113742 (FAQ.md); the first half is graded from
# its kickoff.
KICKOFF_FRAMES = (11185, 113742)


class BundleError(AssertionError):
    pass


def _check(condition, message):
    if not condition:
        raise BundleError(message)


def read_meta(meta_path, expected_slots=EXPECTED_SLOTS):
    """(home_name, away_name, slot_team) from the ATD metadata. Home is the first team listed."""
    root = ET.parse(meta_path).getroot()
    teams = [(team.get("id"), team.findtext("Name")) for team in root.iter("Team")]
    _check(len(teams) == 2, "ATD metadata: expected 2 teams, found {}".format(len(teams)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    _check(len(slots) == expected_slots,
           "ATD metadata: expected {} player slots, found {}".format(expected_slots, len(slots)))
    known = {teams[0][0], teams[1][0]}
    _check(set(slots) <= known, "ATD metadata: a slot refers to an unknown team id")
    slot_team = np.array([H if s == teams[0][0] else A for s in slots], dtype=np.int8)
    return teams[0][1], teams[1][1], slot_team


def read_raw(raw_path, n_slots=EXPECTED_SLOTS, expected_frames=EXPECTED_FRAMES):
    """(ball (n,2), players (n,slots,2)) in normalised units from the raw EPTS text.

    Line layout per the metadata's DataFormatSpecification: "frame:slot;slot;...:ball", every
    group "x,y,s". Speeds are dropped; velocities are recomputed from positions so the competition
    match and the external games go through identical arithmetic.
    """
    with open(raw_path, encoding="ascii", errors="replace") as handle:
        text = handle.read()
    n_cols = 1 + 3 * n_slots + 3
    try:
        table = pd.read_csv(io.StringIO(text.replace(":", ",").replace(";", ",")),
                            header=None, dtype=np.float64, engine="c")
    except Exception as error:
        # pandas can quote the offending token; report the kind of failure only.
        raise BundleError("ATD raw: parse failed ({}); expected {} numeric fields per line"
                          .format(type(error).__name__, n_cols)) from None
    _check(table.shape[1] == n_cols,
           "ATD raw: {} fields per line, expected {}".format(table.shape[1], n_cols))
    values = table.to_numpy()
    frame_id = values[:, 0]
    _check(len(values) == expected_frames,
           "ATD raw: {} lines, expected {}".format(len(values), expected_frames))
    _check(np.array_equal(frame_id, np.arange(1, expected_frames + 1)),
           "ATD raw: frame ids are not 1..{} in order".format(expected_frames))
    slots = values[:, 1:1 + 3 * n_slots].reshape(len(values), n_slots, 3)[:, :, :2]
    ball = values[:, 1 + 3 * n_slots:1 + 3 * n_slots + 2]
    return ball, slots


def read_tags(tag_path, home_name, away_name, fps, check_counts=True):
    root = ET.parse(tag_path).getroot()
    tags, codes, teams, kinds = [], {}, {}, {}
    for inst in root.iter("instance"):
        code = (inst.findtext("code") or "").strip().upper()
        labels = {label.findtext("group"): label.findtext("text")
                  for label in inst.findall("label")}
        team_text = labels.get("Team")
        kind = labels.get("Type")
        codes[code] = codes.get(code, 0) + 1
        if team_text is not None:
            teams[team_text] = teams.get(team_text, 0) + 1
        if kind is not None:
            kinds[kind] = kinds.get(kind, 0) + 1
        try:
            start = int(round(float(inst.findtext("start")) * fps))
            stop = int(round(float(inst.findtext("end")) * fps))
        except (TypeError, ValueError):
            raise BundleError("Smart Tagging: an instance has a non-numeric start/end") from None
        team = {home_name.lower(): H, away_name.lower(): A}.get((team_text or "").strip().lower())
        tags.append(Tag(code=code, start=start, stop=stop, team=team, kind=kind))
    if check_counts:
        _check(codes == EXPECTED_TAG_CODES, "Smart Tagging: code counts differ from DATA.md")
        _check(teams == EXPECTED_TAG_TEAMS, "Smart Tagging: Team counts differ from DATA.md")
        _check(kinds == EXPECTED_TAG_TYPES, "Smart Tagging: Type counts differ from DATA.md")
    _check(all(t.stop >= t.start for t in tags), "Smart Tagging: an interval ends before it starts")
    return tags


def load(data_dir=None):
    """The competition match as a validated `Match`, coordinates in metres."""
    meta_path, raw_path, tag_path = games.require(["atd_meta", "atd_raw", "smart_tagging"],
                                                  GAME, data_dir)
    cfg = games.GAMES[GAME]
    fps = cfg["fps"]
    length, width = cfg["pitch"]

    home_name, away_name, slot_team = read_meta(meta_path)
    ball, slots = read_raw(raw_path, len(slot_team))

    lo, hi = COORD_HARD_BOUNDS
    seen = ~np.isnan(ball[:, 0])
    _check(np.array_equal(seen, ~np.isnan(ball[:, 1])), "ATD raw: ball x and y missing differently")
    _check(np.all((ball[seen] >= lo) & (ball[seen] <= hi)), "ATD raw: ball outside hard bounds")
    present = ~np.isnan(slots[..., 0])
    sx, sy = slots[..., 0][present], slots[..., 1][present]
    _check(np.all((sx >= lo) & (sx <= hi) & (sy >= lo) & (sy <= hi)),
           "ATD raw: player outside hard bounds")
    inside = np.mean((sx >= -0.05) & (sx <= 1.05) & (sy >= -0.05) & (sy <= 1.05))
    _check(inside >= PLAYER_INSIDE_MIN_FRACTION,
           "ATD raw: only {:.3f} of player positions inside the pitch".format(inside))

    periods = [(first, last + 1) for _, first, last in games.half_video_blocks(GAME)]
    graded = np.zeros(len(ball), dtype=bool)
    for start, stop in periods:
        graded[start:stop] = True
    for observed, expected, what in ((seen.sum(), EXPECTED_BALL_SEEN, "all"),
                                     (seen[graded].sum(), EXPECTED_BALL_SEEN_GRADED, "graded")):
        _check(abs(observed - expected) <= BALL_SEEN_TOLERANCE * len(ball),
               "ATD raw: ball seen on {} {} frames, DATA.md says {}".format(observed, what,
                                                                             expected))
    tracked = present[graded].sum(axis=1).mean()
    lo_t, hi_t = EXPECTED_MEAN_TRACKED_GRADED
    _check(lo_t <= tracked <= hi_t,
           "ATD raw: mean tracked objects on graded frames {:.1f} outside [{}, {}]".format(
               tracked, lo_t, hi_t))

    scale = np.array([length, width])
    match = Match(
        name=GAME,
        fps=float(fps),
        pitch=(float(length), float(width)),
        ball=(ball * scale).astype(np.float32),
        players=(slots * scale).astype(np.float32),
        team=slot_team,
        periods=periods,
        kickoffs=list(KICKOFF_FRAMES),
        tags=read_tags(tag_path, home_name, away_name, fps),
    )
    match.home_name, match.away_name = home_name, away_name
    return match.validate()


def summary(match):
    """Aggregate-only description, safe to print."""
    seen = ~np.isnan(match.ball[:, 0])
    graded = match.graded_mask()
    return {
        "frames": match.n_frames,
        "home": getattr(match, "home_name", None),
        "away": getattr(match, "away_name", None),
        "slots_home": int((match.team == H).sum()),
        "slots_away": int((match.team == A).sum()),
        "ball_seen_all": round(float(seen.mean()), 4),
        "ball_seen_graded": round(float(seen[graded].mean()), 4),
        "periods": match.periods,
        "tags": len(match.tags),
    }


if __name__ == "__main__":
    print(summary(load()))
