import json

import numpy as np

from agamemnon.external import idsse, skillcorner
from agamemnon.schema import A, D, H


def write_skillcorner(root, mid="7"):
    folder = root / mid
    folder.mkdir(parents=True)
    (root / "matches.json").write_text(json.dumps([{"id": int(mid)}]))
    info = {"home_team": {"id": 1}, "away_team": {"id": 2}, "pitch_length": 100,
            "pitch_width": 60,
            "players": [{"id": 11, "team_id": 1}, {"id": 21, "team_id": 2}]}
    (folder / "{}_match.json".format(mid)).write_text(json.dumps(info))
    rows = []
    for f in range(40):
        period = 1 if f < 20 else 2
        group = "home team" if f % 20 < 10 else ("away team" if f % 20 < 15 else None)
        rows.append({"frame": f, "period": period,
                     "ball_data": {"x": 0.0, "y": 0.0, "z": 0.0, "is_detected": f % 7 != 0},
                     "possession": {"player_id": None, "group": group},
                     "player_data": [{"x": -10.0, "y": 5.0, "player_id": 11, "is_detected": True},
                                     {"x": 10.0, "y": -5.0, "player_id": 21,
                                      "is_detected": False}]})
    (folder / "{}_tracking_extrapolated.jsonl".format(mid)).write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    (folder / "{}_phases_of_play.csv".format(mid)).write_text(
        "frame_start,frame_end,team_in_possession_id\n2,17,1\n22,37,2\n")
    return mid


def test_skillcorner_convert(tmp_path):
    mid = write_skillcorner(tmp_path)
    m = skillcorner.convert(mid, root=str(tmp_path))
    assert m.fps == 25.0 and m.pitch == (100.0, 60.0)
    assert len(m.periods) == 2 and m.periods[0][0] == 0
    # 19 source frames at 10 fps span 1.9 s -> 48 frames at 25 fps.
    assert m.periods[0][1] == 48
    assert np.allclose(m.players[0, 0], [40.0, 35.0])
    lab = m.labels[m.periods[0][0]:m.periods[0][1]]
    assert lab[0] == D and set(np.unique(lab)) <= {H, A, D}
    # The empty group at the end of the first phase keeps the last team with the ball.
    assert lab[int(1.6 * 25)] == A
    assert m.kickoffs[0] == 5


DFL_INFO = """<?xml version="1.0"?><PutDataRequest><MatchInformation>
<General HomeTeamId="TH" GuestTeamId="TA"/><Environment PitchX="105.00" PitchY="68.00"/>
<Teams><Team TeamId="TH"><Players><Player PersonId="P1"/></Players></Team>
<Team TeamId="TA"><Players><Player PersonId="P2"/></Players></Team></Teams>
</MatchInformation></PutDataRequest>"""


def dfl_positions():
    def frames(n0, n, extra=lambda i: ""):
        return "".join('<Frame N="{}" X="{}" Y="0.0"{}/>'.format(n0 + i, float(i), extra(i))
                       for i in range(n))
    ball = lambda i: ' BallStatus="{}" BallPossession="{}"'.format(int(i >= 2), 1 if i < 6 else 2)
    sets = []
    for section, n0 in (("firstHalf", 10000), ("secondHalf", 100000)):
        sets.append('<FrameSet GameSection="{}" TeamId="BALL">{}</FrameSet>'.format(
            section, frames(n0, 10, ball)))
        sets.append('<FrameSet GameSection="{}" TeamId="TH" PersonId="P1">{}</FrameSet>'.format(
            section, frames(n0 + 1, 8)))
        sets.append('<FrameSet GameSection="{}" TeamId="TA" PersonId="P2">{}</FrameSet>'.format(
            section, frames(n0, 10)))
    return '<?xml version="1.0"?><PutDataRequest><Positions>{}</Positions></PutDataRequest>'.format(
        "".join(sets))


def test_idsse_convert(tmp_path):
    (tmp_path / "DFL_02_01_matchinformation_X_DFL-MAT-M1.xml").write_text(DFL_INFO)
    (tmp_path / "DFL_04_03_positions_raw_observed_X_DFL-MAT-M1.xml").write_text(dfl_positions())
    m = idsse.convert("M1", root=str(tmp_path))
    assert m.periods == [(0, 10), (10, 20)]
    assert m.team.tolist() == [H, A]
    assert m.labels[:10].tolist() == [D, D, H, H, H, H, A, A, A, A]
    assert m.kickoffs == [2, 12]
    assert np.isnan(m.players[0, 0]).all() and np.allclose(m.players[1, 0], [52.5, 34.0])
    assert np.allclose(m.ball[3], [55.5, 34.0])
