import numpy as np

from agamemnon import deadball, features, priors
from agamemnon.schema import A, GOALS, H, SET_PIECES, Match, Tag

FPS = 25.0


def match(ball, players, team, periods=None, tags=()):
    n = len(ball)
    return Match("syn", FPS, (105.0, 68.0), np.asarray(ball, np.float32),
                 np.asarray(players, np.float32), np.asarray(team, np.int8),
                 periods or [(0, n)], [p[0] for p in (periods or [(0, n)])], list(tags)).validate()


def test_interpolate_fills_short_gaps_only_and_never_across_a_period():
    ball = np.column_stack([np.arange(40, dtype=np.float32), np.zeros(40, np.float32)])
    ball[5:8] = np.nan       # 3 frames: filled
    ball[12:20] = np.nan     # 8 frames: too long
    ball[29:31] = np.nan     # straddles the period edge at 30
    filled, age = features.interpolate(ball, 5, [(0, 30), (30, 40)])
    assert np.allclose(filled[5:8, 0], [5, 6, 7])
    assert np.isnan(filled[12:20, 0]).all()
    assert np.isnan(filled[29:31, 0]).all()
    assert age[4] == 0 and age[7] == 3 and age[19] == 8


def test_clean_ball_drops_an_isolated_jump_and_keeps_a_fast_pass():
    x = np.arange(50, dtype=np.float32) * 1.2          # 30 m/s, a hard pass
    ball = np.column_stack([x, np.full(50, 30, np.float32)])
    ball[20] = (90.0, 5.0)                             # one false detection far away
    out = features.clean_ball(ball, FPS, priors.BALL_MAX_SPEED)
    assert np.isnan(out[20]).all()
    assert not np.isnan(np.delete(out, 20, axis=0)).any()


def test_touch_goes_to_the_side_whose_player_is_at_the_ball():
    n = 100
    # The ball runs towards a home player at x = 50, who sends it straight back.
    x = np.where(np.arange(n) < 50, 30 + 0.4 * np.arange(n), 50 - 0.4 * (np.arange(n) - 50))
    ball = np.column_stack([x, np.full(n, 34.0)])
    players = np.empty((n, 2, 2))
    players[:, 0] = (50.5, 34.0)       # home, at the turning point
    players[:, 1] = (20.0, 10.0)       # away, nowhere near
    F = features.compute(match(ball, players, [H, A]), priors)
    touches = np.flatnonzero(F["touch_home"])
    assert len(touches) == 1 and abs(touches[0] - 50) <= 2
    assert F["touch_away"].sum() == 0
    assert F["d_home"][50] < 1.0 and F["d_away"][50] > 30


def test_out_of_bounds_is_remembered_while_the_ball_is_unseen():
    n = 60
    ball = np.column_stack([np.full(n, 50.0), np.linspace(60, 80, n)])   # leaves over y = 68
    ball[40:] = np.nan
    players = np.full((n, 2, 2), 30.0)
    F = features.compute(match(ball, players, [H, A]), priors)
    first_out = int(np.flatnonzero(F["oob"])[0])
    assert ball[first_out, 1] > 68 + priors.OOB_MARGIN >= ball[first_out - 1, 1]
    assert F["last_oob"][45] == 1 and F["seen"][45] == 0
    assert F["far_oob"][39] == 1          # more than FAR_OOB beyond the line by then


def test_attack_direction_follows_where_each_team_stands():
    n = 50
    ball = np.full((n, 2), 50.0)
    players = np.empty((n, 2, 2))
    players[:, 0] = (20.0, 30.0)       # home sits on the left, so it attacks to the right
    players[:, 1] = (80.0, 30.0)
    F = features.compute(match(ball, players, [H, A]), priors)
    assert (F["attack_home"] == 1).all()
    F = features.compute(match(ball, players[:, ::-1], [H, A]), priors)
    assert (F["attack_home"] == -1).all()


def test_tag_features_follow_the_code_and_the_team():
    n = 4000
    tags = [Tag("BUILD UP", 100, 200, H), Tag("DEFENDING IN MIDDLE THIRD", 300, 400, H),
            Tag(SET_PIECES, 1000, 1100, A, "Corner Kick"), Tag(SET_PIECES, 1500, 1600, None, "Throw-in"),
            Tag(GOALS, 2000, 2050, H)]
    m = match(np.zeros((n, 2)), np.zeros((n, 2, 2)), [H, A], tags=tags)
    team, setpiece, post_goal = deadball.tag_features(m, priors)
    assert (team[100:200] == 1).all()             # home attacking
    assert (team[300:400] == -1).all()            # home defending means away has it
    assert (team[1000:1100] == -1).all()          # away's set piece
    assert (team[1500:1600] == 0).all()           # throw-in with no team
    pre = int(priors.SETPIECE_PRE_S * FPS)
    assert setpiece[1000 - pre] == 1 and setpiece[1000 - pre - 1] == 0 and setpiece[1099] == 1
    after = int(priors.POST_GOAL_S * FPS)
    assert post_goal[2049] == 0 and post_goal[2050] == 1 and post_goal[2050 + after - 1] == 1
    assert post_goal[2050 + after] == 0
