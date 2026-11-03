import numpy as np

from agamemnon import decoder, labels
from agamemnon.schema import A, D, H, X

COSTS = [[0, 10, 10], [10, 0, 10], [10, 10, 0]]


def spells(path):
    out, prev = 0, None
    for v in path:
        if v != prev and v in (H, A):
            out += 1
        prev = v
    return out


def emissions(seq, strength=1.0):
    e = np.zeros((len(seq), 3), dtype=np.float32)
    for t, k in enumerate(seq):
        e[t, k] = strength
    return e


def test_brief_interruption_does_not_split_possession():
    seq = [H] * 200 + [A] * 5 + [H] * 200
    path = decoder.viterbi(emissions(seq), COSTS, [(0, len(seq))])
    assert spells(path) == 1 and set(path) == {H}


def test_sustained_change_is_detected():
    seq = [H] * 200 + [A] * 200
    path = decoder.viterbi(emissions(seq), COSTS, [(0, len(seq))])
    assert path.tolist() == seq


def test_dead_ball_between_same_team_gives_two_possessions():
    seq = [H] * 100 + [D] * 100 + [H] * 100
    path = decoder.viterbi(emissions(seq), COSTS, [(0, len(seq))])
    assert path.tolist() == seq and spells(path) == 2


def test_segments_decoded_independently_and_dead_outside():
    seq = [H] * 50 + [A] * 50
    path = decoder.viterbi(emissions(seq), COSTS, [(0, 40), (60, 100)])
    assert (path[40:60] == D).all()
    assert (path[:40] == H).all() and (path[60:] == A).all()


def test_forced_dead_before_kickoff():
    seq = [H] * 100
    forced = np.zeros(100, bool)
    forced[:30] = True
    path = decoder.viterbi(emissions(seq), COSTS, [(0, 100)], forced)
    assert (path[:30] == D).all() and (path[30:] == H).all()


def test_snap_moves_handover_to_first_touch_of_taker():
    lab = np.array([H] * 50 + [A] * 50, dtype=np.int8)
    th = np.zeros(100, np.int8)
    ta = np.zeros(100, np.int8)
    ta[44] = 1
    out = decoder.snap_handovers(lab, th, ta, 10, [(0, 100)])
    assert out[43] == H and out[44] == A and (out[44:] == A).all()


def test_logit_is_order_independent_and_rounded():
    X = np.random.default_rng(0).normal(size=(1000, 5)).astype(np.float32)
    c = (0.1, -2.0, 3.3, 0.0, 1e-3)
    a = decoder.logit(X, c)
    assert np.allclose(a, X.astype(np.float64) @ np.array(c), atol=1e-5)
    assert np.array_equal(a, np.round(a, 6))


def test_finalise_and_csv(tmp_path):
    decoded = np.array([D, H, H, A, D, D], dtype=np.int8)
    mask = np.array([False, True, True, True, True, False])
    codes = labels.finalise(decoded, mask)
    assert codes.tolist() == [X, H, H, A, D, X]
    path = tmp_path / "s.csv"
    labels.write_csv(str(path), codes)
    assert path.read_bytes() == b"frame,label\n0,X\n1,H\n2,H\n3,A\n4,D\n5,X\n"
    assert labels.describe(codes, 25.0)["possessions"] == 2
