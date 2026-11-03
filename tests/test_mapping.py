import numpy as np

from agamemnon.external.mapping import apply_control_rule
from agamemnon.schema import A, D, H


def run(seq, fps=1.0, min_s=3.0):
    lab = np.array(seq, dtype=np.int8)
    return apply_control_rule(lab, [(0, len(lab))], fps, min_s).tolist()


def test_short_touch_is_absorbed():
    assert run([H] * 5 + [A] * 2 + [H] * 5) == [H] * 12


def test_controlled_change_is_kept():
    assert run([H] * 5 + [A] * 4) == [H] * 5 + [A] * 4


def test_restart_possessor_kept_even_if_short():
    assert run([D] * 3 + [A] * 1 + [H] * 6) == [D] * 3 + [A] + [H] * 6


def test_touch_before_stoppage_goes_to_previous_team():
    assert run([H] * 5 + [A] * 1 + [D] * 3) == [H] * 6 + [D] * 3


def test_dead_runs_untouched():
    seq = [H] * 4 + [D] * 1 + [H] * 4
    assert run(seq) == seq


def test_alternating_touches_collapse_to_one_possession():
    seq = [H] * 5 + [A, H, A, H] + [H] * 4
    assert run(seq, min_s=2.0) == [H] * 13
