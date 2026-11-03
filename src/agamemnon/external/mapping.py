"""Map external per-frame possession labels onto the competition's possession rules.

See docs/POSSESSION_RULES.md. External feeds record who touched the ball last; the competition
counts a change of possession only when the other team gains *control*. A spell shorter than the
control threshold is therefore treated as a touch, not a possession, and absorbed by the
possession it interrupts. A spell that starts at a restart is kept whatever its length, because
after a stoppage there is always a clear possessor.
"""

import numpy as np

from ..schema import A, D, H

CONTROL_MIN_S = 2.0


def _spells(seg):
    breaks = np.flatnonzero(seg[1:] != seg[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(seg)]))
    return [(int(bounds[i]), int(bounds[i + 1]), int(seg[bounds[i]]))
            for i in range(len(bounds) - 1)]


def apply_control_rule(labels, periods, fps, min_s=CONTROL_MIN_S):
    """Copy of labels with uncontrolled H/A spells absorbed into the surrounding possession."""
    out = labels.copy()
    min_len = int(round(min_s * fps))
    if min_len <= 1:
        return out
    for start, stop in periods:
        seg = out[start:stop]
        # Repeat until stable: absorbing one touch can join two spells of the same team, which
        # changes the context of the next short spell.
        for _ in range(50):
            spells = _spells(seg)
            changed = False
            for i, (a, b, k) in enumerate(spells):
                if k not in (H, A) or b - a >= min_len:
                    continue
                prev = spells[i - 1][2] if i > 0 else D
                if prev not in (H, A):
                    continue
                seg[a:b] = prev
                changed = True
            if not changed:
                break
        out[start:stop] = seg
    return out
