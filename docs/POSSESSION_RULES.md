# Possession rules

This file states the possession and dead-ball rules the pipeline implements, and how the labels of
each external dataset are mapped onto them. The authority is CHALLENGE.md; where it leaves a choice
open, the choice and the reason for it are written down here.

## The competition definition

- Every graded frame is `H`, `A` or `D`. A possession is a maximal run of one team with the ball
  alive; `H D H` is two possessions.
- While the ball is alive one team always has possession. A contested or loose ball stays with the
  last team that had it.
- Possession changes only when the other team gains *controlled* possession. A touch without
  control, a deflection that does not intercept a pass, or an aerial duel won without the team then
  controlling the ball does not change it.
- After every stoppage there is a clear possessor: the team taking the restart.
- `D` means the ball is not in play. Who "owns" a dead ball is not asked.

## Mapping external labels

### Control rule

External feeds name the team that touched the ball last, which switches on every deflection and
duel. We apply one rule to both datasets (`src/agamemnon/external/mapping.py`):

1. Split each half into runs of `H`, `A` and `D`.
2. An `H` or `A` run shorter than 2.0 s that follows a run of the other team is a touch, not a
   possession: it is relabelled as the team before it.
3. A run that starts right after a `D` run is kept whatever its length, because the restarting team
   is the possessor even if it loses the ball at once.
4. Repeat until nothing changes, since absorbing one touch can merge two runs of the same team.

2.0 s is the low end of the "a player keeps the ball for a few seconds" proxy in CHALLENGE.md. On the
first IDSSE match the rule takes the spell count from 511 to 329 (mean 6.7 s to 10.4 s); raising the
threshold to 3 s or 4 s changes it by less than 10%, so the benchmark is not sensitive to the exact
value.

A touch that happens just before the ball goes out (`H A D`, with `A` short) is given to `H`. This
matches the rule that a loose ball stays with the last team in possession; the team that touched it
last is the one that will *not* take the throw-in, but dead-ball ownership is not scored anyway.

### IDSSE (DFL)

| DFL field | value | label |
|---|---|---|
| `BallStatus` | 0 | `D` |
| `BallStatus` | 1, `BallPossession` 1 | `H` (home as listed in the match information) |
| `BallStatus` | 1, `BallPossession` 2 | `A` |

Then the control rule. Only `firstHalf` and `secondHalf` are used; each is one graded period, and
its first alive frame is the kick-off.

### SkillCorner

| condition | label |
|---|---|
| frame outside every phase of play | `D` |
| inside a phase, possession group `home team` / `away team` | `H` / `A` |
| inside a phase, no group | last team with the ball in this phase, else the phase's team in possession |

Then the control rule. Positions are linearly resampled from 10 fps to 25 fps and labels take the
latest source frame at or before each output frame.
