# Priors

Every constant the pipeline uses. Sources: **A** physics or geometry, **C** a rule, definition or
general football knowledge, **E** fitted or tuned on the external games only (docs/EXTERNAL_DATA.md).
Nothing here was measured on, fitted to or tuned against the competition match or its ground truth.

This file is generated from `src/agamemnon/priors.py` by `scripts/priors_table.py`, and
`tests/test_priors.py` fails if the two disagree. The network weights are in
`src/agamemnon/models/tcn.json`, produced by `scripts/train_tcn.py --final`.

| name | value | source | rationale | produced by |
|---|---|---|---|---|
| `BALL_MAX_SPEED` | `40` | A | fastest shots are around 35 m/s; faster jumps are false detections | - |
| `PLAYER_MAX_SPEED` | `12` | A | sprint speed tops out near 10-11 m/s; faster is a slot swap | - |
| `BALL_INTERP_MAX_S` | `1` | A | a ball unseen for longer may have been played on; do not invent a path | - |
| `BALL_SMOOTH_FRAMES` | `3` | A | removes frame-to-frame jitter before differentiating | - |
| `PLAYER_SMOOTH_FRAMES` | `5` | A | same for players | - |
| `VEL_HALF_WINDOW` | `2` | A | central difference over 0.16 s | - |
| `BALL_STILL_SPEED` | `1` | A | a placed or held ball moves slower than this | - |
| `TOUCH_HALF_WINDOW` | `3` | A | velocity before and after a touch over 0.12 s | - |
| `TOUCH_DV` | `4` | A | velocity change of a pass, trap or deflection | - |
| `TOUCH_RADIUS` | `2` | A | reach of a player playing the ball | - |
| `OOB_MARGIN` | `0.5` | A, C | ball wholly over the line (Law 9) plus tracking jitter | - |
| `FAR_OOB` | `3` | A | unambiguously off the pitch | - |
| `TEAM_DIST_SCALE` | `2` | A | distance advantage per unit of team evidence | - |
| `TEAM_DIST_CLIP` | `3` | A | beyond this the nearer side is clear | - |
| `CLOSE_RADIUS` | `1.5` | A | within playing distance | - |
| `NEAR_RADIUS` | `5` | A | players who can contest the ball within about a second | - |
| `TOUCH_MEMORY_S` | `2` | C | a pass takes one to two seconds to arrive | - |
| `SPREAD_SCALE` | `3` | A | team spread difference per unit | - |
| `SETPIECE_PRE_S` | `2` | C | tagged set pieces mark a window around the restart (starter docs) | - |
| `POST_GOAL_S` | `60` | C | celebration and kick-off take up to about a minute | - |
| `TEAM_COEF` | `(0.1152, 0.041, 0.3059, 0.9613, 0.5459, 0.9801, 0.1021, -3.1858, 0, 0, 0, 2.234)` | E | logistic team model, fitted on all external games | scripts/tune.py --final |
| `DEAD_COEF` | `(8.6778, -0.2149, 0.3634, 1.8751, 1.1234, 1.0611, 1.9755, -2.2193, -3.4979, 6.7944, -1.0086, -0.8435, -3.5865, -1.3592, 0.1562, 0.0806, 3.2228)` | E | logistic dead-ball model, fitted on all external games | scripts/tune.py --final |
| `TEAM_TEMPER` | `0.29` | E | logistic emissions: per-frame log-odds are correlated | scripts/tune.py |
| `DEAD_TEMPER` | `0.3` | E | same, dead-ball model | scripts/tune.py |
| `DEAD_PRIOR` | `0.345` | E | intercept correction of the dead-ball model | scripts/tune.py |
| `SWITCH_COST` | `5.25` | E | Viterbi penalty for H <-> A | scripts/tune.py |
| `END_COST` | `21.75` | E | Viterbi penalty for alive -> D | scripts/tune.py |
| `START_COST` | `3.93` | E | Viterbi penalty for D -> alive | scripts/tune.py |
| `SNAP_WINDOW_S` | `0.83` | E | search window for moving a hand-over to the taker's touch | scripts/tune.py |
| `EMISSION` | `logistic` | E | which emission model the decoder uses | scripts/tune.py --emission network |
| `NN_TEMPER` | `0.3` | E | network emissions: scale of per-frame log-probabilities | scripts/tune.py |
| `NN_DEAD_BIAS` | `0` | E | network emissions: offset of the dead-ball log-probability | scripts/tune.py |

## Amendments

None yet. After the parameters are locked, any change is listed here with its date and a reason
that is not a score.
