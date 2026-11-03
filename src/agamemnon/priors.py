"""Every constant the pipeline uses, in one place.

Each value is either physics/geometry (A), a rule or definition from the competition documents or
general football knowledge (C), or fitted on the external benchmark only (E). docs/PRIORS.md is the
same table with the rationale and provenance of each row, and tests/test_priors.py fails if the two
disagree.
"""

# -- ball and player kinematics (A) ------------------------------------------------------------
BALL_MAX_SPEED = 40.0          # m/s; a struck ball rarely exceeds ~35 m/s
PLAYER_MAX_SPEED = 12.0        # m/s; sprint speed tops out near 10-11 m/s
BALL_INTERP_MAX_S = 1.0        # s; a ball unseen for longer may have been played elsewhere
BALL_SMOOTH_FRAMES = 3
PLAYER_SMOOTH_FRAMES = 5
VEL_HALF_WINDOW = 2
BALL_STILL_SPEED = 1.0         # m/s; a placed or held ball
TOUCH_HALF_WINDOW = 3          # frames each side of a candidate touch
TOUCH_DV = 4.0                 # m/s change of ball velocity that counts as a touch
TOUCH_RADIUS = 2.0             # m; reach of a player playing the ball

# -- pitch geometry (A) ------------------------------------------------------------------------
OOB_MARGIN = 0.5               # m beyond the line before calling a ball out (tracking noise)
FAR_OOB = 3.0                  # m; unambiguously off the pitch

# -- emission features (A, C) ------------------------------------------------------------------
TEAM_DIST_SCALE = 2.0          # m of distance advantage per unit of team evidence
TEAM_DIST_CLIP = 3.0
CLOSE_RADIUS = 1.5             # m; within playing distance
TOUCH_MEMORY_S = 2.0           # s; decay of "who touched it last"
SPREAD_SCALE = 3.0             # m of team-spread difference per unit
SETPIECE_PRE_S = 2.0           # s before a Smart Tagging set piece starts that may still be dead
POST_GOAL_S = 60.0             # s after a goal during which a kick-off restart is expected

# -- emission model coefficients (E, fitted by scripts/tune.py) ----------------------------------
# Logistic models over features.team_design / features.dead_design; see decoder.py.
TEAM_COEF = (0.1152, 0.041, 0.3059, 0.9613, 0.5459, 0.9801, 0.1021, -3.1858, 2.234)
DEAD_COEF = (8.6778, -0.2149, 0.3634, 1.8751, 1.1234, 1.0611, 1.9755, -2.2193, -3.4979, 6.7944,
             -1.0086, -0.8435, -3.5865, -1.3592, 0.1562, 0.0806, 3.2228)

# -- decoder (E, tuned by scripts/tune.py) -------------------------------------------------------
TEAM_TEMPER = 0.29  # per-frame log-odds are correlated; scale before summing
DEAD_TEMPER = 0.30
DEAD_PRIOR = 0.345  # share of dead frames assumed by the dead model's intercept
SWITCH_COST = 5.25  # H <-> A
END_COST = 21.75  # alive -> D
START_COST = 3.93  # D -> alive
SNAP_WINDOW_S = 0.83  # s; move a hand-over to the taker's first touch within this
