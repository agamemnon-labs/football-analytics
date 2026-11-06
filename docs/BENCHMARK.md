# Benchmark

All numbers are on external games (docs/EXTERNAL_DATA.md): 7 IDSSE and 20 SkillCorner matches,
each at four tracking degradation levels (ball missing 20/35/50/60% of frames, plus position
noise, player dropouts, slot swaps and imitation Smart Tagging with timing and team errors).
Scores come from `evaluation/possession_metrics.py`. Every game is scored by a network that never
saw it (5-fold, games grouped). "Balanced" is the mean of the IDSSE and SkillCorner averages, so
the larger SkillCorner set does not dominate.

## Emission models

| emission model | evaluation | balanced spellF1 | IDSSE | SkillCorner | balanced macroF1 |
|---|---|---|---|---|---|
| logistic (team and dead models) + Viterbi | nested leave-one-game-out | 0.501 | | | |
| temporal CNN, 64 channels, no regularisation | 5-fold, nested decoder selection | 0.611 | | | |
| temporal CNN, 96 channels, dropout, team swap (locked) | 5-fold, decoder tuned on all folds | 0.703 | 0.748 | 0.657 | 0.908 |
| same | 5-fold, nested decoder selection | 0.681 (min 0.585, max 0.774) | | | |

The nested figure re-selects the decoder costs without the scored game and is the honest
estimate; the 0.703 row uses the locked costs and is the baseline for the ablations below.

## Decoder ablations

Locked network outputs, one component changed at a time, all 108 game-level cells.

| variant | IDSSE | SkillCorner | balanced spellF1 | balanced macroF1 |
|---|---|---|---|---|
| locked pipeline | 0.748 | 0.657 | 0.703 | 0.908 |
| no Smart Tagging | 0.682 | 0.551 | 0.617 | 0.884 |
| no hand-over snapping | 0.748 | 0.657 | 0.703 | 0.908 |
| no kick-off forcing | 0.748 | 0.657 | 0.702 | 0.908 |
| half switch penalty | 0.747 | 0.646 | 0.696 | 0.908 |
| double switch penalty | 0.740 | 0.660 | 0.700 | 0.907 |

## Network input ablations

Fold 0 only (2 IDSSE and 4 SkillCorner games, 24 cells): the network is retrained with one input
group zeroed in training and inference. The decoder keeps the locked costs, which slightly favours
the full model.

| inputs removed | IDSSE | SkillCorner | balanced spellF1 | without tags at inference |
|---|---|---|---|---|
| none (locked network) | 0.744 | 0.658 | 0.701 | 0.620 |
| proximity (distances, near-player counts) | 0.735 | 0.652 | 0.694 | 0.607 |
| touches | 0.740 | 0.659 | 0.699 | 0.616 |
| team shape and field position | 0.699 | 0.679 | 0.689 | 0.573 |
| stoppage (player and ball speeds) | 0.734 | 0.660 | 0.697 | 0.605 |
| Smart Tagging | 0.668 | 0.558 | 0.613 | 0.613 |

Smart Tagging is the largest single contribution. The other groups overlap: removing any one of
them costs at most 0.012, and team shape is the only group whose removal changes the two datasets
in opposite directions. Without tags the locked network still scores 0.620, against 0.613 for a
network that never saw tags, so it does not depend on them being present.

Reproduce:

    python scripts/fetch_external.py && python scripts/build_cells.py
    python scripts/train_tcn.py --folds 5 --name cv_w96
    python scripts/tune.py --emission network --oof cv_w96
    python scripts/ablation.py --oof cv_w96
    python scripts/train_tcn.py --only-fold 0 --drop GROUP --name abl_GROUP
    python scripts/ablation.py --oof abl_GROUP --within abl_GROUP --variants "locked pipeline" "no Smart Tagging"
