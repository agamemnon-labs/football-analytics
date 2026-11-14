# Benchmark

All numbers are on external games (docs/EXTERNAL_DATA.md): 7 IDSSE and 20 SkillCorner matches,
each at four tracking degradation levels (ball missing 20/35/50/60% of frames, plus position
noise, player dropouts, slot swaps and imitation Smart Tagging with timing and team errors).
Scores come from `evaluation/possession_metrics.py`. Every game is scored by a network that never
saw it (5-fold, games grouped). "Balanced" is the mean of the IDSSE and SkillCorner averages, so
the larger SkillCorner set does not dominate.

The imitation tags were rebuilt on 2026-11-14 to have the structure of the real tagging file
(docs/PRIORS.md, Amendments). Every number below is from the rebuilt benchmark. It is a harder test
than the first version, whose tags were cut at the labelled possession edges and so gave away more
than real tags do; scores from the two versions are not comparable.

## Emission models

| emission model | evaluation | balanced spellF1 | mean over games (min, max) | balanced macroF1 |
|---|---|---|---|---|
| logistic (team and dead models) + Viterbi | nested leave-one-game-out | 0.624 | 0.607 (0.518, 0.688) | |
| temporal CNN, 96 channels (submitted) | 5-fold, nested decoder selection | 0.646 | 0.622 (0.508, 0.727) | |
| same | 5-fold, decoder tuned on all games | 0.646 (IDSSE 0.695, SkillCorner 0.596) | | 0.892 |

The nested rows re-select the decoder costs without the scored game and are the honest estimate.
The last row uses the locked costs and is the baseline for the ablations below. The network is
0.02 ahead of the logistic models, a much smaller gap than on the first version of the benchmark.

## Decoder ablations

Locked network outputs, one component changed at a time, all 108 game-level cells.

| variant | IDSSE | SkillCorner | balanced spellF1 | balanced macroF1 |
|---|---|---|---|---|
| locked pipeline | 0.695 | 0.596 | 0.646 | 0.892 |
| no Smart Tagging | 0.669 | 0.553 | 0.611 | 0.882 |
| no hand-over snapping | 0.696 | 0.596 | 0.646 | 0.892 |
| no kick-off forcing | 0.695 | 0.596 | 0.646 | 0.892 |
| half switch penalty | 0.695 | 0.588 | 0.642 | 0.892 |
| double switch penalty | 0.682 | 0.596 | 0.639 | 0.891 |

## Network input ablations

Fold 0 only (2 IDSSE and 4 SkillCorner games, 24 cells): the network is retrained with one input
group zeroed in training and inference. The decoder keeps the locked costs, which slightly favours
the full model.

| inputs removed | IDSSE | SkillCorner | balanced spellF1 | without tags at inference |
|---|---|---|---|---|
| none (locked network) | 0.692 | 0.591 | 0.641 | 0.612 |
| proximity (distances, near-player counts) | 0.687 | 0.592 | 0.640 | 0.603 |
| touches | 0.694 | 0.591 | 0.643 | 0.618 |
| team shape and field position | 0.637 | 0.593 | 0.615 | 0.573 |
| stoppage (player and ball speeds) | 0.683 | 0.592 | 0.637 | 0.602 |
| Smart Tagging | 0.662 | 0.561 | 0.612 | 0.612 |

Smart Tagging and team shape are the two groups that matter, at 0.029 and 0.026. Team shape only
matters on IDSSE. Proximity, touches and stoppage features each move the score by 0.004 or less:
they overlap, and the touch features in particular add nothing the network cannot get from the
others. Without tags the locked network scores 0.612, the same as a network that never saw tags,
so it does not depend on them being present.

## Tag styles

The tag inputs of every game replaced by tags from a second, independent generator
(`scripts/tag_styles.py`), levels mid and high, out-of-fold networks.

| tag inputs | IDSSE | SkillCorner | balanced spellF1 |
|---|---|---|---|
| tags as trained | 0.694 | 0.602 | 0.648 |
| no tags | 0.669 | 0.559 | 0.614 |
| fixed windows, second generator | 0.688 | 0.599 | 0.644 |
| possession-aligned windows, second generator | 0.706 | 0.608 | 0.657 |

The submitted network holds its score in both styles. The first network, trained on
possession-aligned tags and scored the same way before the rebuild, went from 0.70 to 0.58 when
given fixed windows, which is why the benchmark was rebuilt.

## Stability of the locked decoder

Each tuned constant scaled by 0.5 and 1.5, one at a time, on the out-of-fold outputs of all 108
cells (`python scripts/sweep.py --oof cv_v2`). "Worst cell" is the lowest single game and
degradation level. This documents the shape of the optimum; no locked value was changed.

| variant | balanced spellF1 | worst cell spellF1 | balanced macroF1 |
|---|---|---|---|
| locked | 0.646 | 0.498 | 0.892 |
| NN_TEMPER x0.5 | 0.633 | 0.478 | 0.889 |
| NN_TEMPER x1.5 | 0.644 | 0.489 | 0.893 |
| NN_DEAD_BIAS x0.5 | 0.646 | 0.496 | 0.893 |
| NN_DEAD_BIAS x1.5 | 0.644 | 0.498 | 0.892 |
| SWITCH_COST x0.5 | 0.642 | 0.492 | 0.892 |
| SWITCH_COST x1.5 | 0.641 | 0.489 | 0.892 |
| END_COST x0.5 | 0.644 | 0.497 | 0.892 |
| END_COST x1.5 | 0.645 | 0.495 | 0.892 |
| START_COST x0.5 | 0.645 | 0.497 | 0.892 |
| START_COST x1.5 | 0.645 | 0.495 | 0.892 |
| SNAP_WINDOW_S x0.5 | 0.646 | 0.497 | 0.892 |
| SNAP_WINDOW_S x1.5 | 0.645 | 0.496 | 0.892 |

Reproduce:

    python scripts/fetch_external.py && python scripts/build_cells.py
    python scripts/train_tcn.py --folds 5 --name cv_v2
    python scripts/tune.py --emission network --oof cv_v2
    python scripts/tune.py --emission logistic
    python scripts/ablation.py --oof cv_v2
    python scripts/sweep.py --oof cv_v2
    python scripts/tag_styles.py --oof cv_v2
    python scripts/train_tcn.py --only-fold 0 --drop GROUP --name abl_GROUP
    python scripts/ablation.py --oof abl_GROUP --within abl_GROUP --variants "locked pipeline" "no Smart Tagging"
