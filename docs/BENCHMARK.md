# Benchmark

All numbers are on external games (docs/EXTERNAL_DATA.md) with degraded tracking and imitation
Smart Tagging, scored with `evaluation/possession_metrics.py`. Each game is scored by a model that
never saw it. Work in progress: the final table is written when the configuration is locked.

| emission model | evaluation | balanced spellF1 | IDSSE | SkillCorner |
|---|---|---|---|---|
| logistic (two models) + Viterbi | nested leave-one-game-out, 4 levels | 0.501 | | |
| temporal CNN, 64 channels, no regularisation | 5-fold grouped, nested decoder selection | 0.611 | | |

Reproduce: `python scripts/tune.py` and `python scripts/train_tcn.py --folds 5 --width 64 --steps 4000 --dropout 0 --weight-decay 1e-4 --no-swap && python scripts/tune.py --emission network`.
