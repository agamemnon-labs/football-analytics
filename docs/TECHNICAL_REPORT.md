---
title: "Agamemnon: frame-level ball possession from automatic tracking"
subtitle: "More than a Hack 2027, Phase 1: technical document"
author: "Team Agamemnon: Naufal Fahrozi, Raymond Thymotimmanuel, Tua Delima Sitompul"
date: "November 2026"
---

# 1. Approach

The task is to label each of the 199,319 frames of FC Barcelona vs Athletic Club as `H`, `A`, `D`
or `X`, ranked on `spellF1`, which matches whole possessions one-to-one by temporal IoU. That
metric punishes two things far more than per-frame errors: possessions that are split by
flicker, and boundaries that are off by more than a few frames. Our design follows from it.

1. **Per-frame evidence from geometry and kinematics.** From the automatic tracking we compute,
   per team rather than per slot (slots are not identities): distance from the ball to each side's
   nearest player, players within 1.5 m and 5 m, ball touches detected as abrupt changes of ball
   velocity with a player in reach (the last touch before the frame and the next touch after it),
   each side's spread and the field position of ball and players relative to each side's
   attacking direction, collective player speed over 1 s and 3 s, ball still or not, out-of-bounds,
   and kick-off set-ups. Smart Tagging adds a team prior from phase tags, a window around each SET
   PIECES tag and a window after each GOALS tag.
2. **A temporal network turns the evidence into probabilities.** A dilated residual 1-D
   convolutional network (9 blocks, 96 channels, dilations 1 to 256, about ±20 s of context) maps
   the feature sequence to P(H), P(A), P(D) for every frame. It is trained on external matches only.
3. **Viterbi decoding.** The log-probabilities, tempered, are the emissions of a three-state
   (H, A, D) Viterbi with fixed penalties for H<->A, alive->D and D->alive. The penalties are
   what makes a possession a single interval: a brief run of contrary evidence is not worth two
   switches. Frames before each half's kick-off are forced to `D`.
4. **Boundary snapping.** Each H<->A hand-over is moved to the first touch of the team taking
   over within 0.22 s. The tuned window is short because the network already places most
   hand-overs on the touch.

We never write `X` on a graded frame. Under `spellF1` an `X` is filled with the previous label,
so it can only reproduce a decision the decoder already makes, while `macroF1` charges it as a
miss.

**Why this rather than something else.** A nearest-player rule with smoothing (the published
baseline) cannot represent "the ball is in flight between two team-mates" or "play is stopped";
a frame classifier without a state model flickers; a hand-written rule set needs constants we are
not allowed to fit on this match. A learned emission model plus an explicit state model keeps
every constant either physical or fitted on other games. An earlier version used two logistic
models as emissions; replacing them with the network raised the external `spellF1` from 0.50 to
0.68 (section 5).

# 2. Assets used and dropped

| asset | used | how |
|---|---|---|
| tactical ATD (ball + 46 slots) | yes | all geometric and kinematic features; slot team from the metadata, home = first team |
| tactical Smart Tagging | yes | team prior from phase tags, set-piece and goal windows; weighted evidence, never copied as labels |
| tactical homography | no | the ATD is already in pitch coordinates; nothing we compute needs pixels |
| broadcast ATD, homography, video | no | not synchronised to the tactical timeline and lower coverage |
| tactical video | no | see section 8 |
| ATD speed channel | no | velocities are recomputed from positions so that the competition match and the external games go through identical arithmetic |

Missing and noisy tracking: the ball is missing on about 41% of graded frames. Gaps up to 1 s are
linearly bridged; longer gaps leave the ball features empty and the network falls back on team
shape, collective movement and tags. Ball samples implying more than 40 m/s to both neighbours
are dropped as false detections. Player features are robust to slot swaps because they are per
team and medians.

# 3. External data, training and tuning

All fitting happens on 27 external matches (docs/EXTERNAL_DATA.md). None of the labels were made
by our team, and none come from the provided game.

| dataset | amount | labels used | labels produced by | licence |
|---|---|---|---|---|
| IDSSE, DFL Bundesliga 2022/23 (Bassek et al., *Scientific Data* 12, 195, 2025) | 7 matches, 11.1 h, 1.0 M frames at 25 fps | per-frame ball status (alive/dead) and ball possession (home/away) | the DFL's official position-data feed, published with the dataset | CC BY 4.0 |
| SkillCorner Open Data, A-League 2024/25 | 20 matches, 32.8 h, 3.0 M frames (10 fps resampled to 25) | possession group per frame, phases of play | SkillCorner, as published in its open-data repository | MIT |

Their labels are mapped onto the competition definition: IDSSE possession switches on every
touch, so spells shorter than 2 s are absorbed by the possession they interrupt (except straight
after a restart); SkillCorner frames outside a phase of play are dead. Each match is then degraded
to look like uncorrected automatic tracking at four levels (ball missing in blocks on 20, 35, 50
and 60% of frames, position noise, player dropouts, slot swaps), and imitation Smart Tagging is
generated from its events with timing errors, wrong teams, tags that run into the next possession
and false positives. The pipeline also has to work without tags.

**Training.** The network is trained with PyTorch on a laptop GPU (RTX 4050, 6 GB): 6,000 steps
of 4,096-frame crops, AdamW with weight decay 1e-3, one-cycle learning rate up to 2e-3, dropout
0.25, a team-swap augmentation (home and away inputs and labels exchanged) and tag dropout. The
final model is trained on all 27 matches at all four levels (about 14 minutes). Weights are
committed as JSON; inference is reimplemented in numpy float64 so the submission does not need
PyTorch or a GPU.

**Tuning.** The network is first evaluated by 5-fold cross-validation with matches grouped, so
every match is scored by a network that never saw it. On those out-of-fold outputs a random search
(150 configurations) picks the emission temperature, a dead-ball offset, the three Viterbi
penalties and the snap window, maximising mean `spellF1` minus a penalty on its spread across
matches. A nested leave-one-match-out run of the same search gives the unbiased estimate in
section 5. Every resulting constant, with its source and the script that produced it, is in
docs/PRIORS.md.

# 4. Possession rules implemented

The rules, following CHALLENGE.md and the organisers' answers, are in docs/POSSESSION_RULES.md.
In short:

- **Dead ball** runs from the ball leaving the pitch or play stopping to the restart being taken.
  Evidence: the ball last seen more than 0.5 m beyond a line (Law 9 plus tracking jitter),
  collective walking with the ball still or unseen, kick-off set-ups, and Smart Tagging windows.
  Only the period boundaries are hard rules: frames before each half's kick-off are `D`.
- **Restarts.** After a stoppage the restarting team has possession. A valid tag `Team` pushes the
  first frames towards that team; with `N/A` (all throw-ins, kick-offs after goals) the first
  controlled touch decides. A GOALS tag names the scoring side (verified on the match by counting
  only: 4 consistent, 0 inconsistent), so the kick-off belongs to the other team.
- **Loose and contested balls** stay with the team in possession until sustained evidence for the
  other team outweighs the switch penalty; a change is placed at the taker's first touch.

# 5. Results on the external benchmark

Balanced means average the IDSSE and SkillCorner means, so the 20 SkillCorner matches do not
outweigh the 7 IDSSE ones. Full tables are in docs/BENCHMARK.md.

| emission model | evaluation | balanced spellF1 | balanced macroF1 |
|---|---|---|---|
| logistic + Viterbi | nested leave-one-match-out | 0.501 | |
| temporal CNN (locked) | 5-fold, locked decoder | 0.703 (IDSSE 0.748, SkillCorner 0.657) | 0.908 |
| temporal CNN (locked) | 5-fold, nested decoder selection | **0.681** (match range 0.585 to 0.774) | |

Ablations (locked network, one change at a time): removing Smart Tagging costs 0.086 `spellF1`;
halving or doubling the switch penalty costs 0.007 and 0.003; hand-over snapping and kick-off
forcing change less than 0.002. Retraining without one input group (fold 0) costs at most 0.012
for proximity, touches, team shape or stoppage features, and 0.088 for tags, so apart from tags
the inputs overlap and no single signal carries the model.

**Stability.** Scaling each tuned decoder constant by 0.5 or 1.5, one at a time, moves the
external `spellF1` by at most 0.007 (halving the switch penalty); every other change is within
0.004. The optimum is flat, so the locked values are not sitting on a narrow peak.

**Public 5-minute slice.** Scored once, after the parameters were locked (section 8):

| | spellF1 | macroF1 | possessions (GT 17) |
|---|---|---|---|
| Agamemnon | 0.519 (bootstrap 5-95%: 0.311 to 0.710) | 0.845 (0.605 to 0.924) | 10 |
| best published baseline (`baseline_smooth125`) | 0.424 | 0.581 | 16 |

The slice holds 17 possessions, so its interval is wide and contains the external estimate. It is
a check that the pipeline works on this match, not a prediction of the full-match score.

# 6. Limitations and what we would do next

- **Too few possessions.** On the public slice we predict 10 possessions where the ground truth
  has 17, and on the full match 163 per 90 minutes against about 210 to 240 on the external games.
  Short possessions are merged into their neighbours. The switch penalty is tuned on leagues whose
  possession lengths may differ from this match; we did not adjust it, because doing so on this
  match is not allowed.
- **SkillCorner is weaker than IDSSE** (0.657 against 0.748), in part because its dead-ball labels
  come from phase boundaries rather than a tracking flag.
- **Smart Tagging dependence.** Tags add about 0.09; without them the external score is 0.62. Our
  imitation tags are an assumption about how the real tags err.
- **Next steps:** an explicit duration model (semi-Markov decoding) so short possessions are not
  priced like flicker; more external leagues with ball-status flags; training on degradations
  measured from public automatic-tracking datasets rather than chosen ranges.

# 7. Reproduction

```
git clone https://github.com/agamemnon-labs/football-analytics.git
cd football-analytics
# put the match bundle in data/ (DATA.md) or set MTAH_DATA_DIR
bash scripts/reproduce.sh
```

`reproduce.sh` installs the pinned CPU requirements, runs the tests, writes
`evaluation/results/Agamemnon.csv`, validates it and prints its md5. Python 3.12.12; numpy 2.5.4,
pandas 2.3.3, scipy 1.18.1, lxml 5.4.0 (requirements.txt). On an Intel Core i5-13450HX with 16 GB
of RAM the pipeline takes about 12 s, and the whole script about 30 s with the packages cached.
Two runs from a fresh clone and a fresh virtual environment gave the same md5,
`b627df9ec4abba265ecfdfec2e64271e`; `tests/test_pipeline.py` checks this on every test run.

Retraining is optional (`requirements-train.txt`, PyTorch 2.6 with CUDA):
`python scripts/fetch_external.py`, `python scripts/build_cells.py`,
`python scripts/train_tcn.py --folds 5 --name cv_w96`,
`python scripts/tune.py --emission network --oof cv_w96`, `python scripts/train_tcn.py --final`.
GPU training uses fast non-deterministic kernels, so retrained weights differ in the last digits;
the submission is defined by the committed weights.

This document is `docs/TECHNICAL_REPORT.md`, rendered with
`pip install -r requirements-docs.txt && python scripts/build_report.py` (python-markdown and a
headless Chromium browser).

# 8. Development protocol

- Development code touched `data/` only through the loader, and the developers worked from
  aggregate outputs (counts, shares, scores). Format assumptions were taken from DATA.md and
  checked with hard assertions in the loader.
- Nothing was fitted, tuned or calibrated on the competition match or its ground truth. All
  learned weights and tuned constants come from the external games (docs/PRIORS.md lists every
  constant and its source). The aggregate sanity report on the match (scripts/sanity_report.py)
  was used only to look for bugs; it raised no flags and changed nothing.
- The parameters were locked in commit `f30c2ba` ("docs(priors): lock parameters before public
  evaluation"), pushed to GitHub before the public ground truth was first scored at
  2026-11-06 14:31 UTC. It was scored exactly once; no parameter changed afterwards.
- No hand labels or human-produced data about this match were used.

# 9. Team

| member | role |
|---|---|
| Naufal Fahrozi | team lead and systems architecture: repository, loader, converters, pipeline, packaging |
| Raymond Thymotimmanuel | football domain lead: possession and dead-ball rules, external label mapping, restarts |
| Tua Delima Sitompul | lead data scientist: features, network, decoder, benchmark, tuning and ablations |
