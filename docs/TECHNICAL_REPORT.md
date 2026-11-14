---
title: "Agamemnon: frame-level ball possession from automatic tracking"
subtitle: "More than a Hack 2027, Phase 1: technical document"
author: "Team Agamemnon: Naufal Fahrozi, Raymond Thymotimmanuel, Tua Delima Sitompul"
date: "November 2026"
---

# 1. Approach

We have to label each of the 199,319 frames of FC Barcelona vs Athletic Club as `H`, `A`, `D` or
`X`. The ranking metric, `spellF1`, matches whole possessions one-to-one by temporal overlap, and
that shaped most of our decisions. A single wrong frame barely matters. What hurts is a possession
that flickers to the other team for half a second (one correct possession turns into three wrong
ones) and a boundary that lands a second late. So we never make the call frame by frame. We collect
evidence for every frame and then ask what the most believable sequence of possessions is for the
whole half.

The pipeline has four steps.

1. Per-frame evidence. From the automatic tracking we compute, per team and never per slot (slots
   swap identities), the distance from the ball to each side's nearest player, whether that player
   is within 1.5 m, how many players of each side are within 5 m, ball touches (an abrupt change in ball velocity with a
   player in reach; we keep both the last touch before a frame and the next one after it), each
   side's spread and the field position of ball and players relative to its attacking direction,
   collective player speed over 1 s and 3 s, whether the ball is still, out-of-bounds, and kick-off
   set-ups. Smart Tagging adds a team prior from phase tags, the first 5 s of each SET PIECES
   interval (plus 2 s before it) and the whole of each GOALS interval.
2. A temporal network. A dilated residual 1-D CNN (9 blocks, 96 channels, dilations 1 to 256,
   roughly 20 s of context either side) turns the evidence into P(H), P(A) and P(D) for every
   frame. It is trained on external matches only (section 3).
3. Viterbi decoding. The tempered log-probabilities are emissions of a three-state Viterbi with a
   fixed cost for H<->A, alive->D and D->alive. Those costs are what keep a possession in one
   piece. Frames before each half's kick-off are forced to `D`.
4. Boundary snapping. Each hand-over is moved to the first touch of the team taking over, within
   0.53 s. Switching this step off changes the external score by less than 0.001, because the
   network already puts most hand-overs on the touch.

We never write `X` on a graded frame. The scorer fills an `X` with the previous label before
computing `spellF1`, so abstaining can only repeat a decision the decoder already makes, and
`macroF1` counts it as a miss.

We tried to talk ourselves into simpler designs and couldn't. A nearest-player rule with smoothing,
like the published baseline, has no way to say "the ball is in the air between two team-mates" or
"play has stopped". A frame classifier without a state model flickers like a VAR check that keeps
changing its mind. A hand-written rule set needs
thresholds that we would have had to tune on this match, which the rules forbid. A learned
emission model plus an explicit state model lets every constant be either physical or fitted on
other games. Our first version used two logistic models as emissions. The network is better,
but not by much: 0.646 against 0.624 external `spellF1` (section 5).

![Thirty seconds of an external Bundesliga match as the model receives it, with about a third of the ball positions removed. The bars underneath are the true possession and our labels. The short home possession in the middle is missed, which is the most common kind of error.](assets/demo-still.png)

# 2. Assets used and dropped

| asset | used | how |
|---|---|---|
| tactical ATD (ball + 46 slots) | yes | all geometric and kinematic features; slot team from the metadata, home = first team |
| tactical Smart Tagging | yes | team prior from phase tags, set-piece and goal windows; treated as evidence, never copied into labels |
| tactical homography | no | the ATD is already in pitch coordinates and nothing we compute needs pixels |
| broadcast ATD, homography, video | no | a different timeline from the tactical one, and lower coverage |
| tactical video | no | see section 8 |
| ATD speed channel | no | we recompute velocities from positions, so the competition match and the external games go through the same arithmetic |

The ball is missing on about 41% of graded frames. Gaps up to 1 s are bridged linearly; in longer
gaps the ball features are empty and the network has to rely on team shape, collective movement and
tags. Ball samples that sit more than two frames of travel at 40 m/s away from both neighbours are
dropped as false detections.
Player features are per team and use medians, so a slot swap moves them very little.

# 3. External data, training and tuning

All fitting happens on 27 external matches (docs/EXTERNAL_DATA.md). None of the labels were made by
us, and none come from the provided game.

| dataset | amount | labels used | labels produced by | licence |
|---|---|---|---|---|
| IDSSE, DFL Bundesliga 2022/23 (Bassek et al., *Scientific Data* 12, 195, 2025) | 7 matches, 11.1 h, 1.0 M frames at 25 fps | per-frame ball status (alive/dead) and ball possession (home/away) | the DFL's official position-data feed, published with the dataset | CC BY 4.0 |
| SkillCorner Open Data, A-League 2024/25 | 20 matches, 32.8 h, 3.0 M frames (10 fps resampled to 25) | possession group per frame, phases of play | SkillCorner, as published in its open-data repository | MIT |

Neither dataset uses the competition's definition of possession, so we map them onto it. In both,
a spell shorter than 2 s is given back to the possession it interrupts, unless it comes straight
after a restart; this matters most for IDSSE, where possession switches on every touch. SkillCorner
frames outside a phase of play are dead. We then make each match look like uncorrected automatic
tracking, at four levels: the ball is removed in blocks on 20, 35, 50 or 60% of frames, and we add
position noise, player dropouts and slot swaps.

Neither dataset has Smart Tagging, so we generate imitation tags from each game's labels: set
pieces as fixed 35 s windows that open a few seconds before the restart (5 s for kick-offs), goal
intervals that run from the build-up to after the kick-off, and phase windows of 10 to 30 s that
come in attack/defend pairs and often outlive the possession they start in. Restarts are missed,
teams are missing or wrong, and some phases are false positives. Every configuration is also scored
with no tags at all. This is the second version of the imitation; section 8 explains why the first
was replaced.

The network is trained with PyTorch on a laptop GPU (RTX 4050, 6 GB) for 6,000 steps on crops of
4,096 frames, with AdamW (weight decay 1e-3), a one-cycle learning rate up to 2e-3, dropout 0.25,
tag dropout, and a team swap (home and away inputs and labels exchanged). The final model is
trained on all 27 matches at all four levels in about 14 minutes. The weights are committed as JSON
and inference is reimplemented in numpy float64, so producing the submission needs neither PyTorch
nor a GPU.

For tuning, we first ran 5-fold cross-validation with matches grouped, so every match is scored by
a network that never saw it. On those out-of-fold outputs a random search over 150 configurations
picks the emission temperature, a dead-ball offset, the three Viterbi costs and the snap window,
maximising mean `spellF1` minus half its standard deviation across matches. Repeating the search
with each match left out in turn gives the nested estimate in section 5. Every constant, its source
and the script that produced it are listed in docs/PRIORS.md.

# 4. Possession rules implemented

The full rules, following CHALLENGE.md, are in docs/POSSESSION_RULES.md. The main lines we drew:

A dead ball runs from the moment the ball leaves the pitch or play stops until the restart is
taken. The evidence is the ball last seen more than 0.5 m beyond a line (Law 9 plus tracking
jitter), players walking together while the ball is still or unseen, kick-off set-ups, and Smart
Tagging windows. None of these is a hard rule; only the period boundaries are, so frames before
each half's kick-off are `D`.

After a stoppage the restarting team has the ball. When the tag's `Team` is valid it pushes the
first frames towards that team. With `N/A`, which covers every throw-in and every kick-off after a
goal, the first controlled touch decides. A GOALS tag names the scoring side (we checked this on the
match by counting only: 4 consistent, 0 inconsistent), so the kick-off that follows belongs to the
other team.

Loose and contested balls stay with the team in possession until the evidence for the other team
is strong and lasts long enough to outweigh the switch cost. A change is then placed at the taker's
first touch.

# 5. Results

Balanced means average the IDSSE mean and the SkillCorner mean, so the 20 SkillCorner matches do not
outweigh the 7 IDSSE ones. Full tables are in docs/BENCHMARK.md.

| emission model | evaluation | balanced spellF1 | mean over matches (min, max) | balanced macroF1 |
|---|---|---|---|---|
| logistic + Viterbi | nested leave-one-match-out | 0.624 | 0.607 (0.518, 0.688) | |
| temporal CNN (submitted) | 5-fold, nested decoder selection | 0.646 | 0.622 (0.508, 0.727) | |
| temporal CNN (submitted) | 5-fold, locked decoder | 0.646 (IDSSE 0.695, SkillCorner 0.596) | | 0.892 |

![Left: spellF1 of the logistic and the network emission models on the external matches, nested estimates. Right: spellF1 lost by changing one part of the submitted pipeline.](assets/benchmark-light.png)

Removing Smart Tagging costs 0.035 `spellF1`. Halving or doubling the switch cost costs 0.004 and
0.007, and hand-over snapping and kick-off forcing each change it by less than 0.001. Retraining
the network without one group of inputs on one fold costs 0.029 for tags and 0.026 for team shape,
and 0.004 or less for proximity, touches or stoppage features, which overlap with each other.

Scaling each tuned decoder constant by 0.5 or 1.5, one at a time, moves the external `spellF1` by
at most 0.013 (halving the emission temperature); every other change stays within 0.005. The
optimum is flat, so the locked values are not sitting on a narrow peak.

![The first ten minutes of an external match: true labels, our labels, and the frames in which the ball was tracked.](assets/timeline-light.png)

We scored the public five minutes after the parameters were locked. This happened twice, once for
each locked version (section 8), and neither result was used to choose anything:

| | spellF1 | macroF1 | possessions (ground truth: 17) |
|---|---|---|---|
| Agamemnon, submitted version | 0.457 (bootstrap 5-95%: 0.243 to 0.671) | 0.877 (0.610 to 0.929) | 11 |
| Agamemnon, first locked version | 0.519 (0.311 to 0.710) | 0.845 (0.605 to 0.924) | 10 |
| best published baseline (`baseline_smooth125`) | 0.424 | 0.581 | 16 |

With 17 possessions the intervals are wide, each contains the other version's score, and both
contain the external estimate. We read this as a check that the pipeline works on this match, not
as a prediction of the full-match score and not as a comparison of the two versions.

# 6. Limitations and what we would do next

Our clearest weakness: we find too few possessions. On the public slice we predict 11 where
there are 17, and on the full match 156 per 90 minutes; on the first external match the same
pipeline predicts 187 where the labels have 239. Short possessions get merged into their
neighbours. A lower switch cost would split more of them, but on the external matches it does not
raise `spellF1` (0.646 to 0.642 when halved), because a short possession placed wrongly costs more
than one that is missed.

SkillCorner is weaker than IDSSE (0.596 against 0.695), partly because its dead-ball labels come
from phase boundaries rather than a tracking flag. The imitation tags remain a guess. We know the
window lengths of the real tags, but not how well their timing and teams line up with play, and
that alignment decides what they are worth. Swapping in tags from a second generator, in either the
fixed-window or the older possession-aligned style, moves the external score by about 0.01
(docs/BENCHMARK.md), and without any tags it is 0.61, so the damage from a wrong guess is bounded.

The network adds only 0.02 over two logistic models on this benchmark. Most of the score comes
from the features and the decoder, and that is where further work should go.

With more time we would try a duration model (semi-Markov decoding), so that a short possession is
not priced like flicker, add more external leagues that have a ball-status flag, and measure the
degradation of real automatic tracking on public data instead of choosing ranges.

# 7. Reproduction

```
git clone https://github.com/agamemnon-labs/football-analytics.git
cd football-analytics
# put the match bundle in data/ (DATA.md) or set MTAH_DATA_DIR
bash scripts/reproduce.sh
```

`reproduce.sh` installs the pinned CPU requirements, runs the tests, writes
`evaluation/results/Agamemnon.csv`, validates it and prints its md5. We used Python 3.12.12 with
numpy 2.5.4, pandas 2.3.3, scipy 1.18.1 and lxml 5.4.0 (requirements.txt). On an Intel Core
i5-13450HX with 16 GB of RAM the pipeline takes about 13 s, and the whole script about 30 s once the
packages are cached. A fresh clone with a fresh virtual environment gives the md5
`1b0da3466b8663978d894daa2aa92434`, the same as our working copy, and the file is also identical
when numpy is forced onto other BLAS kernels and thread counts. `tests/test_pipeline.py` repeats the
check on every test run, and `tests/test_determinism.py` checks the network and decoder against a
recorded digest on Linux, Windows and macOS in CI.

Retraining is optional and needs `requirements-train.txt` (PyTorch 2.6 with CUDA):
`python scripts/fetch_external.py`, `python scripts/build_cells.py`,
`python scripts/train_tcn.py --folds 5 --name cv_v2`,
`python scripts/tune.py --emission network --oof cv_v2` and `python scripts/train_tcn.py --final`.
GPU training uses fast non-deterministic kernels, so retrained weights differ in the last digits;
the submission is defined by the committed weights. The figures come from
`python scripts/make_figures.py`, and this document is rendered with
`python scripts/build_report.py` (PDF) and `python scripts/build_docx.py` (Word).

# 8. Development protocol

Our code touched `data/` only through the loader and `scripts/inspect_schema.py`, which prints
column names, counts and ranges, and we worked from aggregate outputs: counts, shares and scores.
Format assumptions came from DATA.md and are checked by hard assertions in the loader.

Nothing was fitted, tuned or calibrated on the competition match or its ground truth. All learned
weights and tuned constants come from the external games, and docs/PRIORS.md lists each one with its
source. We ran an aggregate sanity report on the match (scripts/sanity_report.py) only to look for
bugs; it raised no flags. No hand labels or other human-produced data about this match were used.

The parameters were first locked in commit `f30c2ba` on 2026-11-06, which was on GitHub before the
public ground truth was scored for the first time (2026-11-06 14:31 UTC, tag `phase1-submission`).

On 2026-11-14 we changed the pipeline once, and want to be plain about it. A review of our own
benchmark found that the imitation tags did not have the structure of the tagging we were given.
Ours were short windows cut at the labelled possession edges. The real file, of which we looked at
interval lengths and tag counts only, holds fixed 35 s set-piece windows, goal intervals of one and
a half to four minutes that include the kick-off, and overlapping phase windows of 10 to 30 s. A network
trained on the first kind learns to trust tag edges. Scored out of fold on the external games with
imitation tags of the real structure, the first version fell from 0.70 to 0.58 `spellF1`, below
its own score without tags. So we rebuilt the imitation, changed how the two tag windows are read
(section 1), retrained the network and retuned the decoder, all on the external games. The new
values were locked in commit `c395595`, pushed, and then scored once on the public slice
(2026-11-14 04:26 UTC).

The cause of the change was the structure of an input file, not a score, and the public ground
truth played no part in it. The amended version scores lower on the public five minutes than the
first one did (0.457 against 0.519 `spellF1`, higher on `macroF1`), and we submit it anyway: the
difference is well inside the bootstrap interval, and picking between versions by that number would
be tuning on the ground truth. Both versions can be regenerated from the repository, the first from
the tag and the submitted one from `main`.

# 9. Team

| member | role |
|---|---|
| Naufal Fahrozi | team lead and systems: repository, loader, converters, pipeline, packaging |
| Raymond Thymotimmanuel | football domain: possession and dead-ball rules, external label mapping, restarts |
| Tua Delima Sitompul | data science: features, network, decoder, benchmark, tuning and ablations |
