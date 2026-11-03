# External data

All tuning and model fitting in this repository happens on the games below. None of them is the
competition match, and no data about the competition match is used for fitting anything (see
`docs/PRIORS.md` for where every constant comes from).

`python scripts/fetch_external.py` downloads both used datasets into `external/` (gitignored).
IDSSE files are checked against the md5 sums published by figshare; SkillCorner is pinned to one
commit of its repository.

| dataset | games | rate | labels used | licence | used for | citation |
|---|---|---|---|---|---|---|
| IDSSE, DFL Bundesliga 2022/23 (1st and 2nd division) | 7 full matches | 25 fps, optical (TRACAB-type) tracking of all players and the ball | per-frame `BallStatus` (alive/dead) and `BallPossession` (home/away) from the DFL position feed | CC BY 4.0 ([figshare record](https://doi.org/10.6084/m9.figshare.28196177), licence stated on the record and in the paper) | primary benchmark: fitting the emission models, tuning, cross-validation | Bassek, M., Rein, R., Weber, H., Memmert, D. An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data* 12, 195 (2025). doi:10.1038/s41597-025-04505-y |
| SkillCorner Open Data, A-League 2024/25 | 20 full matches | 10 fps broadcast tracking, resampled to 25 fps | per-frame possession group, phases of play (which exist only while the ball is in play) | MIT ([LICENSE](https://github.com/SkillCorner/opendata/blob/master/LICENSE), commit `4340d27`) | second benchmark: broadcast-tracking noise, check that tuned values transfer to a different league and tracker | SkillCorner Open Data, https://github.com/SkillCorner/opendata. Data courtesy of SkillCorner. |
| Metrica Sports sample data | 3 games | 25 fps | event data, no per-frame possession | no licence file in the repository | not used | — |
| PFF FC World Cup 2022 | 64 games | broadcast tracking | event data | distributed on request under separate terms; not reviewed | not used | — |
| Alfheim (Tromsø IL, ZXY) | 3 games | 20 Hz | none for the ball | ball is not tracked | not used | — |

## How the external labels differ from the competition definition

The competition ground truth comes from corrected tracking: a frame is `D` where that tracking marks
the ball dead, and otherwise takes the team the tracking names as owning the ball, with contested or
loose balls staying with the last team that had it and only a *controlled* gain of the ball counting
as a change (CHALLENGE.md).

**IDSSE.** `BallStatus` is the same kind of flag the ground truth uses for dead ball, so it is taken
as is. `BallPossession` is closer to "who touched it last": it switches on deflections and duel
touches, giving a median spell of about 3 s and roughly 500 spells per match. To bring it to the
competition's controlled-possession rule, a spell shorter than the control threshold is absorbed by
the possession it interrupts, unless it directly follows a restart, where the restarting team is the
clear possessor. The threshold (2 s) is the low end of the CHALLENGE.md proxy "a player keeps the
ball for a few seconds". The mapping and its edge cases are in `docs/POSSESSION_RULES.md`.

**SkillCorner.** There is no ball-status flag. A frame is alive when it lies inside a phase of play,
dead otherwise. Inside a phase the per-frame possession group is used, and frames with an empty group
keep the last team that had the ball. The same control rule as for IDSSE is then applied. Phase
boundaries are less precise than a tracking flag, so SkillCorner is weighted as a robustness check
rather than as the main tuning target.

Neither dataset has Smart Tagging. The benchmark generates imitation tags from each game's own labels
with timing noise, missed tags, `N/A` teams and occasional wrong teams (`src/agamemnon/external/bench.py`),
so that tag-based logic is exercised; every configuration is also scored with no tags at all.

## Attribution

IDSSE data: © DFL Deutsche Fußball Liga, published by Bassek et al. under CC BY 4.0. Changes made
here: coordinates shifted to a corner origin, possession relabelled with the control rule above,
synthetic degradation for benchmarking. SkillCorner data: © SkillCorner, MIT licence; resampled to
25 fps and relabelled as described above.
