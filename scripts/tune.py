"""Fit the emission models and tune the decoder on the external benchmark, leave-one-game-out.

1. For every held-out game, the two logistic emission models are fitted on all *other* games (every
   degradation level, every few frames), so no game is ever scored with weights it helped fit.
2. A seeded random search over the decoder parameters is scored on every (game, level) cell with
   those fold-specific weights.
3. The chosen configuration maximises the mean spellF1 over the two datasets (each dataset counts
   once, whatever its number of games) minus half the standard deviation across games, so a
   setting that wins on some games and collapses on others loses.
4. Nested estimate: for each held-out game, the configuration that is best on the other games is
   scored on it. That number is what the benchmark reports as the honest expectation.

    python scripts/tune.py --configs 150 --workers 8
    python scripts/tune.py --final        # refit emission weights on all games, print priors

Everything is deterministic for a given set of cells and arguments.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agamemnon import priors  # noqa: E402
from agamemnon.external import CACHE_DIR, bench, fit  # noqa: E402
from agamemnon.schema import A, D, H  # noqa: E402

SEARCH = {
    # name: (low, high, log-scale)
    "TEAM_TEMPER": (0.02, 0.6, True),
    "DEAD_TEMPER": (0.02, 0.6, True),
    "SWITCH_COST": (2.0, 40.0, True),
    "END_COST": (2.0, 40.0, True),
    "START_COST": (2.0, 40.0, True),
    "DEAD_PRIOR": (0.2, 0.6, False),
    "SNAP_WINDOW_S": (0.0, 2.0, False),
}
SEARCH_NETWORK = {
    "NN_TEMPER": (0.1, 3.0, True),
    "NN_DEAD_BIAS": (-2.0, 2.0, False),
    "SWITCH_COST": (8.0, 200.0, True),
    "END_COST": (5.0, 150.0, True),
    "START_COST": (5.0, 150.0, True),
    "SNAP_WINDOW_S": (0.0, 1.5, False),
}
SUBSAMPLE = 7
L2 = 1.0
RESULTS = os.path.join(CACHE_DIR, "tune_results_{}.json")


def base_params():
    return {k: getattr(priors, k) for k in dir(priors) if k.isupper()}


def sample_configs(n, seed=0, space=None):
    space = space or SEARCH
    rng = np.random.default_rng(seed)
    configs = [{k: getattr(priors, k) for k in space}]
    for _ in range(n - 1):
        cfg = {}
        for name, (lo, hi, log) in space.items():
            u = rng.random()
            cfg[name] = float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))) if log
                              else lo + u * (hi - lo))
        configs.append(cfg)
    return configs


def training_rows(cells, tags=True):
    """Subsampled design rows and targets from cells, with dataset-balanced weights."""
    team_X, team_y, team_w, dead_X, dead_y, dead_w = [], [], [], [], [], []
    per_dataset = {}
    for cell in cells:
        per_dataset[cell.dataset] = per_dataset.get(cell.dataset, 0) + 1
    for cell in cells:
        Xt, Xd = cell.designs(tags)
        gt = cell.gt
        idx = np.arange(0, len(gt), SUBSAMPLE)
        g = idx[gt[idx] < 3]
        alive = g[(gt[g] == H) | (gt[g] == A)]
        w = 1.0 / per_dataset[cell.dataset]
        team_X.append(Xt[alive]); team_y.append(gt[alive] == H)
        team_w.append(np.full(len(alive), w))
        dead_X.append(Xd[g]); dead_y.append(gt[g] == D); dead_w.append(np.full(len(g), w))
    cat = np.concatenate
    return (cat(team_X), cat(team_y), cat(team_w), cat(dead_X), cat(dead_y), cat(dead_w))


def fit_emissions(cells, tags=True):
    tX, ty, tw, dX, dy, dw = training_rows(cells, tags)
    return (tuple(float(v) for v in fit.logistic(tX, ty, L2, tw)),
            tuple(float(v) for v in fit.logistic(dX, dy, L2, dw)))


def load_cells(keys):
    return [bench.Cell(*k) for k in keys]


def _eval_worker(args):
    keys, coefs, configs, tags = args
    cells = load_cells(keys)
    base = base_params()
    out = []
    for ci, cfg in enumerate(configs):
        for cell in cells:
            prm = SimpleNamespace(**base)
            prm.__dict__.update(cfg)
            if coefs is not None:
                prm.TEAM_COEF, prm.DEAD_COEF = coefs[cell.game]
            s = cell.score(prm, tags)
            out.append((ci, cell.game, cell.dataset, cell.level, s["spell_f1"], s["macro_f1"],
                        s["n_pred"], s["n_gt"]))
    return out


def evaluate(keys, coefs, configs, workers, tags=True):
    chunks = [keys[i::workers] for i in range(workers)]
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        for part in pool.map(_eval_worker, [(c, coefs, configs, tags) for c in chunks if c]):
            rows.extend(part)
    return rows


def summarise(rows, n_configs):
    """Per config: dataset-balanced mean spellF1, std across games, macroF1."""
    table = []
    for ci in range(n_configs):
        r = [x for x in rows if x[0] == ci]
        by_game = {}
        for _, game, ds, _, spell, macro, _, _ in r:
            by_game.setdefault((ds, game), []).append((spell, macro))
        game_spell = {k: np.mean([v[0] for v in vals]) for k, vals in by_game.items()}
        game_macro = {k: np.mean([v[1] for v in vals]) for k, vals in by_game.items()}
        datasets = sorted({k[0] for k in game_spell})
        ds_mean = [np.mean([v for k, v in game_spell.items() if k[0] == d]) for d in datasets]
        spell = float(np.mean(ds_mean))
        std = float(np.std(list(game_spell.values())))
        macro = float(np.mean([np.mean([v for k, v in game_macro.items() if k[0] == d])
                               for d in datasets]))
        table.append({"config": ci, "spell": spell, "std": std, "macro": macro,
                      "objective": spell - 0.5 * std,
                      "per_dataset": dict(zip(datasets, map(float, ds_mean))),
                      "per_game": {k[1]: float(v) for k, v in game_spell.items()}})
    return table


def nested_estimate(table, games):
    """For each held-out game, pick the best config on the others and score it on that game."""
    out = {}
    for g in games:
        best = max(table, key=lambda t: np.mean([v for k, v in t["per_game"].items() if k != g])
                   - 0.5 * np.std([v for k, v in t["per_game"].items() if k != g]))
        out[g] = (best["config"], best["per_game"][g])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", type=int, default=120)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--levels", nargs="+", default=list(bench.LEVELS))
    parser.add_argument("--no-tags", action="store_true")
    parser.add_argument("--emission", choices=("logistic", "network"), default="logistic",
                        help="network: score out-of-fold log-probabilities from train_tcn.py")
    parser.add_argument("--final", action="store_true",
                        help="fit the emission models on every game and print them")
    parser.add_argument("--oof", default="tcn_oof", help="out-of-fold folder for --emission network")
    args = parser.parse_args()
    os.environ["AGAMEMNON_OOF"] = args.oof
    tags = not args.no_tags

    games = bench.all_games()
    keys = [(d, m, lv) for d, m in games for lv in args.levels]
    t0 = time.time()
    if args.final:
        coefs = fit_emissions(load_cells(keys), tags)
        print("TEAM_COEF = {}".format(tuple(round(v, 4) for v in coefs[0])))
        print("DEAD_COEF = {}".format(tuple(round(v, 4) for v in coefs[1])))
        return

    # Fold-specific emission weights: fit without the held-out game.
    coefs = {}
    all_cells = None
    network = args.emission == "network"
    for d, m in ([] if network else games):
        game = "{}-{}".format(d, m)
        train = [k for k in keys if (k[0], k[1]) != (d, m)]
        if all_cells is None:
            all_cells = {k: c for k, c in zip(keys, load_cells(keys))}
        coefs[game] = fit_emissions([all_cells[k] for k in train], tags)
        print("fitted {} in {:.0f}s".format(game, time.time() - t0), flush=True)
    del all_cells

    if network:
        coefs = None
        configs = sample_configs(args.configs, args.seed, SEARCH_NETWORK)
        for cfg in configs:
            cfg["EMISSION"] = "network"
    else:
        configs = sample_configs(args.configs, args.seed)
    rows = evaluate(keys, coefs, configs, args.workers, tags)
    table = summarise(rows, len(configs))
    ranked = sorted(table, key=lambda t: -t["objective"])
    nested = nested_estimate(table, sorted({"{}-{}".format(d, m) for d, m in games}))
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(RESULTS.format(args.emission if args.emission == "logistic" else args.oof), "w") as handle:
        json.dump({"configs": configs, "table": table, "nested": nested, "coefs": coefs,
                   "rows": rows, "levels": args.levels, "tags": tags}, handle)

    print("\nrank  cfg  objective  spellF1  std    macroF1  per dataset")
    for i, t in enumerate(ranked[:10]):
        print("{:>4}  {:>3}  {:.4f}     {:.4f}   {:.3f}  {:.4f}   {}".format(
            i + 1, t["config"], t["objective"], t["spell"], t["std"], t["macro"],
            {k: round(v, 4) for k, v in t["per_dataset"].items()}))
    best = ranked[0]
    print("\nbest config:", json.dumps(configs[best["config"]], indent=1))
    est = [v[1] for v in nested.values()]
    print("nested leave-one-game-out spellF1: mean {:.4f}  min {:.4f}  max {:.4f}".format(
        np.mean(est), np.min(est), np.max(est)))
    print("done in {:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
