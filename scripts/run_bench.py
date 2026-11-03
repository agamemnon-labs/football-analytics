"""Score the locked configuration on every external game and degradation level.

Emission weights are refitted leave-one-game-out, so each game is scored with weights that never
saw it; the decoder parameters are the ones in priors.py. Prints per-game, per-level and mean
spellF1 / macroF1 with tags and without.

    python scripts/fetch_external.py && python scripts/build_cells.py && python scripts/run_bench.py
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from agamemnon import priors  # noqa: E402
from agamemnon.external import bench  # noqa: E402
import tune  # noqa: E402


def _worker(args):
    keys, coefs, tags, overrides = args
    out = []
    for key in keys:
        cell = bench.Cell(*key)
        prm = SimpleNamespace(**tune.base_params())
        prm.__dict__.update(overrides)
        if coefs is not None:
            prm.TEAM_COEF, prm.DEAD_COEF = coefs[cell.game]
        s = cell.score(prm, tags)
        out.append((cell.dataset, cell.game, cell.level, s))
    return out


def run(keys, coefs, tags, workers, overrides=None):
    chunks = [keys[i::workers] for i in range(workers)]
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        for part in pool.map(_worker, [(c, coefs, tags, overrides or {}) for c in chunks if c]):
            rows.extend(part)
    return sorted(rows, key=lambda r: (r[0], r[1], bench.LEVELS.index(r[2])))


def lomo_coefs(games, keys, tags):
    cells = {k: bench.Cell(*k) for k in keys}
    out = {}
    for d, m in games:
        train = [cells[k] for k in keys if (k[0], k[1]) != (d, m)]
        out["{}-{}".format(d, m)] = tune.fit_emissions(train, tags)
    return out


def report(rows, title):
    print("\n## {}".format(title))
    levels = bench.LEVELS
    datasets = sorted({r[0] for r in rows})
    print("\n| game | " + " | ".join(levels) + " | mean |")
    print("|---|" + "---|" * (len(levels) + 1))
    for game in sorted({r[1] for r in rows}):
        vals = [next(r[3]["spell_f1"] for r in rows if r[1] == game and r[2] == lv)
                for lv in levels]
        print("| {} | {} | {:.3f} |".format(game, " | ".join("{:.3f}".format(v) for v in vals),
                                             np.mean(vals)))
    print("\n| dataset | metric | " + " | ".join(levels) + " | mean |")
    print("|---|---|" + "---|" * (len(levels) + 1))
    for ds in datasets:
        for metric in ("spell_f1", "macro_f1"):
            vals = [np.mean([r[3][metric] for r in rows if r[0] == ds and r[2] == lv])
                    for lv in levels]
            print("| {} | {} | {} | {:.3f} |".format(
                ds, metric, " | ".join("{:.3f}".format(v) for v in vals), np.mean(vals)))
    for ds in datasets:
        sub = [r[3] for r in rows if r[0] == ds]
        print("{}: predicted {:.0f} vs true {:.0f} possessions per game, mean {:.1f} s vs {:.1f} s"
              .format(ds, np.mean([s["n_pred"] for s in sub]), np.mean([s["n_gt"] for s in sub]),
                      np.mean([s["dur_pred"] for s in sub]), np.mean([s["dur_gt"] for s in sub])))
    balanced = np.mean([np.mean([r[3]["spell_f1"] for r in rows if r[0] == ds])
                        for ds in datasets])
    print("dataset-balanced mean spellF1 {:.4f}".format(balanced))
    return balanced


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--locked", action="store_true",
                        help="score with the emission weights in priors.py instead of refitting")
    parser.add_argument("--emission", choices=("logistic", "network"), default=priors.EMISSION,
                        help="network: out-of-fold predictions of scripts/train_tcn.py")
    parser.add_argument("--oof", default="tcn_oof")
    args = parser.parse_args()
    os.environ["AGAMEMNON_OOF"] = args.oof
    games = bench.all_games()
    keys = [(d, m, lv) for d, m in games for lv in bench.LEVELS]
    overrides = {"EMISSION": args.emission}
    for tags in (True, False):
        if args.emission == "network" or args.locked:
            coefs = None
        else:
            coefs = lomo_coefs(games, keys, tags)
        rows = run(keys, coefs, tags, args.workers, overrides)
        report(rows, "with imitation Smart Tagging" if tags else "without tags")


if __name__ == "__main__":
    main()
