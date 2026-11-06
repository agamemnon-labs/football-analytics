"""Stability of the locked decoder: each tuned constant scaled by 0.5 and 1.5, one at a time.

Run on the external benchmark with the out-of-fold network outputs. It documents how sharp the
optimum is; it does not change any locked value.

    python scripts/sweep.py --oof cv_w96
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from agamemnon.external import bench  # noqa: E402
from ablation import _worker  # noqa: E402

PARAMS = ("NN_TEMPER", "NN_DEAD_BIAS", "SWITCH_COST", "END_COST", "START_COST", "SNAP_WINDOW_S")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--oof", default="cv_w96")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    os.environ["AGAMEMNON_OOF"] = args.oof
    from concurrent.futures import ProcessPoolExecutor
    keys = [(d, m, lv) for d, m in bench.all_games() for lv in bench.LEVELS]
    chunks = [c for c in (keys[i::args.workers] for i in range(args.workers)) if c]
    variants = [("locked", {})] + [("{} x{}".format(p, f), {"_scale": (p, f)})
                                   for p in PARAMS for f in (0.5, 1.5)]
    print("| variant | balanced spellF1 | worst cell spellF1 | balanced macroF1 |")
    print("|---|---|---|---|")
    with ProcessPoolExecutor(args.workers) as pool:
        for name, overrides in variants:
            rows = [r for part in pool.map(_worker, [(c, overrides) for c in chunks]) for r in part]
            by = {}
            for ds, s in rows:
                by.setdefault(ds, []).append(s)
            spell = np.mean([np.mean([s["spell_f1"] for s in v]) for v in by.values()])
            macro = np.mean([np.mean([s["macro_f1"] for s in v]) for v in by.values()])
            worst = min(s["spell_f1"] for _, s in rows)
            print("| {} | {:.3f} | {:.3f} | {:.3f} |".format(name, spell, worst, macro), flush=True)


if __name__ == "__main__":
    main()
