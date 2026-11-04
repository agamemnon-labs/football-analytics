"""Ablations on the external benchmark: what each component of the locked pipeline is worth.

Decoder-level ablations reuse the out-of-fold network predictions, so every number is on games the
network never saw. Network input ablations retrain fold 0 without a feature group
(`scripts/train_tcn.py --only-fold 0 --drop GROUP`) and are compared on that fold only.

    python scripts/ablation.py --oof cv_w96
    python scripts/ablation.py --oof abl_touches --within abl_touches --variants "locked pipeline"
"""

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from agamemnon.external import CACHE_DIR, bench  # noqa: E402
import tune  # noqa: E402

VARIANTS = {
    "locked pipeline": {},
    "no Smart Tagging": {"_tags": False},
    "no hand-over snapping": {"SNAP_WINDOW_S": 0.0},
    "no kick-off forcing": {"_no_kickoff": True},
    "half switch penalty": {"_scale": ("SWITCH_COST", 0.5)},
    "double switch penalty": {"_scale": ("SWITCH_COST", 2.0)},
}


def _worker(args):
    keys, overrides = args
    out = []
    for key in keys:
        cell = bench.Cell(*key)
        prm = SimpleNamespace(**tune.base_params())
        tags = overrides.get("_tags", True)
        if "_scale" in overrides:
            name, factor = overrides["_scale"]
            setattr(prm, name, getattr(prm, name) * factor)
        for k, v in overrides.items():
            if not k.startswith("_"):
                setattr(prm, k, v)
        if overrides.get("_no_kickoff"):
            cell.kickoffs = [p[0] for p in cell.periods]
        out.append((cell.dataset, cell.score(prm, tags)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--oof", default="cv_w96")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--within", help="score only the games with outputs in this folder, "
                        "e.g. a single-fold run")
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    args = parser.parse_args()
    os.environ["AGAMEMNON_OOF"] = args.oof
    from concurrent.futures import ProcessPoolExecutor
    keys = [(d, m, lv) for d, m in bench.all_games() for lv in bench.LEVELS]
    if args.within:
        folder = os.path.join(CACHE_DIR, args.within)
        keys = [k for k in keys if os.path.isfile(os.path.join(folder, "{}-{}@{}.npz".format(*k)))]
        print("scoring {} cells with outputs in {}".format(len(keys), args.within))
    chunks = [c for c in (keys[i::args.workers] for i in range(args.workers)) if c]
    print("| variant | IDSSE spellF1 | SkillCorner spellF1 | balanced spellF1 | balanced macroF1 |")
    print("|---|---|---|---|---|")
    with ProcessPoolExecutor(args.workers) as pool:
        for name in args.variants:
            overrides = VARIANTS[name]
            rows = [r for part in pool.map(_worker, [(c, overrides) for c in chunks]) for r in part]
            by = {}
            for ds, s in rows:
                by.setdefault(ds, []).append(s)
            spell = {ds: np.mean([s["spell_f1"] for s in v]) for ds, v in by.items()}
            macro = {ds: np.mean([s["macro_f1"] for s in v]) for ds, v in by.items()}
            print("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                name, spell.get("idsse", np.nan), spell.get("skillcorner", np.nan),
                np.mean(list(spell.values())),
                np.mean(list(macro.values()))), flush=True)


if __name__ == "__main__":
    main()
