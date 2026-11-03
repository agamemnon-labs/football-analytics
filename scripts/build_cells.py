"""Precompute benchmark cells (features and design matrices) for every external game and level."""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agamemnon import priors  # noqa: E402
from agamemnon.external import bench  # noqa: E402


def convert(dataset, mid):
    bench.load_game(dataset, mid)
    return dataset, mid


def job(args):
    dataset, mid, level, force = args
    bench.build_cell(dataset, mid, level, priors, force=force)
    return dataset, mid, level


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    games = bench.all_games()
    # Convert each game once before fanning out, so workers only read the cache.
    with ProcessPoolExecutor(args.workers) as pool:
        for done in pool.map(convert, *zip(*games)):
            print("game", *done, flush=True)
        jobs = [(d, m, level, args.force) for d, m in games for level in bench.LEVELS]
        for done in pool.map(job, jobs):
            print("cell", *done, flush=True)
