"""Regenerate the submission from the match bundle.

    python -m agamemnon.pipeline --out evaluation/results/Agamemnon.csv

Reads only the tactical ATD (metadata and raw) and the Smart Tagging XML. Prints aggregate counts
only.
"""

import argparse
import hashlib
import os
import time

from . import decoder, deadball, features, labels, loader, priors


def run(data_dir=None, prm=priors, use_tags=True):
    match = loader.load(data_dir)
    if not use_tags:
        match.tags = []
    F = features.compute(match, prm)
    T = deadball.tag_features(match, prm)
    decoded = decoder.decode(F, match, prm, T)
    return labels.finalise(decoded, match.graded_mask()), match


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(loader.ROOT, "evaluation", "results",
                                                      "Agamemnon.csv"))
    parser.add_argument("--data-dir", default=None, help="defaults to games.DATA_DIR")
    parser.add_argument("--no-tags", action="store_true", help="ignore Smart Tagging")
    args = parser.parse_args()

    t0 = time.time()
    codes, match = run(args.data_dir, use_tags=not args.no_tags)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    labels.write_csv(args.out, codes)
    with open(args.out, "rb") as handle:
        digest = hashlib.md5(handle.read()).hexdigest()
    summary = labels.describe(codes, match.fps)
    print("wrote {}  ({} rows, md5 {})".format(args.out, summary["frames"], digest))
    print("labels {}".format(summary["labels"]))
    print("possessions {}  median {:.1f} s  mean {:.1f} s".format(
        summary["possessions"], summary["median_possession_s"], summary["mean_possession_s"]))
    print("done in {:.0f} s".format(time.time() - t0))


if __name__ == "__main__":
    main()
