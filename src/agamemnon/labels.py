"""Turn decoded states into the submission alphabet and write the CSV."""

import numpy as np

from .schema import LETTERS, X


def finalise(decoded, graded_mask):
    """H/A/D on graded frames, X everywhere else (never scored, so nothing to guess)."""
    out = np.full(len(decoded), X, dtype=np.int8)
    out[graded_mask] = decoded[graded_mask]
    return out


def write_csv(path, codes):
    """frame,label with LF line endings, so the file's md5 does not depend on the platform."""
    letters = LETTERS[codes]
    body = "".join("{},{}\n".format(i, letter) for i, letter in enumerate(letters))
    with open(path, "w", newline="\n", encoding="ascii") as handle:
        handle.write("frame,label\n")
        handle.write(body)
    return path


def describe(codes, fps):
    """Aggregate counts only: label shares and possession statistics."""
    n = len(codes)
    shares = {LETTERS[k]: int(np.sum(codes == k)) for k in range(4)}
    alive = (codes == 0) | (codes == 1)
    change = np.flatnonzero(np.diff(codes) != 0) + 1
    bounds = np.concatenate(([0], change, [n]))
    spells = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if alive[a]]
    lengths = np.array([b - a for a, b in spells]) / fps if spells else np.zeros(0)
    return {
        "frames": n,
        "labels": shares,
        "possessions": len(spells),
        "median_possession_s": float(np.median(lengths)) if len(lengths) else 0.0,
        "mean_possession_s": float(np.mean(lengths)) if len(lengths) else 0.0,
    }
