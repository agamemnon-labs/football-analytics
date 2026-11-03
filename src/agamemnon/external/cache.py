"""Store converted external matches as compressed npz so the benchmark does not re-parse them."""

import json
import os

import numpy as np

from ..schema import Match, Tag
from . import CACHE_DIR

FORMAT_VERSION = 1


def path_for(name):
    return os.path.join(CACHE_DIR, "{}.npz".format(name))


def save(match):
    os.makedirs(CACHE_DIR, exist_ok=True)
    meta = {
        "version": FORMAT_VERSION, "name": match.name, "fps": match.fps,
        "pitch": list(match.pitch), "periods": [list(p) for p in match.periods],
        "kickoffs": list(match.kickoffs),
        "tags": [[t.code, t.start, t.stop, t.team, t.kind] for t in match.tags],
    }
    np.savez_compressed(path_for(match.name), ball=match.ball, players=match.players,
                        team=match.team, labels=match.labels, meta=json.dumps(meta))


def load(name):
    path = path_for(name)
    if not os.path.isfile(path):
        return None
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        if meta.get("version") != FORMAT_VERSION:
            return None
        return Match(
            name=meta["name"], fps=meta["fps"], pitch=tuple(meta["pitch"]),
            ball=data["ball"], players=data["players"], team=data["team"],
            periods=[tuple(p) for p in meta["periods"]], kickoffs=meta["kickoffs"],
            tags=[Tag(*t) for t in meta["tags"]], labels=data["labels"],
        ).validate()
