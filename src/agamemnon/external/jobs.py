"""Bookkeeping for long training jobs: status file, pause marker, atomic writes.

A job is a folder under external/cache/jobs/<name>/ holding:

  status.json   what the job is doing (state, fold, step, speed, command), rewritten atomically
  PAUSE         present = please stop at the next safe point; the job removes it when it starts

The training script polls for PAUSE every few seconds, saves a checkpoint, writes state "paused"
and exits with PAUSED_EXIT. scripts/jobctl.py is the other side: it creates the marker, waits for
the exit and checks that the checkpoint loads before calling the pause safe.
"""

import json
import os
import socket
import sys
import time

from . import CACHE_DIR

JOBS_DIR = os.path.join(CACHE_DIR, "jobs")
PAUSED_EXIT = 75   # EX_TEMPFAIL: stopped on request, resumable


class Paused(Exception):
    """Raised inside a job when a pause was requested and progress has been saved."""


def atomic_write_bytes(path, data):
    """Write to a temporary file, flush it to disk, then rename over the target.

    After a power cut the target is either the old complete file or the new complete file,
    never a torn one; a stray .tmp is the only possible leftover.
    """
    tmp = path + ".tmp"
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_json(path, obj):
    atomic_write_bytes(path, json.dumps(obj, separators=(",", ":")).encode("utf-8"))


class Job:
    def __init__(self, name):
        self.name = name
        self.dir = os.path.join(JOBS_DIR, name)
        os.makedirs(self.dir, exist_ok=True)
        self.status_path = os.path.join(self.dir, "status.json")
        self.pause_path = os.path.join(self.dir, "PAUSE")
        self.fields = {}

    def start(self, **fields):
        """Claim the job for this process. A leftover PAUSE marker belongs to the run that was
        paused, not to this one, so it is cleared."""
        if os.path.isfile(self.pause_path):
            os.remove(self.pause_path)
        self.fields = {"job": self.name, "pid": os.getpid(), "host": socket.gethostname(),
                       "command": " ".join([".venv/Scripts/python", "-u"] + sys.argv),
                       "started": time.time()}
        self.update(state="starting", **fields)

    def update(self, **fields):
        self.fields.update(fields)
        self.fields["updated"] = time.time()
        atomic_write_json(self.status_path, self.fields)

    def pause_requested(self):
        return os.path.isfile(self.pause_path)

    def read(self):
        return read_status(self.name)


def read_status(name):
    path = os.path.join(JOBS_DIR, name, "status.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def all_jobs():
    if not os.path.isdir(JOBS_DIR):
        return []
    return sorted(d for d in os.listdir(JOBS_DIR) if os.path.isdir(os.path.join(JOBS_DIR, d)))
