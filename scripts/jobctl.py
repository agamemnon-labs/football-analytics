"""Status, pause and checkpoint checks for training jobs started by scripts/train_tcn.py.

    python scripts/jobctl.py status [JOB]     what each job is doing, its speed and time left
    python scripts/jobctl.py pause JOB        stop at the next safe point and verify it is safe
    python scripts/jobctl.py check JOB        verify the job's checkpoint without touching it

Resuming is running the job's own command again (printed by `status`); finished folds are
skipped and training continues from the saved step.

`pause` asks the job to stop instead of killing it. It prints SAFE only when the process has
exited on its own, its checkpoint opens as a complete archive and loads with the expected
contents, and the process no longer holds GPU memory.
"""

import argparse
import os
import subprocess
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agamemnon.external import CACHE_DIR  # noqa: E402
from agamemnon.external.jobs import JOBS_DIR, all_jobs, read_status  # noqa: E402

CKPT_KEYS = ("step", "net", "opt", "sched", "rng", "torch_rng")


def alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", "PID eq {}".format(pid), "/NH", "/FO", "CSV"],
                             capture_output=True, text=True).stdout
        return '","{}",'.format(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def gpu_apps():
    """{pid: MiB} of processes holding GPU memory, or None when nvidia-smi is unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    apps = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            apps[int(parts[0])] = int(parts[1]) if parts[1].isdigit() else 0
    return apps


def gpu_summary():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,"
                              "memory.total,temperature.gpu,power.draw",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=10).stdout.strip().splitlines()[0]
        u, m, t, c, w = [x.strip() for x in out.split(",")]
        return "GPU {}% | VRAM {:.1f}/{:.1f} GB | {} C | {} W".format(
            u, int(m) / 1024, int(t) / 1024, c, w)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return "GPU: nvidia-smi unavailable"


def checkpoint_of(status):
    if status.get("mode") == "final":
        return os.path.join(JOBS_DIR, status["job"], "final.ckpt")
    return os.path.join(CACHE_DIR, status["job"], "fold{}.ckpt".format(status.get("fold", 0)))


def check_file(path):
    """(ok, detail) for one checkpoint: complete zip archive, loads, has every expected key."""
    if not os.path.isfile(path):
        return False, "missing"
    try:
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad is not None:
                return False, "CRC error in {}".format(bad)
    except zipfile.BadZipFile:
        return False, "truncated archive"
    try:
        import torch
        state = torch.load(path, map_location="cpu", weights_only=False)
    except ImportError:
        return True, "archive complete (contents not checked: torch missing)"
    except Exception as error:
        return False, "does not load: {}".format(type(error).__name__)
    missing = [k for k in CKPT_KEYS if k not in state]
    if missing:
        return False, "incomplete, missing {}".format(missing)
    return True, "step {} of {}, optimizer, scheduler and RNG state present".format(
        state["step"], state.get("steps", "? (older checkpoint format)"))


def check(name):
    status = read_status(name)
    if status is None:
        return False, "no such job"
    path = checkpoint_of(status)
    ok, detail = check_file(path)
    if not ok and os.path.isfile(path + ".bak"):
        bak_ok, bak_detail = check_file(path + ".bak")
        if bak_ok:
            return True, "main checkpoint {}, backup usable: {}".format(detail, bak_detail)
    if not os.path.isfile(path) and not os.path.isfile(path + ".bak"):
        # Between folds, or the fold had finished training: nothing to resume mid-way.
        return True, "no mid-fold checkpoint needed (fold {} not started or already trained)".format(
            fold_of(status))
    return ok, detail


def fmt_time(seconds):
    if seconds is None:
        return "?"
    if seconds < 90:
        return "{:.0f} s".format(seconds)
    if seconds < 5400:
        return "{:.0f} min".format(seconds / 60)
    return "{:.1f} h".format(seconds / 3600)


def fold_of(status):
    return status.get("fold", 0) + 1


def describe(name):
    s = read_status(name)
    if s is None:
        print("{}: no status".format(name))
        return
    running = alive(s.get("pid"))
    state = s.get("state")
    if running:
        shown = state
    elif state in ("paused", "done", "failed"):
        shown = state
    elif state == "inference":
        shown = ("stopped without pausing during inference (crash, power cut or killed): training "
                 "of fold {} is saved, rerunning finishes the remaining outputs".format(fold_of(s)))
    else:
        shown = ("stopped without pausing (crash, power cut or killed): resumable from step "
                 "{}".format(s.get("checkpoint_step", 0)))
    step, steps, speed = s.get("step", 0), s.get("steps"), s.get("sec_per_step")
    fold, folds = s.get("fold", 0), s.get("folds", 1)
    print("job        {}".format(name))
    print("state      {}".format(shown))
    if steps:
        print("progress   fold {} of {}, step {}/{} ({:.0%} of this fold)".format(
            fold + 1, folds, step, steps, step / steps))
    if speed and steps:
        left_fold = (steps - step) * speed
        # Each later fold costs a full training run plus data loading and inference.
        per_fold = steps * speed + 240
        left = left_fold + max(0, folds - fold - 1) * per_fold
        print("speed      {:.3f} s/step  |  fold left {}  |  job left about {}".format(
            speed, fmt_time(left_fold), fmt_time(left)))
    print("checkpoint step {}".format(s.get("checkpoint_step", "none yet")))
    print("updated    {} ago".format(fmt_time(time.time() - s.get("updated", time.time()))))
    if running and time.time() - s.get("updated", 0) > 300:
        print("WARNING    no update for over 5 minutes: possibly stuck")
    if s.get("error"):
        print("error      {}".format(s["error"]))
    print("resume     {}".format(s.get("command")))
    print(gpu_summary())


def pause(name, max_wait):
    s = read_status(name)
    if s is None:
        sys.exit("NOT SAFE: no job named {}".format(name))
    pid = s.get("pid")
    if not alive(pid):
        print("job {} is not running (state {}); nothing to stop".format(name, s.get("state")))
    else:
        with open(os.path.join(JOBS_DIR, name, "PAUSE"), "w") as handle:
            handle.write(str(time.time()))
        speed = s.get("sec_per_step") or 0.1
        print("pause requested; expected stop within about {} (state: {})".format(
            fmt_time(25 * speed + 10 if s.get("state") == "training" else 90), s.get("state")),
            flush=True)
        t0 = time.time()
        while alive(pid):
            if time.time() - t0 > max_wait:
                sys.exit("NOT SAFE: the job did not stop within {}; it is still running and "
                         "nothing was killed".format(fmt_time(max_wait)))
            time.sleep(1)
        print("process exited after {}".format(fmt_time(time.time() - t0)), flush=True)
    s = read_status(name)
    ok, detail = check(name)
    apps = gpu_apps()
    holding = apps is not None and pid in apps
    print("state      {}".format(s.get("state")))
    print("checkpoint {} - {}".format("OK" if ok else "BAD", detail))
    print("GPU memory of the job {}".format("still held" if holding else "released"))
    print(gpu_summary())
    if ok and not holding and s.get("state") in ("paused", "done"):
        print("SAFE: the laptop can be closed or shut down; resume with:\n  {}".format(
            s.get("command")))
        return 0
    print("NOT SAFE: see the lines above before shutting down")
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("status")
    st.add_argument("job", nargs="?")
    pa = sub.add_parser("pause")
    pa.add_argument("job")
    pa.add_argument("--max-wait", type=float, default=600, help="seconds")
    ch = sub.add_parser("check")
    ch.add_argument("job")
    args = parser.parse_args()
    if args.cmd == "status":
        names = [args.job] if args.job else all_jobs()
        if not names:
            print("no jobs")
        for name in names:
            describe(name)
            print()
        return 0
    if args.cmd == "pause":
        return pause(args.job, args.max_wait)
    ok, detail = check(args.job)
    print("{} - {}".format("OK" if ok else "BAD", detail))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
