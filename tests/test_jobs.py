import json
import os

import pytest

from agamemnon.external import jobs


def test_atomic_write_leaves_no_tmp(tmp_path):
    path = str(tmp_path / "f.json")
    jobs.atomic_write_json(path, {"a": 1})
    jobs.atomic_write_json(path, {"a": 2})
    assert json.load(open(path)) == {"a": 2}
    assert os.listdir(tmp_path) == ["f.json"]


def test_job_start_clears_stale_pause_and_records_status(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOBS_DIR", str(tmp_path))
    job = jobs.Job("demo")
    open(job.pause_path, "w").close()
    job.start(mode="cv", folds=5)
    assert not job.pause_requested()
    job.update(state="training", step=25)
    status = jobs.read_status("demo")
    assert status["state"] == "training" and status["step"] == 25
    assert status["pid"] == os.getpid() and "command" in status
    open(job.pause_path, "w").close()
    assert job.pause_requested()
    assert jobs.all_jobs() == ["demo"]


torch = pytest.importorskip("torch")


def test_checkpoint_falls_back_to_backup_when_main_is_torn(tmp_path):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))
    import numpy as np
    import train_tcn

    net = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.StepLR(opt, 1)
    rng = np.random.default_rng(0)
    path = str(tmp_path / "fold0.ckpt")
    train_tcn.save_checkpoint(path, 500, net, opt, sched, rng)
    train_tcn.save_checkpoint(path, 1000, net, opt, sched, rng)
    assert train_tcn.load_checkpoint(path)["step"] == 1000
    with open(path, "r+b") as handle:          # a power cut in the middle of a write
        handle.truncate(100)
    assert train_tcn.load_checkpoint(path)["step"] == 500
