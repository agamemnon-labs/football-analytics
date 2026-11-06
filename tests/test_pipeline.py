"""End-to-end check on the competition bundle; skipped when the gated data is not present."""

import hashlib

import pytest

from agamemnon import labels, loader, pipeline


def _bundle_present():
    try:
        loader.games.require(["atd_meta", "atd_raw", "smart_tagging"], loader.GAME, None)
        return True
    except BaseException:
        return False


@pytest.mark.skipif(not _bundle_present(), reason="competition data not available")
def test_submission_is_complete_and_deterministic(tmp_path):
    digests = []
    for run in range(2):
        out = tmp_path / "run{}.csv".format(run)
        codes, _ = pipeline.run()
        labels.write_csv(str(out), codes)
        lines = out.read_text(encoding="ascii").splitlines()
        assert lines[0] == "frame,label"
        assert len(lines) - 1 == loader.EXPECTED_FRAMES
        assert {line.rsplit(",", 1)[1] for line in lines[1:]} <= {"H", "A", "D", "X"}
        digests.append(hashlib.md5(out.read_bytes()).hexdigest())
    assert digests[0] == digests[1]
