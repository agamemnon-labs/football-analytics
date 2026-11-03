import numpy as np
import pytest

from agamemnon import loader
from agamemnon.schema import A, H

META = """<?xml version="1.0"?>
<main><Metadata>
<Teams><Team id="T1"><Name>Home FC</Name></Team><Team id="T2"><Name>Away FC</Name></Team></Teams>
<Players>
<Player id="1" teamId="T1"><Name>Track 1</Name></Player>
<Player id="2" teamId="T2"><Name>Track 2</Name></Player>
<Player id="3" teamId="T1"><Name>Track 3</Name></Player>
</Players></Metadata></main>"""

TAGS = """<file><ALL_INSTANCES>
<instance><ID>1</ID><start>1.0</start><end>3.0</end><code>SET PIECES</code>
<label><group>Team</group><text>Away FC</text></label>
<label><group>Type</group><text>Corner Kick</text></label></instance>
<instance><ID>2</ID><start>2.02</start><end>4.0</end><code>Build Up</code>
<label><group>Team</group><text>N/A</text></label></instance>
</ALL_INSTANCES></file>"""


def test_read_meta(tmp_path):
    path = tmp_path / "meta.xml"
    path.write_text(META)
    home, away, team = loader.read_meta(str(path), expected_slots=3)
    assert (home, away) == ("Home FC", "Away FC")
    assert team.tolist() == [H, A, H]


def test_read_meta_rejects_slot_count(tmp_path):
    path = tmp_path / "meta.xml"
    path.write_text(META)
    with pytest.raises(loader.BundleError):
        loader.read_meta(str(path), expected_slots=46)


def test_read_raw(tmp_path):
    lines = ["1:0.1,0.2,1.0;NaN,NaN,NaN;0.5,0.5,0.0:0.3,0.4,2.0",
             "2:0.1,0.2,1.0;0.9,0.8,NaN;0.5,0.5,0.0:NaN,NaN,NaN"]
    path = tmp_path / "raw.txt"
    path.write_text("\n".join(lines) + "\n")
    ball, slots = loader.read_raw(str(path), n_slots=3, expected_frames=2)
    assert ball.shape == (2, 2) and slots.shape == (2, 3, 2)
    assert np.allclose(ball[0], [0.3, 0.4]) and np.isnan(ball[1]).all()
    assert np.isnan(slots[0, 1]).all() and np.allclose(slots[1, 1], [0.9, 0.8])


def test_read_raw_errors_do_not_echo_content(tmp_path):
    path = tmp_path / "raw.txt"
    path.write_text("1:0.1,0.2,1.0:SECRET,0.4,2.0\n")
    with pytest.raises(loader.BundleError) as info:
        loader.read_raw(str(path), n_slots=1, expected_frames=1)
    assert "SECRET" not in str(info.value)


def test_read_raw_rejects_frame_gaps(tmp_path):
    path = tmp_path / "raw.txt"
    path.write_text("1:0.1,0.2,1.0:0.3,0.4,2.0\n3:0.1,0.2,1.0:0.3,0.4,2.0\n")
    with pytest.raises(loader.BundleError):
        loader.read_raw(str(path), n_slots=1, expected_frames=2)


def test_read_tags_maps_teams_and_frames(tmp_path):
    path = tmp_path / "tags.xml"
    path.write_text(TAGS)
    tags = loader.read_tags(str(path), "Home FC", "Away FC", 25, check_counts=False)
    assert [t.code for t in tags] == ["SET PIECES", "BUILD UP"]
    assert tags[0].team == A and tags[0].kind == "Corner Kick"
    assert (tags[0].start, tags[0].stop) == (25, 75)
    assert tags[1].team is None and tags[1].start == 50
