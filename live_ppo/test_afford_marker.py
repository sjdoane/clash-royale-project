"""live_ppo/afford_marker.py (2026-11-26 pm). Run: .venv/bin/python -m pytest -q live_ppo/test_afford_marker.py"""
import json
import os
import shutil
import sys

import pytest
import torch

REPO = "/Users/samueldoane/cr-robot"
sys.path.insert(0, REPO + "/live_ppo")
import afford_marker as AM  # noqa: E402

C5 = REPO + "/runs/bc3/g3j_L13_Wnone_pk_fxafbw/ep4.pth"
C2 = REPO + "/runs/bc3/g3j_L13_Wnone_pk_fx/ep4.pth"
MAIN = REPO + "/runs/g2_ppo_s2r/snapshots/ep0496977.pth"
REH = REPO + "/runs/pocket_1126/rehearsal/reh3_afford/snapshots/ep0000251.pth"


def blob(p):
    return torch.load(p, map_location="cpu", weights_only=False)


def test_unmarked_main_and_c2():
    assert AM.marker(MAIN, blob(MAIN)) is None
    assert AM.check(MAIN, blob(MAIN), False) is None          # the standard command: nothing changes
    assert AM.marker(C2, blob(C2)) is None


def test_c5_marked_by_pocket_json():
    assert "pocket.json" in AM.marker(C5, None)
    assert "Refusing" in AM.check(C5, None, False)
    assert AM.check(C5, None, True) is None


def test_afford_project_refused_on_unmarked():
    assert "no afford_lead marker" in AM.check(MAIN, blob(MAIN), True)


def test_run_snapshot_marked_by_launch_txt():
    assert "launch.txt" in AM.marker(REH, None)


def test_copy_marked_by_ppo_init(tmp_path):
    # a PPO snapshot copied out of its run folder: still marked through blob["ppo"]["init"]
    b = {"ppo": {"init": "runs/bc3/g3j_L13_Wnone_pk_fxafbw/ep4.pth"}}
    p = tmp_path / "copy.pth"
    torch.save(b, p)
    assert "PPO init" in AM.marker(str(p), blob(str(p)))
    assert "Refusing" in AM.check(str(p), blob(str(p)), False)


def test_copy_of_unmarked_stays_unmarked(tmp_path):
    p = tmp_path / "plain.pth"
    torch.save({"ppo": {"init": "runs/bc3/g2_big_L0_H10/ep4.pth"}}, p)
    assert AM.marker(str(p), blob(str(p))) is None


def test_sidecar(tmp_path):
    p = tmp_path / "x.pth"
    torch.save({}, p)
    assert AM.marker(str(p), {}) is None
    (tmp_path / "x.pth.afford_lead.json").write_text(json.dumps({"afford_lead": 1}))
    assert "sidecar" in AM.marker(str(p), {})


def test_launch_txt_last_line_wins(tmp_path):
    (tmp_path / "snapshots").mkdir()
    p = tmp_path / "snapshots" / "ep1.pth"
    torch.save({}, p)
    (tmp_path / "launch.txt").write_text("afford_lead 1\nafford_lead 0\n")
    assert AM.marker(str(p), {}) is None
    (tmp_path / "launch.txt").write_text("afford_lead 0\nafford_lead 1\n")
    assert "launch.txt" in AM.marker(str(p), {})
