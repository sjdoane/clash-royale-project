"""live_ppo/bush_goblins.py and the driver's engine profile (2026-11-27, gen2_l live prep). Run:
.venv/bin/python -m pytest -q live_ppo/eval/test_bush_goblins.py
The rule tests are pure Python. The row, driver and replay tests run the driver / engine in a
subprocess on build/gen2_l with the L20 switches (CR_P0088 latches at import), reading the
recorded YOLO caches in runs/live (no detector process, no phone)."""
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path("/Users/samueldoane/cr-robot")
PY = str(REPO / ".venv/bin/python")
GEN2_L = REPO / "build/gen2_l/engine"
GEN2_K = REPO / "build/gen2_k/engine"
BASE14 = "tornado,jump,log,turret,king,speed,bankrows,stomp,jumpcol,pocketcells,kingpath,jumpedge,walklegs,hogspeed"
L20 = BASE14 + ",cloakboard,bushboard,bushpop,evogb,ghostsplash,ghostcloak"
CACHE = REPO / "runs/live/20261127-010105/raw.yolo.jsonl"
sys.path.insert(0, str(REPO / "live_ppo"))

import bush_goblins as BG  # noqa: E402


# ----------------------------------------------------------------- the rule, on synthetic tracks
def bush(tid, t, tile, born=0.0, side=1):
    return {"track": tid, "cls": BG.BUSH, "side": side, "tile_f": tile, "age": t - born, "held": 0.0,
            "born_tile": tile}


def gob(tid, t, tile, born, cls="goblin", side=1):
    return {"track": tid, "cls": cls, "side": side, "tile_f": tile, "age": t - born, "held": 0.0,
            "born_tile": tile}


def frames(lo, hi, dt=0.1):
    n = int(round((hi - lo) / dt))
    return [round(lo + i * dt, 3) for i in range(n + 1)]


def run(rule, script):
    """script(t) -> dets; returns {t: marked set} and every log event."""
    got, log = {}, []
    for t in frames(0.0, 6.0):
        got[t] = rule.update(script(t), t)
        log += rule.log
    return got, log


def test_pop_marks_the_goblins_born_there():
    rule = BG.BushPops()

    def s(t):
        d = []
        if t <= 2.0:
            d.append(bush(1, t, (3.0, 9.0)))
        if 2.1 <= t <= 4.0:
            d += [gob(2, t, (3.5, 9.5), born=2.1), gob(3, t, (2.4, 9.0), born=2.1, cls="spear-goblin")]
        return d
    got, log = run(rule, s)
    assert all(not got[t] for t in got if t < 2.25)
    assert got[2.3] == {2, 3} and got[4.0] == {2, 3}
    assert got[4.1] == set()                      # the tracks are gone
    pops = [e for e in log if e["ev"] == "pop"]
    assert len(pops) == 1 and pops[0]["bush"] == 1 and pops[0]["t"] == 2.0


def test_marks_capped_closest_in_time_first():
    rule = BG.BushPops()
    borns = {10: 2.0, 11: 2.1, 12: 1.5, 13: 2.2, 14: 0.9, 15: 0.7}

    def s(t):
        d = [bush(1, t, (3.0, 9.0))] if t <= 2.0 else []
        d += [gob(k, t, (3.0, 9.5), born=b) for k, b in borns.items() if b <= t <= 5.0]
        return d
    got, _ = run(rule, s)
    # all six start within -1.4 .. +1.5 s and are on screen when the pop is declared (2.3 s);
    # the cap keeps the 4 closest in time to the pop (2.0 s)
    assert got[3.5] == {10, 11, 12, 13}


def test_far_own_side_old_and_young_are_not_marked():
    rule = BG.BushPops()

    def s(t):
        d = [bush(1, t, (3.0, 9.0))] if t <= 2.0 else []
        if 0.2 <= t <= 1.0:                      # a young bush (0.8 s old when it goes)
            d.append(bush(5, t, (14.0, 9.0), born=0.2))
        if 2.1 <= t <= 4.0:
            d += [gob(20, t, (8.0, 9.0), born=2.1),               # 5 tiles away
                  gob(21, t, (3.0, 9.5), born=2.1, side=0),       # ours
                  gob(22, t, (3.0, 9.5), born=0.4)]               # started 1.6 s before the pop
        if 1.4 <= t <= 2.5:
            d.append(gob(23, t, (14.0, 9.0), born=1.4))          # near the young bush only
        return d
    got, log = run(rule, s)
    assert all(not v for v in got.values())
    assert [e["bush"] for e in log if e["ev"] == "pop"] == [1]   # bush 5 never reached MIN_AGE_S


def test_young_bush_at_threshold():
    rule = BG.BushPops(min_age_s=1.0)
    for t in frames(0.0, 0.9):
        rule.update([bush(1, t, (3.0, 9.0))], t)
    for t in frames(1.0, 2.0):
        rule.update([], t)
        assert not [e for e in rule.log if e["ev"] == "pop"]


def test_flicker_cancels_and_withdraws_marks():
    rule = BG.BushPops()

    def s(t):
        d = []
        if t <= 2.0 or 2.4 <= t <= 4.0:         # the bush blinks out for 0.3 s, then walks on
            d.append(bush(1, t, (3.0, 9.0 - 0.1 * max(0.0, t - 2.0))))
        if 2.1 <= t <= 5.0:
            d.append(gob(2, t, (3.4, 9.0), born=2.1))            # a goblin box on the bush
        return d
    got, log = run(rule, s)
    assert got[2.3] == {2}                        # provisional mark
    assert got[2.4] == set() and got[3.0] == set()
    c = [e for e in log if e["ev"] == "cancel"]
    assert c and c[0]["why"] == "same track" and c[0]["unmarked"] == [2]


def test_new_bush_id_nearby_cancels():
    rule = BG.BushPops()

    def s(t):
        d = []
        if t <= 2.0:
            d.append(bush(1, t, (3.0, 9.0)))
        if 2.3 <= t <= 4.0:
            d.append(bush(9, t, (3.2, 8.5), born=2.3))          # the tracker re-ids the walking bush
        if 2.1 <= t <= 4.0:
            d.append(gob(2, t, (3.0, 9.3), born=2.1))
        return d
    got, log = run(rule, s)
    assert got[2.6] == set() and got[3.5] == set()
    c = [e for e in log if e["ev"] == "cancel" and e["bush"] == 1]
    assert c and c[0]["why"] == "another bush track"


def test_far_bush_does_not_cancel():
    rule = BG.BushPops()

    def s(t):
        d = []
        if t <= 2.0:
            d.append(bush(1, t, (3.0, 9.0)))
        if 2.3 <= t <= 4.0:
            d.append(bush(9, t, (14.0, 9.0), born=2.3))         # the other lane
        if 2.1 <= t <= 4.0:
            d.append(gob(2, t, (3.0, 9.3), born=2.1))
        return d
    got, log = run(rule, s)
    assert got[3.5] == {2}
    assert not [e for e in log if e["ev"] == "cancel" and e["bush"] == 1]


def test_track_that_turns_goblin_starts_then():
    rule = BG.BushPops()

    def s(t):
        d = [bush(1, t, (3.0, 9.0))] if t <= 2.0 else []
        if 0.2 <= t <= 1.0:
            d.append({**gob(30, t, (3.0, 9.4), born=0.2), "cls": "knight"})   # old track, other class
        if 1.1 <= t <= 4.0:
            d.append(gob(30, t, (3.0, 9.4), born=0.2))         # same id, now a goblin box at 1.1
        return d
    got, log = run(rule, s)
    m = [e for e in log if e["ev"] == "mark"]
    assert got[2.3] == {30} and m and abs(m[0]["dt"] - (1.1 - 2.0)) < 1e-6


def test_reset_forgets_everything():
    rule = BG.BushPops()
    for t in frames(0.0, 2.0):
        rule.update([bush(1, t, (3.0, 9.0))], t)
    for t in frames(2.1, 2.4):
        rule.update([gob(2, t, (3.0, 9.4), born=2.1)], t)
    assert rule.marked == {2}
    rule.reset()
    assert (rule.bush, rule.gob, rule.pops, rule.marked, rule.log, rule.bush_hist) == ({}, {}, [], set(), [], [])
    assert rule.update([gob(2, 2.5, (3.0, 9.4), born=2.1)], 2.5) == set()


def test_untracked_dets_are_ignored():
    rule = BG.BushPops()
    assert rule.update([{"cls": "goblin", "side": 1, "tile_f": (3, 9)}], 0.0) == set()
    assert rule.update(None, 0.1) == set()


# ----------------------------------------------------------------- remap
@dataclass(frozen=True)
class U:
    card_sim_id: int
    unit_name: str


@dataclass(frozen=True)
class GS:
    units: tuple


def test_remap_only_bush_goblins_with_goblin_ids():
    gs = GS((U(4, "goblin"), U(23, "spear-goblin"), U(4, BG.NAME), U(23, BG.NAME), U(85, BG.NAME)))
    out = BG.remap(gs)
    assert [u.card_sim_id for u in out.units] == [4, 23, BG.ROW_ID, BG.ROW_ID, 85]
    assert out.units[0] is gs.units[0] and out.units[4] is gs.units[4]


def test_remap_identity_without_bush_goblins():
    gs = GS((U(4, "goblin"), U(85, "suspicious-bush")))
    assert BG.remap(gs) is gs


# ----------------------------------------------------------------- engine and driver (subprocess)
DRV = r"""
import json, os, sys
run_dir, rest = sys.argv[1], sys.argv[2:]
sys.argv = ["driver.py", "--run-dir", run_dir] + rest
sys.path.insert(0, "/Users/samueldoane/cr-robot/live_ppo")
import driver as D
out = {"bush_on": D.BUSH_ON, "cloaked_always": bool(D.ours.cloaked_always),
       "build": None if D.PROFILE is None else D.PROFILE.build,
       "trained_on": D.ours.trained_on, "bush_flag": D.ours.bush_goblins}
if os.environ.get("BG_ROWS"):
    from python_ai.models import perception_encoder as PE
    stock = set(PE.card_table())
    enc = D.EvoEncoder(PE)
    tab = PE.card_table()
    r = tab.get(1011)
    out["rows_added"] = sorted(set(tab) - stock)
    out["has85"] = 85 in tab
    out["row"] = None if r is None else {k: getattr(r, k) for k in
                 ("type_offset", "max_hp", "units", "flying", "antiair", "dps", "attack_range", "speed",
                  "is_building")}
    out["yolo"] = {c: D.yolo_card_id(c) for c in ("suspicious-bush", "goblin", "spear-goblin")}
print("RESULT " + json.dumps(out))
"""


def driver(tmp_path, engine, p0088, flags, ckpt=None, env=None):
    e = dict(os.environ, CR_ENGINE=str(engine), CR_YOLO_CACHE=str(CACHE))
    e.pop("CR_LIVE_TRAINED_ON", None)
    if p0088 is None:
        e.pop("CR_P0088", None)
    else:
        e["CR_P0088"] = p0088
    e.update(env or {})
    rd = tmp_path / f"rd{len(list(tmp_path.iterdir()))}"
    rd.mkdir()
    args = [PY, "-c", DRV, str(rd), *flags] + (["--", "--checkpoint", str(ckpt)] if ckpt else [])
    r = subprocess.run(args, cwd=str(Path(engine) / "perception"), env=e, capture_output=True, text=True)
    res = None
    for ln in r.stdout.splitlines():
        if ln.startswith("RESULT "):
            res = json.loads(ln[7:])
    return r, res


TRACK = ["--units", "yolo", "--track-units"]
PK = TRACK + ["--afford-project"]          # flags for the pocket-checkpoint copies


def _t(date):
    return time.mktime(time.strptime(date, "%Y-%m-%d %H:%M:%S"))


SRC_CK = REPO / "runs/pocket_1126/g2_pocket_fresh/snapshots/ep0101074.pth"   # read only, copied


@pytest.fixture
def l_run(tmp_path):
    """A run folder with one gen2_k block (resume 0) and one gen2_l L20 block (resume 1000); its
    snapshots are copies of a real pocket checkpoint (the driver loads the checkpoint at import)."""
    import shutil  # noqa: PLC0415
    d1, d2 = "2026-11-27 10:00:00", "2026-11-27 20:00:00"
    run = tmp_path / "g2_l_fake"
    (run / "snapshots").mkdir(parents=True)
    (run / "launch.txt").write_text(
        f"date {d1}\nengine {GEN2_K}\nsw {BASE14}\nafford_lead 1\nCR_P0088={BASE14}\n"
        f"date {d2}\nengine {GEN2_L}\nsw {L20}\nafford_lead 1\nCR_P0088={L20}\n")
    recs = [(_t(d1) + 60, 50, 50), (_t(d2) + 60, 1050, 50)]
    (run / "progress.jsonl").write_text("".join(json.dumps({"t": t, "episodes": e, "episodes_this_update": n})
                                                + "\n" for t, e, n in recs))
    for name, mt in (("ep0000900.pth", _t(d1) + 3000), ("ep0001100.pth", _t(d2) + 300), ("latest.pth", _t(d2) + 400)):
        f = run / "snapshots" / name
        shutil.copyfile(SRC_CK, f)
        os.utime(f, (mt, mt))
    return run


def test_driver_flags_defaults(tmp_path):
    r, res = driver(tmp_path, GEN2_L, L20, TRACK)
    assert r.returncode == 0, r.stderr[-2000:]
    assert res["bush_flag"] == "auto" and res["trained_on"] is None and res["build"] is None
    assert res["bush_on"] is False and res["cloaked_always"] is False
    assert "no launch.txt profile" in r.stdout


def test_gen2_l_rows_and_classes(tmp_path):
    r, res = driver(tmp_path, GEN2_L, L20, TRACK + ["--bush-goblins", "on"], env={"BG_ROWS": "1"})
    assert r.returncode == 0, r.stderr[-2000:]
    assert res["has85"] and 1011 in res["rows_added"] and 136 in res["rows_added"]
    row = res["row"]
    assert row["type_offset"] == 0 and row["max_hp"] == pytest.approx(304, abs=0.5)
    assert row["dps"] == pytest.approx(0.30476, abs=1e-4) and row["attack_range"] == pytest.approx(0.06667, abs=1e-4)
    assert row["speed"] == pytest.approx(0.08, abs=1e-4) and not row["flying"] and not row["is_building"]
    assert res["yolo"] == {"suspicious-bush": 85, "goblin": 4, "spear-goblin": 23}


def test_main_checkpoint_unchanged(tmp_path):
    ck = REPO / "runs/g2_ppo_s2r/snapshots/ep0496977.pth"
    r, res = driver(tmp_path, REPO / "build/gen2_c/engine", None, TRACK, ckpt=ck)
    assert r.returncode == 0, r.stderr[-2000:]
    assert res["build"] is None and res["bush_on"] is False and res["cloaked_always"] is False
    assert "no launch.txt profile" in r.stdout and "ghost mode: reveal gate" in r.stdout


def test_gen2_l_checkpoint_profile(tmp_path, l_run):
    r, res = driver(tmp_path, GEN2_L, L20, PK, ckpt=l_run / "snapshots/ep0001100.pth")
    assert r.returncode == 0, r.stderr[-2000:]
    assert res == {**res, "build": "gen2_l", "bush_on": True, "cloaked_always": True}
    assert "[forced by cloakboard]" in r.stdout and "bush goblins: on (--bush-goblins auto)" in r.stdout
    # the switch order does not matter
    r, res = driver(tmp_path, GEN2_L, ",".join(reversed(L20.split(","))), PK,
                    ckpt=l_run / "snapshots/ep0001100.pth")
    assert r.returncode == 0 and res["build"] == "gen2_l"


def test_gen2_k_checkpoint_profile(tmp_path, l_run):
    r, res = driver(tmp_path, GEN2_K, BASE14, PK, ckpt=l_run / "snapshots/ep0000900.pth")
    assert r.returncode == 0, r.stderr[-2000:]
    assert res["build"] == "gen2_k" and res["bush_on"] is False and res["cloaked_always"] is False
    assert "ghost mode: reveal gate" in r.stdout


@pytest.mark.parametrize("case", ["engine", "p0088_missing", "p0088_short", "no_tracker", "cloaked_rows_off",
                                  "cloaked_always_on_k", "latest", "override_disagrees"])
def test_driver_refusals(tmp_path, l_run, case):
    ck_l, ck_k = l_run / "snapshots/ep0001100.pth", l_run / "snapshots/ep0000900.pth"
    engine, p88, flags, ck, env = GEN2_L, L20, list(PK), ck_l, None
    if case == "engine":
        engine, p88 = GEN2_K, L20
    elif case == "p0088_missing":
        p88 = None
    elif case == "p0088_short":
        p88 = BASE14
    elif case == "no_tracker":
        flags = ["--units", "yolo", "--afford-project"]
    elif case == "cloaked_rows_off":
        flags += ["--cloaked-rows", "off"]
    elif case == "cloaked_always_on_k":
        engine, p88, ck = GEN2_K, BASE14, ck_k
        flags += ["--cloaked-always"]
    elif case == "latest":
        ck = l_run / "snapshots/latest.pth"
    elif case == "override_disagrees":
        engine, p88, ck = GEN2_K, BASE14, ck_l
        flags += ["--trained-on", "gen2_k"]
    r, res = driver(tmp_path, engine, p88, flags, ckpt=ck, env=env)
    assert r.returncode != 0 and res is None, (case, r.stdout[-1500:])
    want = {"engine": "the checkpoint was trained on", "p0088_missing": "CR_P0088=None, the checkpoint was trained with",
            "p0088_short": f"CR_P0088='{BASE14}', the checkpoint was trained with",
            "no_tracker": "--bush-goblins needs --track-units",
            "cloaked_rows_off": "--cloaked-rows off on a checkpoint trained with cloakboard",
            "cloaked_always_on_k": "--cloaked-always on a checkpoint trained without cloakboard",
            "latest": "cannot tell which launch.txt block", "override_disagrees": "override 'gen2_k' disagrees"}[case]
    assert want in r.stderr, (case, r.stderr[-1500:])


def test_latest_with_override(tmp_path, l_run):
    r, res = driver(tmp_path, GEN2_L, L20, PK + ["--trained-on", "gen2_l"], ckpt=l_run / "snapshots/latest.pth")
    assert r.returncode == 0, r.stderr[-2000:]
    assert res["build"] == "gen2_l" and res["bush_on"] and res["cloaked_always"]
    r, res = driver(tmp_path, GEN2_L, L20, PK, ckpt=l_run / "snapshots/latest.pth",
                    env={"CR_LIVE_TRAINED_ON": "build/gen2_l/engine"})
    assert r.returncode == 0 and res["build"] == "gen2_l"


def test_bush_goblins_off_warns(tmp_path, l_run):
    r, res = driver(tmp_path, GEN2_L, L20, PK + ["--bush-goblins", "off"], ckpt=l_run / "snapshots/ep0001100.pth")
    assert r.returncode == 0 and res["bush_on"] is False and "WARNING: --bush-goblins off" in r.stdout


# ----------------------------------------------------------------- recorded sessions
# Each window holds the pops measured 2026-11-27 (bush_goblins.py docstring). Track ids are the
# tracker's, counted from the window start.
EXPECT = {
    "20261124-094533:500:540": {"pops": 3, "cancels": 2, "named": [36, 37, 38]},
    "20261125-010327:235:310": {"pops": 5, "cancels": 3, "named": [39, 40, 118, 122, 125, 126]},
    "20261127-010105:1670:1800": {"pops": 3, "cancels": 1, "named": [10, 11, 16, 17, 69, 73]},
    "20261123-101008:0:100000": {"pops": 0, "cancels": 0, "named": []},
}


@pytest.fixture(scope="module")
def replay(tmp_path_factory):
    out = tmp_path_factory.mktemp("bg") / "replay.json"
    e = dict(os.environ, CR_ENGINE=str(GEN2_L), CR_P0088=L20)
    r = subprocess.run([PY, str(REPO / "live_ppo/eval/bush_goblins_replay.py"), *EXPECT, "--out", str(out)],
                       cwd=str(GEN2_L / "perception"), env=e, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(out.read_text())


@pytest.mark.parametrize("spec", list(EXPECT))
def test_recorded_pops(replay, spec):
    got, want = replay[spec], EXPECT[spec]
    evs = [e for _, e in got["events"]]
    assert sum(e["ev"] == "pop" for e in evs) == want["pops"]
    assert sum(e["ev"] == "cancel" for e in evs) == want["cancels"]
    named = sorted(int(k) for k, r in got["tracks"].items() if r["named"])
    assert named == want["named"]
    for k in named:          # opponent tracks; only their goblin-class emissions get the name
        r = got["tracks"][str(k)]
        assert set(r["side"]) == {"1"}
        assert r["named"] <= sum(n for c, n in r["cls"].items() if c in BG.GOBLIN_CLASSES)


def test_played_bush_goblin_boxes_and_false_bush_not_named(replay):
    t = replay["20261127-010105:1670:1800"]["tracks"]
    # nothing named after the last real pop (1737 s): the 1788 s single-box false bush
    assert not [k for k, r in t.items() if r["named"] and r["first"] > 1745]
    # goblin boxes on a bush as it is played (within 0.6 s of the bush's birth) are not named
    bushes = [r for r in t.values() if BG.BUSH in r["cls"]]
    for k, r in t.items():
        if r["named"]:
            assert all(abs(r["start"] - b["start"]) > 0.6 for b in bushes), k
