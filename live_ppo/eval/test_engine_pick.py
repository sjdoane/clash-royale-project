"""live_ppo/engine_pick.py and its live_session.sh block (2026-11-27, gen2_l live prep).

    .venv/bin/python -m pytest -q live_ppo/eval/test_engine_pick.py

Pure Python (no engine import). The live_session.sh test runs only the engine-pick block cut out
of the script (between its "# Engine per checkpoint" comment and the first "fi" after it), never
the script itself: nothing is paused, launched or tapped.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path("/Users/samueldoane/cr-robot")
sys.path.insert(0, str(REPO / "live_ppo"))
import engine_pick as P  # noqa: E402

MAIN = REPO / "runs/g2_ppo_s2r/snapshots/ep0496977.pth"
C5 = REPO / "runs/bc3/g3j_L13_Wnone_pk_fxafbw/ep4.pth"
RUN = REPO / "runs/pocket_1126/g2_pocket_fresh"
BASE14 = "tornado,jump,log,turret,king,speed,bankrows,stomp,jumpcol,pocketcells,kingpath,jumpedge,walklegs,hogspeed"
L20 = BASE14 + ",cloakboard,bushboard,bushpop,evogb,ghostsplash,ghostcloak"
J11 = "tornado,jump,log,turret,king,speed,bankrows,stomp,jumpcol,pocketcells,kingpath"
GEN2_K = (REPO / "build/gen2_k/engine").resolve()
GEN2_L = (REPO / "build/gen2_l/engine").resolve()


def test_no_launch_txt_no_profile():
    assert P.pick(MAIN) is None                 # main ep0496977: today's behaviour
    assert P.pick(C5) is None                   # imitation model
    with pytest.raises(P.PickError):            # an override only chooses among a run's blocks
        P.pick(MAIN, "gen2_l")


def test_real_restart_run_blocks():
    snaps = sorted((RUN / "snapshots").glob("ep*.pth"))
    assert snaps
    n = {"gen2_j": 0, "gen2_k": 0}
    for s in snaps:                              # gen2_k from the bridge-patch switch at ep 16,181
        p = P.pick(s)
        ep = int(s.stem[2:])
        want = (GEN2_K, BASE14) if ep > 16181 else ((REPO / "build/gen2_j/engine").resolve(), J11)
        assert p.engine == want[0] and P.same_sw(p.sw, want[1]) and not p.notes, s
        assert not p.has("cloakboard") and not p.has("bushpop")
        n[p.build] += 1
    assert n["gen2_j"] and n["gen2_k"]
    p = P.pick(RUN / "state_ep0016181_pre_gen2k1126.pth")       # the state the gen2_k block resumed from
    assert p.build == "gen2_j" and P.same_sw(p.sw, J11)
    p = P.pick(RUN / "latest_ep0008286_pre_bld1126.pth")
    assert p.build == "gen2_j" and p.block_date == "2026-11-26 13:50:22"
    p = P.pick(RUN / "state_ep0152502_pre_r2anchor1127.pth")
    assert p.build == "gen2_k" and p.block_date == "2026-11-26 15:28:50"


def test_real_run_cannot_tell_latest():
    with pytest.raises(P.PickError, match="CR_LIVE_TRAINED_ON"):
        P.pick(RUN / "latest.pth")
    p = P.pick(RUN / "latest.pth", "gen2_k")
    assert p.engine == GEN2_K and "override" in p.how
    p = P.pick(RUN / "latest.pth", "build/gen2_k/engine")
    assert p.engine == GEN2_K
    with pytest.raises(P.PickError, match="no block"):
        P.pick(RUN / "latest.pth", "gen2_l")
    snap = sorted((RUN / "snapshots").glob("ep*.pth"))[-1]
    with pytest.raises(P.PickError, match="disagrees"):
        P.pick(snap, "gen2_j")
    assert P.pick(snap, "gen2_k").engine == GEN2_K   # an override that agrees is fine


def _block(date, engine, sw):
    return (f"date {date}\ninit runs/bc3/x/ep4.pth\nengine {engine}\nsw {sw}\nafford_lead 1\n"
            f"CR_P0088={sw}\nstage C pid 1 {date} flags --kl-end 0.15\n")


def _t(date):
    return time.mktime(time.strptime(date, "%Y-%m-%d %H:%M:%S"))


@pytest.fixture
def fake_run(tmp_path):
    """gen2_k block at D1 (resume 0), gen2_l block at D2 (resume 1000)."""
    d1, d2 = "2026-11-27 10:00:00", "2026-11-27 20:00:00"
    run = tmp_path / "g2_fake"
    (run / "snapshots").mkdir(parents=True)
    (run / "launch.txt").write_text(
        _block(d1, REPO / "build/gen2_k/engine", BASE14)
        + "# a comment; engine_gen2_l lines with suffixes are not engine lines\nstart_ref_gen2_l x\n"
        + _block(d2, REPO / "build/gen2_l/engine", L20))
    recs = [(_t(d1) + 60, 50, 50), (_t(d1) + 3600, 1000, 50), (_t(d2) + 60, 1050, 50), (_t(d2) + 600, 1500, 50)]
    (run / "progress.jsonl").write_text("".join(json.dumps({"t": t, "episodes": e, "episodes_this_update": n}) + "\n"
                                                for t, e, n in recs))
    for name, mt in (("ep0000900.pth", _t(d1) + 3000), ("ep0001000.pth", _t(d1) + 3600),
                     ("ep0001100.pth", _t(d2) + 300), ("ep0001200.pth", _t(d1) + 3700)):
        f = run / "snapshots" / name
        f.write_bytes(b"x")
        os.utime(f, (mt, mt))
    return run


def test_fake_run_blocks(fake_run):
    s = fake_run / "snapshots"
    p = P.pick(s / "ep0000900.pth")
    assert p.engine == GEN2_K and P.same_sw(p.sw, BASE14)
    p = P.pick(s / "ep0001000.pth")              # the episode the gen2_l block resumed from: gen2_k wrote it
    assert p.engine == GEN2_K
    p = P.pick(s / "ep0001100.pth")
    assert p.engine == GEN2_L and P.same_sw(p.sw, L20) and p.p0088 == L20
    assert p.has("cloakboard") and p.has("bushboard") and p.has("bushpop")
    with pytest.raises(P.PickError, match="older than that block"):
        P.pick(s / "ep0001200.pth")              # gen2_l by episode, but written before the gen2_l block
    assert P.pick(s / "ep0001200.pth", "gen2_l").engine == GEN2_L   # explicit: allowed
    with pytest.raises(P.PickError, match="disagrees"):
        P.pick(s / "ep0000900.pth", "gen2_l")


def test_fake_run_without_progress(fake_run):
    (fake_run / "progress.jsonl").unlink()
    with pytest.raises(P.PickError, match="cannot tell"):
        P.pick(fake_run / "snapshots" / "ep0001100.pth")
    # an mtime before the gen2_l block leaves only the gen2_k block as a candidate
    assert P.pick(fake_run / "snapshots" / "ep0000900.pth").engine == GEN2_K


def test_same_profile_blocks_need_no_episode(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    (run / "launch.txt").write_text(_block("2026-11-27 10:00:00", REPO / "build/gen2_k/engine", BASE14)
                                    + _block("2026-11-27 11:00:00", REPO / "build/gen2_k/engine", BASE14))
    (run / "latest.pth").write_bytes(b"x")
    p = P.pick(run / "latest.pth")
    assert p.engine == GEN2_K and "same engine" in p.how


def test_inconsistent_block_refused(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    (run / "launch.txt").write_text(f"date 2026-11-27 10:00:00\nengine {REPO}/build/gen2_k/engine\nsw {BASE14}\n"
                                    f"CR_P0088={L20}\n")
    (run / "latest.pth").write_bytes(b"x")
    with pytest.raises(P.PickError, match="sw"):
        P.pick(run / "latest.pth")


def test_env_check(fake_run):
    p = P.pick(fake_run / "snapshots" / "ep0001100.pth")
    assert P.env_check(None, "build/gen2_k/engine", "x") is None        # no profile: never refuses
    assert P.env_check(p, None, None) is None
    assert P.env_check(p, "build/gen2_l/engine", ",".join(reversed(L20.split(",")))) is None
    assert "CR_LIVE_ENGINE" in P.env_check(p, "build/gen2_k/engine", None)
    assert "CR_P0088" in P.env_check(p, None, BASE14)
    assert "CR_P0088" in P.env_check(p, None, "")


def _shell(ckpt, env=None):
    e = {k: v for k, v in os.environ.items() if k not in ("CR_LIVE_ENGINE", "CR_P0088", "CR_LIVE_TRAINED_ON")}
    e.update(env or {})
    return subprocess.run([sys.executable, str(REPO / "live_ppo/engine_pick.py"), "--shell", str(ckpt)],
                          capture_output=True, text=True, env=e)


def test_shell_mode(fake_run):
    r = _shell(MAIN)
    assert r.returncode == 0 and r.stdout.startswith("PICK_KIND=none\n")
    r = _shell(fake_run / "snapshots" / "ep0001100.pth")
    assert r.returncode == 0
    v = dict(ln.split("=", 1) for ln in r.stdout.splitlines())
    assert v["PICK_KIND"] == "run" and v["PICK_ENGINE"] == str(GEN2_L) and v["PICK_CLOAKED_ALWAYS"] == "1"
    assert v["PICK_P0088"] == L20
    r = _shell(fake_run / "snapshots" / "ep0001100.pth", {"CR_LIVE_ENGINE": "build/gen2_k/engine"})
    assert r.returncode == 2 and "CR_LIVE_ENGINE" in r.stderr
    r = _shell(RUN / "latest.pth")
    assert r.returncode == 2 and "CR_LIVE_TRAINED_ON" in r.stderr
    r = _shell(RUN / "latest.pth", {"CR_LIVE_TRAINED_ON": "gen2_k"})
    assert r.returncode == 0 and "PICK_CLOAKED_ALWAYS=0" in r.stdout


def _session_block():
    s = (REPO / "live_ppo/live_session.sh").read_text()
    i = s.index("# Engine per checkpoint")
    j = s.index("\nfi\n", i) + 4
    return s[i:j]


def _run_block(args, env=None):
    e = {k: v for k, v in os.environ.items() if k not in ("CR_LIVE_ENGINE", "CR_P0088", "CR_LIVE_TRAINED_ON")}
    e.update(env or {})
    script = (f'cd {REPO}\nROOT=$PWD\nset -- "$@"\n' + _session_block()
              + '\necho "LIVE_ENGINE=$LIVE_ENGINE"\necho "P88=${CR_P0088-(unset)}"\necho "EFF=$EFF"\n')
    return subprocess.run(["bash", "-c", script, "live_session_block", *args], capture_output=True, text=True, env=e)


def test_live_session_block(fake_run):
    newest = sorted((REPO / "runs/g2_ppo_s2r/snapshots").glob("ep*.pth"), key=lambda f: f.stat().st_mtime)[-1]
    # Sam's standard command (no --checkpoint): engine exactly as before, CR_P0088 untouched
    r = _run_block(["--act"])
    assert r.returncode == 0, r.stderr
    assert f"LIVE_ENGINE={REPO}/build/gen2_c/engine\n" in r.stdout and "P88=(unset)\n" in r.stdout
    assert f"EFF={newest}\n" in r.stdout
    r = _run_block(["--act"], {"CR_LIVE_ENGINE": "build/gen2_j/engine", "CR_P0088": J11})
    assert f"LIVE_ENGINE={REPO}/build/gen2_j/engine\n" in r.stdout and f"P88={J11}\n" in r.stdout
    r = _run_block(["--checkpoint", "runs/g2_ppo_s2r/snapshots/ep0496977.pth", "--act"])
    assert f"LIVE_ENGINE={REPO}/build/gen2_c/engine\n" in r.stdout and "P88=(unset)\n" in r.stdout
    # a gen2_l checkpoint: its engine and switches, both forms of --checkpoint, the last one wins
    ck = fake_run / "snapshots" / "ep0001100.pth"
    for args in (["--checkpoint", str(ck)], [f"--checkpoint={ck}"],
                 ["--checkpoint", str(MAIN), "--seconds", "600", "--checkpoint", str(ck)]):
        r = _run_block(args)
        assert r.returncode == 0, r.stderr
        assert f"LIVE_ENGINE={GEN2_L}\n" in r.stdout and f"P88={L20}\n" in r.stdout, (args, r.stdout)
    r = _run_block(["--checkpoint", str(ck)], {"CR_LIVE_ENGINE": "build/gen2_l/engine", "CR_P0088": L20})
    assert r.returncode == 0 and f"LIVE_ENGINE={GEN2_L}\n" in r.stdout
    # mismatches refuse before anything else runs
    for env in ({"CR_LIVE_ENGINE": "build/gen2_k/engine"}, {"CR_P0088": BASE14}):
        r = _run_block(["--checkpoint", str(ck)], env)
        assert r.returncode == 1 and "refusing to start" in r.stdout and "LIVE_ENGINE=" not in r.stdout, env
    # the current pocket run: gen2_k + BASE14
    snap = sorted((RUN / "snapshots").glob("ep*.pth"))[-1]
    r = _run_block(["--checkpoint", str(snap)])
    assert f"LIVE_ENGINE={GEN2_K}\n" in r.stdout and f"P88={BASE14}\n" in r.stdout
