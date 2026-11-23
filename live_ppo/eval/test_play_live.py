"""The command line play_live.py builds, on the driver path and on the --stock path.

    .venv/bin/python -m pytest -q live_ppo/eval/test_play_live.py

Nothing is launched: build_cmd is pure.
"""
import shlex
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "live_ppo")]
import play_live as P  # noqa: E402
import state_pipeline  # noqa: E402

RUN = Path("/runs/live/x")
CAPTURE = ["--capture", "mirror", "--ios-geometry", "/geo.json"]


def cmd(*argv):
    a = P.parse_args(list(argv))
    return a, P.build_cmd(a, RUN, CAPTURE, a.checkpoint)


def split(c):
    """(driver flags, loop args) of a driver-path command."""
    assert c[:3] == [sys.executable, "-u", str(P.DRIVER)]
    k = c.index("--")
    return c[3:k], c[k + 1:]


def test_flags_are_the_pipeline_constant():
    assert P.DEFAULT_FLAGS is state_pipeline.DEFAULT_FLAGS


def test_stock_path_is_the_old_command():
    a, c = cmd("--stock")
    assert a.checkpoint == P.STOCK_CKPT
    assert c == [sys.executable, "-u", str(P.LOOP), "--policy", "neural", *CAPTURE,
                 "--checkpoint", str(P.STOCK_CKPT), "--seconds", "420.0", "--reserve", "4.0",
                 "--confirm-log", str(RUN / "placements.jsonl")]


def test_driver_path_default():
    a, c = cmd()
    drv, loop = split(c)
    assert a.checkpoint == P.DRIVER_CKPT
    flags = shlex.split(state_pipeline.DEFAULT_FLAGS)
    assert drv[:2] == ["--run-dir", str(RUN)]
    assert drv[2:2 + len(flags)] == flags
    rest = drv[2 + len(flags):]
    assert rest == ["--temp", "0.5", "--det-weights", str(P.DEFAULT_DET), "--record"]
    # the loop gets what the stock path gives it, checkpoint and tactical layer aside
    _, stock = cmd("--stock", "--checkpoint", str(P.DRIVER_CKPT))
    assert loop == stock[3:] + ["--no-tactical"]
    assert "--act" not in c            # dry run unless asked


def test_tactical_layer_is_off_on_the_driver_path_only():
    assert "--no-tactical" in split(cmd()[1])[1]
    assert "--no-tactical" not in split(cmd("--tactical")[1])[1]
    assert "--no-tactical" not in cmd("--stock")[1]
    assert cmd("--stock", "--no-tactical")[1].count("--no-tactical") == 1
    assert split(cmd("--act")[1])[1][-2:] == ["--act", "--no-tactical"]


def test_act_reaches_the_loop_only():
    for extra in ((), ("--stock",)):
        _, c = cmd("--act", *extra)
        assert c.count("--act") == 1
        if not extra:
            drv, loop = split(c)
            assert "--act" in loop and "--act" not in drv


def test_loop_options_pass_through_on_both_paths():
    opts = ["--seconds", "300", "--reserve", "3", "--no-tactical", "--ignore-staleness"]
    _, stock = cmd("--stock", *opts)
    _, c = cmd(*opts, "--checkpoint", str(P.STOCK_CKPT))
    assert split(c)[1] == stock[3:]
    assert stock[-2:] == ["--no-tactical", "--ignore-staleness"]
    assert stock[stock.index("--seconds") + 1] == "300.0"


def test_record_and_detector_flags():
    drv, _ = split(cmd("--no-record")[1])
    assert "--record" not in drv
    drv, _ = split(cmd("--record-every", "2")[1])
    assert drv[drv.index("--record") + 1:][:2] == ["--record-every", "2"]
    drv, _ = split(cmd("--det-weights", str(P.DET_B), "--temp", "1")[1])
    assert drv[drv.index("--det-weights") + 1] == str(P.DET_B)
    assert drv[drv.index("--temp") + 1] == "1.0"
    _, stock = cmd("--stock", "--det-weights", str(P.DET_B))
    assert "--det-weights" not in stock and "--record" not in stock


def test_replay_and_extra_driver_flags():
    drv, loop = split(cmd("--replay", "runs/live/rec_dry1", "--replay-from", "650",
                          "--replay-realtime", "--driver-flags=--tap-mode drag")[1])
    i = drv.index("--loop-frames")
    assert drv[i:i + 4] == ["--loop-frames", str((Path.cwd() / "runs/live/rec_dry1").resolve()),
                            "--loop-from", "650"]
    assert "--loop-realtime" in drv and drv[-2:] == ["--tap-mode", "drag"]
    assert "--act" not in loop


@pytest.mark.parametrize("argv", [
    ["--replay", "runs/live/rec_dry1", "--act"],      # a replay never taps
    ["--driver-flags=--act"],                         # --act only through the confirmation gate
    ["--driver-flags=--tap-mode drag -- --act"],
    ["--driver-flags=--ensure-match"],
    ["--replay", "runs/live/rec_dry1", "--stock"],
    ["--device", "android"],                          # the driver needs the iPhone frame
    ["--stock", "--driver-flags=--greedy"],
])
def test_refused(argv):
    with pytest.raises(SystemExit):
        P.parse_args(argv)


def test_decks_per_path():
    assert P.driver_train_deck() == [125, 168, 129, 15, 40, 72, 33, 7]
    assert P.DECK_26 == [15, 6, 25, 40, 24, 72, 33, 7]


def test_mirror_preflight_never_captures_in_the_launcher(tmp_path, monkeypatch, capsys):
    """The launcher hands the window capture to a helper process.

    A process that has captured the window makes the same call hang in every
    child it starts later (macOS 26.6.2), and the driver is such a child.
    """
    geo = tmp_path / "geometry.json"
    geo.write_text("{}")
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        return P.subprocess.CompletedProcess(argv, 0, stdout="device: window\n")

    monkeypatch.setattr(P.subprocess, "run", fake_run)
    for name in ("Quartz", "capture.mirror"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    P.check_mirror(geo, tmp_path / "start.png")
    assert calls == [[sys.executable, str(Path(P.__file__).resolve()), P.MIRROR_SHOT_FLAG,
                      str(geo), str(tmp_path / "start.png")]]
    assert "Quartz" not in sys.modules and "capture.mirror" not in sys.modules
    assert "device: window" in capsys.readouterr().out


def test_mirror_preflight_reports_the_helper_failure(tmp_path, monkeypatch, capsys):
    geo = tmp_path / "geometry.json"
    geo.write_text("{}")
    monkeypatch.setattr(P.subprocess, "run", lambda argv, **kw: P.subprocess.CompletedProcess(
        argv, 1, stdout="no iPhone Mirroring window on screen\n"))
    with pytest.raises(SystemExit) as exc:
        P.check_mirror(geo, tmp_path / "start.png")
    assert exc.value.code == 2
    assert "no iPhone Mirroring window on screen" in capsys.readouterr().err
