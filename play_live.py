#!/usr/bin/env python3
"""Run a trained checkpoint in a live Clash Royale match, through our perception stack.

By default this launches live_ppo/driver.py: the engine's live loop
(engine/perception/live/mvp_loop.py) with our state estimation hooked in (YOLO
units + tracker, HUD clock and tower numerals, screen hand, opponent plays), run
with the production flag set state_pipeline.DEFAULT_FLAGS. It runs the preflight
checks first and keeps everything a run produces, including a recording of every
frame perception saw, so a run can be audited afterwards:

    python play_live.py                       # dry run: watch and decide, no taps
    python play_live.py --act                 # place real cards (asks first)
    python play_live.py --act --seconds 300
    python play_live.py --replay runs/live/rec_dry1 --replay-from 650 --seconds 60
                                              # no phone: recorded frames, never taps
    python play_live.py --stock               # the old path: mvp_loop alone, stock perception

Flags that choose what runs:
    --checkpoint P     default remote_results/ppo_league26_ep138708.pth (driver path;
                       the round-robin winner, training/REPORT_netv2_ppo.md) or
                       checkpoints/partner/model_weights_ep76728.pth (--stock)
    --temp T           sampling temperature of the driver's policy (default 0.5, the
                       setting that checkpoint was selected at; also 0.5 for a generation-2
                       checkpoint, where it sets which card and where, since 2026-11-23)
    --det-weights W    unit detector weights for the YOLO side-car (default: DEFAULT_DET
                       below, chosen by the unit and opponent-play measurements in LIVE.md)
    --record / --no-record   save every perception frame + timestamps to the run dir
                       (default on; layout in live_ppo/RECORDING.md); --record-every N
    --driver-flags S   extra live_ppo/driver.py flags, e.g. "--tap-mode drag"
    --stock            run mvp_loop directly with its stock readers (no driver, no
                       detector side-car, no recording), for comparison

Default device `mirror` (ios/SETUP.md): the iPhone shown in macOS's iPhone
Mirroring window, clicked with the mouse. `iphone` uses WebDriverAgent instead
(needs Xcode). Both need ios/geometry.json from tools/calibrate_ios.py, and
preflight 4-5 become: the device answers, the geometry fits the screen.
`android` (adb) works with --stock only: the driver's readers need the iPhone frame.

A generation-2 (NetV3, `arch: v3`) checkpoint needs the generation-2 engine, chosen with
the CR_ENGINE environment variable (LIVE.md, "Generation-2 checkpoints"):

    CR_ENGINE=build/gen2/engine python play_live.py --checkpoint runs/bc3/g2_big_L0_H10/ep4.pth

The driver then takes its defaults from the checkpoint: 2 decisions a second, the
observation tail and the opponent-state section on, no gen-1 crutches (tactical layer,
solvency gate, leak floor), plays in flight masked. --tactical turns the tactical
officer and the solvency gate back on for it.

Preflight (any failure stops the run before a frame is read):
    1. engine patches applied (patches/apply_patches.py --check)
    2. the checkpoint loads, carries the deck its path expects and matches the net:
       driver path, the ppo_bc26 training deck (2.6 Hog Cycle with the Evolution /
       Hero ids, read from live_ppo/driver.py's TRAIN_DECK) and the one-ability-slot
       net; --stock, the base-id 2.6 deck and the stock net
    3. driver path: the detector weights and the side-car's interpreter exist
    4. exactly one device over adb (BlueStacks lists itself twice; deduped)
    5. the device is 1080x1920, the size the actuator's tap mapping assumes
    6. a screenshot is saved to the run directory

With --act it then shows that screenshot and asks you to confirm the
opponent is a Training Camp "Trainer" (or a friend in a Friendly Battle who
agreed to it). Nothing is tapped until you type "yes". --replay never taps
and refuses --act.

Everything lands in runs/live/<timestamp>/:
    start.png          the screen at launch (+ start_frame.png, what the bot sees)
    live.log           the loop's full output, errors and tracebacks included
    placements.jsonl   one line per issued placement (--confirm-log)
    run.json           checkpoint, sha256, device, args, command, exit code
  and on the driver path:
    perception.jsonl   one line per decision: everything the policy was fed
    overlay/           one annotated frame per decision
    latency.jsonl, board_age.csv   timing (live_ppo/latency_report.py reads them)
    raw/, raw.yolo.jsonl           the recording: every frame, a timestamp per frame,
                       the detector's live boxes (live_ppo/RECORDING.md)
    match_sbs.mp4      rendered after the run ends (also after Ctrl-C; --no-video skips):
                       the game with detector boxes beside the policy's board
                       (live_ppo/match_video.py)

The match runs about 3-5 minutes with overtime; --seconds defaults to 420 so
the loop outlasts it. Start the loop with the battle already loading or on
screen. --ensure-match is not exposed: its taps are not scaled for
1080x1920 yet.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
# CR_ENGINE: an engine root other than REPO/engine (the generation-2 copy, build/gen2/engine).
# live_ppo/driver.py and training/common.py read the same variable; the driver is started in
# that engine's perception/ directory.
ENGINE = (Path(os.environ["CR_ENGINE"]).expanduser().resolve() if os.environ.get("CR_ENGINE")
          else REPO / "engine")
LOOP = ENGINE / "perception" / "live" / "mvp_loop.py"
DRIVER = REPO / "live_ppo" / "driver.py"
sys.path.insert(0, str(REPO / "live_ppo"))
from state_pipeline import DEFAULT_FLAGS  # noqa: E402  the production driver flag set

# --stock: the partner checkpoint, base-id 2.6 deck, stock net.
STOCK_CKPT = REPO / "checkpoints" / "partner" / "model_weights_ep76728.pth"
DECK_26 = [15, 6, 25, 40, 24, 72, 33, 7]
# Driver path: the round-robin winner among the ppo_bc26-deck checkpoints
# (training/REPORT_netv2_ppo.md, remote_results/rr/final.log), selected at T = 0.5.
DRIVER_CKPT = REPO / "remote_results" / "ppo_league26_ep138708.pth"
DRIVER_TEMP = 0.5
# A generation-2 (v3) checkpoint: WHAT / WHERE temperature 0.5 (WHEN stays 1.0, bias 0). Until 2026-11-23
# this was 1.0, the training temperature, and both live sessions of that day ran at 1.0. Measured on
# ep0121202, 1,000 held-out matches under the live2 condition: 59.7 % at 1.0, 69.5 % at 0.5 (+9.8 +- 1.5),
# 70.6 % at 0.25; with act delay 4,7,1: 60.2 % at 1.0, 67.7 % at 0.5
# (evals/ghost_g2_ppo_s2r_ep0121202_live2_*_ec_20261123.json). --temp 1.0 gives the old behaviour.
V3_TEMP = 0.5
# Unit detector weights for the driver's YOLO side-car.
DET_V2 = REPO / "data" / "det" / "runs" / "v2" / "weights" / "best.pt"     # 191 classes
DET_B = REPO / "runs" / "det_v3_pod" / "ab" / "B" / "weights" / "last.pt"  # 198 classes
# Round 3 (2026-11-20): a copy of runs/det_r3_pod/train/weights/last.pt, 198 classes.
# Why it is the default: LIVE.md "Detector".
DET_R3 = REPO / "data" / "det" / "runs" / "r3" / "weights" / "best.pt"
# Round 5 (2026-11-23): round 3 fine-tuned on our own live matches, a copy of
# runs/det_r5_pod_live2/live/weights/last.pt, same 198 classes. Default since 2026-11-23.
DET_R5 = REPO / "data" / "det" / "runs" / "r5" / "weights" / "best.pt"
DEFAULT_DET = DET_R5
SIDECAR_PYTHON = REPO / ".venv-det" / "bin" / "python"
SCREEN = (1080, 1920)
IOS_GEOMETRY = REPO / "ios" / "geometry.json"


def fail(msg: str) -> None:
    print(f"preflight FAILED: {msg}", file=sys.stderr)
    sys.exit(2)


def check_patches() -> None:
    out = subprocess.run([sys.executable, str(REPO / "patches" / "apply_patches.py"),
                          "--check"], capture_output=True, text=True)
    lines = [l for l in out.stdout.splitlines() if l.strip()]
    bad = [l for l in lines if not l.rstrip().endswith(": applied")]
    if out.returncode != 0 or bad or not lines:
        fail("engine patches not all applied:\n  " + "\n  ".join(bad or [out.stderr])
             + "\n  run: python patches/apply_patches.py")
    print(f"patches: {len(lines)} applied")


def driver_train_deck() -> list:
    """TRAIN_DECK as live_ppo/driver.py defines it (its adapter refuses any other
    deck). Read from the source: importing the driver starts it."""
    import ast
    for node in ast.parse(DRIVER.read_text()).body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "TRAIN_DECK" for t in node.targets):
            return list(ast.literal_eval(node.value))
    fail(f"TRAIN_DECK not found in {DRIVER}")


def check_checkpoint_v3(path: Path, blob: dict) -> dict:
    """A generation-2 (NetV3) checkpoint: the generation-2 engine is selected, the deck
    it was trained on is the driver's, and it loads through the loader the driver uses
    (training/obs_compat.load_policy_v3: observation length, card ids, the hash of
    training/net_v3.py, the state dict)."""
    import policy_v3                                   # live_ppo/, on sys.path above
    sys.path[:0] = [str(ENGINE), str(ENGINE / "python_ai")]
    try:
        import clash_royale_env as E
    except ImportError as exc:
        fail(f"no built engine at {ENGINE}: {exc}")
    gen2 = hasattr(E.ClashRoyaleEnv, "LEVEL_START")
    had = "CLASH_DECK" in os.environ
    try:
        policy_v3.check(blob, path=str(path), engine_gen2=gen2, engine=str(ENGINE),
                        driver_deck=driver_train_deck())
        sys.path.append(str(REPO))
        from training import common as TC, obs_compat as OC
        policy_v3.check(blob, path=str(path), engine_gen2=gen2, engine=str(ENGINE),
                        driver_deck=driver_train_deck(), train_deck=TC.HOG26_DECK)
        _, meta = OC.load_policy_v3(str(path), blob)
    except ValueError as exc:                          # V3Error is one
        fail(f"{exc}\n  a generation-2 checkpoint runs as:\n  CR_ENGINE={REPO / 'build' / 'gen2' / 'engine'} "
             f".venv/bin/python play_live.py --checkpoint {path}")
    finally:
        if not had:     # training.common sets a default the hand reader must not see
            os.environ.pop("CLASH_DECK", None)
    info = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "deck": policy_v3.parse_deck(blob.get("deck") or TC.HOG26_DECK),
        "arch": "v3",
        "engine": str(ENGINE),
        "obs_dim": meta["obs_dim"], "decision_ticks": meta["decision_ticks"],
        "names": meta["names"], "dataset": meta["dataset"], "dataset_fp": meta["dataset_fp"],
        "epoch": blob.get("epoch"),
    }
    print(f"checkpoint: {path.name}, v3 (NetV3), observation {info['obs_dim']}, "
          f"{info['decision_ticks']} ticks per decision, tail names {info['names']!r}, "
          f"dataset {info['dataset']}; engine {ENGINE}")
    return info


def check_checkpoint(path: Path, stock: bool = False) -> dict:
    """The checkpoint loads, carries the deck its path expects and matches the net
    that path builds. Driver path: the ppo_bc26 deck and the one-ability-slot net
    (or a net_v2), as live_ppo/driver.py's PPOPolicy loads it; a generation-2
    checkpoint goes to check_checkpoint_v3. --stock: the base-id 2.6 deck and the
    stock net."""
    import torch
    if not path.is_file():
        fail(f"no checkpoint at {path}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and blob.get("arch") == "v3":
        if stock:
            fail(f"{path.name} is a generation-2 (v3) checkpoint: it runs on the driver "
                 "path only, drop --stock")
        return check_checkpoint_v3(path, blob)
    if not isinstance(blob, dict) or "model" not in blob:
        fail(f"{path.name} is a bare state dict (no deck, no episode count); the live "
             "loop needs a trainer checkpoint")
    deck = list(blob.get("deck") or [])
    sys.path[:0] = [str(ENGINE)]
    from python_ai.models.net import MicroRoyaleNet
    state = blob["model"]
    arch = "v1"
    if stock:
        if deck != DECK_26:
            fail(f"checkpoint deck {deck} is not the 2.6 Hog Cycle {DECK_26}; the live "
                 "loop reads the hand as the 2.6 deck"
                 + (" (this is a driver-path checkpoint: drop --stock)"
                    if deck == driver_train_deck() else ""))
        net = MicroRoyaleNet()
    else:
        want = driver_train_deck()
        if deck != want:
            fail(f"checkpoint deck {deck} is not the deck live_ppo/driver.py was built "
                 f"for, {want} (2.6 Hog Cycle with the Evolution / Hero ids)"
                 + ("; this is a base-id checkpoint for the stock loop: add --stock"
                    if deck == DECK_26 else ""))
        if blob.get("arch") == "net_v2" or "board.stem.weight" in state:
            arch = "net_v2"
            had = "CLASH_DECK" in os.environ
            sys.path.append(str(REPO))
            from training import net_v2
            if not had:     # training.common sets a default the hand reader must not see
                os.environ.pop("CLASH_DECK", None)
            net = net_v2.NetV2(num_ability_slots=1, **dict(blob.get("cfg") or {}))
        else:
            net = MicroRoyaleNet(num_ability_slots=1)
    try:
        missing, unexpected = net.load_state_dict(state, strict=False)
    except RuntimeError as exc:
        if not os.environ.get("CR_ENGINE"):
            raise
        fail(f"{path.name} does not fit the net of the engine CR_ENGINE selects ({ENGINE}); "
             "a checkpoint that is not generation-2 runs on engine/, without CR_ENGINE\n  "
             + str(exc).splitlines()[0])
    missing = [k for k in missing if not k.startswith(("place_hires", "place_ctx_hi"))]
    if missing or unexpected:
        fail(f"checkpoint does not match the net: missing {missing}, "
             f"unexpected {unexpected}")
    info = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "deck": deck,
        "arch": arch,
        "episodes_completed": blob.get("episodes_completed"),
        "curriculum_stage": blob.get("curriculum_stage"),
        "phase": blob.get("phase"),
    }
    print(f"checkpoint: {path.name}, {arch}, episode {info['episodes_completed']}, "
          f"phase {info['phase']}, rung {info['curriculum_stage']}, deck {deck}")
    return info


def check_detector(weights: Path) -> None:
    if not SIDECAR_PYTHON.is_file():
        fail(f"no {SIDECAR_PYTHON}: the YOLO side-car runs in .venv-det "
             "(ultralytics + cv2)")
    if not weights.is_file():
        fail(f"no detector weights at {weights}")
    print(f"detector: {weights}")


def check_replay(frames: Path, start: int, dest: Path) -> int:
    """--replay: the frame directory has frames; the first one served stands in
    for the launch screenshot."""
    files = sorted(frames.glob("f*.jpg"))
    if len(files) <= start:
        fail(f"{frames} has {len(files)} f*.jpg frames; --replay-from {start} leaves none")
    shutil.copy2(files[start], dest.with_name("start_frame.jpg"))
    print(f"replay: {len(files) - start} frames from {files[start].name} in {frames} "
          "(no phone, no taps)")
    return len(files) - start


def pick_device(adb: str) -> str:
    out = subprocess.run([adb, "devices"], capture_output=True, text=True).stdout
    serials = sorted({l.split()[0] for l in out.splitlines()[1:]
                      if l.strip().endswith("\tdevice")})
    wanted = os.environ.get("ANDROID_SERIAL")
    if wanted:
        if wanted not in serials:
            fail(f"ANDROID_SERIAL={wanted} is not attached (adb sees {serials})")
        return wanted
    # BlueStacks Air shows up as both emulator-5554 and 127.0.0.1:5555; they
    # are the same device, and the emulator- name is the one that works.
    emus = [s for s in serials if s.startswith("emulator-")]
    if len(emus) == 1:
        return emus[0]
    if len(serials) == 1:
        return serials[0]
    if not serials:
        fail("no device over adb. Start BlueStacks Air and enable ADB in its "
             "settings (Advanced > Android Debug Bridge)")
    fail(f"several devices {serials}; set ANDROID_SERIAL to one")


def check_screen(adb: str, serial: str) -> None:
    out = subprocess.run([adb, "-s", serial, "shell", "wm", "size"],
                         capture_output=True, text=True).stdout
    sizes = re.findall(r"(\d+)x(\d+)", out)
    if not sizes:
        fail(f"could not read the screen size: {out.strip()!r}")
    w, h = map(int, sizes[-1])  # an override size is listed last
    if (w, h) != SCREEN:
        fail(f"device screen is {w}x{h}; the tap mapping assumes "
             f"{SCREEN[0]}x{SCREEN[1]} portrait. Set BlueStacks to that resolution")
    print(f"device: {serial}, {w}x{h}")


def screenshot(adb: str, serial: str, dest: Path) -> None:
    png = subprocess.run([adb, "-s", serial, "exec-out", "screencap", "-p"],
                         capture_output=True, timeout=15).stdout
    if not png.startswith(b"\x89PNG"):
        fail("screencap returned no PNG")
    dest.write_bytes(png)
    print(f"screenshot: {dest}")


def check_iphone(url: str, geometry: Path, dest: Path) -> None:
    sys.path[:0] = [str(ENGINE / "perception")]
    import cv2
    import numpy as np
    from capture.wda import IosGeometry, WdaClient, WdaError
    client = WdaClient(url, timeout=10)
    try:
        st = client.status()
        png = client.screenshot_png()
    except WdaError as exc:
        fail(f"{exc}\n  see ios/SETUP.md (start WDA, then `iproxy 8100 8100`)")
    ios = st.get("os", {})
    print(f"device: iPhone over WDA, iOS {ios.get('version', '?')}")
    if not geometry.is_file():
        fail(f"no geometry at {geometry}. During a Training Camp battle run:\n  "
             "cd engine/perception && ../../.venv/bin/python tools/calibrate_ios.py "
             f"--out {geometry}")
    geo = IosGeometry.load(geometry)
    shot = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    try:
        frame = geo.frame(shot)
    except WdaError as exc:
        fail(f"{exc}; the screen changed size or orientation. Re-run calibrate_ios.py")
    dest.write_bytes(png)
    cv2.imwrite(str(dest.with_name("start_frame.png")), frame)
    print(f"screenshot: {dest} (+ start_frame.png, what the bot sees)")


MIRROR_SHOT_FLAG = "--mirror-shot"      # internal: run mirror_shot() and exit


def mirror_shot(geometry: Path, dest: Path) -> int:
    """The window capture half of check_mirror. Runs in its own process."""
    sys.path[:0] = [str(ENGINE / "perception")]
    import cv2
    from capture.mirror import MirrorError, MirrorWindow
    from capture.wda import IosGeometry, WdaError
    perms = MirrorWindow.permissions()
    missing = [k for k, ok in perms.items() if not ok]
    if missing:
        print(f"this terminal lacks {', '.join(missing)} permission. System Settings > "
              "Privacy & Security > Screen Recording and Accessibility: add the "
              "terminal app, then quit and reopen it")
        return 1
    try:
        win = MirrorWindow()
        shot = win.capture()
    except MirrorError as exc:
        print(exc)
        return 1
    print(f"device: iPhone Mirroring window at {win.bounds}")
    try:
        frame = IosGeometry.load(geometry).frame(shot)
    except WdaError as exc:
        print(f"{exc}; the window was resized. Re-run calibrate_ios.py --mirror")
        return 1
    cv2.imwrite(str(dest), shot)
    cv2.imwrite(str(dest.with_name("start_frame.png")), frame)
    print(f"screenshot: {dest} (+ start_frame.png, what the bot sees)")
    return 0


def check_mirror(geometry: Path, dest: Path) -> None:
    """Preflight for iPhone Mirroring, with the capture done by a helper process.

    The launcher must never capture the window itself. Measured on macOS 26.6.2
    (2026-11-20): once a process has called CGWindowListCreateImage, the same
    call hangs for good in any child it starts afterwards, so the driver sat on
    its first frame, recorded nothing and placed nothing. A helper that has
    exited does not affect the driver.
    """
    if not geometry.is_file():
        fail(f"no geometry at {geometry}. During a Training Camp battle run:\n  "
             "cd engine/perception && ../../.venv/bin/python tools/calibrate_ios.py "
             f"--mirror --out {geometry}")
    try:
        out = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), MIRROR_SHOT_FLAG,
             str(geometry), str(dest)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        fail("the iPhone Mirroring window did not capture within 30 s. Quit and "
             "reopen the terminal and the iPhone Mirroring app, then try again")
    if out.returncode:
        fail(out.stdout.strip() or f"mirror preflight failed (exit {out.returncode})")
    print(out.stdout, end="")


def confirm_opponent(shot: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", str(shot)])
    print()
    print("Before any card is placed, look at the screenshot (just opened).")
    print("Real-player modes (ladder, Classic/Grand Challenges, 2v2 with")
    print("randoms) are not supported: Supercell's ToS bans bots there, and it")
    print("risks this account. Training Camp, or a Friendly Battle with a")
    print("teammate who agreed, only.")
    answer = input('Is the opponent a Training Camp "Trainer" (or an agreed '
                   'friendly)? type yes: ')
    if answer.strip().lower() != "yes":
        print("not confirmed; nothing was tapped")
        sys.exit(1)


def driver_temp(args, arch: str = "v1") -> float:
    """--temp, or the default of the checkpoint's kind."""
    if args.temp is not None:
        return args.temp
    return V3_TEMP if arch == "v3" else DRIVER_TEMP


def build_cmd(args, run_dir: Path, capture: list, checkpoint: Path, arch: str = "v1") -> list:
    """The command line of the run. The mvp_loop arguments are the same on both
    paths; the driver path puts them after `--`, behind the driver's own flags.
    `arch` is the checkpoint's kind from check_checkpoint: a v3 checkpoint gets its
    own default temperature, and --tactical reaches it as the driver's own flags
    (the driver keeps the tactical layer off for v3 whatever the loop is told)."""
    loop = ["--policy", "neural", *capture,
            "--checkpoint", str(checkpoint),
            "--seconds", str(args.seconds), "--reserve", str(args.reserve),
            "--confirm-log", str(run_dir / "placements.jsonl")]
    if args.act:
        loop.append("--act")
    # The driver's checkpoint was trained and evaluated without the engine's tactical
    # layer, and loses with it (LIVE.md, "The tactical layer"); the stock net needs it.
    if args.no_tactical or not (args.stock or args.tactical):
        loop.append("--no-tactical")
    if args.ignore_staleness:
        loop.append("--ignore-staleness")
    if args.stock:
        return [sys.executable, "-u", str(LOOP), *loop]
    drv = ["--run-dir", str(run_dir), *shlex.split(DEFAULT_FLAGS),
           "--temp", str(driver_temp(args, arch)),
           "--det-weights", str(args.det_weights.resolve())]
    if arch == "v3" and args.tactical and not args.no_tactical:
        drv += ["--tactical-officer", "--solvency-gate"]
    if args.record:
        drv.append("--record")
        if args.record_every != 1:
            drv += ["--record-every", str(args.record_every)]
    if args.replay is not None:
        drv += ["--loop-frames", str(args.replay.resolve()), "--loop-from", str(args.replay_from)]
        if args.replay_realtime:
            drv.append("--loop-realtime")
    drv += shlex.split(args.driver_flags)
    return [sys.executable, "-u", str(DRIVER), *drv, "--", *loop]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=("mirror", "iphone", "android"),
                    default="mirror")
    ap.add_argument("--wda-url", default="http://127.0.0.1:8100")
    ap.add_argument("--geometry", type=Path, default=IOS_GEOMETRY)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help=f"default {DRIVER_CKPT.relative_to(REPO)}, or with --stock "
                         f"{STOCK_CKPT.relative_to(REPO)}")
    ap.add_argument("--act", action="store_true", help="place real cards")
    ap.add_argument("--seconds", type=float, default=420.0)
    ap.add_argument("--no-tactical", action="store_true",
                    help="network alone picks every cell (no Cannon/Fireball "
                         "advisor, no solvency gate); already the default on the "
                         "driver path, so this only changes --stock")
    ap.add_argument("--tactical", action="store_true",
                    help="driver path: turn the engine's Cannon/Fireball advisor and "
                         "solvency gate back on (the default before 2026-11-20)")
    ap.add_argument("--reserve", type=float, default=4.0,
                    help="elixir the solvency gate holds while nothing attacks")
    ap.add_argument("--ignore-staleness", action="store_true")
    ap.add_argument("--stock", action="store_true",
                    help="the old path: mvp_loop alone with its stock perception "
                         "(no driver, no YOLO side-car, no recording)")
    ap.add_argument("--temp", type=float, default=None,
                    help=f"driver path: policy sampling temperature (default {DRIVER_TEMP}; "
                         f"{V3_TEMP} for a generation-2 checkpoint, where it is the WHAT / "
                         "WHERE temperature)")
    ap.add_argument("--det-weights", type=Path, default=DEFAULT_DET,
                    help="driver path: unit detector weights for the YOLO side-car "
                         f"(default {DEFAULT_DET.relative_to(REPO)})")
    ap.add_argument("--record", action=argparse.BooleanOptionalAction, default=True,
                    help="driver path: save every perception frame with a timestamp "
                         "to RUN/raw/ for the offline audit (live_ppo/RECORDING.md)")
    ap.add_argument("--record-every", type=int, default=1, metavar="N",
                    help="--record: keep every Nth in-game frame (default all)")
    ap.add_argument("--replay", type=Path, default=None, metavar="FRAMES_DIR",
                    help="driver path, no phone: serve these recorded f*.jpg frames "
                         "instead of the mirror window. Never taps; refuses --act")
    ap.add_argument("--replay-from", type=int, default=0, metavar="N",
                    help="--replay: start at the Nth frame file")
    ap.add_argument("--replay-realtime", action="store_true",
                    help="--replay: one board per recorded frame at the recording's "
                         "4 fps (default: each frame four times, at loop speed)")
    ap.add_argument("--video", action=argparse.BooleanOptionalAction, default=True,
                    help="after the run, render runs/live/<ts>/match_sbs.mp4: the game with "
                         "detector boxes beside the policy's board (live_ppo/match_video.py); "
                         "driver path with --record only")
    ap.add_argument("--driver-flags", default="", metavar="STR",
                    help='extra live_ppo/driver.py flags, e.g. "--tap-mode drag"')
    args = ap.parse_args(argv)
    if args.checkpoint is None:
        args.checkpoint = STOCK_CKPT if args.stock else DRIVER_CKPT
    extra = shlex.split(args.driver_flags)
    if "--" in extra or any(f.split("=")[0] in ("--act", "--ensure-match") for f in extra):
        # the driver hands flags it does not know to the loop, so these would
        # reach it without the confirmation below
        ap.error("--driver-flags takes driver.py flags only; --act is play_live's own flag")
    if args.replay is not None:
        if args.act:
            ap.error("--replay never taps: drop --act")
        if args.stock or args.device != "mirror":
            ap.error("--replay runs through the driver's mirror path: no --stock, no --device")
    if args.stock and extra:
        ap.error("--driver-flags does nothing with --stock")
    if args.device == "android" and not args.stock:
        ap.error("--device android needs --stock: the driver's readers and detector "
                 "expect the 549x976 iPhone frame")
    return args


def main() -> int:
    args = parse_args()

    iphone = args.device in ("iphone", "mirror")
    adb = None if iphone else shutil.which("adb")
    if not iphone and not adb:
        fail("adb not on PATH (brew install android-platform-tools)")

    check_patches()
    if os.environ.get("CR_ENGINE"):
        print(f"engine: {ENGINE} (CR_ENGINE; the patch check above is engine/'s)")
    ckpt = check_checkpoint(args.checkpoint.resolve(), stock=args.stock)
    if not args.stock:
        check_detector(args.det_weights.resolve())
    if not iphone:
        serial = pick_device(adb)
        check_screen(adb, serial)

    run_dir = REPO / "runs" / "live" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    shot = run_dir / "start.png"
    if args.replay is not None:
        serial = f"replay:{args.replay.resolve()}"
        check_replay(args.replay.resolve(), args.replay_from, shot)
        if not args.geometry.is_file():
            fail(f"no geometry at {args.geometry} (the loop loads it even for a replay)")
        capture = ["--capture", "mirror", "--ios-geometry", str(args.geometry.resolve())]
    elif args.device == "mirror":
        serial = "iphone-mirroring"
        check_mirror(args.geometry.resolve(), shot)
        capture = ["--capture", "mirror", "--ios-geometry", str(args.geometry.resolve())]
    elif iphone:
        serial = "iphone"
        check_iphone(args.wda_url, args.geometry.resolve(), shot)
        capture = ["--capture", "wda", "--wda-url", args.wda_url,
                   "--ios-geometry", str(args.geometry.resolve())]
    else:
        screenshot(adb, serial, shot)
        capture = ["--capture", "adb"]
    if args.act:
        confirm_opponent(shot)

    cmd = build_cmd(args, run_dir, capture, args.checkpoint.resolve(), arch=ckpt["arch"])
    meta = {"started": dt.datetime.now().isoformat(timespec="seconds"),
            "path": "stock" if args.stock else "driver",
            "checkpoint": ckpt, "device": serial, "act": args.act, "cmd": cmd}
    if not args.stock:
        meta.update(driver_flags=DEFAULT_FLAGS, temp=driver_temp(args, ckpt["arch"]),
                    det_weights=str(args.det_weights.resolve()), record=args.record,
                    replay=str(args.replay.resolve()) if args.replay else None)
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2))
    print(f"path: {'stock mvp_loop' if args.stock else 'live_ppo/driver.py ' + DEFAULT_FLAGS}")
    print(f"mode: {'ACT - real taps' if args.act else 'dry run - no taps'}"
          f"{'' if args.stock or not args.record else ', recording to raw/'}")
    print(f"run dir: {run_dir}\n")

    env = dict(os.environ, ANDROID_SERIAL=serial, PYTHONUNBUFFERED="1")
    with open(run_dir / "live.log", "w") as log:
        proc = subprocess.Popen(cmd, cwd=LOOP.parent.parent, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            code = proc.wait()
        except KeyboardInterrupt:
            # Ctrl-C reached the loop too: it prints its summary and the driver
            # closes the recording. Only kill it if that takes too long.
            try:
                for line in proc.stdout:
                    sys.stdout.write(line)
                    log.write(line)
                code = proc.wait(timeout=30)
            except (KeyboardInterrupt, subprocess.TimeoutExpired):
                proc.terminate()
                code = proc.wait()

    placements = run_dir / "placements.jsonl"
    n = sum(1 for _ in placements.open()) if placements.exists() else 0
    meta.update(exit_code=code, placements=n,
                ended=dt.datetime.now().isoformat(timespec="seconds"))
    manifest = run_dir / "raw" / "manifest.json"
    if manifest.is_file():
        m = json.loads(manifest.read_text())
        meta["recording"] = {k: m.get(k) for k in ("count", "in_game_count", "in_game_span_s",
                                                    "achieved_fps", "frames_seen", "dropped", "bytes")}
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2))
    print(f"\nexit {code}, {n} placements issued; log in {run_dir / 'live.log'}")
    if "recording" in meta:
        r = meta["recording"]
        print(f"recording: {r['count']} frames, {r['bytes'] / 1e6:.0f} MB in {run_dir / 'raw'} "
              "(live_ppo/RECORDING.md)")
    if args.video and args.record and not args.stock:
        render_video(run_dir)
    return code


def render_video(run_dir: Path) -> None:
    """Side-by-side match video from the run's recording (files only, no capture)."""
    vid = REPO / "live_ppo" / "match_video.py"
    manual = f"{sys.executable} {vid} {run_dir}"
    if not (run_dir / "raw" / "frames.jsonl").is_file():
        print("video: no recording in this run, skipped")
        return
    print("\nrendering match video (Ctrl-C skips; rerun later with: " + manual + ")")
    sys.stdout.flush()
    try:
        out = subprocess.run([sys.executable, str(vid), str(run_dir)], text=True)
        if out.returncode != 0:
            print(f"video: render failed (exit {out.returncode}); rerun with: {manual}")
    except KeyboardInterrupt:
        print(f"video: skipped; render later with: {manual}")


if __name__ == "__main__":
    if sys.argv[1:2] == [MIRROR_SHOT_FLAG]:
        sys.exit(mirror_shot(Path(sys.argv[2]), Path(sys.argv[3])))
    sys.exit(main())
