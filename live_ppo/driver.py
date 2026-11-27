#!/usr/bin/env python3
"""Run engine/perception/live/mvp_loop.py with the BC->PPO (ppo_bc26) checkpoint,
without editing engine/.

Everything is done by wrapping the loop's module globals before calling its main():

* Checkpoint adapter. ppo_bc26 is trained on deck 125,168,129,15,40,72,33,7
  (Evo Skeletons, Hero Musketeer, Evo Cannon, ...): the net has the Hero
  Musketeer ability head and its hand one-hots were only ever 125/168/129, never
  the base ids 24/6/25 the live hand reader produces. `PPOPolicy` builds the net
  with num_ability_slots=1 (loads with zero missing / unexpected keys) and remaps
  the hand to the training ids before encoding. CRBAB templates and the hand
  tracker keep the base 2.6 deck (CLASH_DECK unset at import).
* Per-decision perception log (perception.jsonl), an overlay frame per decision
  (overlay/), and one short "DECIDE" line per decision in live.log.
* Optional fixes behind flags (off = the stock loop's observation):
    --fill-clock    seconds_elapsed from the match start (stock: always 0, so the
                    time scalar and the 2x/3x elixir-phase scalar never move)
    --units yolo    units from video_dataset/detector v2 (YOLOv8s + side net)
                    instead of CRBAB's detector
    --tile-map tower  (default) detector row -> engine row by the tower-anchored map
                    for the YOLO units and the tap inverse (live_ppo/tile_map.py),
                    the same map as the hog26 BC data and the IL export; the stock
                    offset1 = det + 1 puts every tower one row off
    --ground calib  (default) YOLO unit ground point = box bottom-centre + the
                    dev-fitted per-kind offset (unit_tracker.GROUND_OFFSET; troops
                    +0.42 rows, flyers +0.36, buildings +1.52); --ground box = raw
    --track-units   YOLO units through live_ppo/unit_tracker.py (low-confidence boxes
                    count once they repeat, class/side vote, ground-point smoothing)
    --tower-latch   a Princess tower whose bar reads unreadable after having read
                    low is latched destroyed (stock: never destroyed, reads 1.0)
    --hud-towers    opponent Princess HP from the numeral above each bar, as a
                    fraction of the first full reading (live_ppo/hud_live.py;
                    stock CRBAB never reads the ladder arena's bars); with it, our
                    own Princesses from the numeral on our bars too (hud_live.
                    OwnTowers; --no-own-towers leaves them to stock CRBAB) and
                    our King from its numeral once its bar is drawn (hud_live.
                    OwnKing; --no-own-king-num keeps the stock bar reading)
    --hud-clock     seconds_elapsed from the on-screen countdown, overtime aware
                    (falls back to the --fill-clock wall clock until it locks)
    --elixir-est    digit | bar: the policy's elixir is a fractional estimate
                    (live_ppo/elixir_estimate.py: dead reckoning between the
                    integer reading's tick-overs; bar: plus the purple bar's
                    fill) instead of the integer. Needs a match clock
                    (--hud-clock or --fill-clock) for the 2x / 3x rate. Default
                    off. The simulator's twin is --sim2real-elixir-est.
    --opp-plays     opponent plays from YOLO enemy births (live_ppo/opp_plays.py)
                    -> engine play recorder: fills the opponent cycle block and
                    opp_elixir_spent (stock: both always zero)
    --opp-spells    with --opp-plays: opponent spell casts too (live_ppo/opp_spells.py)
    --no-opp-watch  with --opp-plays: skip the deploy-stopwatch path (live_ppo/opp_watch.py;
                    on by default: time + tile of each troop/building play from the
                    game's red stopwatch, card from the YOLO bodies born at it)
    --no-opp-evidence        with --opp-plays: the readers without the evidence rules of
                    2026-11-19 (spell casts, forms, YOLO-only plays; on by default,
                    measured by live_ppo/eval/opp_plays_ab.sh)
    --no-opp-spell-evidence  with --opp-spells: only the spell-cast rules off
    --opp-state     with --opp-plays: the observation's opponent-state section
                    (engine item 46, patch 0012; live_ppo/opp_state.py): opponent
                    elixir estimate + confidence, Evolution counters, Hero/Champion
                    ability state, our own evo charge and ability readiness
    --save-raw N    save every Nth raw in-game frame to RUN/raw/
    --record        the audit recording (live_ppo/RECORDING.md): every frame this hook sees,
                    saved off the perception thread to RUN/raw/f<index>.jpg with a
                    timestamp row per frame in RUN/raw/frames.jsonl, and the detector's
                    live boxes in RUN/raw.yolo.jsonl. In-game frames are the exact JPEG
                    bytes the YOLO side-car was sent. --record-every N keeps every Nth
                    in-game frame. play_live.py turns it on by default.
    --dump-obs      append every decision's observation to RUN/obs.f32
    --obs-v2 MODE   observation v2 tail (training/obs_v2.py, live_ppo/obs_tail.py): the
                    last plays of both sides (card, tile, age), our next card, our spells
                    in the air, appended after the engine observation. auto (default):
                    on when the checkpoint was trained with it; on / off. --obs-v2-names
                    all | spells | none picks which opponent card names it carries
                    (default spells: troops and buildings show as an unknown card).

Usage: driver.py --run-dir DIR [driver flags] -- <mvp_loop args>
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import time
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# CR_ENGINE: an engine root other than REPO/engine, e.g. a scratch copy with
# patches/pending applied and built (gen-2). Its perception/ and python_ai/ are used.
ENGINE = Path(os.environ.get("CR_ENGINE") or (REPO / "engine")).resolve()
PERC = ENGINE / "perception"
os.environ.pop("CLASH_DECK", None)       # base 2.6 deck for the CRBAB hand reader
sys.path[:0] = [str(PERC), str(ENGINE), str(ENGINE / "python_ai")]

import argparse  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", type=Path, required=True)
ap.add_argument("--fill-clock", action="store_true")
ap.add_argument("--units", choices=("crbab", "yolo"), default="yolo",
                help="unit source for the policy: yolo = our detector + tracker; crbab = the engine's stock detector")
ap.add_argument("--tile-map", choices=("offset1", "tower"), default="tower",
                help="detector row -> engine row for the YOLO units AND the tap inverse "
                     "(live_ppo/tile_map.py). offset1 = stock engine = det + 1; tower = the "
                     "hog26 tower-anchored map, which matches the measured tower footprints")
ap.add_argument("--ground", choices=("calib", "box"), default="calib",
                help="YOLO unit ground point: calib = box bottom-centre + the dev-fitted "
                     "per-kind offset (unit_tracker.GROUND_OFFSET: troop / flyer / building); "
                     "box = the side-car's raw box bottom-centre tile (old behaviour)")
ap.add_argument("--track-units", action="store_true",
                help="YOLO units through live_ppo/unit_tracker.py: the side-car also returns "
                     "low-confidence boxes, which count once they repeat on the same track; "
                     "class and side are voted over the unit's life, a building's side is the "
                     "half it stands in, and the ground point is smoothed")
ap.add_argument("--det-weights", default=None,
                help="detector weights for the YOLO side-car (default: its own, v2)")
ap.add_argument("--det-conf", type=float, default=0.35,
                help="confidence a box needs to count on its own (opponent plays, overlay, and "
                     "untracked units). With --track-units the side-car runs at the tracker's "
                     "lower threshold and only the tracker sees the boxes below this")
ap.add_argument("--own-deck-prior", action="store_true",
                help="--track-units: a unit on our side must be one of our deck's (off: no "
                     "measured gain on the held-out videos)")
ap.add_argument("--cloaked-rows", choices=("on", "off"), default="on",
                help="on (default since 2026-11-27): give the cloaked cards (Royal Ghost 47 / "
                     "136) the observation row the engine shows when the ghost is revealed "
                     "(live_ppo/cloaked_rows.py); the stock card table has none, so every "
                     "detected ghost was dropped. off: the old behaviour (dropped)")
ap.add_argument("--cloaked-always", action="store_true",
                help="with --cloaked-rows on, pass EVERY ghost box, the cloaked (walking) ones "
                     "too. Off (default): a ghost box reaches the observation only while the "
                     "simulator would show it, from its first hit to 1.8 s after its last "
                     "(live_ppo/cloaked_gate.py)")
ap.add_argument("--body-rows", choices=("on", "on-confirmed", "rows", "off"), default="on-confirmed",
                help="rows: give the bodies a card leaves behind (Golemite, Lava Pups, Elixir "
                     "Golem splits, Cursed Hog, the Rascal Boy and Girls, the reborn Phoenix "
                     "and its egg) the observation row the engine shows for that body instead "
                     "of the parent card's (live_ppo/body_rows.py); every other unit is "
                     "encoded byte for byte as before. on: rows, plus the classes dropped "
                     "today (Elixir Golem big / mid, Goblin Brawler, Phoenix, the Hero "
                     "Barbarian Barrel's Barbarian) reach the tracker and the observation; "
                     "they then take part in the tracker's class-agnostic NMS and matching, "
                     "so units overlapping them can change. on-confirmed is the default since "
                     "2026-11-27 (coordinator ruling; --body-rows rows is the revert to the earlier "
                     "default); off: the encoder as before "
                     "(runs/live_1127/encoder_fixes). on-confirmed: on without Goblin Brawler "
                     "(39 %% of its boxes wrong), and an Elixir Golem / Phoenix / Hero Barbarian "
                     "box reaches the tracker only when a box of the same card stood within 1.5 "
                     "tiles on one of the 3 frames before (body_rows.ProxyConfirm)")
ap.add_argument("--trained-on", default=os.environ.get("CR_LIVE_TRAINED_ON") or None, metavar="BUILD",
                help="2026-11-27: the loop's --checkpoint picks the engine, CR_P0088 and the Ghost mode "
                     "it was trained with from its run's launch.txt (live_ppo/engine_pick.py); the "
                     "driver refuses an engine / CR_P0088 that differ. A checkpoint without a "
                     "launch.txt (the main run g2_ppo_s2r) keeps today's behaviour. Use this (or "
                     "CR_LIVE_TRAINED_ON) only when the launch.txt block cannot be told, e.g. "
                     "latest.pth: gen2_l names the run's last block on build/gen2_l")
ap.add_argument("--bush-goblins", choices=("auto", "on", "off"), default="auto",
                help="the two goblins a Suspicious Bush lets out (boxed as goblin / spear-goblin) "
                     "get the training engine's Bush Goblin row, id 1011 (live_ppo/bush_goblins.py; "
                     "needs --track-units). auto (default): on for a checkpoint trained with "
                     "bushpop (gen2_l), off otherwise, so other checkpoints encode as before")
ap.add_argument("--compare-yolo", action="store_true",
                help="run YOLO beside CRBAB for the overlay/log, without feeding it")
ap.add_argument("--tower-latch", action="store_true")
ap.add_argument("--greedy", action="store_true", help="argmax instead of sampling")
ap.add_argument("--decision-hz", type=float, default=0.0,
                help="decisions per second (0 = --decision-ticks, else the checkpoint's "
                     "decision_ticks or decision_hz, else 1.0)")
ap.add_argument("--decision-ticks", type=int, default=0,
                help="engine ticks (0.1 s) per decision, e.g. 5 = 2 Hz (0 = the checkpoint's "
                     "`decision_ticks` metadata, which gen-2 checkpoints carry; a checkpoint "
                     "without it runs at 1 Hz as before). live_ppo/decision_rate.py has the order "
                     "and the measured loop budget")
ap.add_argument("--temp", type=float, default=1.0,
                help="sampling temperature for both heads (FirstLight-Zero: 0.5 beat 1.0 "
                     "by 66 Elo and argmax by 18); ignored with --greedy. A v3 checkpoint: the "
                     "WHAT (which card) and WHERE temperature, set inside the net; its WHEN "
                     "head has --when-temp / --when-bias")
ap.add_argument("--when-temp", type=float, default=None,
                help="v3 checkpoints only: temperature of the WHEN (play now?) coin, "
                     "p = sigmoid((w + bias) / temp). Default 1 = as trained")
ap.add_argument("--when-bias", type=float, default=None,
                help="v3 checkpoints only: added to the WHEN logit before the temperature "
                     "(> 0 plays more often). Default 0 = as trained")
ap.add_argument("--tactical-officer", action="store_true",
                help="v3 checkpoints only: turn the tactical officer back on (the advisor "
                     "that places Cannon / Fireball / Giant instead of the WHERE head). Off by "
                     "default for v3 whatever the loop's --no-tactical says (live_ppo/"
                     "policy_v3.py); for older checkpoints the loop's flag decides, as before")
ap.add_argument("--pending-mask", choices=("auto", "on", "off"), default="auto",
                help="a card whose placement is still in flight (issued, not yet seen leaving "
                     "its slot: play_watch) cannot be chosen again and is shown as the "
                     "simulator's actuation delay shows it (slot cost 0; elixir paid). auto: on "
                     "for a v3 checkpoint, which decides twice a second, off for the others")
ap.add_argument("--solvency-gate", action="store_true",
                help="re-enable the live-only elixir-reserve veto (refuse a card that leaves "
                     "< 4 elixir while nothing attacks). Off by default: training never used "
                     "it (shipping.USE_SOLVENCY_GATE = False) and it vetoed plays the net chose")
ap.add_argument("--leak-floor", type=float, default=None,
                help="at or above this elixir the no-op is masked, so the net must pick a card "
                     "(its own card and placement heads choose which/where). The gen-1 policy "
                     "holds with P(no-op) ~0.9 on a full bar when the opponent is passive, in the "
                     "sim too. 0 or >10 disables. Default 10; for a v3 checkpoint the default is "
                     "off, and a value forces its WHEN coin to 'play' instead")
ap.add_argument("--overlay-every", type=int, default=1)
ap.add_argument("--hand", choices=("screen", "tracker"), default="screen",
                help="hand fed to the policy and used for taps: the per-frame screen read "
                     "(tracker fills only unreadable slots), or the engine's FIFO tracker")
ap.add_argument("--hand-cycle", action="store_true",
                help="hand from the deck cycle (live_ppo/state_filter.HandCycle): a slot that "
                     "goes blank holds the next card at once, as in the engine")
ap.add_argument("--elixir-bar", action="store_true",
                help="cross-check the elixir digit against the purple bar; replace a "
                     "misread (drop flash read 0, a '0' read 1) by the bar's value")
ap.add_argument("--elixir-est", choices=("off", "digit", "bar"), default="off",
                help="the policy's elixir: a fractional estimate (live_ppo/elixir_estimate.py) "
                     "instead of the integer reading. digit: dead reckoning between the "
                     "reading's tick-overs; bar: plus the purple bar's fill")
ap.add_argument("--ledger-tol", type=float, default=None,
                help="ElixirLedger cost tolerance (stock 1.0: with 1-cost cards in the deck "
                     "every non-rising frame books a fake 1-elixir play)")
ap.add_argument("--no-play-watch", action="store_true",
                help="off: live_ppo/play_watch.py, which reads from the hand tray whether each "
                     "placement landed, charges the policy's elixir only for plays the bar has "
                     "not caught up with, and prints the landed count at the end")
ap.add_argument("--allow-empty-slot", action="store_true",
                help="off: the rule that a slot whose raw read is blank (its card still "
                     "arriving) cannot be chosen. Live 093139: 4 placements tapped an empty slot")
ap.add_argument("--no-front-guard", action="store_true",
                help="send the taps even when iPhone Mirroring could not be brought to the "
                     "front (default: no tap is sent, since it would click another app)")
ap.add_argument("--loop-frames", type=Path, default=None,
                help="soak test: with --capture mirror, serve these recorded frames "
                     "(cycled, live-paced) instead of the window. Never touches the phone.")
ap.add_argument("--loop-from", type=int, default=0)
ap.add_argument("--inject-stall-at", type=float, default=None,
                help="guard test, --loop-frames only: the frame source blocks forever "
                     "this many seconds in (reproduces 2026-11-17's frozen capture)")
ap.add_argument("--max-board-age-ms", type=float, default=1000.0,
                help="hard tap guard: no tap on a board older than this (stock 2000)")
ap.add_argument("--tap-gap-ms", type=int, default=150,
                help="pause between the card tap and the arena tap (stock 80; at 80 the "
                     "game showed 'No card selected' twice in one Training Camp match)")
ap.add_argument("--tap-hold-ms", type=int, default=80, help="touch hold per tap (stock 60)")
ap.add_argument("--tap-mode", choices=("click", "drag"), default="click",
                help="click: card tap then arena tap (stock). drag: one touch, pressed on the "
                     "card and dragged to the target, the game's own drag-to-play; no gap "
                     "between two taps, so no 'No card selected'")
ap.add_argument("--drag-ms", type=int, default=60,
                help="--tap-mode drag: duration of the drag from the card to the target")
ap.add_argument("--pocket-mask", action="store_true",
                help="open the policy's WHERE mask to the pocket (engine rows 16-22 of a lane whose enemy "
                     "Princess is dead) once the death is CONFIRMED (live_ppo/pocket_live.py, needs "
                     "--hud-towers); a lane closes again after the game refuses a pocket tile. Default off")
ap.add_argument("--pocket-confirm-s", type=float, default=1.0,
                help="--pocket-mask: seconds the destroyed latch must hold with no numeral read")
ap.add_argument("--hud-towers", action="store_true",
                help="opponent Princess HP fraction from the tower numerals")
ap.add_argument("--no-own-towers", action="store_true",
                help="with --hud-towers: leave our own Princesses to stock CRBAB (default: "
                     "read them from our numerals, hud_live.OwnTowers)")
ap.add_argument("--no-own-king-num", action="store_true",
                help="with --hud-towers: keep the stock King bar reading for our King (default: "
                     "its numeral once the bar is drawn, hud_live.OwnKing)")
ap.add_argument("--hud-hz", type=float, default=16.0,
                help="perception frames per second, for --hud-towers' destroyed latch (1.5 s "
                     "without a numeral) and --hand-cycle's debounce/lag windows (tuned at 4 Hz, "
                     "scaled to this rate); state_pipeline passes the video's frame rate")
ap.add_argument("--hud-clock", action="store_true",
                help="seconds_elapsed from the on-screen countdown (overtime aware)")
ap.add_argument("--opp-plays", action="store_true",
                help="opponent plays from YOLO births -> cycle block + opp_elixir_spent")
ap.add_argument("--opp-spells", action="store_true",
                help="--opp-plays also takes opponent spell casts (implies --opp-plays)")
ap.add_argument("--keep-spent", action="store_true",
                help="old behaviour: our elixir-spent total keeps counting across the matches of one "
                     "launch (the ledger is created once). Default: the total restarts at each match, "
                     "as the simulator's does. Live 20261123-004549: by match 4 the policy was fed "
                     "186 spent at 35 s and stopped playing.")
ap.add_argument("--no-opp-watch", action="store_true",
                help="--opp-plays without the deploy-stopwatch path (YOLO births only)")
ap.add_argument("--no-opp-evidence", action="store_true",
                help="--opp-plays without the evidence rules of 2026-11-19 (the readers as before)")
ap.add_argument("--no-opp-spell-evidence", action="store_true",
                help="--opp-spells without the spell-cast evidence rules; the troop rules stay")
ap.add_argument("--opp-rules", default="own_units,badge_marks,three_musk,rage,gang_bodies",
                help="comma list of play-reader rules of 2026-11-27 to switch on; all five on by "
                     "default since 2026-11-27 (coordinator ruling); --opp-rules "
                     "own_units,badge_marks,three_musk is the revert to the earlier default, "
                     "--opp-rules '' turns every rule off "
                     "(live_ppo/opp_plays.py, test live_ppo/eval/test_opp_rules.py, replay "
                     "live_ppo/eval/opp_rules_replay.py): own_units (our own deck's units near "
                     "their stopwatch do not name it), badge_marks (a red stopwatch drawn over "
                     "their unit's level badge on our half is not a play), three_musk (2+ "
                     "musketeers born together on their half are Three Musketeers), rage (a stopwatch with a YOLO rage box on it is their Rage, unless "
                     "a known deck lacks Rage), gang_bodies (Goblin Gang needs 4 of its bodies)")
ap.add_argument("--opp-state", action="store_true",
                help="fill the opponent-state section (engine item 46; needs an engine with "
                     "patch 0012; implies --opp-plays). On by itself, with --opp-spells, for a "
                     "generation-2 (v3) checkpoint: it was trained with that section filled")
ap.add_argument("--no-opp-state", action="store_true",
                help="v3 checkpoint: leave the opponent-state section at zero and do not turn "
                     "--opp-plays / --opp-spells on for it (the model never saw that; for "
                     "comparisons only)")
ap.add_argument("--obs-v2", choices=("auto", "on", "off"), default="auto",
                help="observation v2 tail (training/obs_v2.py: recent plays of both sides, our next "
                     "card; live_ppo/obs_tail.py) appended after the engine observation. auto: on "
                     "when the checkpoint says obs_v2 (blob['obs_v2'] or its cfg); a network "
                     "trained without it ignores it either way")
ap.add_argument("--obs-v2-names", choices=("all", "spells", "none"), default=None,
                help="which opponent card names the tail carries: spells only (default; troop and "
                     "building plays show as an unknown card with time and tile: on live match 2 "
                     "the reader named 19 of 25 troop plays right), all, or none. Must match the "
                     "`names` of the obs_v2_noise the policy was trained with: a v3 checkpoint "
                     "carries it, so its value is the default and any other is refused")
ap.add_argument("--save-raw", type=int, default=0, metavar="N",
                help="save every Nth raw in-game frame to RUN/raw/ (0 = off)")
ap.add_argument("--record", action="store_true",
                help="audit recording: every frame the perception hook sees -> RUN/raw/ "
                     "(JPEG + raw/frames.jsonl timestamps + raw.yolo.jsonl live boxes), "
                     "written by a background thread; layout in live_ppo/RECORDING.md")
ap.add_argument("--record-every", type=int, default=1, metavar="N",
                help="--record: save every Nth in-game frame (default 1 = all); the index "
                     "still has a row for every frame")
ap.add_argument("--levels", default=None, metavar="OC,OT,PC,PT",
                help="gen-2 engines only (engine item 44): own card, own tower, opponent card, "
                     "opponent tower level for the observation's level section, e.g. 11,11,15,15 "
                     "(the King level is shown on each King Tower). Default: the engine's "
                     "tournament level for all four. Ignored by an engine without the section.")
ap.add_argument("--dump-obs", action="store_true",
                help="append each decision's observation vector (float32) to RUN/obs.f32")
ap.add_argument("--loop-realtime", action="store_true",
                help="--loop-frames: serve the recording at its own 4 fps, one board per "
                     "recorded frame, frame index = recorded frame number (the HUD clock "
                     "filter needs match time to advance at wall speed)")
ap.add_argument("--loop-timestamps", action="store_true",
                help="--loop-frames: serve a run's raw/ recording at its recorded times "
                     "(frames.jsonl beside the frames), frame index = recorded frame number, "
                     "once through; where the recording has a gap, or is over, the last frame "
                     "is served again at 4 fps. For recordings that are not 4 fps (a live "
                     "run's is about 11): without it such a match replays in slow motion and "
                     "every timed rule sees stretched time")
ap.add_argument("--ability", choices=("off", "log", "tap"), default="log",
                help="generation-2 (v3) checkpoints only: the Hero ability button. log (default) = "
                     "read the button every frame (live_ppo/ability_button.py), fill the "
                     "observation's own-ability fields from it, mask the policy's ability head by "
                     "it, and log / draw the press it would make -- nothing is sent. tap = also "
                     "send the press (one tap on the button) when the loop runs with --act. "
                     "off = as before this reader existed (the head's draw is logged only). "
                     "Ignored for any other checkpoint")
ap.add_argument("--afford-project", action="store_true",
                help="DEFAULT OFF; not for the standard command. Projected affordability "
                     "(runs/imit_timing_1126/AFFORD_RULE.txt s1/s4): a card is offered when its cost "
                     "<= P = min(E_view + (1/28)*MULT*13, 10) (E_view = the observation's elixir after "
                     "the plays in flight; 0 -> no projection), since the game queues a release that is a "
                     "fraction of a second short. Only for a checkpoint trained with the rule. Refuses "
                     "--solvency-gate; PlayWatch waits 1.3 s longer for the slot to empty; logs E_view, "
                     "MULT, P, cost per decision (perception.jsonl 'afford') and RUN/afford.json")
ours, loop_args = ap.parse_known_args()
_OPP_RULE_ON = [x.strip() for x in ours.opp_rules.split(",") if x.strip()]
for _r in _OPP_RULE_ON:
    if _r not in ("own_units", "badge_marks", "three_musk", "rage", "gang_bodies"):
        ap.error(f"--opp-rules: unknown rule {_r!r} (known: own_units, badge_marks, three_musk, rage, gang_bodies)")
if ours.afford_project and ours.solvency_gate:
    raise SystemExit("--afford-project: the solvency gate checks the bar against the cost; it is not "
                     "written for the projected rule, so the two cannot be combined (AFFORD_RULE.txt s4b)")
ours.opp_plays = ours.opp_plays or ours.opp_spells or ours.opp_state
try:
    # clashroyalebuildabot's detector logs "Setting state..." at DEBUG on every frame.
    from loguru import logger as _loguru
    _loguru.remove()
    _loguru.add(sys.stderr, level="INFO")
except ImportError:
    pass
if loop_args and loop_args[0] == "--":
    loop_args = loop_args[1:]

if ours.inject_stall_at is not None and ours.loop_frames is None:
    raise SystemExit("--inject-stall-at is a test hook and needs --loop-frames")
RUN = ours.run_dir
ACT = "--act" in loop_args
(RUN / "overlay").mkdir(parents=True, exist_ok=True)
PLOG = open(RUN / "perception.jsonl", "a", buffering=1)

import live.mvp_loop as L  # noqa: E402
from clashroyalebuildabot.constants import SCREENSHOT_HEIGHT, SCREENSHOT_WIDTH  # noqa: E402
from contracts import TowerObservation  # noqa: E402
from live.actuator import engine_tile_centre, DISPLAY_WIDTH, DISPLAY_HEIGHT  # noqa: E402
from live.unit_to_card import UnitMappingError, card_id_for, _norm  # noqa: E402

import clash_royale_env as E  # noqa: E402

# Base hand ids (what the CRBAB reader maps to) -> the ids the PPO net trained on.
TRAIN_DECK = [125, 168, 129, 15, 40, 72, 33, 7]
BASE_TO_TRAIN = {24: 125, 6: 168, 25: 129}
NAME = {i: E.get_card_info(i)["name"] for i in TRAIN_DECK + [24, 6, 25]}

# ----------------------------------------------------------------- engine per checkpoint
# 2026-11-27 (gen2_l live prep): the loop's --checkpoint names the engine, the CR_P0088 switches and
# the Ghost mode it was trained with (live_ppo/engine_pick.py, from its run's launch.txt). Without a
# launch.txt (the main run runs/g2_ppo_s2r, imitation models) or without --checkpoint (state_pipeline)
# there is no profile and nothing here changes. A profile must match the engine loaded above
# (CR_ENGINE) and CR_P0088, or the driver refuses; cloakboard forces --cloaked-always (that sim draws
# every Ghost, cloaked too); bushboard needs the stock row 85 (the walking Bush); bushpop turns
# --bush-goblins auto on.
import engine_pick  # noqa: E402

PROFILE = None
_P88_ENV = os.environ.get("CR_P0088")
if "--checkpoint" in loop_args:
    try:
        PROFILE = engine_pick.pick(loop_args[loop_args.index("--checkpoint") + 1], ours.trained_on)
    except engine_pick.PickError as ex:
        raise SystemExit(f"engine pick refused: {ex}") from ex
elif ours.trained_on:
    print(f"--trained-on / CR_LIVE_TRAINED_ON={ours.trained_on}: ignored, no --checkpoint")
if PROFILE is None:
    print(f"engine: {ENGINE} (CR_ENGINE), CR_P0088={_P88_ENV!r}; no launch.txt profile for the "
          f"checkpoint: engine, switches and Ghost mode as given (unchanged behaviour)")
else:
    _bad = []
    if ENGINE != PROFILE.engine:
        _bad.append(f"CR_ENGINE is {ENGINE}, the checkpoint was trained on {PROFILE.engine}")
    if not engine_pick.same_sw(_P88_ENV or "", PROFILE.sw):
        _bad.append(f"CR_P0088={_P88_ENV!r}, the checkpoint was trained with {PROFILE.p0088!r}")
    elif PROFILE.sw:
        if not hasattr(E, "p0088_active"):
            _bad.append(f"the engine {ENGINE} has no CR_P0088 switches (p0088_active)")
        else:
            _act = E.p0088_active()
            _miss = [s for s in PROFILE.sw if s != "all" and _act.get(s) is not True]
            if not engine_pick.same_sw(str(_act.get("raw", "")), PROFILE.sw) or _miss:
                _bad.append(f"the engine latched CR_P0088={_act.get('raw')!r}; switches not on: {_miss}")
    if _bad:
        raise SystemExit("engine pick refused (" + PROFILE.describe() + "): " + "; ".join(_bad))
    if PROFILE.has("cloakboard"):
        if ours.cloaked_rows == "off":
            raise SystemExit("--cloaked-rows off on a checkpoint trained with cloakboard (the sim draws every "
                             "Ghost): refusing")
        if not ours.cloaked_always:
            ours.cloaked_always = True
    elif ours.cloaked_always:
        raise SystemExit("--cloaked-always on a checkpoint trained without cloakboard (its sim shows a Ghost "
                         "only while revealed, the cloaked gate's job): refusing")
    if PROFILE.has("bushboard"):
        from python_ai.models import perception_encoder as _PE  # noqa: PLC0415
        if 85 not in _PE.card_table():
            raise SystemExit(f"bushboard profile but the card table of {ENGINE} has no row 85 (Suspicious Bush)")
    print(f"engine (from the checkpoint's training run): {PROFILE.describe()}")
    for _n in PROFILE.notes:
        print(f"  engine pick note: {_n}")
BUSH_ON = ours.bush_goblins == "on" or (ours.bush_goblins == "auto" and PROFILE is not None
                                         and PROFILE.has("bushpop"))
print(f"ghost mode: {'every ghost box (--cloaked-always)' if ours.cloaked_always else 'reveal gate' if ours.cloaked_rows == 'on' else 'dropped (--cloaked-rows off)'}"
      f"{' [forced by cloakboard]' if PROFILE is not None and PROFILE.has('cloakboard') else ''}; "
      f"bush goblins: {'on' if BUSH_ON else 'off'} (--bush-goblins {ours.bush_goblins})")
if PROFILE is not None and PROFILE.has("bushpop") and not BUSH_ON:
    print("WARNING: --bush-goblins off on a checkpoint trained with bushpop: popped Bush Goblins encode "
          "as Goblins / Spear Goblins, which its training never showed")
if BUSH_ON and not ours.track_units:
    raise SystemExit("--bush-goblins needs --track-units (the rule reads the tracker's tracks)")


# ----------------------------------------------------------------- YOLO side-car
JPEG_QUALITY = 92


class Yolo:
    def __init__(self, conf=0.35, weights=None):
        cmd = [str(REPO / ".venv-det" / "bin" / "python"), str(REPO / "live_ppo" / "yolo_server.py"),
               "--conf", str(conf)]
        if weights:
            cmd += ["--weights", str(weights)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        hello = json.loads(self.p.stdout.readline())
        print(f"yolo side-car ready on {hello['device']}: {hello.get('weights')} conf {hello.get('conf')}")
        self.hello = hello
        self.ms = []

    def __call__(self, bgr, jpeg=None):
        # jpeg: this frame already encoded by the recorder (same call, same quality),
        # so the file on disk is byte for byte what the side-car decodes
        b = jpeg if jpeg is not None else encode_jpeg(bgr)
        t = time.perf_counter()
        self.p.stdin.write(struct.pack(">I", len(b)) + b)
        self.p.stdin.flush()
        r = json.loads(self.p.stdout.readline())
        self.ms.append((time.perf_counter() - t) * 1000)
        return r["dets"]


def encode_jpeg(bgr):
    """The one JPEG encoding of a perception frame: sent to the YOLO side-car and,
    with --record, written to RUN/raw/ unchanged."""
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes()


_YOLO_ID = {}


def yolo_card_id(cls: str):
    """YOLO class (hog-rider, skeleton-ev1, musketeer-hero) -> engine base card id,
    or None for spells/projectiles/unmapped. Base ids: the encoder's attribute table
    excludes Evolutions."""
    if cls in _YOLO_ID:
        return _YOLO_ID[cls]
    base = cls
    for suf in ("-ev1", "-hero"):
        if base.endswith(suf):
            base = base[: -len(suf)]
    cid = None
    try:
        cid = card_id_for(base.replace("-", "_"))
    except UnitMappingError:
        by = {_norm(E.get_card_info(c)["name"]): c for c in E.get_all_card_ids()}
        cid = by.get(_norm(base)) or by.get(_norm(base + "s"))
    if cid is not None and E.get_card_info(cid).get("is_spell"):
        cid = None
    _YOLO_ID[cls] = cid
    return cid


if ours.body_rows != "off":
    import body_rows  # noqa: E402
if ours.body_rows == "on":
    # live_ppo/body_rows.py: classes that map to no card today get a real proxy id of the
    # body's own kind (troop / fly), for the unit path only; their rows come in EvoEncoder
    _YOLO_ID.update(body_rows.PROXY)
PROXY_GATE = None
if ours.body_rows == "on-confirmed":
    # as "on" without goblin-brawler; each proxy box must be confirmed by the frames before
    _YOLO_ID.update(body_rows.PROXY_CONFIRMED)
    PROXY_GATE = body_rows.ProxyConfirm()


import live.wda_actuator as _WA  # noqa: E402
_WA.TAP_GAP_S = ours.tap_gap_ms / 1000
_WA.TOUCH_HOLD_S = ours.tap_hold_ms / 1000

# Command -> execution latency log (live_ppo/latency_report.py reads it). One JSON
# line per event, all times time.perf_counter() seconds: "board" = a perceived
# in-game board (capture time, screen hand, elixir), "decide" = a play the policy
# chose on a board, "enq" = the play handed to the actuator, "tap" = the click pair
# actually sent (start/end).
import threading  # noqa: E402

LAT = open(RUN / "latency.jsonl", "a", buffering=1)
_LAT_LOCK = threading.Lock()   # written from the perception, policy and actuator threads


def lat(kind, **f):
    f["k"] = kind
    with _LAT_LOCK:
        LAT.write(json.dumps(f) + "\n")


sys.path.insert(0, str(REPO / "live_ppo"))
import play_watch  # noqa: E402

PLAYS = None if ours.no_play_watch else play_watch.PlayWatch()
# --afford-project (AFFORD_RULE.txt; same constants as the sim hook runs/pocket_1126/fix/afford_lead.py)
AFF_REGEN = 1.0 / 28.0           # elixir per tick at 1x (gen2_j GameManager.h:251; 2.8 s per elixir)
AFF_DMIN = 13                    # ticks (low end of the trained act delay 13,15,1; a constant)
AFF_QUEUE_S = AFF_DMIN / 10.0    # longest queue wait the projection allows (shortfall <= G = 13 ticks of regen)
_AFF = {"last": None, "decisions": 0, "differ": 0, "plays": [], "p_only": 0}
if ours.afford_project and PLAYS is not None:
    # a queued release empties its slot up to AFF_QUEUE_S later than usual; its cost stays owed while open
    PLAYS.LAND_S = play_watch.PlayWatch.LAND_S + AFF_QUEUE_S
_LAST_DECIDE = {}                # slot -> (card name, tile) of the policy's last choice
EMPTY_MAX_S = 3.0                # a slot blank longer than this is no longer held back (a card
                                 # left selected reads blank too, and must stay playable)
_BLANK_SINCE = [None] * 4        # per slot: when its raw read turned blank
_EMPTY = {"slots": []}           # slots the policy may not choose on the board being decided
_INFLIGHT = {"slots": {}, "applied": []}   # slots holding a card already on its way (--pending-mask)

# ----------------------------------------------------------------- the checkpoint's kind
# Some defaults depend on what the loop is about to load (live_ppo/policy_v3.py): a
# generation-2 (v3) checkpoint carries its tail names and runs without the gen-1
# crutches. The blob named by the loop's --checkpoint is read once, the first time a
# default is needed; without --checkpoint (state_pipeline) nothing is read and every
# default is the old one.
import policy_v3  # noqa: E402

_CK = {}


def _engine_gen2():
    return hasattr(E.ClashRoyaleEnv, "LEVEL_START")     # as training/obs_compat.IS_GEN2


def _ckpt():
    """(blob, kind) of the loop's --checkpoint; (None, None) without one. A v3
    checkpoint this process cannot run is refused here, before anything starts."""
    if "v" not in _CK:
        blob = kind = None
        if "--checkpoint" in loop_args:
            import torch as _t  # noqa: PLC0415
            path = loop_args[loop_args.index("--checkpoint") + 1]
            blob = _t.load(path, map_location="cpu", weights_only=False)
            kind = policy_v3.arch(blob)
            if kind == policy_v3.V3:
                try:
                    policy_v3.check(blob, path=path, engine_gen2=_engine_gen2(), engine=str(ENGINE),
                                    obs_v2=ours.obs_v2, names=ours.obs_v2_names,
                                    driver_deck=TRAIN_DECK)
                except policy_v3.V3Error as ex:
                    raise SystemExit(f"v3 checkpoint refused: {ex}") from ex
            # afford_lead marker <-> --afford-project (live_ppo/afford_marker.py, 2026-11-26 pm; read-only)
            import afford_marker  # noqa: PLC0415
            _am = afford_marker.check(path, blob, ours.afford_project)
            if _am:
                raise SystemExit(_am)
        _CK["v"] = (blob, kind)
    return _CK["v"]


def _settings(kind, blob):
    """policy_v3.settings from this command line, for a checkpoint of this kind."""
    try:
        return policy_v3.settings(
            kind, blob, names=ours.obs_v2_names, leak_floor=ours.leak_floor,
            tactical_loop="--no-tactical" not in loop_args,
            tactical_officer=ours.tactical_officer, pending_mask=ours.pending_mask,
            play_watch=PLAYS is not None)
    except policy_v3.V3Error as ex:
        raise SystemExit(str(ex)) from ex


def SET():
    """The settings for the loop's --checkpoint (old defaults without one)."""
    if "set" not in _CK:
        blob, kind = _ckpt()
        _CK["set"] = _settings(kind, blob)
    return _CK["set"]


# A v3 checkpoint was trained on the engine's opponent-state section (418 floats: the
# opponent's elixir estimate, our Evolution and Hero state, theirs) and on a tail whose
# opponent part is fed by the opponent's plays, spells included. Both readers follow
# the checkpoint, here, before anything below looks at the three flags.
if ours.opp_state and ours.no_opp_state:
    raise SystemExit("--opp-state and --no-opp-state together")
if _ckpt()[1] == policy_v3.V3:
    if ours.no_opp_state:
        print("WARNING: v3 checkpoint with --no-opp-state: the opponent-state section stays "
              "zero, which this model never saw in training")
    else:
        if not (ours.opp_state and ours.opp_spells):
            print("v3 checkpoint: --opp-state --opp-plays --opp-spells turned on (the model was "
                  "trained with the opponent-state section and the opponent's plays filled)")
        ours.opp_state = ours.opp_plays = ours.opp_spells = True


class Recorder:
    """--record: the audit recording of a run (layout: live_ppo/RECORDING.md).

    frame() is called once per frame the perception hook (bgs) sees, on the perception
    thread; it only queues. One background thread writes the JPEGs, one row per frame in
    raw/frames.jsonl (after that frame's file is complete) and the live detector boxes in
    raw.yolo.jsonl. In-game frames arrive already encoded: the same bytes go to the YOLO
    side-car, so recording adds no encode to the perception thread. If the disk falls
    more than MAX_PENDING frames behind, frames are dropped (row kept, "dropped": true)
    rather than slowing perception."""

    MAX_PENDING = 256           # ~40 MB of queued JPEGs
    MENU_TAIL_S = 3.0           # non-game frames: all of them this long after the last in-game frame,
    MENU_EVERY_S = 1.0          # then one per second (menus, loading, result screens)

    def __init__(self, run, every=1, meta=None):
        import queue  # noqa: PLC0415
        self.dir = run / "raw"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.every = max(1, int(every))
        self.q = queue.Queue()
        self.pending = 0
        self.idx = open(self.dir / "frames.jsonl", "a", buffering=1)
        self.dets_f = open(run / "raw.yolo.jsonl", "a", buffering=1)
        self.epoch_off = time.time() - time.perf_counter()   # perf_counter s -> unix s
        self.seq = self.n_in = self.saved = self.saved_in = self.dropped = self.bytes = 0
        self.last_in_ms = self.last_menu_ms = None
        self.first = self.last = None        # (wall_time_ms) of the first / last saved in-game frame
        self.rows = []
        (self.dir / "record.json").write_text(json.dumps(dict(
            meta or {}, jpeg_quality=JPEG_QUALITY, every=self.every,
            started_unix=round(time.time(), 3), perf_to_unix_offset_s=round(self.epoch_off, 6)), indent=1))
        self.th = threading.Thread(target=self._run, name="recorder", daemon=True)
        self.th.start()

    def frame(self, fi, wall_ms, in_game, screen, match, since_start_s, rgb, bgr):
        """Queue one perception frame. Returns the JPEG bytes when this frame was
        encoded here (an in-game frame that is being saved), for the YOLO side-car."""
        t0 = _W.get("t0")
        t_perf = (t0 + wall_ms / 1000.0) if t0 is not None else time.perf_counter()
        row = {"seq": self.seq, "frame": int(fi), "file": None, "wall_time_ms": round(float(wall_ms), 3),
               "t_perf": round(t_perf, 6), "t_unix": round(t_perf + self.epoch_off, 6),
               "in_game": bool(in_game), "screen": screen, "match": int(match),
               "since_match_start_s": round(float(since_start_s), 3)}
        self.seq += 1
        jpeg = payload = None
        if in_game:
            save = self.n_in % self.every == 0
            self.n_in += 1
            self.last_in_ms = wall_ms
        else:
            tail = self.last_in_ms is not None and wall_ms - self.last_in_ms <= self.MENU_TAIL_S * 1000
            save = tail or self.last_menu_ms is None or wall_ms - self.last_menu_ms >= self.MENU_EVERY_S * 1000
            if save:
                self.last_menu_ms = wall_ms
        if save and self.pending >= self.MAX_PENDING:
            save = False
            row["dropped"] = True
            self.dropped += 1
        if save:
            row["file"] = f"f{int(fi):06d}.jpg"
            if in_game:
                jpeg = payload = encode_jpeg(bgr)
            else:
                payload = rgb                      # encoded by the writer thread
            self.pending += 1
        self.q.put(("f", row, payload))
        return jpeg

    def dets(self, fi, wall_ms, dets):
        """The boxes the live side-car returned for this frame (everything it sent,
        i.e. down to the side-car's own --conf), in the .yolo.jsonl cache format."""
        self.q.put(("d", {"frame": int(fi), "t": round(float(wall_ms) / 1000.0, 3), "battle": True,
                          "dets": dets}, None))

    def _run(self):
        while True:
            kind, row, payload = self.q.get()
            try:
                if kind == "stop":
                    return
                if kind == "d":
                    self.dets_f.write(json.dumps(row) + "\n")
                    continue
                if payload is not None:
                    if not isinstance(payload, (bytes, bytearray)):
                        payload = encode_jpeg(np.ascontiguousarray(payload[:, :, ::-1]))
                    with open(self.dir / row["file"], "wb") as fh:
                        fh.write(payload)
                    row["bytes"] = len(payload)
                    self.bytes += len(payload)
                    self.saved += 1
                    if row["in_game"]:
                        self.saved_in += 1
                        self.first = row["wall_time_ms"] if self.first is None else self.first
                        self.last = row["wall_time_ms"]
                    self.rows.append(row)
                    self.pending -= 1
                self.idx.write(json.dumps(row) + "\n")
            except Exception as exc:  # noqa: BLE001  a full disk must not take the match down
                print(f"RECORDER ERROR: {exc!r}", flush=True)
                if kind == "f" and payload is not None:
                    self.pending -= 1
            finally:
                self.q.task_done()

    def close(self, timeout_s=30.0):
        """Flush the queue, write raw/manifest.json (record_mirror.py's format), print a summary."""
        self.q.put(("stop", None, None))
        self.th.join(timeout_s)
        span = (self.last - self.first) / 1000.0 if self.saved_in > 1 else 0.0
        fps = (self.saved_in - 1) / span if span > 0 else 0.0
        t_first = self.rows[0]["wall_time_ms"] if self.rows else 0.0
        (self.dir / "manifest.json").write_text(json.dumps({
            "frames": [{"file": r["file"], "index": r["frame"], "wall_time_ms": r["wall_time_ms"],
                        "t_since_start_s": round((r["wall_time_ms"] - t_first) / 1000.0, 4)}
                       for r in self.rows],
            "count": len(self.rows), "in_game_count": self.saved_in,
            "elapsed_s": round((self.rows[-1]["wall_time_ms"] - t_first) / 1000.0, 3) if self.rows else 0.0,
            "in_game_span_s": round(span, 3), "achieved_fps": round(fps, 2),
            "frames_seen": self.seq, "dropped": self.dropped, "bytes": self.bytes,
            "source": "driver.py --record"}, indent=0))
        self.idx.close()
        self.dets_f.close()
        print(f"recording: {self.saved} frames ({self.saved_in} in-game at {fps:.1f} fps over {span:.0f} s) "
              f"of {self.seq} seen, {self.bytes / 1e6:.0f} MB, {self.dropped} dropped -> {self.dir}")


# ----------------------------------------------------------------- the Hero ability
# Generation-2 (v3) checkpoints only: with any other checkpoint, or --ability off,
# ABIL stays None and nothing below is installed. ability_button.py reads whether
# our Hero's button is on screen and ready; ability_watch.py follows a press the
# way play_watch follows a placement; ability_tap.py sends the one tap through the
# actuator's own worker. --ability log (the default) does everything but send.
ABIL = None            # ability_watch.AbilityWatch: the press tracker
ABIL_BTN = None        # ability_button.ButtonFilter: the reader's time rule
AB = None              # the ability_button module
_ABIL = {"mode": "off", "card": None, "slot": None, "cost": None, "actuator": None, "ledger": None,
         "in_game": False, "in_match": False, "tapping": False, "dec": None, "log": None,
         "frames": 0, "states": {}, "ms": [], "ready_dec": 0, "want": 0, "held": {}}
_ABILITY_TAG = "ability"     # the ledger tag of an ability press (placements are tagged with ints)


def _ability_in_battle():
    """The loop's debounced match gate AND the newest frame's own screen read."""
    return bool(_ABIL["in_match"] and _ABIL["in_game"])


def _ability_setup():
    global ABIL, ABIL_BTN, AB
    if ours.ability == "off" or _ckpt()[1] != policy_v3.V3:
        return
    import ability_button as _AB  # noqa: PLC0415
    slots = [s for s in (1, 2) if s < len(TRAIN_DECK)
             and (E.get_card_info(TRAIN_DECK[s]).get("is_champion")
                  or E.get_card_info(TRAIN_DECK[s]).get("is_hero"))]
    names = [E.get_card_info(TRAIN_DECK[s])["name"] for s in slots]
    safe = _AB.tap_is_safe()
    if names != [_AB.HERO_NAME] or not safe[0] or not _AB.ASSET.exists():
        why = (f"the deck's ability cards are {names}, the reader knows the {_AB.HERO_NAME}'s button only"
               if names != [_AB.HERO_NAME] else f"tap point refused: {safe[1]}" if not safe[0]
               else f"{_AB.ASSET} is missing (live_ppo/eval/build_ability_button.py)")
        print(f"hero ability: OFF -- {why}")
        return
    import ability_tap  # noqa: PLC0415
    import ability_watch  # noqa: PLC0415
    _AB.templates()
    cost = float(E.get_card_stats(TRAIN_DECK[slots[0]])["base"]["ability_cost"])
    AB, ABIL_BTN, ABIL = _AB, _AB.ButtonFilter(), ability_watch.AbilityWatch(cost)
    mode = "tap" if (ours.ability == "tap" and ACT and ours.loop_frames is None) else "log"
    _ABIL.update(mode=mode, card=TRAIN_DECK[slots[0]], slot=slots[0], cost=cost,
                 log=open(RUN / "ability.jsonl", "a", buffering=1))
    x, y = AB.tap_display()
    print(f"hero ability: {names[0]} (engine slot {slots[0]}, cost {cost:g} from the engine's card data); "
          f"button read every in-game frame; tap point ({x}, {y}) in the {DISPLAY_WIDTH}x{DISPLAY_HEIGHT} "
          f"display space = frame px {AB.tap_frame_px()}; "
          + ("PRESSES ARE SENT (--ability tap with --act)" if mode == "tap"
             else "presses are logged and drawn, NOT sent"
                  + (" (--ability tap needs --act and a live source)" if ours.ability == "tap"
                     else " (--ability log; --ability tap sends them)")))

    _orig_ms_update = L.MatchState.update

    def _ms_update(self, screen_name):
        out = _orig_ms_update(self, screen_name)
        _ABIL["in_match"] = bool(out)
        return out

    L.MatchState.update = _ms_update
    if mode != "tap":
        return

    # ---- sending: everything below exists only with --ability tap and --act
    from live import elixir_ledger as _ELA  # noqa: PLC0415
    record_play, update = _ELA.ElixirLedger.record_play, _ELA.ElixirLedger.update

    def _check(press):
        return ability_tap.last_check(
            press=press, in_battle=_ability_in_battle(),
            ready_now=ABIL_BTN.ready_now(time.perf_counter()),
            plays_in_flight=len(PLAYS.in_flight()) if PLAYS is not None else 0,
            safe=AB.tap_is_safe(), disabled=ABIL.disabled)

    def _sent(press):
        # The bar will drop by the ability's cost. The ledger is told, so the drop
        # is matched to this press and not booked as a card (a 3 would read as a
        # Cannon); _ledger_update takes it out of `spent` again, which in the
        # simulator counts cards only (ElixirStatsCollector::onCardPlayed).
        led = _ABIL["ledger"]
        if led is not None:
            record_play(led, press.cost, tag=_ABILITY_TAG)
        print(f"ABILITY press #{press.i}: tapped ({x}, {y})", flush=True)

    def _tap(actuator, press):
        _ABIL["tapping"] = True
        t0 = time.perf_counter()
        sent = False
        try:
            sent = ability_tap.tap(actuator, press, watch=ABIL, point=_WA.Tap(x, y), check=_check,
                                   hold_ms=int(_WA.TOUCH_HOLD_S * 1000),
                                   gap_ms=int(_WA.TAP_GAP_S * 1000), on_sent=_sent)
        finally:
            _ABIL["tapping"] = False
            lat("ability_tap", t0=t0, t1=time.perf_counter(), press=press.i, sent=sent)
            if not sent:
                print(f"ABILITY press #{press.i}: NOT SENT ({press.why})", flush=True)

    _WA.WdaActuator._run = lambda self: ability_tap.worker_loop(
        self, _tap, lambda: (int(_WA.TOUCH_HOLD_S * 1000), int(_WA.TAP_GAP_S * 1000)))

    def _ledger_update(self, elixir, now=None, multiplier=1.0):
        _ABIL["ledger"] = self
        return ability_tap.ledger_update(update, self, elixir, cost=cost, tag=_ABILITY_TAG,
                                         now=now, multiplier=multiplier)

    _ELA.ElixirLedger.update = _ledger_update


_ability_setup()


def _ability_frame(in_game, bgr, cap, fi, elixir, hud):
    """One board for the button reader and the press tracker (perception thread)."""
    t = time.perf_counter() if cap is None else cap
    _ABIL["in_game"] = bool(in_game)
    t0 = time.perf_counter()
    r = ABIL_BTN.update(AB.read(bgr) if in_game and bgr is not None else None, t, fi)
    ABIL.frame(t, r.state, elixir if in_game else None)
    if not in_game:
        return
    _ABIL["ms"].append((time.perf_counter() - t0) * 1000.0)
    _ABIL["frames"] += 1
    _ABIL["states"][r.state] = _ABIL["states"].get(r.state, 0) + 1
    row = {"state": r.state, "ready": bool(ABIL_BTN.ready), "present": ABIL.present(t),
           "used": bool(ABIL.used), "open": ABIL.open()}
    hud["ability"] = row
    _ABIL["log"].write(json.dumps({"frame": fi, "t": round(t, 3), "match_s": round(float(_OWN_CLOCK["s"]), 2),
                                   "elixir": elixir, **r.row(), "why": r.why, **row}) + "\n")


def _ability_now(gs):
    """The ability's state for one decision: what the observation's own-ability
    fields say and what masks the policy's ability head (the same values)."""
    now = time.perf_counter()
    present = ABIL.present(now)
    blocked = ABIL.blocked()
    ready, why = ability_tap.readiness(button_ready=ABIL_BTN.ready_now(now), elixir=float(gs.my_elixir),
                                       cost=_ABIL["cost"], blocked=blocked)
    return {"present": bool(present), "ready": bool(ready), "why": why, "button": ABIL_BTN.state,
            "frame": ABIL_BTN.frame, "used": bool(ABIL.used)}


def _ability_decide(gs, d, ab):
    """After the card decision: issue the press the policy asked for, or say why not."""
    if not ab.get("want"):
        return
    _ABIL["want"] += 1
    why = ability_tap.hold_reason(card_this_decision=d.slot is not None,
                                  plays_in_flight=len(PLAYS.in_flight()) if PLAYS is not None else 0)
    if not why and not _ability_in_battle():
        why = "not in a battle"
    if why:
        ab["why"] = f"held: {why}"
        _ABIL["held"][why] = _ABIL["held"].get(why, 0) + 1
        return
    send = _ABIL["mode"] == "tap"
    tap = AB.tap_display()
    press = ABIL.issue(time.perf_counter(), match_s=round(float(_OWN_CLOCK["s"]), 1), frame=int(gs.frame_index),
                       p=ab["p"], cost=_ABIL["cost"] if send else None, tap=tap)
    ab.update(press=press.i, tap=list(tap), tap_frame_px=list(AB.tap_frame_px()))
    if send:
        ab["sent"] = ability_tap.send(_ABIL["actuator"], press, ABIL)
        if not ab["sent"]:
            ab["why"] = press.why
    print(f"ABILITY press #{press.i} at {press.match_s:.0f} s (frame {press.frame}, p={ab['p']:.3f}): "
          f"tap ({tap[0]}, {tap[1]}) display = frame px {AB.tap_frame_px()} -- "
          + ("queued for the actuator" if ab["sent"] else
             f"NOT SENT ({press.why})" if send else "NOT SENT (dry: this is where it would tap)"), flush=True)


import ability_tap  # noqa: E402  (no engine import; used by the functions above)

_orig_wa_init = _WA.WdaActuator.__init__


def _wa_init(self, client, *a, **k):
    tseq = getattr(client, "tap_sequence", None)
    if tseq is not None:
        def tap_sequence(points, *ta, **tk):
            t0 = time.perf_counter()
            err = None
            try:
                return tseq(points, *ta, **tk)
            except Exception as exc:
                # the actuator thread swallows this, so say it here
                print(f"[tap] FAILED: {type(exc).__name__}: {exc}", flush=True)
                lat("tap_err", t0=t0, err=f"{type(exc).__name__}: {exc}")
                err = str(exc) or type(exc).__name__
                raise
            finally:
                lat("tap", t0=t0, t1=time.perf_counter())
                # (an ability tap reports to its own tracker: ability_tap.tap)
                if PLAYS is not None and not _ABIL["tapping"]:
                    PLAYS.tap_done(time.perf_counter(), err)
        client.tap_sequence = tap_sequence
    _orig_wa_init(self, client, *a, **k)
    _ABIL["actuator"] = self


_WA.WdaActuator.__init__ = _wa_init
_orig_play = _WA.WdaActuator.play


def _wa_play(self, slot, tile_x, tile_y, **k):
    t = time.perf_counter()
    lat("enq", t=t, slot=int(slot))
    play = None
    if PLAYS is not None:
        card, _tile = _LAST_DECIDE.get(int(slot), (None, None))
        play = PLAYS.issue(t, slot, card=card, tile=(int(tile_x), int(tile_y)),
                           match_s=round(float(_OWN_CLOCK["s"]), 1))
    before = self.dropped
    out = _orig_play(self, slot, tile_x, tile_y, **k)
    if play is not None and self.dropped > before:
        PLAYS.dropped(play)
    return out


_WA.WdaActuator.play = _wa_play

# One row mapping for observations and taps (live_ppo/tile_map.py). offset1 leaves
# the engine's actuator untouched; tower replaces its engine->detector inverse so a
# policy's engine tile lands on the arena row the units were read from.
sys.path.insert(0, str(REPO / "live_ppo"))
from tile_map import det_to_engine_row, engine_to_det_row  # noqa: E402
if ours.tile_map == "tower":
    import live.actuator as _ACT  # noqa: E402

    def _engine_tile_centre_tower(tile_x, tile_y):
        return _ACT.tile_centre(tile_x, int(engine_to_det_row(int(tile_y), "tower")))

    def _engine_row_is_tappable_tower(tile_y):
        return 0 <= int(tile_y) < 34

    _ACT.engine_tile_centre = _WA.engine_tile_centre = _engine_tile_centre_tower
    _ACT.engine_row_is_tappable = _engine_row_is_tappable_tower
    engine_tile_centre = _engine_tile_centre_tower

from unit_tracker import ground_tile  # noqa: E402

UNIT_KIND = None
if ours.ground == "calib":
    def UNIT_KIND(cls):
        """troop / fly / bld for the ground-point offset (engine card table), None = unmapped."""
        cid = yolo_card_id(cls)
        if cid is None:
            return None
        from python_ai.models import perception_encoder as PE  # noqa: PLC0415
        a = PE.card_table().get(cid)
        if E.get_card_info(cid).get("is_building") or (a is not None and a.is_building):
            return "bld"
        return "fly" if a is not None and a.flying > 0 else "troop"

UNIT_TRACKER = None
if ours.track_units:
    from unit_tracker import UnitTracker  # noqa: E402

    def _is_building(cls):
        cid = yolo_card_id(cls)
        return cid is not None and bool(E.get_card_info(cid).get("is_building"))

    _TRAIN_TO_BASE = {v: k for k, v in BASE_TO_TRAIN.items()}
    _OWN_BASE_IDS = {_TRAIN_TO_BASE.get(c, c) for c in TRAIN_DECK}

    class _OwnDeck:
        """`base class in deck` for the tracker: our deck's units, by engine base card id."""
        def __contains__(self, base_cls):
            return yolo_card_id(base_cls) in _OWN_BASE_IDS

    _OWN_DECK = _OwnDeck()
    _POCKET = {"open": False}          # a Princess tower is down: buildings may cross the river
    UNIT_TRACKER = UnitTracker(
        hi=ours.det_conf, is_building=_is_building, kind=UNIT_KIND,
        pocket_open=lambda: _POCKET["open"],
        allowed=(lambda side: _OWN_DECK if side == 0 else None) if ours.own_deck_prior else None)

# A cloaked unit (Royal Ghost) is left out of the simulator's observation while it walks;
# the detector boxes it in both states. The gate passes a ghost box only while it would be
# revealed in the simulator (live_ppo/cloaked_gate.py). Off with --cloaked-rows off (the
# ghost has no row, every box is dropped) or --cloaked-always.
CLOAK_GATE = None
if ours.cloaked_rows == "on" and not ours.cloaked_always:
    import cloaked_gate  # noqa: E402
    CLOAK_GATE = cloaked_gate.RevealGate(deploy=cloaked_gate.deploy_s(E))
    print(f"cloaked gate: on (deploy {CLOAK_GATE.deploy:.1f} s, reveal {CLOAK_GATE.reveal:.1f} s)")

# --bush-goblins (live_ppo/bush_goblins.py): goblin tracks born at a Bush's pop are its Bush Goblins
BUSH_POPS = None
if BUSH_ON:
    import bush_goblins  # noqa: E402
    BUSH_POPS = bush_goblins.BushPops()

# iPhone Mirroring tap path, timed. The stock MirrorClient.tap_sequence activates the
# app before every placement, and its frontmost check reads NSWorkspace, which never
# updates in a process without a run loop, so every placement paid activate + a fixed
# 0.25 s. Here the frontmost check reads the window server's on-screen order (a few
# ms). Each posted event's duration is logged ("post") to find where the rest goes.
import capture.mirror as _CMR  # noqa: E402


def _front_owner(win):
    """The app that owns the front normal window (the window server's order)."""
    Q = win.Q
    for w in Q.CGWindowListCopyWindowInfo(
            Q.kCGWindowListOptionOnScreenOnly | Q.kCGWindowListExcludeDesktopElements,
            Q.kCGNullWindowID) or []:
        if w.get("kCGWindowLayer", 0) == 0:
            return w.get("kCGWindowOwnerName")
    return None


def _mirror_frontmost(win):
    return _front_owner(win) == win.owner


def _wait_front(win, seconds):
    end = time.perf_counter() + seconds
    while not _mirror_frontmost(win):
        if time.perf_counter() >= end:
            return False
        time.sleep(0.03)
    return True


def _bring_front(win):
    """Bring iPhone Mirroring to the front and see that it got there. Returns
    how ("activate" / "open") or None. The stock MirrorWindow.activate() is not
    used: it returns at once when NSWorkspace says the app is in front, and in
    this process (no run loop) NSWorkspace still says what was in front when
    the run started. Live 093139: another window was in front for 13 s, four
    placements were clicked into it and none reached the game."""
    try:
        from AppKit import NSWorkspace  # noqa: PLC0415
        for app in NSWorkspace.sharedWorkspace().runningApplications():
            if app.localizedName() == win.owner:
                app.activateWithOptions_(1 << 1)             # ignoring other apps
                break
    except Exception:                                        # noqa: BLE001
        pass
    if _wait_front(win, 0.3):
        return "activate"
    try:                    # LaunchServices: works from a background process
        subprocess.run(["/usr/bin/open", "-a", win.owner], timeout=3, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:                                        # noqa: BLE001
        pass
    return "open" if _wait_front(win, 1.0) else None


def _mirror_post(win, src, kind, at, posts):
    Q = win.Q
    t = time.perf_counter()
    Q.CGEventPost(Q.kCGHIDEventTap, Q.CGEventCreateMouseEvent(src, kind, at, Q.kCGMouseButtonLeft))
    posts.append(round((time.perf_counter() - t) * 1000, 2))


def _mirror_tap_sequence(self, points, hold_ms=60, gap_ms=80):
    win, Q = self.window, self.window.Q
    t0 = time.perf_counter()
    activated = not _mirror_frontmost(win)
    front = raised = None
    if activated:
        front = _front_owner(win)
        raised = _bring_front(win)
        if raised is None and not ours.no_front_guard:
            lat("post", mode=ours.tap_mode, activated=True, sent=False, front=front,
                prep_ms=round((time.perf_counter() - t0) * 1000, 1))
            raise RuntimeError(f"iPhone Mirroring is not the front window ({front or 'another app'} is) "
                               "and could not be brought forward: no tap sent")
        time.sleep(0.08)                                     # focus settles before input
    ox, oy, _, _ = win.refresh()
    t_ready = time.perf_counter()
    src = Q.CGEventSourceCreate(Q.kCGEventSourceStateHIDSystemState)
    posts = []
    pts = [(ox + x, oy + y) for x, y in points]
    if ours.tap_mode == "drag" and len(pts) == 2:
        (x0, y0), (x1, y1) = pts
        _mirror_post(win, src, Q.kCGEventMouseMoved, (x0, y0), posts)
        _mirror_post(win, src, Q.kCGEventLeftMouseDown, (x0, y0), posts)
        time.sleep(hold_ms / 2000)
        steps = 6
        for i in range(1, steps + 1):
            f = i / steps
            _mirror_post(win, src, Q.kCGEventLeftMouseDragged, (x0 + (x1 - x0) * f, y0 + (y1 - y0) * f), posts)
            time.sleep(ours.drag_ms / 1000 / steps)
        _mirror_post(win, src, Q.kCGEventLeftMouseUp, (x1, y1), posts)
    else:
        for i, (x, y) in enumerate(pts):
            if i:
                time.sleep(gap_ms / 1000)
            _mirror_post(win, src, Q.kCGEventMouseMoved, (x, y), posts)
            time.sleep(0.02)
            _mirror_post(win, src, Q.kCGEventLeftMouseDown, (x, y), posts)
            time.sleep(hold_ms / 2000)
            _mirror_post(win, src, Q.kCGEventLeftMouseDragged, (x + 1, y), posts)
            time.sleep(hold_ms / 2000)
            _mirror_post(win, src, Q.kCGEventLeftMouseUp, (x, y), posts)
    lat("post", mode=ours.tap_mode, activated=activated, front=front, raised=raised,
        prep_ms=round((t_ready - t0) * 1000, 1), posts_ms=posts,
        total_ms=round((time.perf_counter() - t0) * 1000, 1))


_CMR.MirrorClient.tap_sequence = _mirror_tap_sequence

if ours.ledger_tol is not None:
    from live import elixir_ledger as _EL  # noqa: E402
    _tol = ours.ledger_tol
    _orig_post = _EL.ElixirLedger.__post_init__

    def _post(self):
        self.tolerance = _tol
        _orig_post(self)
    _EL.ElixirLedger.__post_init__ = _post

if PLAYS is not None:
    # The policy's elixir is the bar minus what is promised to taps the bar has
    # not caught up with. The ledger keeps a play subtracted until a bar drop
    # matches its cost, else for 4 s; the bar drains over two frames, so 14 of
    # the 36 plays that landed in live 093139 never matched and were charged
    # twice for 4 s, and the 9 that did not land were charged for 4 s too.
    from live import elixir_ledger as _ELW  # noqa: E402
    _orig_record_play = _ELW.ElixirLedger.record_play

    def _record_play(self, cost, now=None, tag=None):
        PLAYS.cost(cost)
        return _orig_record_play(self, cost, now=now, tag=tag)

    _ELW.ElixirLedger.record_play = _record_play
    if ABIL is not None and _ABIL["mode"] == "tap":
        # The simulator debits the ability's cost on the tick it is activated
        # (GameManager::activateChampionAbility); live it is owed from the press
        # until the bar shows it, like a placement's cost.
        _ELW.ElixirLedger.unconfirmed_cost = property(lambda self: PLAYS.owed() + ABIL.owed())
    else:
        _ELW.ElixirLedger.unconfirmed_cost = property(lambda self: PLAYS.owed())

if ours.hand == "screen":
    # The actuator taps the physical slot, so the hand the policy picks from must be
    # the hand on screen. The FIFO tracker matched the screen on 27/102 frames in the
    # dry replay while the screen read was 120/120, i.e. most taps played a card the
    # net did not choose. Keep the tracker only as a fallback for an unreadable slot.
    from contracts import UNKNOWN_CARD_SIM_ID as _UNK  # noqa: E402

    class ScreenHand(L.HandTracker):
        _last_obs = ()

        def update(self, observed_hand, new_plays):
            prev, self._last_obs = self._last_obs, tuple(observed_hand)
            super().update(observed_hand, new_plays)
            if OPP is not None and OPP.state is not None:
                # --opp-state: our own plays, from the screen hand (a card that left
                # a fully read hand was played) -> our Evolution counter
                if len(prev) == 4 and len(self._last_obs) == 4 and _UNK not in prev + self._last_obs:
                    for c in set(prev) - set(self._last_obs):
                        OPP.state.own_play(BASE_TO_TRAIN.get(c, c), _OWN_CLOCK["s"])

        def as_tuple(self):
            obs = self._last_obs
            if len(obs) != 4:
                return super().as_tuple()
            fifo = tuple(self.hand) if self.seeded and len(self.hand) == 4 else (_UNK,) * 4
            return tuple(o if o != _UNK else f for o, f in zip(obs, fifo))

    # Before the tracker seeds, mvp_loop already uses the screen read as is.
    L.HandTracker = ScreenHand

# Alternate hand-card art. The stock pool has one template per card, cut from plain
# art, so a charged Evo Skeletons (purple evo art) or the Hero Musketeer art scores
# ~0.46 against everything and Evo Skeletons was read as Musketeer. These extra
# templates (live_ppo/templates/hand_alts, cut from iPhone Mirroring frames) score
# a card as the best of its templates.
import live.deck_hand as _DH  # noqa: E402

_ALTS = REPO / "live_ppo" / "templates" / "hand_alts"


class AltDeckHandDetector(_DH.DeckHandDetector):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        rows, group = list(self._templates), list(range(len(self._names)))
        alts = json.loads((_ALTS / "alts.json").read_text()) if (_ALTS / "alts.json").exists() else {}
        for fn, meta in alts.items():
            if meta["name"] in self._names:
                img = cv2.imread(str(_ALTS / fn), cv2.IMREAD_COLOR)
                v = _DH.normalise_icon(img).ravel()
                rows.append(v / (np.linalg.norm(v) + 1e-6))
                group.append(self._names.index(meta["name"]))
        self._all = np.asarray(rows, dtype=np.float32)
        self._group = np.asarray(group)
        print(f"hand templates: {len(self._all) - len(self._names)} alternate art "
              f"({', '.join(m['name'] for m in alts.values())})")

    def _score(self, crop):
        s = self._all @ self._vec(crop)
        out = np.full(len(self._names), -1.0, dtype=np.float32)
        np.maximum.at(out, self._group, s)
        return out

    # An unaffordable card is drawn in greyscale and washed out: its best template
    # correlation sits at 0.40-0.47, around MIN_SCORE (0.45), while the runner-up
    # is far below (0.44 vs 0.06 for a dimmed Cannon). A present slot (badge or
    # greyscale) whose best is clear of the rest by GREY_MARGIN is that card,
    # not blank. Tuned on the dev videos (live_ppo/eval/replay_hand.py).
    GREY_MIN_SCORE = 0.35
    GREY_MARGIN = 0.20
    LAST_INFO = None          # the last run()'s per-window details (see _hungarian)

    def _hungarian(self, windows):
        from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

        reads = [_DH.SlotRead(_DH.Cards.BLANK, 0.0, 0.0, w.badge) for w in windows]
        # per window, for the hand-cycle filter (state_filter.HandCycle) and offline
        # evals: [assigned card's name, its score, runner-up, present, colour std,
        # best-matching card, its score]. A window left out of the assignment (no
        # badge, not greyscale) is still matched, unassigned: one that reads blank
        # while it still shows the card it held is not an empty slot.
        self.last_info = [[None, 0.0, 0.0, bool(w.present), round(float(w.colour), 2), None, 0.0]
                          for w in windows]
        AltDeckHandDetector.LAST_INFO = self.last_info
        for i, w in enumerate(windows):
            if i and not w.present:
                sc = self._best_over_lift(w.tall, w.base, w.height)[0]
                j = int(np.argmax(sc))
                self.last_info[i][5:] = [self._names[j], round(float(sc[j]), 3)]
        live = [i for i, w in enumerate(windows) if i == 0 or w.present]
        if not live:
            return reads
        scores = np.stack([self._best_over_lift(windows[i].tall, windows[i].base, windows[i].height)[0]
                           for i in live])
        rows, cols = linear_sum_assignment(-scores)
        for r, c in zip(rows, cols):
            row = scores[r]
            best = float(row[c])
            second = float(np.max(np.delete(row, c))) if len(row) > 1 else 0.0
            ok = best >= self.min_score or (best >= self.GREY_MIN_SCORE and best - second >= self.GREY_MARGIN)
            card = self._cards[self._names[c]] if ok else _DH.Cards.BLANK
            reads[live[r]] = _DH.SlotRead(card, best, second, windows[live[r]].badge)
            self.last_info[live[r]][:3] = [self._names[c], round(best, 3), round(second, 3)]
            j = int(np.argmax(row))
            self.last_info[live[r]][5:] = [self._names[j], round(float(row[j]), 3)]
        return reads

    def _best_over_lift(self, tall, base, height):
        direct = self._score(tall[base:base + height])
        if float(direct.max()) >= _DH.SEARCH_TRIGGER:
            return direct, 0
        best, best_dy = direct, 0
        for dy in range(_DH.LIFT_SEARCH[0], _DH.LIFT_SEARCH[1]):
            if dy == 0 or base + dy < 0 or base + dy + height > tall.shape[0]:
                continue
            scores = self._score(tall[base + dy:base + dy + height])
            if float(scores.max()) > float(best.max()):
                best, best_dy = scores, dy
        return best, best_dy


_DH.DeckHandDetector = AltDeckHandDetector

class CachedYolo:
    """Replays only (CR_YOLO_CACHE=a .yolo.jsonl made by the same side-car on the same
    frames): dets by frame index, so offline evals are deterministic and fast. The
    live loop never sets it."""

    def __init__(self, path):
        self.by_frame = {}
        for line in open(path):
            r = json.loads(line)
            self.by_frame[int(r["frame"])] = r["dets"]
        self.frame = None
        self.ms = []

    def __call__(self, bgr, jpeg=None):
        return self.by_frame.get(self.frame, [])


_NEED_YOLO = ours.units == "yolo" or ours.compare_yolo or ours.opp_plays
YOLO = (CachedYolo(os.environ["CR_YOLO_CACHE"]) if _NEED_YOLO and os.environ.get("CR_YOLO_CACHE")
        else Yolo(min(ours.det_conf, UNIT_TRACKER.lo) if UNIT_TRACKER is not None else ours.det_conf,
                  ours.det_weights) if _NEED_YOLO else None)

# ----------------------------------------------------------------- HUD readers
sys.path.insert(0, str(REPO / "live_ppo"))
import state_filter  # noqa: E402
from live.unit_to_card import hand_card_id_for  # noqa: E402

if ours.elixir_bar:
    # Before the ledger sees it: mvp_loop's perceive hands the detector's digit
    # to the ElixirLedger first, then to build_game_state.
    _orig_detector_run = L.Detector.run

    def _detector_run(self, image):
        st = _orig_detector_run(self, image)
        if st is not None and st.screen.name == "in_game":
            d = float(st.numbers.elixir.number)
            e, _bar, fixed = state_filter.fix_elixir(d, state_filter.to_ref(image, rgb=True))
            if fixed:
                st.numbers = replace(st.numbers, elixir=replace(st.numbers.elixir, number=e))
        return st

    L.Detector.run = _detector_run
ELIXIR_EST = None
if ours.elixir_est != "off":
    # The policy was trained on the fractional bar; the reading is its floor.
    # One estimator class here and in training/sim2real.py (--sim2real-elixir-est).
    import elixir_estimate  # noqa: E402
    ELIXIR_EST = elixir_estimate.ElixirEstimator()
HAND_CYCLE = (state_filter.HandCycle([getattr(c, "name", c) for c in L.DECK], hz=ours.hud_hz)
              if ours.hand_cycle else None)
HUD_TOWERS = HUD_OWN = HUD_KING = HUD_CLOCK = OPP = _OWN_KING = None
if ours.hud_towers:
    import hud_live  # noqa: E402
    HUD_TOWERS = hud_live.OppTowers(hz=ours.hud_hz)
    if not ours.no_own_towers:
        HUD_OWN = hud_live.OwnTowers(hz=ours.hud_hz)
        _OWN_KING = hud_live.OwnKingDips()     # the deploy stopwatch over our King's badge
        if not ours.no_own_king_num:
            HUD_KING = hud_live.OwnKing(hz=ours.hud_hz)
if ours.hud_clock:
    import hud_live  # noqa: E402,F811
    HUD_CLOCK = hud_live.MatchClock()
_OPP_RULES = {"evidence": not ours.no_opp_evidence,
              "spell_evidence": False if ours.no_opp_spell_evidence else None}
_OPP_RULES.update({r: True for r in _OPP_RULE_ON})
print(f"opp-play rules (--opp-rules): {','.join(_OPP_RULE_ON) or 'none'}"
      f"{'' if ours.opp_plays else ' (unused: --opp-plays is off)'}; body rows (--body-rows): {ours.body_rows}")
if ours.opp_plays:
    import opp_plays  # noqa: E402
    OPP = opp_plays.OppModel(TRAIN_DECK, spells=ours.opp_spells, state=ours.opp_state,
                             watch=not ours.no_opp_watch, **_OPP_RULES)
_OWN_CLOCK = {"s": 0.0}     # match seconds of the last frame (--opp-state own plays)
# Observation v2 tail: fed by bgs() on every in-game frame, read by EvoEncoder.encode.
TAIL = None
_TAIL_CLK = {"s": 0.0}      # the tail's clock (OPP's) at the last in-game frame
_TAIL_IDS: dict = {}


def _tail_start(why):
    global TAIL
    if TAIL is None:
        import obs_tail  # noqa: PLC0415
        names = SET()["names"]
        TAIL = obs_tail.LiveTail(E, tile_map=ours.tile_map, names=names)
        print(f"obs v2 tail: ON ({why}); opponent names: {names}"
              + ("" if OPP is not None and ours.opp_spells else
                 "  WARNING: without --opp-plays --opp-spells its opponent part stays empty"))


def _tail_card_id(name):
    """Engine card id (training deck) of a hand-reader or engine card name, or None."""
    if not name:
        return None
    if not _TAIL_IDS:
        for c in TRAIN_DECK:
            for n in (E.get_card_info(int(c))["name"],
                      E.get_card_info(int(TAIL.bk.face.get(int(c), int(c))))["name"]):
                _TAIL_IDS["".join(ch for ch in n.lower() if ch.isalnum())] = int(c)
    cid = hand_card_id_for(name)
    if cid is not None and cid >= 0:
        return int(BASE_TO_TRAIN.get(cid, cid))
    return _TAIL_IDS.get("".join(ch for ch in str(name).lower() if ch.isalnum()))


if ours.obs_v2 == "on":
    _tail_start("--obs-v2 on")
# OPP's clock: the HUD countdown reads whole seconds and MatchClock re-anchors on
# every read, so its elapsed() is a 1 s staircase. OppModel dates plays and runs
# its sub-second windows (GROUP_S, VOTE_S, ...) on its clock, and was tuned on a
# continuous one, so OPP gets the staircase de-quantised: anchored at the frame
# the countdown value CHANGES (true elapsed = the read there), advanced by wall
# time, kept in [read, read + 1). Same convention as seconds_elapsed (which is
# unchanged; only OPP sees this).
_OPP_CLK = {"left": None, "el": None, "t": None}
if ours.save_raw and not ours.record:
    (RUN / "raw").mkdir(exist_ok=True)
REC = None
if ours.record:
    _h = getattr(YOLO, "hello", None) or {}
    REC = Recorder(RUN, ours.record_every, meta={
        "argv": sys.argv[1:], "act": ACT,
        "detector": {"weights": _h.get("weights"), "sidecar_conf": _h.get("conf"),
                     "det_conf": ours.det_conf, "device": _h.get("device"),
                     "cache": os.environ.get("CR_YOLO_CACHE")},
        "replay_of": str(ours.loop_frames) if ours.loop_frames else None})
HUDLOG: OrderedDict = OrderedDict()
_match = {"id": 0}
_SPENT = {"match": 0, "base": 0.0}   # ledger total at the first frame of the current match


def _new_match():
    global OPP
    _match["id"] += 1
    if HUD_TOWERS is not None:
        HUD_TOWERS.reset()
    if HUD_OWN is not None:
        HUD_OWN.reset()
        _OWN_KING.reset()
    if HUD_KING is not None:
        HUD_KING.reset()
    if HUD_CLOCK is not None:
        HUD_CLOCK.reset()
    if ELIXIR_EST is not None:
        ELIXIR_EST.reset()
    _latch.clear()              # a tower latched in the last match is alive again
    if POCKET_G is not None:
        POCKET_G.reset()
    _KING.reset()
    if HAND_CYCLE is not None:
        HAND_CYCLE.reset()
    if UNIT_TRACKER is not None:
        UNIT_TRACKER.reset()
    if CLOAK_GATE is not None:
        CLOAK_GATE.reset()
    if BUSH_POPS is not None:
        BUSH_POPS.reset()
    if PROXY_GATE is not None:
        PROXY_GATE.reset()
    if OPP is not None:
        OPP = opp_plays.OppModel(TRAIN_DECK, spells=ours.opp_spells, state=ours.opp_state,
                                 watch=not ours.no_opp_watch, **_OPP_RULES)
    _OPP_CLK.update(left=None, el=None, t=None)
    if TAIL is not None:
        TAIL.reset()
    if ABIL is not None:
        ABIL_BTN.reset()
        ABIL.reset()


def _opp_seconds(match_s, wall_s):
    """match_s with the HUD clock's 1 s steps smoothed out (see _OPP_CLK)."""
    if HUD_CLOCK is None or HUD_CLOCK.elapsed_at_read is None:
        return match_s
    if HUD_CLOCK.read_ok and HUD_CLOCK.last_left != _OPP_CLK["left"]:
        _OPP_CLK.update(left=HUD_CLOCK.last_left, el=HUD_CLOCK.elapsed_at_read, t=wall_s)
    if _OPP_CLK["t"] is None:
        return match_s
    return min(max(_OPP_CLK["el"] + (wall_s - _OPP_CLK["t"]), match_s), match_s + 0.99)

# ----------------------------------------------------------------- state hooks
CACHE: OrderedDict = OrderedDict()
_clock = {"start": None, "last_in": None}
_latch = {}
_KING = state_filter.KingFloor()
_orig_bgs = L.build_game_state


def _seconds(state, wall_ms):
    in_game = state.screen.name == "in_game"
    t = wall_ms / 1000.0
    if in_game:
        if _clock["start"] is None or (_clock["last_in"] is not None and t - _clock["last_in"] > 20):
            _clock["start"] = t
            _new_match()
        _clock["last_in"] = t
        return t - _clock["start"]
    return 0.0


def bgs(state, frame, small, **kw):
    secs = _seconds(state, kw.get("wall_time_ms", 0.0))
    in_game = state.screen.name == "in_game"
    fi = kw.get("frame_index", 0)
    hud = {}
    bgr = np.ascontiguousarray(frame[:, :, ::-1]) if in_game else None
    jpeg = None
    if REC is not None:
        # before any reader runs, so a frame that makes one raise is on disk too
        jpeg = REC.frame(fi, kw.get("wall_time_ms", 0.0), in_game, state.screen.name,
                         _match["id"], secs, frame, bgr)
    elif ours.save_raw and in_game and fi % ours.save_raw == 0:
        cv2.imwrite(str(RUN / "raw" / f"f{fi:06d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ours.fill_clock:
        kw["seconds_elapsed"] = secs
    if HUD_CLOCK is not None and in_game:
        el = HUD_CLOCK.update(bgr, kw.get("wall_time_ms", 0.0) / 1000.0)
        if HUD_CLOCK.restart:
            # the countdown restarted with no menu in between (recorded videos):
            # a new match, as if the screen had left the game
            _clock["start"] = kw.get("wall_time_ms", 0.0) / 1000.0
            rr = HUD_CLOCK.restart_read
            _new_match()
            HUD_CLOCK._accept(*rr)          # already read twice: no warm-up
            el = HUD_CLOCK.elapsed(rr[1])
        hud["clock"] = {"left": HUD_CLOCK.last_left, "read_ok": HUD_CLOCK.read_ok,
                        "elapsed": None if el is None else round(el, 2),
                        "ot": HUD_CLOCK.overtime, "badge": HUD_CLOCK.mult_badge}
        if el is not None:
            kw["seconds_elapsed"] = el
    match_s = kw.get("seconds_elapsed", secs)
    _OWN_CLOCK["s"] = float(match_s)
    if not ours.keep_spent and "my_elixir_spent" in kw:
        # mvp_loop makes one ElixirLedger per launch, so its total runs on through
        # every match played in that launch; the policy's feature is per match.
        if _SPENT["match"] != _match["id"]:
            _SPENT.update(match=_match["id"], base=float(kw["my_elixir_spent"]))
        kw["my_elixir_spent"] = max(0.0, float(kw["my_elixir_spent"]) - _SPENT["base"])
    gs, rep = _orig_bgs(state, frame, small, **kw)
    elixir_est = None
    if ELIXIR_EST is not None and in_game:
        # Every frame, on the capture clock. The reading stays the integer for
        # the ledger, PlayWatch and the logs; mvp_loop takes PlayWatch.owed()
        # off this value afterwards, as it does off the integer.
        te = kw.get("wall_time_ms", 0.0) / 1000.0
        if (PLAYS is not None and any(p.state == "open" or (p.state == "landed" and not p.paid)
                                      for p in PLAYS.plays[-6:])) \
                or (ABIL is not None and ABIL.owed() > 0):
            ELIXIR_EST.own_play(te)       # whole-elixir changes of the reading are ours for now
        fill = None
        if ours.elixir_est == "bar":
            fill = elixir_estimate.bar_fill(state_filter.to_ref(bgr, rgb=False))[0]
        elixir_est = ELIXIR_EST.update(te, digit=float(state.numbers.elixir.number), fill=fill,
                                       mult=elixir_estimate.multiplier(float(match_s)))
        if elixir_est is not None:
            gs = replace(gs, my_elixir=float(elixir_est))
            hud["elixir_est"] = round(float(elixir_est), 3)
    if HAND_CYCLE is not None and in_game:
        # per-window match details (AltDeckHandDetector) for the soft-blank rules
        names, nxt = HAND_CYCLE.update([c.name for c in state.cards[1:5]], state.cards[0].name,
                                       AltDeckHandDetector.LAST_INFO)
        hud["hand_cycle"] = {"hand": names, "next": nxt}
        gs = replace(gs, my_hand=tuple(hand_card_id_for(n) for n in names))
        # the cycle fills a slot with its incoming card at once, but that card
        # takes ~0.7 s to arrive: until the slot shows it, it cannot be tapped
        now_s = kw.get("wall_time_ms", 0.0) / 1000.0
        for s, c in enumerate(state.cards[1:5]):
            if c.name != state_filter.BLANK:
                _BLANK_SINCE[s] = None
            elif _BLANK_SINCE[s] is None or now_s < _BLANK_SINCE[s]:
                _BLANK_SINCE[s] = now_s
        hud["empty_slots"] = [b is not None and now_s - b < EMPTY_MAX_S for b in _BLANK_SINCE]
    for tw in (HUD_TOWERS, HUD_OWN):            # opponent, then our own Princesses
        if tw is None or not in_game:
            continue
        filt = tw.update(bgr)
        upd = {}
        for key, f in filt.items():
            _bar, num = tw.raw[key]
            hud.setdefault("towers", {})[key] = {
                "num": None if num is None else num.value, "hp": f.hp, "max": f.max_hp,
                "f": round(f.frac, 4), "m": f.measured, "dead": f.destroyed,
                "stock": round(float(getattr(gs, key).hp_fraction), 3)}
            if f.from_bar:                     # the value is the bar's, the numeral is covered
                hud["towers"][key]["bar"] = True
            # until the max latches, keep the stock reading
            if f.measured or f.destroyed:
                upd[key] = TowerObservation(0.0 if f.destroyed else float(f.frac),
                                            hp_measured=f.measured, destroyed=f.destroyed)
        if upd:
            gs = replace(gs, **upd)
    if POCKET_G is not None and in_game:
        _pt = kw.get("wall_time_ms", 0.0) / 1000.0
        for _k in pocket_live.LANES:
            _e = (hud.get("towers") or {}).get(_k)
            if _e is not None:
                POCKET_G.see_tower(_k, bool(_e["dead"]), _e["num"], _pt)
        hud["pocket"] = POCKET_G.snapshot()
    ydets = ydets_all = None
    if YOLO is not None and in_game:
        if hasattr(YOLO, "by_frame"):
            YOLO.frame = fi        # CR_YOLO_CACHE (replays only): boxes by frame index
        ydets_all = YOLO(bgr, jpeg)
        if REC is not None:
            REC.dets(fi, kw.get("wall_time_ms", 0.0), ydets_all)
        # boxes that count on their own; the ones below det_conf are for the tracker only
        ydets = [d for d in ydets_all if d["conf"] >= ours.det_conf]
        if OPP is not None:
            new = OPP.update(ydets, _opp_seconds(match_s, kw.get("wall_time_ms", 0.0) / 1000.0),
                             frame=bgr)
            if TAIL is not None:
                TAIL.see_opp(OPP, new)
            hud["opp_new"] = [p.name for p in new]
            hud["opp"] = {"n": len(OPP.det.plays), "spent": OPP.spent,
                          "elixir": round(OPP.elixir.value, 2), "neg": len(OPP.elixir.negative_events)}
    if OPP is not None:
        gs = replace(gs, opp_elixir_spent=float(OPP.spent))
        # ydets is None on a frame the screen classifier did not call in_game
        # (a transient misread): keep the stock units rather than crash the thread.
        if ours.units == "yolo" and ydets is not None:
            from contracts import UnitObservation  # noqa: PLC0415
            units = []
            udets = ydets
            _bush_marked = ()
            if PROXY_GATE is not None:          # --body-rows on-confirmed: unconfirmed proxy boxes out
                _pdrop = PROXY_GATE.drop(ydets_all, kw.get("wall_time_ms", 0.0) / 1000.0)
                if _pdrop:
                    udets = [d for d in ydets if id(d) not in _pdrop]
                    ydets_all = [d for d in ydets_all if id(d) not in _pdrop]
            if UNIT_TRACKER is not None:
                t0 = time.perf_counter()
                _POCKET["open"] = any(getattr(gs, k).destroyed for k in (
                    "own_princess_left", "own_princess_right", "opp_princess_left", "opp_princess_right"))
                udets = UNIT_TRACKER.update(
                    [d for d in ydets_all if yolo_card_id(d["cls"]) is not None],
                    t=kw.get("wall_time_ms", 0.0) / 1000.0)
                hud["track_ms"] = round((time.perf_counter() - t0) * 1000, 3)
                if BUSH_POPS is not None:       # --bush-goblins: tracks marked as a Bush's goblins
                    _bush_marked = BUSH_POPS.update(udets, kw.get("wall_time_ms", 0.0) / 1000.0)
                    if BUSH_POPS.log:
                        hud["bush_pop"] = BUSH_POPS.log
            cloak_log = []
            for d in udets:
                cid = yolo_card_id(d["cls"])
                if cid is None:
                    continue
                if CLOAK_GATE is not None and cid in cloaked_gate.CLOAKED_CARDS:
                    shown = CLOAK_GATE.keep(d, kw.get("wall_time_ms", 0.0) / 1000.0)
                    cloak_log.append(CLOAK_GATE.log[-1])
                    if not shown:
                        continue
                # the tracker already applied the ground offset (kind=UNIT_KIND)
                tx, ty = (ground_tile(d, UNIT_KIND(d["cls"])) if UNIT_KIND is not None and UNIT_TRACKER is None
                          else d["tile"])
                units.append(UnitObservation(
                    card_sim_id=cid, team=int(d["side"]),
                    unit_name=(bush_goblins.NAME if _bush_marked and d.get("track") in _bush_marked
                               and d["cls"] in bush_goblins.GOBLIN_CLASSES else d["cls"]),
                    tile_x=int(tx),
                    tile_y=int(det_to_engine_row(int(ty), ours.tile_map)),
                    hp_fraction=1.0, hp_measured=False, confidence=float(d["conf"]),
                    team_from_badge=None))
            if cloak_log:
                hud["cloaked"] = cloak_log
            gs = replace(gs, units=tuple(units))
    # the King "no fill" floor on video arenas -> full (state_filter.KingFloor)
    if in_game:
        kupd = {}
        for key in ("own_king", "opp_king"):
            if key == "opp_king" and HUD_TOWERS is not None and HUD_TOWERS.king is not None \
                    and HUD_TOWERS.king.f.measured:
                # the value is OppKing's numeral (hud_live.OPP_KING_READER), not king_hp.py's
                # bar: the floor is for that bar's "no fill" misread and would turn a King really
                # under 2.5 % (or one that stays there over HOLD_S) into full
                continue
            t = getattr(gs, key)
            f = _KING.update(key, float(t.hp_fraction), bool(t.hp_measured), float(match_s))
            if f is not None:
                kupd[key] = TowerObservation(f, hp_measured=True)
        if kupd:
            gs = replace(gs, **kupd)
            hud["king_floor_full"] = sorted(kupd)
        if _OWN_KING is not None and gs.own_king.hp_measured:
            t = gs.own_king
            f = _OWN_KING.update(float(t.hp_fraction), float(match_s))
            if f != float(t.hp_fraction):
                gs = replace(gs, own_king=TowerObservation(f, hp_measured=True))
                hud["own_king_dip"] = round(float(t.hp_fraction), 3)
        if HUD_KING is not None:
            # our King's numeral, once its bar is drawn (a Princess down, or the
            # King hit); before that the stock reading above stands
            kf = HUD_KING.update(bgr)
            _kbar, knum = HUD_KING.raw
            hud.setdefault("towers", {})["own_king"] = {
                "num": None if knum is None else knum.value, "hp": kf.hp, "max": kf.max_hp,
                "f": round(kf.frac, 4), "m": kf.measured,
                "stock": round(float(gs.own_king.hp_fraction), 3)}
            if kf.measured:
                gs = replace(gs, own_king=TowerObservation(float(kf.frac), hp_measured=True))
    # in_game only: a menu / loading frame's "bars" once read 0.23 then
    # unreadable and latched a full-HP tower destroyed for the whole match
    if ours.tower_latch and in_game:
        upd = {}
        for key in ("own_princess_left", "own_princess_right",
                    "opp_princess_left", "opp_princess_right"):
            if (HUD_TOWERS if key.startswith("opp") else HUD_OWN) is not None:
                continue                       # the HUD filter latches these itself
            t = getattr(gs, key)
            st = _latch.setdefault(key, {"low": False, "dead": False})
            if t.hp_measured and t.hp_fraction < 0.25:
                st["low"] = True
            if t.hp_measured and t.hp_fraction > 0.5:
                st["low"] = False
            if not t.hp_measured and st["low"]:
                st["dead"] = True
            if st["dead"]:
                upd[key] = TowerObservation(0.0, hp_measured=False, destroyed=True)
        if upd:
            gs = replace(gs, **upd)
    if TAIL is not None and in_game:
        # own plays (a card left the hand; tile from the placement issued for it),
        # our next card, and the clock the tail is built on: OPP's, so both sides'
        # plays are dated alike
        ts = float(_opp_seconds(match_s, kw.get("wall_time_ms", 0.0) / 1000.0))
        TAIL.see_own(PLAYS.plays if PLAYS is not None else (), _tail_card_id, ts, HAND_CYCLE)
        nxt_name = (hud.get("hand_cycle") or {}).get("next") if HAND_CYCLE is not None else state.cards[0].name
        TAIL.set_next(_tail_card_id(nxt_name))
        _TAIL_CLK["s"] = ts
        hud["tail_s"] = round(ts, 3)
    CACHE[kw.get("frame_index", 0)] = (state, frame, ydets, secs)
    while len(CACHE) > 40:
        CACHE.popitem(last=False)
    HUDLOG[fi] = hud
    while len(HUDLOG) > 400:
        HUDLOG.popitem(last=False)
    if ABIL is not None:
        t0 = _W.get("t0")
        _ability_frame(in_game, bgr, None if t0 is None else t0 + kw.get("wall_time_ms", 0.0) / 1000.0,
                       fi, float(state.numbers.elixir.number) if in_game else None, hud)
    if in_game:
        t0 = _W.get("t0")
        cap = None if t0 is None else t0 + kw.get("wall_time_ms", 0.0) / 1000.0
        if PLAYS is not None:
            PLAYS.frame(time.perf_counter() if cap is None else cap,
                        [c.name for c in state.cards[1:5]], float(state.numbers.elixir.number))
        lat("board", fi=fi, cap=cap, proc=time.perf_counter(),
            hand=[c.name for c in state.cards[1:5]],
            elixir=float(state.numbers.elixir.number),
            **({} if elixir_est is None else {"elixir_est": round(float(elixir_est), 3)}))
    return gs, rep


L.build_game_state = bgs


if ours.loop_frames is not None:
    import capture.mirror as _CM  # noqa: E402
    from capture.source import Frame  # noqa: E402

    class _LoopSource:
        def __init__(self, *_a, **_k):
            fs = sorted(ours.loop_frames.glob("f*.jpg"))[ours.loop_from:]
            self._ts = None
            if ours.loop_timestamps:
                rec = {r["file"]: r for r in map(json.loads, open(ours.loop_frames / "frames.jsonl"))
                       if r.get("file")}
                self._files = [f for f in fs if f.name in rec]
                self._nums = [rec[f.name]["frame"] for f in self._files]
                self._ts = [rec[f.name]["wall_time_ms"] - rec[self._files[0].name]["wall_time_ms"]
                            for f in self._files]
                self._k, self._served = -1, -1e9
                fs = self._files[:1]                # the rest are read as they come due
            self._imgs = [cv2.imread(str(f)) for f in fs]
            self._t0 = time.perf_counter()
            self._i = 0
            self.size = self._imgs[0].shape[1::-1]
            self.fps = 16.0
            self.frame_count = -1

        def read_new(self, timeout_s=1.0):
            if (ours.inject_stall_at is not None
                    and time.perf_counter() - self._t0 > ours.inject_stall_at):
                print(f"INJECTED STALL: frame source blocked at {ours.inject_stall_at:.0f} s", flush=True)
                threading.Event().wait()            # never returns
            if self._ts is not None:
                # each recorded frame at its recorded time; a frame is skipped when
                # perception is late for it, as a live capture would skip it
                import bisect
                while True:
                    el = (time.perf_counter() - self._t0) * 1000
                    k = bisect.bisect_right(self._ts, el) - 1
                    if k != self._k or el - self._served >= 250:
                        break
                    time.sleep(0.004)
                if k != self._k:
                    self._img = cv2.imread(str(self._files[k]))
                self._k, self._served = k, el
                return Frame(index=self._nums[k], wall_time_ms=el, image=self._img)
            if ours.loop_realtime:
                # one board per recorded frame at the recording's 4 fps; the frame
                # index is the recorded frame number, so logs line up with labels
                while True:
                    el = time.perf_counter() - self._t0
                    k = int(el * 4) % len(self._imgs)
                    if k != self._i - 1 or self._i == 0:
                        break
                    time.sleep(0.01)
                self._i = k + 1
                return Frame(index=ours.loop_from + k, wall_time_ms=el * 1000, image=self._imgs[k])
            time.sleep(0.015)                       # ~ the window-capture wait
            img = self._imgs[(self._i // 4) % len(self._imgs)]   # 4 fps recording
            f = Frame(index=self._i, wall_time_ms=(time.perf_counter() - self._t0) * 1000, image=img)
            self._i += 1
            return f

        read = read_new

        def close(self):
            pass

    class _NoWindow:
        bounds = (0, 0, 0, 0)

        def __init__(self, *a, **k):
            pass

    class _NoClient:
        def __init__(self, *a, **k):
            pass

        def tap_sequence(self, *a, **k):
            raise RuntimeError("loop-frames soak must never tap")

    _CM.MirrorSource, _CM.MirrorWindow, _CM.MirrorClient = _LoopSource, _NoWindow, _NoClient

# ----------------------------------------------------------------- stall guard
import threading  # noqa: E402
from live.action_gate import GateDecision  # noqa: E402
import traceback  # noqa: E402

AGE = {"ms": 0.0}
AGES = open(RUN / "board_age.csv", "a", buffering=1)
_W = {"w": None}


class LoudWorker(L.PerceptionWorker):
    """The stock worker counts exceptions in `errors` and never prints them, so a
    capture that raises on every frame looks exactly like a hang. Log each new
    error type with its traceback, and every 50th repeat."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        _W["w"] = self
        _W["t0"] = getattr(self._source, "_t0", None)   # capture clock origin
        self._seen = {}
        orig = self._adapt

        def adapt(frame):
            try:
                return orig(frame)
            except Exception as exc:
                self._report(exc)
                raise
        self._adapt = adapt
        orig_read = self._source.read_new

        def read_new(timeout_s=1.0):
            try:
                return orig_read(timeout_s=timeout_s)
            except Exception as exc:
                self._report(exc)
                raise
        self._source.read_new = read_new

    def _report(self, exc):
        key = type(exc).__name__ + str(exc)[:80]
        n = self._seen.get(key, 0) + 1
        self._seen[key] = n
        if n == 1 or n % 50 == 0:
            print(f"PERCEPTION ERROR x{n}: {exc!r}\n" + "".join(traceback.format_exc()), flush=True)


L.PerceptionWorker = LoudWorker


def _watchdog():
    """No board for 3 s -> dump every thread's stack once per stall, so a hang
    names the call it is stuck in (last night's stall left no trace at all)."""
    dumped = False
    while True:
        time.sleep(0.5)
        w = _W["w"]
        snap = w.latest() if w else None
        if snap is None:
            continue
        age = snap.age_ms()
        if ACT and age > 10000:
            print("WATCHDOG: perception stalled >10 s during --act; stopping the run.", flush=True)
            os._exit(4)
        if age > 3000 and not dumped:
            dumped = True
            print(f"WATCHDOG: no new board for {age:.0f} ms (frames {w.frames}, errors {w.errors}, "
                  f"alive {w.alive}, last_error {w.last_error!r})", flush=True)
            for tid, fr in sys._current_frames().items():
                name = next((t.name for t in threading.enumerate() if t.ident == tid), tid)
                print(f"--- thread {name}\n" + "".join(traceback.format_stack(fr)), flush=True)
        elif age < 1000:
            dumped = False


threading.Thread(target=_watchdog, name="watchdog", daemon=True).start()


class FreshGate(L.ActionGate):
    """Stock gate plus: a 1 s board-age limit (stock 2 s) and a check that
    the board the tap is decided on is cached."""

    def __init__(self, *a, **k):
        k["max_age_ms"] = ours.max_board_age_ms
        super().__init__(*a, **k)

    def check(self, board_index, age_ms):
        AGE["ms"] = age_ms
        st = CACHE.get(board_index)
        if st is None or st[1] is None:
            return GateDecision(False, "board-unchecked")
        return super().check(board_index, age_ms)


L.ActionGate = FreshGate


def board_age_ms():
    w = _W["w"]
    snap = w.latest() if w else None
    return snap.age_ms() if snap is not None else 0.0


# ----------------------------------------------------------------- policy
import encoder_fixes  # noqa: E402


class EvoEncoder:
    def __init__(self, enc):
        self.enc = enc
        from python_ai.models import perception_encoder as PE  # noqa: PLC0415
        if ours.cloaked_rows == "on":
            # live_ppo/cloaked_rows.py: rows for cards the stock table skips because the
            # probe cannot see them cloaked (Royal Ghost). Adds rows only; no other card moves.
            import cloaked_rows  # noqa: PLC0415
            added = cloaked_rows.install(enc)
            print(f"cloaked rows: added {sorted(added)} to the card table")
        if ours.body_rows != "off":
            # live_ppo/body_rows.py: rows for spawned / second bodies, measured from the engine;
            # synthetic ids >= 1000, so no existing row moves
            added = body_rows.install(enc)
            print(f"body rows: added {sorted(added)} to the card table")
        if BUSH_POPS is not None:
            # live_ppo/bush_goblins.py: the Bush Goblin row (body -30 of card 85), measured from the engine
            added = bush_goblins.install(enc)
            print(f"bush goblins: added {sorted(added)} to the card table")
        e = PE.SPATIAL_SIZE + 1 + PE.HAND_SIZE + PE.HAND_SIZE * PE.NUM_CARD_IDS
        C = E.ClashRoyaleEnv
        # the cycle block sits right after the extra scalars in both layouts
        assert e == C.EXTRA_SCALARS_START and e + C.NUM_EXTRA_SCALARS == C.CYCLE_START, \
            (e, C.EXTRA_SCALARS_START, C.CYCLE_START)
        self.cyc = slice(C.CYCLE_START, C.CYCLE_START + C.CYCLE_BLOCK_SIZE)
        self.spatial = int(PE.SPATIAL_SIZE)     # the elixir scalar; the 4 slot costs follow
        # Level section (engine item 44): appended after the cycle blocks, so a
        # forward offset; absent on a gen-1 engine.
        n = getattr(C, "NUM_LEVEL_SCALARS", 0)
        self.lvl = slice(C.LEVEL_START, C.LEVEL_START + n) if n else None
        if self.lvl is not None:
            levels = ([int(v) for v in ours.levels.split(",")] if ours.levels
                      else [E.TOURNAMENT_LEVEL] * n)
            assert len(levels) == n, f"--levels wants {n} values, got {levels}"
            self.lvl_vals = np.array([E.level_scalar(v) for v in levels], np.float32)
            print(f"level section: levels {levels} -> {self.lvl_vals.tolist()}")
        # Opponent-state section (engine item 46): appended after the level section.
        self.opp_state = None
        if ours.opp_state:
            assert hasattr(C, "OPP_STATE_SIZE"), \
                "--opp-state needs an engine with patch 0012 (no OPP_STATE_SIZE in this one)"
            self.opp_state = slice(C.OPP_STATE_START, C.OPP_STATE_START + C.OPP_STATE_SIZE)
        elif getattr(C, "OPP_STATE_SIZE", 0):
            print("WARNING: engine has the opponent-state section (item 46) but --opp-state is "
                  "off: it stays zero, which a model trained on it never saw")

    def encode(self, gs):
        hand = tuple(BASE_TO_TRAIN.get(c, c) if c is not None else c for c in gs.my_hand)
        # live_ppo/encoder_fixes: a destroyed tower's cells clear in every channel,
        # as the engine's do (live_ppo/eval/live_vs_sim.py)
        if ours.body_rows != "off":
            gs = body_rows.remap(gs)        # only units of the body classes change id
        if BUSH_POPS is not None:
            gs = bush_goblins.remap(gs)     # only units named bush-goblin change id
        obs = encoder_fixes.encode(replace(gs, my_hand=hand), self.enc)
        if self.lvl is not None:
            obs[self.lvl] = self.lvl_vals
        if OPP is not None:
            # the encoder leaves the opponent-cycle block at zero; the engine's own
            # play recorder (mirror env fed by --opp-plays) fills it
            obs[self.cyc] = OPP.block(float(gs.seconds_elapsed))
        if self.opp_state is not None and OPP is not None and OPP.state is not None:
            if ABIL is not None and _ABIL["dec"] is not None:
                # our ability's "deployed" and "ready" from the button reader
                # (one reading per decision: PPOPolicy.decide)
                OPP.state.own_button = {_ABIL["card"]: (_ABIL["dec"]["present"], _ABIL["dec"]["ready"])}
            obs[self.opp_state] = OPP.state.section(hand, float(gs.my_elixir),
                                                    float(gs.seconds_elapsed))
        if _INFLIGHT["slots"]:
            # --pending-mask: a card already on its way, as the simulator's actuation
            # delay shows it (training/reward.py inflight_obs)
            _INFLIGHT["applied"] = policy_v3.inflight_obs(obs, self.spatial, _INFLIGHT["slots"])
        if TAIL is not None:
            # the observation v2 tail, on the clock of the frame `gs` was read from
            ts = (HUDLOG.get(gs.frame_index) or {}).get("tail_s", _TAIL_CLK["s"])
            obs = np.concatenate([np.asarray(obs, np.float32), TAIL.build(float(ts))])
        return obs


_t0 = time.perf_counter()
_n = {"i": 0}


def _net_v2_module():
    """training.net_v2, imported without leaking training.common's CLASH_DECK
    default into this process (the CRBAB hand reader needs the base deck)."""
    had = "CLASH_DECK" in os.environ
    if str(REPO) not in sys.path:
        sys.path.append(str(REPO))
    from training import net_v2 as N2  # noqa: PLC0415
    if not had:
        os.environ.pop("CLASH_DECK", None)
    return N2


def _checkpoint_arch(blob):
    """(is_v2, cfg) of a checkpoint blob: by `arch`, else by net_v2's keys."""
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    if (isinstance(blob, dict) and blob.get("arch") == "net_v2") or (
            isinstance(state, dict) and "board.stem.weight" in state):
        return True, dict(blob.get("cfg") or {})
    return False, None


def _load_v3(path, blob):
    """A v3 checkpoint -> (NetV3 drawing its own WHEN coin, loader info), or
    SystemExit saying why not: training/obs_compat.load_policy_v3 checks the
    engine, the observation length, the card ids, net_v3.py's code hash and the
    weights; policy_v3.check the engine, the tail flags and the deck.

    A v3 checkpoint has no `deck` key. What stands in for it is the deck the
    training code fixes, training/common.HOG26_DECK, which must be TRAIN_DECK.

    The loader routes the ability heads by the process's default deck, which in
    this process is the BASE 2.6 deck (CLASH_DECK is unset for the hand reader)
    and has no Hero: the slots are set again from TRAIN_DECK."""
    had = "CLASH_DECK" in os.environ
    if str(REPO) not in sys.path:
        sys.path.append(str(REPO))
    try:
        policy_v3.check(blob, path=path, engine_gen2=_engine_gen2(), engine=str(ENGINE),
                        obs_v2=ours.obs_v2, names=ours.obs_v2_names, driver_deck=TRAIN_DECK)
        try:
            from training import common as TC  # noqa: PLC0415
            from training import obs_compat as OC  # noqa: PLC0415
        finally:
            if not had:
                os.environ.pop("CLASH_DECK", None)
        policy_v3.check(blob, path=path, engine_gen2=_engine_gen2(), engine=str(ENGINE),
                        obs_v2=ours.obs_v2, names=ours.obs_v2_names, driver_deck=TRAIN_DECK,
                        train_deck=TC.HOG26_DECK)
        net, info = OC.load_policy_v3(str(path), blob, when_internal=True)
    except ValueError as ex:
        raise SystemExit(f"v3 checkpoint refused: {ex}") from ex
    from python_ai.rl import abilities as A  # noqa: PLC0415
    net.set_ability_slots(A.ability_engine_slots(TRAIN_DECK))
    return net, info


# --pocket-mask (default off): runs/pocket_1126/fix/pocket_mask.py opens the pocket in
# MicroRoyaleNet.placement_mask; the lanes it may open are the confirmed, not refused ones.
POCKET_G = None
_PKT = {"state": 0}       # lanes open on the decision being made (bit 1 LEFT, 2 RIGHT)
_PK = None
if ours.pocket_mask:
    import pocket_live  # noqa: E402
    sys.path.insert(0, str(REPO / "runs" / "pocket_1126" / "fix"))
    import pocket_mask as _PK  # noqa: E402
    if HUD_TOWERS is None:
        print("WARNING: --pocket-mask without --hud-towers: no tower death can be confirmed, "
              "the pocket stays closed", flush=True)
    POCKET_G = pocket_live.PocketGuard(confirm_s=ours.pocket_confirm_s)
    _PK_ORIG_STATE = _PK._enemy_state

    def _pk_state(self, obs):
        # obs says dead AND the guard confirmed it AND no refusal closed the lane
        return _PK_ORIG_STATE(self, obs) & int(POCKET_G.open_bits())

    _PK._enemy_state = _pk_state
    _PK.install(table="verified")     # the table the pocket-trained models learned under (DATA_SPEC 4)
    # Refuse an engine with no pocket legality (e.g. build/gen2_c, live_session.sh's default): the table would be empty
    # and --pocket-mask would silently open nothing. The pocket-trained nets were trained on build/gen2_j.
    _pk_t = _PK.build_pocket_add(int(E.ClashRoyaleEnv.NUM_CARD_IDS), 32, int(E.ClashRoyaleEnv.BOARD_WIDTH))
    _pk_w = int(E.ClashRoyaleEnv.BOARD_WIDTH)
    if int(_pk_t[..., 16 * _pk_w:23 * _pk_w].sum()) == 0:     # pocket rows 16-22 only (gen2_c has 204 deploy cells at rows 25-29)
        raise SystemExit(f"--pocket-mask: the engine {os.environ.get('CR_ENGINE', 'engine/')} opens no pocket cell "
                         f"(no pocket legality); run with the training engine set: CR_LIVE_ENGINE=build/gen2_j/engine "
                         f"CR_P0088=tornado,jump,log,turret,king,speed,bankrows,stomp,jumpcol,pocketcells,kingpath")
    print(f"pocket mask: ON, table {_PK.TABLE} (confirm {ours.pocket_confirm_s:g} s, closes a lane after a refused pocket tile)",
          flush=True)


class _PocketLegal(dict):
    """The live loop's third mask (engine legality on a fresh board, mvp_loop._legal_by_card) plus the
    pocket cells of the lanes open on this decision."""
    net = None

    def get(self, k, default=None):
        base = dict.get(self, k, default)
        st = _PKT["state"]
        if base is None or not st:
            return base
        table = _PK._pocket_table(self.net)
        cid = BASE_TO_TRAIN.get(k, k)
        closed = _PK.build_closed(self.net.num_card_ids, self.net.placement_rows, self.net.board_width)
        return base | (table[st, cid] & ~closed[cid]).unsqueeze(0).to(base.device)


def _pocket_lanes(card_id, tile):
    """Lane bits whose pocket holds this cell for this (training-id) card."""
    net = POCKET_G.net
    table = _PK._pocket_table(net)
    cell = int(tile[1]) * net.board_width + int(tile[0])
    return [b for b in (1, 2) if bool(table[b, card_id, cell])]


#: what a v3 run did, for the end-of-run line (v3_summary)
_V3 = {"decisions": 0, "plays": 0, "pending": 0, "forced": 0, "ability_want": 0,
       "ability_p_max": 0.0, "obs_len": None}


class PPOPolicy(L.NeuralPolicy):
    def __init__(self, checkpoint, tactical=True, reserve=4.0, **kw):
        import torch
        import python_ai.models.net as netmod
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.kind = policy_v3.arch(blob)
        v3 = self.kind == policy_v3.V3
        info = None
        if v3:
            # Its defaults (tail names, sampling) were resolved from the loop's
            # --checkpoint when the driver started, so it must be the one named there.
            if _ckpt()[1] != policy_v3.V3:
                raise SystemExit(f"{checkpoint} is a v3 checkpoint: name it with --checkpoint on "
                                 f"the driver's command line (its defaults are set at start)")
            built, info = _load_v3(checkpoint, blob)
            self.set = _settings(self.kind, blob)
            tactical = self.set["tactical"]
            is_v2, cfg, probe = False, None, None
            _tail_start(f"the checkpoint was trained with it: {info['obs_dim']} floats")
            if TAIL.names != blob["names"]:
                raise SystemExit(f"obs v2 tail runs with opponent names '{TAIL.names}', the "
                                 f"checkpoint was trained with '{blob['names']}'")
        else:
            self.set = _settings(self.kind, blob)
            deck = list(blob.get("deck") or [])
            if deck != TRAIN_DECK:
                raise SystemExit(f"checkpoint deck {deck} != ppo_bc26 deck {TRAIN_DECK}")
            is_v2, cfg = _checkpoint_arch(blob)
            if ours.obs_v2 == "auto" and (blob.get("obs_v2") or (blob.get("cfg") or {}).get("obs_v2")):
                _tail_start("the checkpoint was trained with it")
        if is_v2:
            # NeuralPolicy builds `python_ai.models.net.MicroRoyaleNet(...)` and
            # strict-loads. NetV2 is a subclass (every method the loop calls
            # exists), but it cannot be constructed while that name is swapped:
            # MicroRoyaleNet.__init__ calls super(MicroRoyaleNet, self). So the
            # NetV2 is built first and the swapped name just hands it back.
            N2 = _net_v2_module()
            built = N2.NetV2(num_ability_slots=1, **cfg)
            probe = N2.NetV2(num_ability_slots=1, **cfg)
        elif not v3:
            probe = netmod.MicroRoyaleNet(num_ability_slots=1)
        if is_v2 or v3:
            # v3 the same way: the loaded NetV3 (also a MicroRoyaleNet subclass) is
            # handed to the base class, which loads the same weights into it again.

            class _Prebuilt:
                def __new__(cls, *a, **k):
                    return built
            netmod.MicroRoyaleNet = _Prebuilt
        try:
            super().__init__(checkpoint, deck_ability_slots=1, tactical=tactical, reserve=reserve)
        finally:
            netmod.MicroRoyaleNet = L_ORIG_NET
        if v3:
            print(f"ckpt adapter: arch v3 ({type(self.net).__name__}), observation "
                  f"{info['obs_dim']} = engine {blob['base_obs_dim']} + tail, tail names "
                  f"'{info['names']}', {info['decision_ticks']} ticks per decision, lead "
                  f"{info['lead']}, dataset {info['dataset']} ({info['dataset_fp']}), ability "
                  f"slots {self.net.ability_slots} (deck: training/common.HOG26_DECK); hand remap "
                  f"{BASE_TO_TRAIN}; decision rate {L.DECISION_HZ} Hz")
        else:
            r = probe.load_state_dict(blob["model"], strict=False)
            print(f"ckpt adapter: arch {'net_v2' if is_v2 else 'v1'} ({type(self.net).__name__}), "
                  f"num_ability_slots=1, missing {list(r.missing_keys)}, "
                  f"unexpected {list(r.unexpected_keys)}; hand remap {BASE_TO_TRAIN}; "
                  f"decision rate {L.DECISION_HZ} Hz")
        self.encoder = EvoEncoder(self.encoder)
        if POCKET_G is not None:
            POCKET_G.net, POCKET_G.classify = self.net, _pocket_lanes
            _PocketLegal.net = self.net
            self._legal_by_card = _PocketLegal(self._legal_by_card)
        if ours.afford_project:
            # Innermost wrapper (the empty-slot, pocket and leak-floor masks stay around it):
            # cost <= E_view becomes cost <= P (AFFORD_RULE.txt s1); the no-op column is the base's.
            aff_base, aff_net = self.net.affordability_mask, self.net
            from python_ai.models import perception_encoder as _PE  # noqa: PLC0415
            aff_e0 = _PE.SPATIAL_SIZE + 1 + _PE.HAND_SIZE + _PE.HAND_SIZE * _PE.NUM_CARD_IDS   # = C.EXTRA
            print(f"projected affordability: ON (R 1/28, DMIN {AFF_DMIN}, mult at obs[{aff_e0}+9]; "
                  f"PlayWatch LAND_S {PLAYS.LAND_S if PLAYS is not None else None} s)")

            def _projected_mask(obs):
                m = aff_base(obs)
                s, h = aff_net.spatial_size, aff_net.hand_size
                e_view = obs[:, s:s + 1] * 10.0
                mult = obs[:, aff_e0 + 9:aff_e0 + 10] * 3.0
                p = torch.where(e_view > 0, torch.clamp(e_view + AFF_REGEN * mult * AFF_DMIN, max=10.0),
                                torch.zeros_like(e_view))
                costs = obs[:, s + 1:s + 1 + h] * 10.0
                out = m.clone()
                out[:, :h] = (costs > 0.0) & (costs <= p + 1e-6)
                _AFF["last"] = {"e_view": round(float(e_view[0, 0]), 3), "mult": round(float(mult[0, 0]), 3),
                                "p": round(float(p[0, 0]), 3), "costs": [round(float(c), 1) for c in costs[0]],
                                "base": [bool(b) for b in m[0, :h]], "proj": [bool(b) for b in out[0, :h]]}
                return out
            self.net.affordability_mask = _projected_mask
        if not ours.allow_empty_slot:
            base_mask = self.net.affordability_mask

            def _present_mask(obs):
                # live only, like the tappable-cell mask: in the engine a played
                # card's replacement is in the hand at once
                m = base_mask(obs)
                for s in _EMPTY["slots"]:
                    m[:, s] = False
                return m
            self.net.affordability_mask = _present_mask
        if not ours.solvency_gate:
            self._gate = None
        elif v3 and self._gate is None:
            # the base class builds the gate only with the tactical officer on
            from python_ai.advisors import tactics  # noqa: PLC0415
            self._gate = tactics.SolvencyGate(reserve=reserve)
        print(f"solvency gate: {'ON' if self._gate is not None else 'OFF'}")
        floor = self.set["leak_floor"]
        self._v3_floor = 0.0
        self._knobs = None
        if v3:
            # NetV3 never reads the no-op column of the mask, so the floor cannot
            # work as it does below: with --leak-floor the WHEN coin is forced.
            self._v3_floor = floor if 0 < floor <= 10 else 0.0
            self._knobs = dict(_SAMPLING["set_sampling"])
            self.net.set_sampling(**self._knobs)
            self._ab_rng = __import__("random").Random(0)   # "would it press" draws, own stream
            self._ab_want = False
            print(f"v3 policy: {_SAMPLING['note']}; tactical officer "
                  f"{'ON (--tactical-officer)' if tactical else 'OFF'}; leak floor "
                  + (f"forces a play at elixir >= {self._v3_floor:g}" if self._v3_floor else "OFF")
                  + f"; plays in flight {'masked' if self.set['pending_mask'] else 'NOT masked'}"
                  + "; hero ability: " + ("logged only (--ability off, or no reader for this deck)"
                                          if ABIL is None else
                                          "masked by the button reader, presses SENT" if _ABIL["mode"] == "tap"
                                          else "masked by the button reader, presses logged and drawn, not sent"))
        elif 0 < floor <= 10:
            net, orig = self.net, self.net.affordability_mask

            def _no_leak_mask(obs):
                # Elixir is /10 in the observation (net.affordability_mask).
                m = orig(obs)
                full = obs[:, net.spatial_size] * 10.0 >= floor - 1e-4
                any_card = m[:, :-1].any(dim=1)
                m[:, -1] = m[:, -1] & ~(full & any_card)
                return m
            self.net.affordability_mask = _no_leak_mask
            print(f"leak floor: no-op masked at elixir >= {floor:g}")
        if self.set["pending_mask_lost"]:
            print("WARNING: plays in flight are NOT masked (--no-play-watch): at this decision "
                  "rate the policy can ask for a card that is already on its way")

    def _v3_peek(self, gs, obs, feats, mask):
        """What the coming v3 decision is drawn from, for the log, from the raw
        heads (no random draw, so the acting forward's coin is the only one), and
        the --leak-floor force. Returns (five-way probs, the log's v3 block)."""
        torch = self.torch
        o = self.net.heads(feats, mask)
        legal = bool(o["has_legal"][0])
        knobs = dict(self._knobs)
        forced = bool(self._v3_floor and legal
                      and float(obs[0, self.net.spatial_size]) * 10.0 >= self._v3_floor - 1e-4)
        if forced:
            knobs["when_bias"] = policy_v3.FORCE_PLAY_BIAS
            self.net.set_sampling(**knobs)         # decide() puts the knobs back
        pr = policy_v3.card_probs(float(o["when"][0]), o["what"][0].tolist(), legal, knobs)
        ability = None
        for slot in self.net.ability_slots:
            # The simulator samples this head where the ability is ready
            # (pool_env.OppPolicy.act: softmax(logits)[1] * ready, then a coin) and
            # activates it. Live "ready" is the button reader's (_ability_now); the
            # press itself is issued after the card decision (_ability_decide).
            p = policy_v3.ability_prob(o["ability"][slot - 1][0].tolist())
            draw = self._ab_rng.random() < p
            dec = _ABIL["dec"] if ABIL is not None and slot == _ABIL["slot"] else None
            _V3["ability_p_max"] = max(_V3["ability_p_max"], p)
            if dec is None:
                # --ability off (or no reader for this deck): the draw is logged only
                ability = {"slot": int(slot), "p": round(p, 4), "want": bool(draw), "sent": False}
                _V3["ability_want"] += draw
                if draw and not self._ab_want:
                    print(f"ABILITY slot {slot}: the policy's draw says press it (p={p:.3f}) -- NOT SENT, "
                          f"the button is not read (--ability off)")
                self._ab_want = bool(draw)
                continue
            want = bool(dec["ready"] and draw)
            ability = {"slot": int(slot), "p": round(p, 4), "button": dec["button"],
                       "present": dec["present"], "ready": dec["ready"],
                       "not_ready": dec["why"], "want": want, "sent": False, "why": ""}
            _ABIL["ready_dec"] += dec["ready"]
            _V3["ability_want"] += want
        v3 = policy_v3.decision_record(
            pr, [_cname(c) for c in gs.my_hand], ability=ability, pending=_INFLIGHT["applied"],
            knobs={k: v for k, v in knobs.items() if k != "when_internal"},
            obs_len=int(obs.shape[1]), forced=forced)
        _V3["decisions"] += 1
        _V3["pending"] += bool(_INFLIGHT["applied"])
        _V3["forced"] += forced
        _V3["obs_len"] = int(obs.shape[1])
        return pr["p"], v3, forced

    def decide(self, gs, ready, now):
        torch = self.torch
        age = board_age_ms()
        if age > ours.max_board_age_ms:
            # A frozen board fed to the LSTM again and again is not what it trained
            # on, and the tap would be refused anyway: hold and do not step.
            d = L.Decision(None, L.DEFAULT_TILE, f"hold: board {age:.0f} ms old")
            log_decision(gs, d, [0.0] * 5, None, age)
            return d
        if self._hx is None:
            self.reset_hidden()
        _AFF["last"] = None
        _EMPTY["slots"] = [s for s, e in enumerate((HUDLOG.get(gs.frame_index) or {}).get("empty_slots") or ())
                           if e]
        # One reading of the plays in flight for this decision: the encoder runs
        # twice (here and in the base class) and the perception thread moves them.
        _INFLIGHT["applied"] = []
        _INFLIGHT["slots"] = (policy_v3.blocked_slots(PLAYS.in_flight(), [_cname(c) for c in gs.my_hand])
                              if self.set["pending_mask"] else {})
        hx, cx = self._hx.clone(), self._cx.clone()
        if ABIL is not None:
            # one reading of the ability button for this decision, like the plays in flight
            _ABIL["dec"] = _ability_now(gs)
        obs = torch.as_tensor(self.encoder.encode(gs)).unsqueeze(0)
        if POCKET_G is not None:
            if PLAYS is not None:
                POCKET_G.see_plays(list(PLAYS.plays))
            _PKT["state"] = int(_PK_ORIG_STATE(self.net, obs)[0]) & int(POCKET_G.open_bits())
        v3, forced = None, False
        with torch.no_grad():
            feats, _e, _s = self.net.extract_features(obs)
            mask = self.net.affordability_mask(obs)
            if self.kind == policy_v3.V3:
                probs, v3, forced = self._v3_peek(gs, obs, feats, mask)
            else:
                logits, *_ = self.net.step_lstm_and_card(feats, (hx, cx), mask)
                probs = torch.softmax(logits, -1)[0].tolist()
        t_dec = time.perf_counter()
        try:
            d = super().decide(gs, ready, now)
        finally:
            if forced:
                self.net.set_sampling(**self._knobs)
        if d.slot is not None:
            card = _cname(gs.my_hand[d.slot]) if d.slot < len(gs.my_hand) else None
            _LAST_DECIDE[int(d.slot)] = (card, tuple(d.tile))
            if POCKET_G is not None and d.slot < len(gs.my_hand) and gs.my_hand[d.slot] is not None:
                _cid = int(gs.my_hand[d.slot])
                POCKET_G.note_decision(d.slot, d.tile, BASE_TO_TRAIN.get(_cid, _cid), _PKT["state"])
                if v3 is not None:
                    _tid = BASE_TO_TRAIN.get(_cid, _cid)
                    _tab = _PK._pocket_table(self.net)
                    _tap = engine_tile_centre(int(d.tile[0]), int(d.tile[1]))
                    v3["pocket"] = {"state": _PKT["state"], "card": _tid,
                                    "pocket_cells_legal": int(_tab[_PKT["state"], _tid].sum()) if _PKT["state"] else 0,
                                    "tile_in_pocket": bool(_PKT["state"] and _tab[_PKT["state"], _tid, int(d.tile[1]) * self.net.board_width + int(d.tile[0])]),
                                    "tap_display": [int(_tap.x), int(_tap.y)],
                                    "tap_frame_px": [round(_tap.x / DISPLAY_WIDTH * 549), round(_tap.y / DISPLAY_HEIGHT * 976)]}
            lat("decide", t=t_dec, fi=gs.frame_index, slot=int(d.slot), card=card, tile=list(d.tile))
            _V3["plays"] += v3 is not None
        if ABIL is not None and v3 is not None and v3.get("ability") is not None and "ready" in v3["ability"]:
            _ability_decide(gs, d, v3["ability"])
        log_decision(gs, d, probs, obs[0].numpy(), age, v3=v3)
        return d


def v3_summary():
    """One paragraph on a v3 run, or None when no v3 decision was made."""
    n = _V3["decisions"]
    if not n:
        return None
    return (f"v3 policy: {n} decisions on a {_V3['obs_len']}-float observation, {_V3['plays']} play "
            f"requests ({n - _V3['plays']} holds); a play was in flight (its slot masked) on "
            f"{_V3['pending']} decisions; leak floor forced {_V3['forced']}\n"
            + (f"hero ability: the policy's draw said press on {_V3['ability_want']} decisions "
               f"(highest p {_V3['ability_p_max']:.2f}); none sent -- the button was not read "
               f"(--ability off, or no reader for this deck)" if ABIL is None else ability_summary()))


def ability_summary():
    """The ability paragraph of a v3 run with the button reader on."""
    st = _ABIL["states"]
    ms = np.array(_ABIL["ms"]) if _ABIL["ms"] else np.zeros(1)
    held = "; ".join(f"{k}: {v}" for k, v in _ABIL["held"].items())
    return (f"hero ability button: read on {_ABIL['frames']} in-game frames "
            f"({', '.join(f'{k} {v}' for k, v in sorted(st.items())) or 'none'}; "
            f"{ms.mean():.1f} ms a frame); ready on {_ABIL['ready_dec']} of {_V3['decisions']} decisions "
            f"(highest p {_V3['ability_p_max']:.2f}); the policy asked for a press on {_ABIL['want']}"
            + (f" (held on: {held})" if held else "") + "\n"
            + ABIL.summary(sent=_ABIL["mode"] == "tap"))


import python_ai.models.net as _netmod  # noqa: E402
L_ORIG_NET = _netmod.MicroRoyaleNet
L.NeuralPolicy = PPOPolicy


def _decision_hz():
    """--decision-hz, else --decision-ticks, else the checkpoint's own
    `decision_ticks` (engine ticks per decision; gen 2 trains at 5) or
    `decision_hz`, else 1 Hz (every PPO run before gen 2 trains at
    SKIP_TICKS=10): decision_rate.decision_hz. Set before the loop starts: it
    reads the module global L.DECISION_HZ at runtime."""
    import decision_rate  # noqa: PLC0415
    blob = None
    if not ours.decision_hz and not ours.decision_ticks and "--checkpoint" in loop_args:
        blob = _ckpt()[0]
    hz, why = decision_rate.decision_hz(ours.decision_hz, ours.decision_ticks, blob)
    if why != "default":
        print(f"decision rate: {hz:g} Hz ({why})")
    return hz


L.DECISION_HZ = _decision_hz()

# How the flags sample (policy_v3.sampling). Older checkpoints, as before: --greedy
# is the argmax, else --temp != 1 tempers the card and cell logits, both through
# torch's Categorical. A v3 checkpoint takes its temperatures inside the net
# (PPOPolicy sets them), so the Categorical is never tempered for it, and
# --greedy leaves its WHEN coin alone.
try:
    _SAMPLING = policy_v3.sampling(_ckpt()[1] or policy_v3.V1, temp=ours.temp, greedy=ours.greedy,
                                   when_temp=ours.when_temp, when_bias=ours.when_bias)
except policy_v3.V3Error as ex:
    raise SystemExit(str(ex)) from ex
SET()       # the checkpoint's settings, resolved (and refused) before the loop starts

if _SAMPLING["patch"] == "argmax":
    import torch as _torch

    class _Argmax(_torch.distributions.Categorical):
        def sample(self, *a, **k):
            return self.logits.argmax(-1)

    _torch.distributions.Categorical = _Argmax
elif _SAMPLING["patch"] == "tempered":
    import torch as _torch
    _T = _SAMPLING["patch_temp"]

    class _Tempered(_torch.distributions.Categorical):
        # Masked logits are -inf and stay -inf after the division.
        def __init__(self, probs=None, logits=None, validate_args=None):
            super().__init__(logits=logits / _T, validate_args=validate_args)

    _torch.distributions.Categorical = _Tempered


# ----------------------------------------------------------------- logging/overlay
def _tw(t):
    return {"f": round(float(t.hp_fraction), 3), "m": bool(t.hp_measured), "dead": bool(t.destroyed)}


def _cname(c):
    if c is None or c < 0:
        return "?"
    return E.get_card_info(int(c))["name"]


def log_decision(gs, d, probs, obs, age=0.0, v3=None):
    _n["i"] += 1
    st = CACHE.get(gs.frame_index)
    state, frame, ydets, secs = st if st else (None, None, None, 0.0)
    raw_hand = [c.name for c in state.cards[1:5]] if state else []
    nxt = state.cards[0].name if state else None
    units = [{"cls": u.unit_name, "id": u.card_sim_id, "side": u.team, "tile": [u.tile_x, u.tile_y],
              "conf": round(float(u.confidence), 2), "hp": round(float(u.hp_fraction), 2),
              "badge": u.team_from_badge} for u in gs.units]
    crbab = []
    if state is not None:
        for team, us in ((0, state.allies), (1, state.enemies)):
            for u in us:
                p = u.position
                crbab.append({"cls": u.unit.name, "side": team, "tile": [p.tile_x, p.tile_y + 1],
                              "bbox": [round(v, 1) for v in p.bbox], "conf": round(float(p.conf), 2)})
    card = None
    if d.slot is not None and d.slot < len(gs.my_hand):
        card = _cname(gs.my_hand[d.slot])
    from python_ai.models import perception_encoder as PE  # noqa: PLC0415
    e0 = PE.SPATIAL_SIZE + 1 + PE.HAND_SIZE + PE.HAND_SIZE * PE.NUM_CARD_IDS
    rec = {
        "i": _n["i"], "t": round(time.perf_counter() - _t0, 2), "frame": gs.frame_index,
        "screen": state.screen.name if state else None,
        "hand": [_cname(c) for c in gs.my_hand], "hand_raw": raw_hand, "next": nxt,
        "elixir_raw": float(state.numbers.elixir.number) if state else None,
        "elixir": round(float(gs.my_elixir), 2), "spent": round(float(gs.my_elixir_spent), 1),
        "seconds_elapsed_fed": round(float(gs.seconds_elapsed), 1), "since_match_start": round(secs, 1),
        "towers": {k: _tw(getattr(gs, k)) for k in ("own_king", "own_princess_left", "own_princess_right",
                                                     "opp_king", "opp_princess_left", "opp_princess_right")},
        "units": units, "crbab": crbab, "yolo": ydets,
        "decision": {"slot": d.slot, "card": card, "tile": list(d.tile), "why": d.why},
        "p": [round(p, 3) for p in probs],
        "obs_time": None if obs is None else round(float(obs[e0 + 0]), 4),
        "obs_mult": None if obs is None else round(float(obs[e0 + 9]), 4),
        "obs_spent": None if obs is None else round(float(obs[e0 + 1]), 4),
        "obs_opp_spent": None if obs is None else round(float(obs[e0 + 2]), 4),
        "obs_opp_towers": None if obs is None else [round(float(obs[e0 + 6 + k]), 4) for k in range(3)],
        "obs_cycle_nnz": None if obs is None else int(np.count_nonzero(
            obs[E.ClashRoyaleEnv.CYCLE_START:E.ClashRoyaleEnv.CYCLE_START + E.ClashRoyaleEnv.CYCLE_BLOCK_SIZE])),
        "hud": HUDLOG.get(gs.frame_index),
        "opp_plays": None if OPP is None else OPP.log[-12:],
        "board_age_ms": round(age, 1),
    }
    if TAIL is not None and obs is not None:
        import obs_tail  # noqa: PLC0415
        tail = np.asarray(obs[-obs_tail.V.TAIL_SIZE:], np.float32)
        dec = obs_tail.V.decode(tail, 0.0)
        rec["obs_v2"] = {
            "tail_s": (HUDLOG.get(gs.frame_index) or {}).get("tail_s"),
            "opp": [[_cname(p.card), p.x, p.y, round(-p.t, 1)] for p in dec["opp"]],
            "own": [[_cname(p.card), p.x, p.y, round(-p.t, 1)] for p in dec["own"]],
            "next": _cname(dec["next"])}
    if ours.afford_project and _AFF["last"] is not None:
        a = dict(_AFF["last"])
        a["offered_by_p_only"] = [i for i, (b, q) in enumerate(zip(a["base"], a["proj"])) if q and not b]
        _AFF["decisions"] += 1
        _AFF["differ"] += bool(a["offered_by_p_only"])
        if d.slot is not None and d.slot < len(a["costs"]):
            a["chosen"] = {"slot": int(d.slot), "card": card, "cost": a["costs"][d.slot],
                           "p_only": d.slot in a["offered_by_p_only"]}
            _AFF["p_only"] += a["chosen"]["p_only"]
            _AFF["plays"].append(dict(a["chosen"], t=time.perf_counter(), e_view=a["e_view"], mult=a["mult"],
                                      p=a["p"], match_s=round(secs, 1)))
        rec["afford"] = a
    if v3 is not None:
        # a v3 decision: p(play now), the cards by p(card | play), the cell, the ability
        v3["chosen"] = None if d.slot is None else {"slot": int(d.slot), "card": card,
                                                    "tile": list(d.tile)}
        rec["v3"] = v3
        _c = E.ClashRoyaleEnv
        if ABIL is not None and obs is not None and ours.opp_state and hasattr(_c, "OWN_ABILITY_OFFSET"):
            # the ability fields the net was shown on this decision: ours per slot
            # (has one, deployed, ready, cooldown), the opponent's per card
            # (seen, cooldown, cost/10, recency)
            _a = _c.OPP_STATE_START + _c.OWN_ABILITY_OFFSET
            _b = _c.OPP_STATE_START + _c.OPP_ABILITY_OFFSET
            rec["obs_ability"] = {
                "own": [round(float(x), 3) for x in obs[_a:_a + 2 * _c.ABILITY_FEATURES]],
                "opp": [round(float(x), 3)
                        for x in obs[_b:_b + _c.NUM_OPP_ABILITY_CARDS * _c.ABILITY_FEATURES]]}
    if ours.dump_obs and obs is not None:
        # obs.f32 keeps the engine observation's width (its readers reshape by
        # it); the observation v2 tail goes to obs_tail.f32, row for row
        row = np.asarray(obs, dtype=np.float32)
        if TAIL is not None:
            import obs_tail  # noqa: PLC0415
            with open(RUN / "obs_tail.f32", "ab") as fh:
                row[-obs_tail.V.TAIL_SIZE:].tofile(fh)
            row = row[:-obs_tail.V.TAIL_SIZE]
        with open(RUN / "obs.f32", "ab") as fh:
            row.tofile(fh)
    PLOG.write(json.dumps(rec) + "\n")
    AGES.write(f"{rec['t']},{age:.1f},{gs.frame_index}\n")
    pn = probs[-1]
    if v3 is not None:
        top = ", ".join(f"{c} {p:.2f}" for c, p in v3["what"][:3]) or "nothing playable"
        ab = "" if v3["ability"] is None else f"; ability p={v3['ability']['p']:.2f}"
        if v3["ability"] is not None and "button" in v3["ability"]:
            a = v3["ability"]
            ab += (f" button {a['button']}" + (", pressable" if a["ready"] else "")
                   + (f" PRESS #{a['press']}" if "press" in a else f" ({a['why']})" if a["why"] else ""))
        pend = "" if not v3["pending"] else f"; in flight: slot {[p[0] for p in v3['pending']]}"
        what = (f"hold  ({d.why}" if d.slot is None
                else f"PLAY {card} -> tile {tuple(d.tile)} ({d.why}")
        print(f"DECIDE {rec['t']:6.1f}s {what}; p(play)={v3['p_play']:.2f}; {top}{ab}{pend}) "
              f"elixir {gs.my_elixir:.1f} age {age:.0f}ms")
    elif d.slot is None:
        print(f"DECIDE {rec['t']:6.1f}s hold  ({d.why}; P(no-op)={pn:.2f}) elixir {gs.my_elixir:.1f} age {age:.0f}ms")
    else:
        print(f"DECIDE {rec['t']:6.1f}s PLAY {card} -> tile {tuple(d.tile)} "
              f"(P(card)={probs[d.slot]:.2f}, P(no-op)={pn:.2f}, elixir {gs.my_elixir:.1f}, age {age:.0f}ms; {d.why})")
    if frame is not None and (_n["i"] % ours.overlay_every == 0
                              or (v3 is not None and "press" in (v3.get("ability") or {}))):
        draw_overlay(rec, frame, state, ydets, d)      # a pressing decision is always drawn


def draw_overlay(rec, frame, state, ydets, d):
    img = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]
    sx, sy = W / SCREENSHOT_WIDTH, H / SCREENSHOT_HEIGHT
    fx, fy = W / DISPLAY_WIDTH, H / DISPLAY_HEIGHT
    if state is not None:
        for team, us in ((0, state.allies), (1, state.enemies)):
            # stock engine detector, drawn grey for comparison only (the policy reads it only with --units crbab)
            col = (150, 150, 150) if ours.units == "yolo" else (255, 160, 0) if team == 0 else (0, 0, 255)
            for u in us:
                l, t, r, b = u.position.bbox
                p1, p2 = (int(l * sx), int(t * sy)), (int(r * sx), int(b * sy))
                cv2.rectangle(img, p1, p2, col, 2)
                lab = (f"{'stock:' if ours.units == 'yolo' else ''}{u.unit.name[:10]} {'A' if team == 0 else 'E'}"
                       f" ({u.position.tile_x},{u.position.tile_y + 1})")
                cv2.putText(img, lab, (p1[0], max(10, p1[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
    if ydets:
        for y in ydets:
            x1, y1, x2, y2 = map(int, y["box"])
            col = (0, 220, 0) if y["side"] == 0 else (0, 200, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 1)
            cv2.putText(img, f"Y:{y['cls'][:12]} {'A' if y['side'] == 0 else 'E'} ({y['tile'][0]},{y['tile'][1] + 1})",
                        (x1, min(H - 4, y2 + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1)
    if d.slot is not None:
        tap = engine_tile_centre(*d.tile)
        cv2.drawMarker(img, (int(tap.x * fx), int(tap.y * fy)), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
    ab = (rec.get("v3") or {}).get("ability") if ABIL is not None else None
    if ab is not None and "button" in ab:
        # Where the ability tap lands, through the same display -> frame mapping as
        # the card marker above: a yellow cross whenever the button is on screen
        # (it must sit on the button), a thick red ring on the decision that presses.
        ax, ay = AB.tap_display()
        at = (int(ax * fx), int(ay * fy))
        press = "press" in ab
        if ab["present"] or press:
            cv2.drawMarker(img, at, (0, 255, 255), cv2.MARKER_CROSS, 16, 1)
            cv2.circle(img, at, int(AB.RADIUS * W / AB.FRAME_W), (0, 255, 255) if not press else (0, 0, 255),
                       1 if not press else 3)
            lab = (f"ABILITY PRESS #{ab['press']}" + (" SENT" if ab["sent"] else " (not sent)") if press
                   else f"ability {ab['button']}" + (", pressable" if ab["ready"] else ""))
            cv2.putText(img, lab, (max(4, at[0] - 150), at[1] - int(AB.RADIUS) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 0), 3)
            cv2.putText(img, lab, (max(4, at[0] - 150), at[1] - int(AB.RADIUS) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 255) if press else (0, 255, 255), 1)
    tw = rec["towers"]
    lines = [
        f"#{rec['i']} t={rec['t']}s fed_clock={rec['seconds_elapsed_fed']}s since_start={rec['since_match_start']}s",
        f"elixir raw {rec['elixir_raw']} fed {rec['elixir']}  next {rec['next']}",
        "hand " + ",".join(h[:8] for h in rec["hand"]),
        "raw  " + ",".join(h[:8] for h in rec["hand_raw"]),
        "opp K {:.2f} L {:.2f}{} R {:.2f}{}".format(tw["opp_king"]["f"], tw["opp_princess_left"]["f"],
                                                    "" if tw["opp_princess_left"]["m"] else "?",
                                                    tw["opp_princess_right"]["f"],
                                                    "" if tw["opp_princess_right"]["m"] else "?"),
        "own K {:.2f} L {:.2f}{} R {:.2f}{}".format(tw["own_king"]["f"], tw["own_princess_left"]["f"],
                                                    "" if tw["own_princess_left"]["m"] else "?",
                                                    tw["own_princess_right"]["f"],
                                                    "" if tw["own_princess_right"]["m"] else "?"),
        f"DECIDE {rec['decision']['card'] or 'hold'} {tuple(rec['decision']['tile']) if rec['decision']['slot'] is not None else ''} pNoop={rec['p'][-1]:.2f}",
    ]
    h = rec.get("hud") or {}
    if "clock" in h:
        c = h["clock"]
        lines.append(f"clk left {c['left']} el {c['elapsed']} badge x{c['badge']}{' OT' if c['ot'] else ''}")
    if "towers" in h:
        tl, tr = h["towers"]["opp_princess_left"], h["towers"]["opp_princess_right"]
        lines.append(f"hud opp L {tl['hp']}/{tl['max']} R {tr['hp']}/{tr['max']}")
    if rec.get("opp_plays") is not None:
        o = h.get("opp", {})
        lines.append(f"opp spent {o.get('spent', 0):.0f} elx {o.get('elixir', 0):.1f} cyc {rec['obs_cycle_nnz']}: "
                     + ",".join(p["name"][:8] for p in rec["opp_plays"][-4:]))
    y0 = 300
    for i, s in enumerate(lines):
        cv2.putText(img, s, (4, y0 + 13 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3)
        cv2.putText(img, s, (4, y0 + 13 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    cv2.imwrite(str(RUN / "overlay" / f"d{rec['i']:04d}_f{rec['frame']}.jpg"), img,
                [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    sys.argv = [str(PERC / "live" / "mvp_loop.py")] + loop_args
    try:
        code = L.main()
    finally:
        if REC is not None:
            REC.close()
        if PLAYS is not None and PLAYS.plays:
            sent = "--act" in loop_args
            print("\n" + PLAYS.summary(sent=sent))
            if sent:
                print("  (this is the count to trust. The loop's lines above - \"unit appeared\", "
                      "\"THE GAME REFUSED IT\", \"issued plays never confirmed\" - under-count: "
                      "they miss units among other units and bar drops spread over two frames.)")
            (RUN / "plays.json").write_text(json.dumps(PLAYS.rows(), indent=1))
        if ours.afford_project:
            rows = PLAYS.rows() if PLAYS is not None else []
            used = set()
            for a in _AFF["plays"]:
                # the PlayWatch issue for this decision: same slot, issued within 1 s after it
                hit = next((p for p in (PLAYS.plays if PLAYS is not None else ())
                            if p.i not in used and p.slot == a["slot"] and 0 <= p.t - a["t"] < 1.0), None)
                if hit is not None:
                    used.add(hit.i)
                    r = rows[hit.i]
                    a["seq"], a["state"] = hit.i, r["state"]
                    a["tap_to_blank_s"] = (None if hit.t_tap is None or hit.t_blank is None
                                           else round(hit.t_blank - hit.t_tap, 3))
                a["t"] = round(a["t"] - _t0, 2)
            q = [a for a in _AFF["plays"] if a["cost"] > a["e_view"] + 1e-6]
            print(f"\nprojected affordability: {_AFF['decisions']} decisions, P offered a card the base mask "
                  f"did not on {_AFF['differ']}; {len(_AFF['plays'])} plays chosen, {_AFF['p_only']} of them "
                  f"only under P; {len(q)} chosen with cost > E_view (queued in the game)")
            (RUN / "afford.json").write_text(json.dumps(
                {"regen": AFF_REGEN, "dmin": AFF_DMIN, "land_s": None if PLAYS is None else PLAYS.LAND_S,
                 "decisions": _AFF["decisions"], "differ": _AFF["differ"], "p_only": _AFF["p_only"],
                 "plays": _AFF["plays"]}, indent=1))
        if v3_summary():
            print("\n" + v3_summary())
        if ABIL is not None:
            (RUN / "ability.json").write_text(json.dumps(
                {"mode": _ABIL["mode"], "card": _ABIL["card"], "cost": _ABIL["cost"],
                 "tap_display": list(AB.tap_display()), "tap_frame_px": list(AB.tap_frame_px()),
                 "frames": _ABIL["frames"], "states": _ABIL["states"], "ready_decisions": _ABIL["ready_dec"],
                 "wanted": _ABIL["want"], "held": _ABIL["held"], "disabled": ABIL.disabled,
                 "presses": ABIL.rows()}, indent=1))
        if YOLO is not None and YOLO.ms:
            a = np.array(YOLO.ms)
            print(f"yolo side-car round trip: mean {a.mean():.0f} ms, p95 {np.percentile(a, 95):.0f} ms, n={len(a)}")
    sys.exit(code)
