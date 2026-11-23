# Live play

`play_live.py` runs a trained checkpoint in a real match. Since 2026-11-19 it goes through
**our perception stack** by default: it launches `live_ppo/driver.py` (the engine's live loop
with our state estimation hooked in) with the production flag set
`state_pipeline.DEFAULT_FLAGS`, and it records every frame perception saw so the run can be
audited afterwards. The old path, the engine's `mvp_loop.py` alone with its stock readers, is
still there as `--stock`.

## Running it

The game runs on a real iPhone, driven from the Mac through macOS iPhone
Mirroring (or WebDriverAgent). Setup and calibration are in `ios/SETUP.md`. Clash Royale
refuses to run on BlueStacks Air and on the Android emulator: both crash at
launch, which is the game's own emulator check. `--device android` remains
for a real Android phone over adb, with `--stock` only (the driver's readers need the iPhone
frame).

```bash
.venv/bin/python play_live.py            # dry run: perception + decisions, no taps
.venv/bin/python play_live.py --act      # real taps, after a typed confirmation
```

Checklist before `--act`:

1. The iPhone Mirroring window open and showing the game, and
   `ios/geometry.json` calibrated (`calibrate_ios.py --mirror`). Check that
   `runs/live/<ts>/start_frame.png` looks like a normal full battle.
2. The in-game battle deck is exactly Hog Rider, Musketeer, Cannon, Ice
   Golem, Skeletons, Ice Spirit, The Log, Fireball. The hand reader only
   knows these eight (plus the alternate card art of Skeletons, Musketeer and Cannon).
3. Do one dry run during a Training Camp match first and read `live.log`:
   the hand should be read as the cycle, elixir should track, and the `DECIDE` lines should
   name plays that make sense. The last lines give the perception rate and
   `recording: N frames (...)`.
4. Start the battle (Training Camp, or a Friendly Battle with a teammate who
   agreed), then launch with `--act` and confirm the opponent at the prompt.

5. Leave the Mac alone during an `--act` match. Every tap is a mouse click at the iPhone
   Mirroring window's position on screen. Before each tap the driver checks that this window is
   the front one and brings it forward if it is not; if that fails it sends no click and the play
   is counted as not landed. A window dragged over the game, or the mouse in use, still costs
   plays.

Nothing is tapped in a dry run, and with `--act` nothing is tapped until you type `yes` at the
opponent prompt. Ctrl-C stops the run; the launcher gives the driver up to 30 s to print its
summary and close the recording.

The launcher can be started on the menu or mid-battle. Outside a battle the table prints
`not in game` once a second and nothing is decided, but `--seconds` counts from launch, so
start it when the battle is about to begin. A healthy start prints a line beginning
`capture: iPhone Mirroring window` within a few seconds of the `DRY RUN` / `ACTING` banner.

If the log stops at that banner and no `capture:` line follows, the driver is stuck on its
first window capture. On macOS 26.6.2 a process that has captured the window makes the same
call hang in every child it starts afterwards, so the launcher takes its `start.png` in a
helper process that exits before the driver starts (`check_mirror` in `play_live.py`). Any
new wrapper around the driver has to follow the same rule. This cost two live runs on
2026-11-20, which recorded no frames and placed no cards.

Ladder, Classic and Grand Challenges, and any other matchmade mode are
excluded on purpose. Those opponents are real players who didn't agree to
face a bot, and Supercell's ToS bans bots there. A ban would cost the
account.

### Options

| flag | default | what it does |
|---|---|---|
| `--checkpoint P` | `remote_results/ppo_league26_ep138708.pth` (`--stock`: the partner checkpoint) | policy weights; preflight checks deck and architecture for the path |
| `--temp T` | 0.5 | sampling temperature of the driver's policy |
| `--det-weights W` | `data/det/runs/r3/weights/best.pt` | unit detector for the YOLO side-car (see "Detector" below) |
| `--record` / `--no-record` | on | save every perception frame + timestamps (`--record-every N` for every Nth) |
| `--driver-flags=STR` | none | extra `live_ppo/driver.py` flags, e.g. `--driver-flags="--tap-mode drag"`. `--act` is refused here: it only goes through the confirmation |
| `--driver-flags="--no-front-guard"` | guard on | click even when iPhone Mirroring could not be brought to the front (the behaviour before 2026-11-20) |
| `--driver-flags="--allow-empty-slot"` | off | let the policy choose a hand slot whose card has not arrived yet (before 2026-11-20) |
| `--driver-flags="--no-play-watch"` | watch on | no `plays.json`, no landed count, and the engine ledger's own 4 s elixir debit |
| `--driver-flags="--no-own-king-num"` | numeral on | our King Tower's HP from the engine's bar reader alone (before 2026-11-21; see "Screen reading in the second live match") |
| `--driver-flags="--decision-ticks N"` | 0 (not given) | decide every N engine ticks (10 ticks a second). Without it or `--decision-hz` the driver takes the checkpoint's `decision_ticks` (gen 2: 5, so 2 Hz), else `decision_hz`, else 1 Hz as before (`live_ppo/decision_rate.py`; see "Deciding twice a second") |
| `--driver-flags="--ability tap"` | `log` | gen-2 checkpoints: send the Hero ability press (one tap on the button) with `--act`. `log` reads the button, masks the head and draws the press without sending it; `off` is the behaviour before 2026-11-22 (see "Generation-2 checkpoints") |
| `--tactical` | off | driver path: turn the engine's Cannon/Fireball tile advisor and solvency gate back on (see "The tactical layer") |
| `--stock` | off | run `mvp_loop.py` directly: stock perception, no detector, no recording |
| `--replay DIR` | off | no phone: feed recorded frames (below) |
| `--seconds`, `--reserve`, `--no-tactical`, `--ignore-staleness`, `--device`, `--geometry`, `--wda-url` | as before | passed to the loop unchanged |

### Without the phone (replay)

```bash
.venv/bin/python play_live.py --replay runs/live/rec_dry1 --replay-from 650 --replay-realtime --seconds 75
```

Runs the same launcher path (preflight, driver, policy, logs, recording) on a directory of
recorded frames instead of the mirror window. It never taps and refuses `--act`. Without
`--replay-realtime` the frames are fed as fast as perception takes them. Any `raw/` directory of an
earlier run works as `DIR`.

Both modes assume a 4 frames-per-second recording like `rec_dry1`. A live run records about 11
frames per second: replay it with `--driver-flags="--loop-timestamps"` (since 2026-11-21), which
serves each frame at its recorded time from `raw/frames.jsonl` and logs the recorded frame
number. Without it each frame of such a recording is served four times, the match runs several
times slower than it was played, and every timed rule (flight times, latches, the elixir debit)
sees stretched time.

```bash
.venv/bin/python play_live.py --replay runs/live/<ts>/raw --seconds 192 --no-record --driver-flags="--loop-timestamps"
```

## What a run leaves

Everything lands in `runs/live/<timestamp>/`:

| file | content |
|---|---|
| `run.json` | checkpoint + sha256, path (driver / stock), driver flags, temp, detector weights, device, act, the command, exit code, and the recording totals |
| `start.png`, `start_frame.png` | the screen at launch and the cropped frame the bot sees (`start_frame.jpg` in a replay) |
| `live.log` | the driver's full output: `DECIDE` lines, errors, the end-of-run timing summary |
| `placements.jsonl` | one line per issued placement (the loop's own verdicts; see "Did a play land?" below) |
| `plays.json` | one row per issued placement with the verdict to trust: `landed`, `failed` (with the reason) or `unseen` |
| `perception.jsonl` | one line per decision: everything the policy was fed |
| `ability.jsonl`, `ability.json` | gen-2 checkpoints: the Hero ability button as read on every in-game frame, and the presses chosen (sent or not) with the tap pixel and the verdict |
| `overlay/` | one annotated frame per decision |
| `latency.jsonl`, `board_age.csv` | timing; `live_ppo/latency_report.py runs/live/<ts>` reads them |
| `raw/` | **the recording**: one JPEG per perception frame (the exact bytes the detector got), `frames.jsonl` with a timestamp per frame, `record.json`, `manifest.json` |
| `raw.yolo.jsonl` | the boxes the live detector returned for each in-game frame |
| `match_sbs.mp4` | rendered when the run ends, Ctrl-C included (`--no-video` skips): the recorded game at real speed with the detector's boxes on the left, and on the right a top-down board drawn from `perception.jsonl` (units, tower HP, clock, elixir, hand, slot probabilities, the chosen play). `live_ppo/match_video.py runs/live/<ts>` re-renders it |

The last four rows exist on the driver path only. The recording layout is specified in
`live_ppo/RECORDING.md`. Cost: about 145 KB per frame, so roughly 260 MB for a 3-minute match
and 440 MB for a 5-minute one at 10 frames per second; saving is asynchronous and costs the loop
about 1 ms per frame (numbers in RECORDING.md). `runs/` is gitignored; delete `raw/` of runs
you no longer need.

### Did a play land?

The line to read at the end of `live.log` is `placements that landed (the card left its slot):
N/M`, followed by each play that did not land and why. It comes from `live_ppo/play_watch.py`: a
card that is played leaves its hand slot within about half a second, and one that is not played
stays. The same rows are in `plays.json`.

The loop's older lines above it under-count and should not be used to judge a match:
`unit appeared`, `THE GAME REFUSED IT`, and `issued plays never confirmed`. On the first full
live match (`runs/live/20261120-093139`, 45 placements, checked frame by frame) 36 landed. The
loop printed `unit appeared 10/44`, `THE GAME REFUSED IT 19` and `23 issued plays never
confirmed`; 14 of those 23 had landed. The elixir bar drains over two frames, so the drop
matched no single card cost.

What the nine lost plays were, and what changed:

| lost | cause | now |
|---|---|---|
| 4 (Hog Rider, four tries in 13 s) | iPhone Mirroring was not the front window; the engine's check that it was used a stale answer, so the clicks went to another window | the driver checks the window list itself and brings Mirroring forward before each tap, or sends nothing |
| 4 | the policy chose a slot whose next card had not arrived yet (the slot still read empty) | such a slot cannot be chosen (for up to 3 s) |
| 1 | issued as the match ended | none needed |

The elixir fed to the policy also changed: a play is charged until the bar has dropped for it or
until it is known not to have landed (about 1.5 s), instead of for the engine ledger's fixed 4 s.

Not yet run in a real `--act` match: the front-window guard and the new elixir debit. Both were
checked by replaying the match above through the driver (no taps) and by
`live_ppo/eval/test_play_watch.py`; bringing the window forward from a background process was
tested on this Mac.

Our own tower HP: in the same match a Goblin Barrel on our left tower put goblin level badges
over the HP numeral ("3013" read as "313"), and the tower was held as destroyed from 10 s on; it
fell at 99 s. `hud_live.TowerHPFilter` now rejects a reading that is the held value with a digit
missing, undoes a destroyed latch when a numeral comes back and holds, and for our own towers
does not latch while the health bar is still drawn. Replayed on that match's frames the left
tower now reads 3013, 1482 after the first barrel, 1076, 201 after the second, and destroyed at 99 s. Tests:
`live_ppo/eval/test_tower_filter.py`.

### Screen reading in the second live match

`runs/live/20261120-185026` (2026-11-20). Our three towers' real HP was read by eye for every
frame (`live_ppo/eval/labels_own_towers_live_20261120-185026.json`). Three things the policy
was fed were wrong; all three were changed on 2026-11-21:

| fed in the match | cause | now |
|---|---|---|
| our right Princess Tower at 0.07-0.10 from 31 to 37 s (really 1.00 falling to 0.67); our left at 0.04-0.07 from 73 to 77 s and at 147-148 s (really 0.80 to 0.52, then 0.29 to 0.25) | a goblin's red level badge ("11") stood on our HP numeral and was read as digits: 3052 as 312, 2448 as 218, 1000 as 110. After three such reads the filter took the low value | `hud_live.OwnTowerReader` treats a numeral with badge red on it as hidden instead of reading it; while it is hidden `TowerHPFilter` follows the tower's health bar; a numeral far below what the bar shows is not taken |
| our King Tower at 1.00 from 159 to 166 s (really 0.87 falling to 0.69), at 0.22-0.63 from 167 to 172 s (really 0.69), and at 1.00 from 175 s to the end (really 0.56 falling to 0.14, 0.01 on the last frame) | the King's HP came from the engine's bar reader only; its numeral was not read | `hud_live.OwnKing` reads the King's numeral and bar through the same filter once the bar is drawn (`--driver-flags="--no-own-king-num"` gives the old reading) |
| 12 opponent Goblin Barrels for the 6 thrown | a barrel was logged when it left their King Tower and again about 3 s later when its goblins stood on our side | `opp_spells.OppSpellDetector`: the same thrown spell seen again within 4 s of its cast is that throw landing. The play keeps its time and its point moves to where it came down (`FLIGHT_S`; off with `--driver-flags="--no-opp-spell-evidence"`) |

Measured by replaying the recorded frames through the driver at their recorded times (no taps;
163 decisions): the fed value is within 5 % of the tower's real HP at 99.4 % of the decisions for
the left Princess Tower (96.3 % in the match), 98.8 % for the right (95.7 %) and 100 % for the
King (89.0 %); 6 Goblin Barrel plays (12). The same replay with the code as it was gives 95.7 %,
96.9 %, 87.2 % and 12. The three decisions still off are a hit taken under a troop (fed 0.58 for
0.52) and a destroyed tower fed at 0.08 for 1.5 s before it latches. Tests and fixtures:
`live_ppo/eval/test_live_match2.py`.

Looked at and not changed: both of our Cannons that vanished 11 s and 16 s after they were
placed had been destroyed (their health bars run down over the last 3 s); and in the last 30 s
the detector misnamed opponent units in the crowd at our towers (goblin-type units as
Barbarians or Elite Barbarians, some at confidence 0.6-0.8, for seconds), which the tracker
passes on.

### Opponent spells in the air and stacked plays (2026-11-22)

Two more faults of the same match, changed on 2026-11-22:

| fed in the match | cause | now |
|---|---|---|
| a thrown opponent spell's tile 4 s after the throw (a Goblin Barrel lands 2.9-3.2 s after it), and never as a spell in the air | the reader's point followed the sprite down the field, so the tile was withheld until the flight was over | `live_ppo/spell_flight.py` fits the flight's start (a ballistic arc from their King Tower, fitted per spell) and gives the landing point and time; `opp_spells` attaches it to the play once one third of the board (left lane / King Tower / right lane) has 90 % of the probability on at least 3 sightings; `obs_tail` then shows the spell in `OPP_SPELLS` with tile and time to impact |
| 5 of 30 opponent troop plays missed, all a 2nd or 3rd card dropped on a tile whose deploy stopwatch was still running | the stopwatch's face goes back to white and fills again; the reader took it for the same watch | `opp_watch.OppWatches`: a face that falls to at most half the colour it had reached, with at least 45 % white, is a new play there (`OppWatches(restart=False)` is the reader as before) |

Replaying the match's frames through the reader and the tail (`live_ppo/eval/tail_replay.py`):
the four plain Goblin Barrels are in `OPP_SPELLS` 0.26, 0.28, 0.36 and 0.44 s after the throw, on
the right side from the first frame, 0.3-0.7 tiles from where the goblins came down and within
0.13 s of the landing time. The two EVOLVED barrels get no estimate (two barrels drawn as one
widening box that the detector loses after 1.9 s) and behave as before; so do Rocket and Arrows
(no fit: the detector loses a Rocket 1.3-1.4 s into a 3.2 s flight). Troop plays: 28 of 30 found
(25 before), 3 false plays (3 before), 21 of 28 named right (19 of 25), every found play within
0.18 s and half a tile. Still missed: a Fire Spirit dropped on an Ice Spirit's tile with no
visible restart, and one of three cards dropped on one tile within 0.4 s. In the first live match
(no labels) the reader has 32 plays for 30; both new ones are real second stopwatches.

Accuracy of the estimate on every throw of the two live runs and ten training videos, and
(reported only) three held-out videos: `live_ppo/eval/eval_spell_flight.py`. Tests:
`live_ppo/eval/test_spell_flight.py`, `test_opp_watch.py`, `test_live_match2.py`.
`training/obs_noise.py` has the same behaviour for training as the preset `"reader2"`
(`obs_v2_noise: "reader2"`); the default tail noise is still the reader as it was on 2026-11-21.

### Deciding twice a second

Gen-2 checkpoints decide every 5 engine ticks. From the two live runs' own timing files
(`.venv/bin/python live_ppo/decision_rate.py runs/live/20261120-185026 runs/live/20261120-093139 --hz 2`):

| | median | 90th pct | at 2 Hz |
|---|---|---|---|
| frame period = perception per board | 90 ms | 97 ms | 5.6 boards per decision |
| board age at the decision | 79 ms | 88 ms | |
| decision (encode + policy + log; the current net) | 4 ms | 5 ms | none over 500 ms |
| decision to tap finished (two clicks, 381 ms) | 385 ms | 392 ms | 0 of 77 taps still going out at the next decision; shortest margin 99 ms |
| decision to the loop holding a board without the played card | 471 ms | 549 ms | not yet there at the next decision for 40 % of 73 plays (1 % at 1 Hz) |

So the loop keeps up at 2 Hz, with two things to know. A decision right after a play often still
sees the played card in the hand; the elixir is already debited (`play_watch`), but for a gen-1
checkpoint the slot is not masked, so picking the same slot again would tap whichever card has
cycled in (a gen-2 checkpoint masks it: "Generation-2 checkpoints" below). And the loop's
wait for a fresher board (47 ms on average) is added to every interval: 1.05 s between decisions
at 1 Hz, so about 0.55 s (5.5 ticks) at 2 Hz unless the wait cap is lowered.

### The tactical layer

The engine's live loop can lay two rules over a policy: an advisor that picks the tile for
Cannon and Fireball, and a solvency gate that refuses spends leaving less than 4 elixir while
nothing is attacking. They were written for the partner's stock net, and `--stock` keeps them.
The driver's checkpoint was trained and evaluated without them, and since 2026-11-20 the driver
path runs without them (`--tactical` turns them back on).

Why: in the first live match every Cannon tile came from the advisor, which goes for the cell
with the most approaching enemy HP in range. It put two of the five Cannons that landed on the
river row beside the bridge, (11,15) and (13,15), and two level with the left Princess Tower, (6,6).
Replaying the same match with the advisor off, the net put all eight of its Cannons in the
middle, columns 8-11, rows 9-13. In the simulator (`training/advisor_ab.py`, 600 seeded matches
per arm against the mirror-deck teacher rungs 0-10, same seeds, temperature 0.5; rows in
`evals/advisor_ab_20261120.json`):

| arm | win rate | matches won that the net alone lost / lost that it won | tower damage taken |
|---|---|---|---|
| net alone | 0.613 | | 1.55 |
| advisor Cannon tile | 0.507 | 37 / 101 (sign test p < 0.001) | 1.77 |
| solvency gate | 0.585 | 79 / 96 (p = 0.23) | 1.50 |
| both, plus advisor Fireball aim (the old live default) | 0.545 | 69 / 110 (p = 0.003) | 1.66 |

The advisor's Cannon tile loses at every rung. The gate is mixed: worse against rungs 2 and 4
(0.89 and 0.67 against 0.99 and 0.89), better against rungs 8 and 10 (0.27 and 0.39 against
0.18 and 0.31), no clear difference overall. The engine switches both with one flag, so the gate
alone is not available from the launcher. Not measured: the real-deck opponents (the local
engine build lacks some of their cards) and a live match without the layer.

### Handing a run to the audit tool

The offline audit (`live_ppo/eval/match_audit.py`, its docstring has the full workflow) takes
the run directory, re-runs the detector and tracker on the saved frames at their true
timestamps, and writes to `runs/match_audit/<name>/`:

```bash
P=.venv/bin/python ; M=live_ppo/eval/match_audit.py
$P $M detect runs/live/<ts> --name <name>                         # re-detect every frame (its default weights)
$P $M detect runs/live/<ts> --name <name> --weights <W.pt>        # ... with another detector
$P $M detect runs/live/<ts> --name <name> --from-cache runs/live/<ts>/raw.yolo.jsonl   # the live boxes, no detector
$P $M flicker <name>                                              # then sample / stats / tracker-report
```

To replay the run through the whole launcher again:
`play_live.py --replay runs/live/<ts>/raw --driver-flags="--loop-timestamps"` (see "Without the
phone").

## The checkpoints

**Driver path (default): `remote_results/ppo_league26_ep138708.pth`** (gitignored; sha256
`544b954a…`; league PPO run, episode 138,708). It is the round-robin winner among the
checkpoints on this Mac and the one `training/REPORT_netv2_ppo.md` recommends for live play,
at `--temp 0.5`, which is why the launcher passes that temperature by default. The driver
needs a checkpoint trained on the `ppo_bc26` deck (2.6 Hog Cycle with the Evolution / Hero card
ids, `TRAIN_DECK` in `live_ppo/driver.py`) with the one-ability-slot net (v1 or net_v2);
preflight checks exactly that. Also accepted: `remote_results/ppo_bc26_final.pth`,
`remote_results/v2_league_a.pth` (net_v2).

### Generation-2 checkpoints (NetV3, since 2026-11-22)

A generation-2 checkpoint (`arch: v3`, `training/net_v3.py`, e.g.
`runs/bc3/g2_big_L0_H10/ep4.pth`) runs on the driver path with the generation-2 engine, which
is a separate built copy chosen by the `CR_ENGINE` environment variable:

```bash
CR_ENGINE=build/gen2/engine .venv/bin/python play_live.py --checkpoint runs/bc3/g2_big_L0_H10/ep4.pth          # dry run
CR_ENGINE=build/gen2/engine .venv/bin/python play_live.py --checkpoint runs/bc3/g2_big_L0_H10/ep4.pth --act    # real taps, after the typed confirmation
```

Without `CR_ENGINE` preflight refuses it and prints this command; a gen-1 checkpoint is refused
the other way round (it does not fit the gen-2 net sizes). Nothing changes for the default
checkpoint: with `CR_ENGINE` unset the launcher's command is what it was.

What the driver takes from the checkpoint instead of from flags (`live_ppo/policy_v3.py` has
the rules, `live_ppo/eval/test_policy_v3.py` the tests):

| | gen-1 (v1, net_v2) | gen-2 (v3) |
|---|---|---|
| observation | engine's | engine's 14,459 + the 148-float tail = 14,607; tail on, opponent names as the checkpoint's `names` (`--obs-v2 off` or other `--obs-v2-names` are refused) |
| opponent-state section (418 floats) | not in the engine | filled: `--opp-state --opp-plays --opp-spells` come on by themselves (`--no-opp-state` leaves it zero, for comparisons) |
| decisions | 1 per second | `decision_ticks` 5 = 2 per second |
| temperature | `--temp 0.5` | `--temp 1.0` = WHAT and WHERE, set inside the net; `--when-temp` (default 1) and `--when-bias` (default 0) shape the WHEN coin, p(play now) = sigmoid((w + bias) / temp). These are the settings of its simulator evaluation |
| `--greedy` | argmax of both heads | argmax of WHAT and WHERE; WHEN stays a coin at its trained probability (an argmax there would mean never playing while p < 0.5, which is nearly always, or always playing) |
| tactical officer, solvency gate | off (launcher), `--tactical` turns both on | off; `--tactical` passes the driver's `--tactical-officer --solvency-gate` |
| leak floor (no-op masked on a full bar) | 10 | off; `--leak-floor N` forces the WHEN coin to "play" at N elixir |
| a play still in flight | elixir debited only | its slot cannot be chosen and its cost is off the elixir the net sees, as the simulator's actuation delay shows it (`--pending-mask off` to drop this) |
| deck | checkpoint's `deck` key | v3 checkpoints have no `deck` key: the training code's `training/common.HOG26_DECK` stands in and must equal the driver's `TRAIN_DECK` |

The three gen-1 crutches are off because this net was cloned from a teacher that plays without
them and its WHEN head already decides when to hold; each has its flag for an A/B.

A play in flight: from the moment a placement is issued until `play_watch` has seen the card
leave its slot (about 0.5 s) or given it up (1.2 s after the tap ended), that hand slot is
masked and its cost is taken off the elixir scalar. At 2 decisions a second the next decision
usually falls inside that window (52 of 314 decisions in the replay below), and without the
mask it could pick the same slot again and tap whichever card had cycled in.

The Hero ability (since 2026-11-22; `--driver-flags="--ability off|log|tap"`, default `log`,
v3 checkpoints with the Hog 2.6 deck only; any other checkpoint ignores the flag):

| `--ability` | button read | observation and mask | the press |
|---|---|---|---|
| `log` (default) | every in-game frame | from the reader | logged and drawn, **never sent**, also with `--act` |
| `tap` | the same | the same | one tap on the button, and only when the loop runs with `--act` |
| `off` | no | the estimate used before the reader existed | the head's draw is logged only |

* **The reader** (`live_ppo/ability_button.py`). While our Hero Musketeer is alive the game draws
  a round button above the right end of the card tray (centre at frame pixel (472, 759.5),
  radius about 31, an elixir badge reading 3). The reader calls it `absent`, `charging` (grey:
  less than 3 elixir), `ready` (light blue, magenta badge), `used` (navy) or `unknown`. `ready`
  needs the pixels to match the recorded ready button AND the colours to be right, on two
  frames in a row no more than 0.5 s apart; anything else is not ready. `--ability log` in a
  replay: 0 frames read as ready that are not ready by hand label, on 1,364 and 1,530 frames
  of the two recorded matches (table below).
* **The mask** is what the simulator's `is_champion_ability_ready` is: button ready, the elixir
  the policy is shown at least the engine's `ability_cost` (3), and no press of ours open or
  already counted. The same value fills the observation's "ready" field.
* **The tap** (`live_ppo/ability_tap.py`) goes through the card taps' own worker, so it can
  never fall between a placement's two clicks. It is ONE click at display (619, 996) = frame
  pixel (472, 759), the button's centre: 41 px above the card band, clear of the emote button.
  It is sent only if, at the moment of sending, the loop is in a battle, the newest frame is
  in game, the button is still read as ready and no card placement is in flight. A press
  waits a decision when a card is chosen on the same decision.
* **After the tap** (`live_ppo/ability_watch.py`): the 3 elixir is taken off what the policy
  is shown until the bar shows the drop; the press is `confirmed` when the button leaves
  ready and the bar dropped (or it shows `used`), `failed` after 2 s. Two presses in a row
  that did not register turn ability taps off for the rest of the run.
* **A dry run shows where it would tap**: every overlay with the button on screen carries a
  yellow cross and ring at the tap point (it must sit on the button), the pressing decision a
  red ring and `ABILITY PRESS #n (not sent)`; `live.log` has one `ABILITY press` line per
  press with the pixel, `ability.jsonl` one line per frame read, `ability.json` the presses.
* **Window size**: the templates are at the frame scale of the usual iPhone Mirroring window
  (326 x 720 points, every recorded run). At another window size the button reads `unknown`
  and the ability is simply not pressed; the state counts at the end of `live.log` show it.
* **Only this phone's picture reads as ready.** The ready test is a pixel match to the button
  as this phone draws it through iPhone Mirroring (steady ready: 1-2 grey levels from the
  template, limit 6). Footage of the same button from other phones (YouTube videos) sits at
  17-19 and reads `unknown`: 0 `ready` on 12,531 ready-looking frames of three held-out videos
  and 0 on 4,096 of the training video `4s1o1sk-hPM`. Never a wrong ready on any of them
  (43,733 held-out video frames and the 957 frames of `rec_dry1`), and never a right one.
* **The pressed look** (`used`: navy ring, grey badge) has never been recorded on this phone.
  Its colour rule comes from the training video `4s1o1sk-hPM` (383 navy frames, all read
  `used`; our two recorded matches, where nothing was pressed: no `used` frame).

Measured without the phone on the recording of the second live match (`runs/live/20261120-185026`,
a gen-1 match, so the hand and the elixir are what gen-1's plays left; detector boxes from the
recording's `raw.yolo.jsonl`, no side-car):

```bash
CR_ENGINE=build/gen2/engine CR_YOLO_CACHE=runs/live/20261120-185026/raw.yolo.jsonl \
  .venv/bin/python play_live.py --checkpoint runs/bc3/g2_big_L0_H10/ep4.pth \
  --replay runs/live/20261120-185026/raw --seconds 192 --no-record --driver-flags="--loop-timestamps"
```

| | this replay (`runs/live/20261122-174340`) | the same checkpoint in the simulator (`evals/ghost_g2_big_L0_ep4_t1.0_e884_20261122.json`, 1000 matches) |
|---|---|---|
| observation length | 14,607 on all 314 logged decisions | 14,607 |
| decisions per second | 1.81 (347 in 192 s; none over 1 s; the loop's wait for a fresh board adds about 45 ms to each interval) | 2.0 |
| plays requested per minute | 7.7 (22 in 171 s of match) | 10.3 |
| mean elixir at a play request | 4.9 (the recording's bar averages 4.0: gen-1 was spending it) | 7.2 |
| cards | The Log 6, Hog Rider 5, Skeletons 4, Ice Golem 3, Fireball 2, Ice Spirit 1, Cannon 1; Musketeer never | all eight |
| decisions held | 93 % | about 91 % (10.3 plays in 120 decisions) |

A replay cannot close the loop: none of the requested plays happens, so the hand does not cycle
and the bar does not drop for them. The play rate and the card mix here say the policy runs and
plays, not how it plays a match of its own.

The ability in the same kind of replay (2026-11-22, `--ability log`, both recorded matches; the
hand labels are `live_ppo/eval/labels_ability_button.json`, every frame of every window in
which the button is on screen):

| | `20261120-185026` (the reader's templates were cut from it) | `20261120-093139` (no frame of it used) |
|---|---|---|
| in-game frames read (6 ms each) | 1,364: absent 1,099, charging 101, ready 111, unknown 53 | 1,530: absent 1,080, charging 246, ready 172, unknown 32 |
| read as ready (two-frame rule) but not ready by label | 0 | 0 |
| frames ready by label -> ready by the rule | 107 of 127 | 163 of 182 |
| decisions with the observation's "ready" = 1 | 20 of 302, all 20 on a frame labelled ready | 32 of 315, all 32 on a frame labelled ready |
| p(press) on those decisions | median 0.005, highest 0.016 | median 0.002, highest 0.018 |
| presses the policy drew | 1 (frame 1312, ready by label), drawn on the button | 1 (frame 1205, ready by label), drawn on the button |

The misses are the first frames of each ready spell (the two-frame rule, and the white flash
as the button appears, which reads `unknown`). Timing against the simulator, measured on the
six Musketeer placements of `20261120-185026`: the game draws the button 1.5-1.8 s after the
placement is issued (fully drawn at 2.1-2.3 s), the reader has it at 2.1-2.7 s and calls it
ready at 2.8-3.3 s. The simulator reports deployed and ready on the tick after the play, so
the policy sees "ready" about 3 s later live than it did in training, about 2 s of which is
the game itself.

The opponent's ability fields are what they were in training: the engine's simulated
detector never reports an ability (`OppDetectionNoise.ability_detected = 0`), so the policy
only ever saw "this Champion was played" and its ability cost, with cooldown and recency at
0. Live fills the same two fields when an opponent Champion or Hero play is identified and
leaves the other two at 0; nothing reads an opponent ability being used. Measured once on the
held-out recording `rec_dry1` (opponent Archer Queen, played twice; `state_pipeline.py` with
the gen-2 engine and the round-3 detector's boxes): the two opponent slots equal what the
training simulator would show for the hand-labelled plays (all zero before the Queen's first
play, `[1, 0, 0.1, 0 | 0, 0, 0, 0]` from it on) on 404 of the 410 match frames; the 6 frames
that differ are the 1.5 s between
the Queen's deploy stopwatch appearing and the play being emitted. Whether she used her ability
in that match was not labelled and is not in the observation, live or in training.

**`--stock`: `checkpoints/partner/model_weights_ep76728.pth`** (gitignored; sha256
`69a25f0a…2ba3877`). Trained on the Windows box under
`_runs/resume_ep18046_3000`: phase 1 `mirror`, curriculum `curriculum_stage`
3 (the "stage 4" you see counting from 1), 9 demotions, deck = 2.6 Hog Cycle with the base
card ids. Same architecture as our pinned engine (`2f17292`); it loads with no missing
or extra tensors, both in `play_match.py` and in the live loop's `NeuralPolicy`. The driver
rejects it (different deck ids, no ability head), and the stock path rejects the driver's
checkpoints; preflight says which flag to use.

Partner checkpoint in the simulator (2026-11-17, 30 seeded games each, seed 1000, greedy):

| opponent | win rate | crowns for / against |
|---|---|---|
| random | 1.00 | 3.00 / 0.03 |
| scripted Rusher / Cycler | 1.00 / 1.00 | 3.00 / 0.17, 2.90 / 0.00 |
| scripted Defender / Counter | 0.97 / 0.97 | 2.30 / 0.10, 2.67 / 0.23 |
| C++ heuristic | 1.00 | 2.53 / 0.43 |
| teacher rung 0 / 3 | 1.00 / 0.97 | 2.23 / 0.67, 2.57 / 1.07 |
| **teacher rung 6** | **0.23** | 1.30 / 1.87 |
| **teacher rung 10** | **0.23** | 1.33 / 2.20 |

Raw rows in `evals/partner_vs_*.json`. It beats every rules-based opponent
and loses about 3 in 4 to the lookahead teacher from 5 s up. Expect the real game to be harder
than any of these: the engine's known gaps (no card levels, spells at full tower damage,
1 decision/s) are in README.md.

## Detector

`--det-weights` picks the unit detector the side-car loads. The candidates:

* **r3** (default since 2026-11-20): `data/det/runs/r3/weights/best.pt`, a copy of
  `runs/det_r3_pod/train/weights/last.pt` (198 classes). Round 3: B fine-tuned on synthetic
  frames with crowded scenes and deploy stopwatches, the KataCR human-labelled frames and
  track-labelled real frames.
* **v2**: `data/det/runs/v2/weights/best.pt` (191 classes), the default until 2026-11-20.
* **B**: `runs/det_v3_pod/ab/B/weights/last.pt` (198 classes), the round-2 reference.
* **r4**: `runs/det_r4_pod/train/weights/last.pt`, round 3 fine-tuned further on more real
  frames. Not the default: see below.

### Why r3 is the default (2026-11-20)

Units, through the tracker with the driver's settings (`training/det_track_check.py`; strict
F1 counts a unit only when it is found with the right card and side, and falls with false
tracks; inference at 896 px, the side-car's size):

| | v2 | B | r3 | r4 |
|---|---|---|---|---|
| held-out videos, 99 frames / 966 units: strict F1 | 0.791 | 0.805 | 0.823 | 0.826 |
| &nbsp;&nbsp;units found / false tracks | 821 / 59 | 866 / 42 | 894 / 36 | 904 / 43 |
| fresh videos, 120 frames / 806 units: strict F1 | 0.681 | 0.783 | 0.806 | 0.811 |
| &nbsp;&nbsp;units found / false tracks | 597 / 36 | 709 / 40 | 730 / 37 | 738 / 34 |

r3 against v2, the old default, is +0.032 held-out and +0.125 on the fresh videos (P(gain) 1.00
each; the fresh labels were pre-filled from B's boxes, which favours B's family over v2 there).
r3 against B is a gain on both sets (+0.018 and +0.023, bootstrap P(gain) 0.99 each). r4 against
r3 is not (+0.002 and +0.005, P 0.66 and 0.73): it finds a few more units, with more false
tracks on the held-out set and slightly fewer on the fresh one.
The fresh videos (`LekrcVeQjiw`, `kaLMeKpsuGY`) were never used to train, tune or pick; their
labels are agent-made and still need a human check (`runs/match_audit/fresh_test/check/CHECK.md`).
Inference at 1024 px gave r3 nothing over 896 (held-out 0.821).

Opponent plays with the evidence rules on, the driver's configuration (current table for all
four detectors: `runs/opp_plays_ab/table.md`):

| held-out pair, 825 plays | v2 | B | r3 | r4 |
|---|---|---|---|---|
| false plays (of them spell casts) | 14 (3) | 26 (12) | 18 (6) | 27 (10) |
| spell casts found, of 25 | 19 | 23 | 23 | 23 |
| card correct, of matched plays with a known card | 33/44 | 37/50 | 40/48 | 38/48 |
| form (base / evo / hero) correct | 41/44 | 41/50 | 39/48 | 39/48 |
| opponent elixir, mean abs error | 3.70 | 4.44 | 3.19 | 3.94 |

So r3 reads units better than v2 and B and no worse than r4, and reads opponent plays better
than B or r4. Against v2 on
opponent plays it is a trade: four more false plays in 825 (three of them spell casts) and more
base units read as Evolution or Hero forms, for four more real casts found, more cards named
right and a better elixir estimate.

End to end: `play_live.py --replay runs/live/rec_dry1 --replay-from 650 --replay-realtime
--seconds 75` runs clean with r3; the side-car round trip is 31 ms mean, 47 ms p95 on this Mac.
First live match through the mirror with it: `runs/live/20261120-093139` (2026-11-20), see
"Did a play land?" above.

### The earlier v2 / B comparison (2026-11-19)

Before round 3 the launcher default was decided on opponent-play reading, the part of the state that feeds
the opponent's hand, cycle and elixir. Both detectors were run over the four labelled sources
(`rec_dry1`, `vid_QD_1suWFNKo`, `vid_iVgmnDX-GPY`, `vid_uVf1EUszTIM`; 1,226 labelled plays,
about 80 with a known card) through the driver's own JPEG path at the side-car's 0.1
threshold, and scored with `live_ppo/eval/eval_opp_plays.py` in the driver's configuration
(`--watch`, detector cut 0.35, tolerances 1.0 s / 1.5 tiles):

| all four sources | v2 | B |
|---|---|---|
| play recall | 0.895 (1097/1226) | 0.897 (1100/1226) |
| play precision | **0.968** (1097/1133) | 0.939 (1100/1171) |
| false plays | **36** | 71 |
| card correct, of matched plays with a known card | 0.823 (65/79) | 0.807 (67/83) |
| form (base / evo / hero) correct | **0.962** (76/79) | 0.831 (69/83) |
| strict (time, tile and card all right) | 0.879 (1078/1226) | 0.876 (1074/1226) |
| spell recall | 0.880 (22/25) | 0.920 (23/25) |
| spell precision | **0.880** (22/25) | 0.434 (23/53) |
| opponent elixir, mean abs error | **2.66** | 3.04 |

Held-out pair only (`vid_uVf1EUszTIM`, `vid_iVgmnDX-GPY`): recall 0.947 / 0.952, precision
0.969 / 0.937, form 0.936 / 0.784, spell precision 0.88 / 0.45, elixir error 3.81 / 4.52
(v2 / B).

**B was worse here, so the default stayed v2.** Recall is the same; B invents twice as many
plays, 30 of them spell casts that never happened (Earthquake 6, Freeze 5, Goblin Barrel,
Arrows, Barbarian Barrel, Poison and Tornado 3 each), and it reads base units as Evolution or
Hero forms. Raising B's cut to 0.5 or 0.65 does not close it (precision 0.939 / 0.953, spell
precision 23/52 and 24/44). With the stopwatch reader off (not how the driver runs) B has the
higher recall, 0.650 against 0.546, at precision 0.734 against 0.752.

To run a match with B anyway (it is the better unit detector, so a comparison run is useful):

```bash
.venv/bin/python play_live.py --det-weights runs/det_v3_pod/ab/B/weights/last.pt
```

The table above is the readers as they were before 2026-11-19. Since then they carry evidence
rules (spell casts: `opp_spells.OppSpellDetector`; forms and YOLO-only plays: `opp_plays.py`;
constants and reasons at the top of each class; `--no-opp-evidence` gives the readers above,
`--no-opp-spell-evidence` switches the spell rules off alone). With them, on all four sources,
B makes 34 false plays instead of 71 (12 false casts instead of 30) and its play precision is
0.970 against v2's 0.980, at the same recall. B's spell precision is still lower, 23/35 against
v2's 19/22, and under the rules v2 loses three real casts of 25 on the held-out matches (two
Poison/Graveyard seen in two frames only, one Poison thrown onto a Graveyard). Current numbers,
development and held-out apart: `runs/opp_plays_ab/table.md`.

Since 2026-11-21 a thrown spell is also one play from launch to landing (`FLIGHT_S`, see "Screen
reading in the second live match"). On the labelled sources' saved boxes, `eval_spells.py` goes
from 56 detected casts to 52 with the same 50 of 51 labelled casts found (two false Goblin
Barrels and two false Fireballs gone), and `eval_opp_plays.py` prints the same numbers as before,
with and without `--watch`. `runs/opp_plays_ab/table.md` was not rebuilt.

If a retrained detector is to replace r3, repeat both measurements before changing
`DEFAULT_DET` in `play_live.py` (and `DEFAULT_WEIGHTS` in `live_ppo/eval/match_audit.py`):
`training/det_track_check.py NEW.pt --vs data/det/runs/r3/weights/best.pt` for units, and for
opponent plays one command builds the box caches for any weights and prints the whole table,
old and new rules side by side (`--rules all` adds the troop rules without the spell rules):

```bash
live_ppo/eval/opp_plays_ab.sh data/det/runs/v2/weights/best.pt <new weights.pt> [more ...]
```
