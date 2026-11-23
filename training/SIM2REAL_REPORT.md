# Sim-to-real randomization for gen-2 training (preset `v1`)

Written 2026-11-22. Code: `training/sim2real.py` (the layer), `training/priv_critic.py`
(privileged critic). Wired into `training/pool_env.py`, `training/ppo_pool.py` and
`training/eval_ghost.py`. Everything is opt-in: without a `--sim2real...` or
`--priv-critic` flag the trainer and the eval run exactly as before.

Recorded data only was used. Nothing here was measured on the phone during this work.

## 1. What is randomized

Every episode draws a strength `s = scale x U(scale_range)` and its own systematic
parameters ("this match the digit ticks late, this reader is poor"). Inside the episode,
errors are events with a duration in engine ticks, so they are time-correlated.

- `s = 0` is exactly the clean environment (checked on 1,000 eval matches: identical rows).
- Structural differences (integer elixir, whole-second clock, opponent tracking from
  detected plays) are on with probability `min(1, s)` per episode.
- Rates and durations scale linearly with `s`.
- Each row below has its own switch: `--sim2real-only a,b` / `--sim2real-off a,b`.

Sources: **A** = `runs/live/20261120-185026`, **B** = `runs/live/20261120-093139`,
**C** = `runs/live/20261122-174340` (driver logs `latency.jsonl`, `perception.jsonl`,
`plays.json`); **P** = `runs/perception/STATE_ACCURACY_REPORT.md` (labelled reader
accuracy on the perception test recordings; used for measurement only).

| Flag | What it models | Drawn | Distribution at `s = 1` | Measurement behind it |
|---|---|---|---|---|
| `elixir` | The driver feeds the HUD digit, an integer, not the fractional bar. | match + step | Digit = floor(true + offset), offset U(-0.25, 0.05) per match. After each own play, with per-match probability U(0, 0.3), the digit is off by one (-1 with share 0.8) for 0.3-1.0 s. Misread 0.05 % per decision. | A, C: integer on 476 of 476 decisions. A, 1,927 frames against a reconstructed bar: equal to floor 79 %, one below 14.6 %, one above 3.6 %, mean(read - true) -0.56; mismatch 41 % of frames within 1 s of an own play, 15 % elsewhere. The reconstructed truth drifts, so the mismatch shares are upper bounds. Digit accuracy on labelled frames 99.95 % (P). |
| `refill` | The hand slot stays empty after a card leaves it. | match + play | U(1.2, 1.6) s per match, +-0.1 s per play. | A: median 1.37 s, p5-p95 1.25-1.55, n = 32. B: median 1.34 s, n = 41. |
| `hand` | A slot read as empty or as another card for a few frames. | match + step | 0.4 % per slot per decision x logU(1/3, 3) per match; lasts 0.5 s x Geometric(mean 2); half blank, half another card of our deck. | P: slot accuracy 99.13 % [98.96, 99.29] on human play. About half of those errors come from card drags the bot never makes (judgment), hence 0.4 %. |
| `clock` | The clock is read in whole seconds and can be off. | match | Whole seconds; offset U(-1, +1) s per match. | A, B: fed in whole seconds on 91.4 % / 93.0 % of decisions (extrapolated on the rest); spread of (fed - frame time) 1.07 / 1.10 s. P: within 1 s on 99.68 % of labelled frames. |
| `towers` | A tower numeral held at its last value while unreadable. | match + step | 0.5 % per tower per decision x logU(1/3, 3), x4 for our right Princess Tower; held 0.5 s x Geometric(mean 3). | P: numerals right on 99.28-99.77 % of frames, our right Princess 97.76 %. "HP went up" 9 times in 652 tower-decisions in A (old driver), 0 in 1,252 in C. |
| `opptrack` | Opponent cycle block and opponent spend come from detected plays, late and incomplete, not from the truth. | match | Built from the engine's simulated detector reports (the same ones that already feed the opponent-state block). | Structural: this is how the live driver builds them. Live emission latency: median 1.52 s (n = 41, B) and 1.54 s (n = 37, A); simulated detector: median 1.50 s. |
| `oppdet` | How good that detector is this match. | match | Each rate uniform from the engine default up to a top value: miss 0.03-0.04 to 0.25, identity unknown 0.02 to 0.40, false plays 0.02 to 0.13, late 0.05 to 0.15. | P, 89 gold plays: miss 0.146 [0.09, 0.23], identity wrong 0.213 [0.12, 0.35] (37 of 47 right), false 0.073 [0.03, 0.15]. Ranges are centred on these. "Late" is not measured. |
| `tapfail` | A tap that does not register: nothing is played, the driver shows the card in flight for about 1.6 s. | match + play | Share of taps that fail U(0, 6 %) per match; P(fail after a fail) = 0.5 (bursts). | Current driver: 0 of 32 (95 % upper bound 9 %). Old driver: 9 of 45, four in a row. Not measurable more tightly from recordings. |
| `place` | A placement lands on a neighbouring tile. | match + play | U(0, 5 %) of plays per match, one tile in a random direction, only if legal. | n = 26 plays, 12 stationary troops: 10 on the commanded tile, 2 doubtful. No tile error evidenced; conservative. |
| `timing` | The driver decides about every 0.55 s, not 0.5 s. | match + step | Extra U(0.03, 0.08) s per decision per match, jittered per step, rounded to engine ticks. | C, n = 312 intervals: median 0.550 s, mean 0.548, p10/p90 0.52/0.58, max 0.60. |
| `hold` | A decision skipped because the board frame was stale. | match + step | U(0, 0.4 %) of decisions per match; P(next one skipped too) 0.5, at most 6. | 0 of 645 recorded decisions (95 % upper bound 0.46 %). Not evidenced; conservative. |
| `levels` | Opponent units one level above or below what the observation says. | match | 30 % of matches, +1 or -1 with equal chance. | Not measurable from recordings (friendly play is level 11 both sides). This is the engine's only per-episode handle on unit-stat error; see section 6. |

Already in the trainer before this work and left alone: board noise on the unit planes
(`training/obs_noise.py`, measured), tail noise on the recent-plays block, and the
0.3-0.6 s actuation delay (`--act-delay 3,6,1`).

## 2. Implementation

- The layer is a mixin over `GhostEnv` / `PoolEnv`, used only when the env config has a
  `sim2real` key. It changes the actor's observation and its action. The reward, the
  shaping terms, the opponent and `info["priv"]` keep seeing the true state.
- It reads from the engine only what the live driver has (`ENGINE_READS`, held by a test).
- Everything it may edit is one contiguous stretch of the observation
  (`sim2real.clean_span()`, 1,189 floats: elixir, slot costs, hand one-hots, time /
  spend / tower scalars, cycle block, level entries).
- Trainer and eval flags: `--sim2real v1` (or a JSON path), `--sim2real-scale X` or
  `lo:hi` (per-episode range), `--sim2real-only`, `--sim2real-off`. Trainer only:
  `--sim2real-widen N` (the upper end of the range grows linearly from `lo` to `hi` over
  the run's first N episodes), `--sim2real-anchor-clean`, `--priv-critic`,
  `--priv-critic-lr`.
- `--sim2real-anchor-clean`: the trainer's KL anchor compares the learner with the frozen
  clone. With the layer on, the clone would be evaluated on readings it never trained on,
  and the anchor would pull the learner toward that degraded reaction. With this flag the
  clone is given the true values of the layer's span instead (a privileged teacher); the
  learner still sees only the readings.
- `--priv-critic`: `V = V_net(obs) + g(priv[:18])`, where `priv[:18]` is the true elixir
  of both sides, the opponent's hand and next card and the six true tower health
  fractions. `g` is a separate small network with a zero-initialised last layer and its
  own optimizer. The policy net, its inputs and its state dict are untouched. Judged a
  small, safe change and implemented.

Tests (all pass): `training/tests/test_sim2real.py` (37) and
`training/tests/test_sim2real_trainer.py` (4). They cover: every randomization exactly
off at scale 0; the clean observation returned as the same object; each randomization
touches only its own entries; seeds reproduce; reward and privileged vector unchanged;
the engine read list; with `--priv-critic` the rollout buffer (policy inputs and actions)
and the first policy update are identical to a run without it; the clean-anchor rows are
aligned and reach only the reference policy. Existing files re-run after the edits:
`test_ppo_pool.py` (8), `test_pool_env.py` and `test_eval_ghost.py` (33), all pass.

## 3. Validation

Policy: the clone `runs/bc3/g2_big_L0_H10/ep4.pth`, `--temp 1.0 --when-bias 0.4
--act-delay 3,6,1`, 1,000 matches against held-out human replays per condition, the same
1,000 jobs in every condition. Differences are paired against the clean run; "±" is one
standard error clustered on the replay. A single win rate has se ≈ 2.0 pp at n = 1,000.
Leak is elixir leaked per match as `reward.py` counts it. Files: `evals/s2r/ep4_*.json`.

### 3.1 The clone (never trained with the layer)

| Condition | Win % | Δ win, pp | Crowns scored (Δ) | Crowns conceded (Δ) | Plays/min | Leak (Δ) |
|---|---|---|---|---|---|---|
| clean | 47.4 ± 2.1 | | 1.377 | 1.698 | 11.19 | 1.21 |
| preset 0x | 47.4 | 0 (identical rows) | 1.377 | 1.698 | 11.19 | 1.21 |
| preset 0.5x | 42.1 | -5.3 ± 1.8 | 1.355 (-0.02 ± 0.04) | 1.812 (+0.11 ± 0.04) | 10.67 | 2.16 (+0.95 ± 0.12) |
| **preset 1x** | 37.6 | **-9.8 ± 1.7** | 1.361 (-0.02 ± 0.04) | 1.982 (+0.28 ± 0.04) | 9.89 | 3.57 (+2.36 ± 0.15) |
| preset 2x | 30.4 | -17.0 ± 1.9 | 1.186 (-0.19 ± 0.04) | 2.170 (+0.47 ± 0.04) | 9.22 | 5.37 (+4.15 ± 0.20) |
| preset, scale drawn 0-1.5 | 40.6 | -6.8 ± 1.7 | 1.352 (-0.03 ± 0.04) | 1.868 (+0.17 ± 0.04) | 10.35 | 2.63 (+1.42 ± 0.14) |
| preset 1x + board noise | 36.4 | -11.0 ± 1.8 | 1.341 (-0.04 ± 0.04) | 2.044 (+0.35 ± 0.04) | 9.86 | 3.61 (+2.40 ± 0.16) |
| only `elixir` | 42.0 | **-5.4 ± 1.5** | 1.340 (-0.04 ± 0.03) | 1.779 (+0.08 ± 0.03) | 10.83 | 2.29 (+1.08 ± 0.15) |
| only `opptrack` | 43.9 | -3.5 ± 1.6 | 1.432 (+0.06 ± 0.03) | 1.818 (+0.12 ± 0.03) | 10.78 | 1.44 (+0.22 ± 0.13) |
| only `timing` | 44.2 | -3.2 ± 1.7 | 1.351 (-0.03 ± 0.03) | 1.741 (+0.04 ± 0.04) | 10.96 | 1.63 (+0.42 ± 0.12) |
| only `tapfail` | 45.0 | -2.4 ± 0.9 | 1.371 (-0.01 ± 0.02) | 1.750 (+0.05 ± 0.02) | 11.07 | 1.60 (+0.39 ± 0.09) |
| only `refill` | 45.5 | -1.9 ± 1.4 | 1.372 (-0.01 ± 0.03) | 1.725 (+0.03 ± 0.03) | 11.12 | 1.19 (-0.02 ± 0.10) |
| only `hand` | 46.3 | -1.1 ± 1.0 | 1.382 (+0.01 ± 0.02) | 1.714 (+0.02 ± 0.02) | 11.19 | 1.25 (+0.03 ± 0.05) |
| only `place` | 46.8 | -0.6 ± 0.6 | 1.387 (+0.01 ± 0.01) | 1.714 (+0.02 ± 0.01) | 11.20 | 1.22 (+0.01 ± 0.05) |
| only `oppdet` | 46.9 | -0.5 ± 1.4 | 1.401 (+0.02 ± 0.03) | 1.662 (-0.04 ± 0.03) | 11.18 | 1.26 (+0.05 ± 0.10) |
| only `clock` | 47.1 | -0.3 ± 0.3 | 1.374 (0.00 ± 0.01) | 1.704 (+0.01 ± 0.01) | 11.18 | 1.24 (+0.03 ± 0.03) |
| only `hold` | 47.4 | 0.0 ± 0.9 | 1.380 (0.00 ± 0.02) | 1.707 (+0.01 ± 0.02) | 11.12 | 1.24 (+0.02 ± 0.06) |
| only `towers` | 47.4 | 0 (identical rows) | 1.377 | 1.698 | 11.19 | 1.21 |
| only `levels` | 48.3 | +0.9 ± 1.0 | 1.414 (+0.04 ± 0.02) | 1.659 (-0.04 ± 0.02) | 11.09 | 1.19 (-0.02 ± 0.06) |

Layer statistics at 1x: 3.3 % of taps failed, 2.1 % of plays moved one tile, 0.37 % of
decisions held, 0.557 s per decision, opponent level off in 29 % of matches.

What the table says:

- Degradation is smooth: about -5, -10, -17 points at 0.5x, 1x, 2x. No cliff. At 2x the
  policy still plays 9.2 cards a minute and wins 30 %.
- The cost shows as conceded crowns and leaked elixir (it plays less and later), not as
  lost attack.
- Three structural differences carry most of it: the integer elixir digit, opponent
  tracking from late detected plays, and the 0.55 s decision step. All three are how the
  live driver works today, so the clone is already paying them on the phone.
- The single costs add up to about -18 points, the preset costs -10: they overlap.
- `towers` fires (a stale tower value on 1.1 % of decisions) but the clone ignores it.
  Even at 20x only 3 of 60 matches change.

### 3.2 A policy trained with (part of) the layer

A full PPO run with the layer was not possible within the limits of this work (CPU only,
20 minutes per run). As a partial stand-in, the clone was fine-tuned by behaviour cloning
for 17 minutes on CPU (344 steps, 704,512 rows) with only the stateless part of the layer
applied to the stored rows: integer elixir with the digit offset, and the whole-second
clock. Same eval, first 600 jobs of the same panel, paired; se about 2.1 pp. Files:
`evals/s2r/ft_*.json`.

| Condition | Clone | Fine-tuned | Fine-tuned - clone |
|---|---|---|---|
| clean | 45.8 | 50.0 | +4.2 ± 2.2 |
| only `elixir` | 40.7 | 48.7 | **+8.0 ± 2.1** |
| preset 0.5x | 42.8 | 43.5 | +0.7 ± 2.1 |
| preset 1x | 38.0 | 39.8 | +1.8 ± 1.9 |
| preset 2x | 29.8 | 30.7 | +0.8 ± 2.1 |

- The cost of the integer elixir digit alone falls from -5.1 points (clone, these 600
  jobs) to -1.3 ± 2.1 (fine-tuned, against its own clean run). That one is learnable
  quickly, and the clean result did not suffer.
- **The full preset is not recovered.** At 1x the fine-tuned net is +1.8 ± 1.9 over the
  clone, not distinguishable from zero, and it loses 10.2 ± 2.1 points against its own
  clean run. The fine-tune saw none of the other randomizations.
- So the requirement "a policy trained with the layer loses little at the live level" is
  **not shown here**. It needs the PPO run. Section 5 gives the check to run on its
  checkpoints.

## 4. Realism check

Live: what the driver logged in recorded matches A, B, C (section 1). Sim: the clone
playing 100 eval matches with `--sim2real v1` at 1x, every observation and action logged
against the truth (25,824 decisions, 2,461 play attempts, 1,838 opponent plays).

| Quantity | Live (recorded) | Simulator, preset 1x | Verdict |
|---|---|---|---|
| Elixir value is an integer | 476 of 476 decisions | 100 % | matches |
| Mean (shown - true) elixir | -0.56 | -0.57 | matches |
| Digit equals floor / one below / one above | 79 % / 14.6 % / 3.6 % (mismatches are upper bounds) | 88.7 % / 10.1 % / 0.4 % | one-below inside the bound; **one-above is rarer in sim** |
| Digit mismatch within 1 s of an own play vs elsewhere | 41 % vs 15 % (upper bounds) | 22.9 % vs 9.2 % | same direction and ratio, about half the level |
| Decision interval | median 0.550 s, mean 0.548, p10/p90 0.52/0.58, max 0.60 (n = 312) | mean 0.554 s; 0.5 s on 48 % and 0.6 s on 52 % of decisions | mean matches; **shape does not** (engine ticks are 0.1 s) |
| Decisions held on a stale frame | 0 of 645 | 0.40 % | inside the 95 % bound (0.46 %), not evidenced |
| Decision to card leaving the hand | about 0.45-0.54 s (tap to blank slot median 0.37-0.38 s, n = 32 and 41) | median 0.50 s, p5-p95 0.30-0.70 | centre matches; **sim is wider** (existing `--act-delay 3,6,1`, not this layer) |
| Slot empty after the card leaves | median 1.37 s (1.25-1.55) and 1.34 s | drawn 1.1-1.7 s, centre 1.4 s | matches |
| Frames with at least one empty slot | 24.8 % (A, 11.2 plays/min), 27.8 % (B) | 23.1 % (9.9 plays/min) | matches (independent check) |
| Taps that never register | 0 of 32 current driver; 9 of 45 old driver | 3.7 % | inside the bound; not evidenced |
| Play lands one tile off | none shown in 12 stationary troops | 2.0 % | conservative; not evidenced |
| Opponent plays reported (recall) | 0.854 | 0.845 | matches |
| Reported plays that are real (precision) | 0.927 | 0.935 | matches |
| Reported play without the right card | 21.3 % (wrong or unknown) | 21.4 % (unknown) | rate matches; **sim never reports a wrong card**, only "unknown" |
| Delay from opponent play to report | median 1.52 s (p5-p95 0.09-4.54) and 1.54 s | median 1.50 s (p5-p95 0.60-4.80) | median and tail match; **sim lacks the fast reports** (live p5 0.09 s) |
| Clock value | whole seconds on 91-93 % of decisions, extrapolated otherwise | always whole seconds, offset up to ±1 s | close; sim never shows a fractional second |
| A tracked unit missing for 1 / 2 / 3 / 4+ decisions (existing board noise, not this layer) | 63 % / 13 % / 7 % / 17 %, mean 1.8 decisions (tracker gaps, no ground truth) | about 78 % / 17 % / 4 % / 1 %, mean 1.3 | **sim gaps are shorter**; the live figure is inflated by tracker artefacts by an unknown amount |

Where it does not match, plainly:

1. Decision timing has the right mean but only two values in sim.
2. Wrong-card opponent reports do not exist in the engine's detector.
3. Fast opponent reports (spells seen at cast) are missing in sim.
4. Unit-missing runs in the existing board noise are shorter than live tracker gaps.
5. Tap failure, placement error and held decisions are in the preset at rates the
   recordings allow but do not show. They are insurance, not measurements.

## 5. Recommended training configuration

Add to the gen-2 `ppo_pool` command (everything else as now, including the default board
and tail noise and the actuation delay):

    --sim2real v1 --sim2real-scale 0:1.5 --sim2real-widen 10000 --sim2real-anchor-clean --priv-critic

- **Preset `v1`, all twelve on.** None of them alone costs the untrained clone more than
  5.4 points, and the three largest are real properties of the live driver.
- **Scale drawn per match from 0 to 1.5.** The clean case (0) and the measured live level
  (1) are both inside, with a 50 % margin above. Two thirds of matches have the
  structural differences on. The untrained clone loses 6.8 points over this range, a
  handicap it can learn against, not a collapse. Consequence to know: systematic values
  such as the refill lag are scaled too, so their mean over training is 0.75 of the
  measured value.
- **Widen over the first 10,000 episodes** (of 50,000): the upper end grows from 0 to 1.5
  while the critic settles. The count is the run's total episodes, so on a resumed run
  the ramp starts part-way.
- **`--sim2real-anchor-clean` on.** Otherwise the KL anchor ties the learner to the
  clone's reaction to readings the clone was never trained on.
- **`--priv-critic` on.** The actor now sees degraded readings; the critic should not have
  to. It starts as a zero residual, so it cannot hurt the first update.
- The run already in progress was started without these flags and is unaffected. Start
  the layer in a new run. Resuming a run with `--priv-critic` added is tested; resuming
  with `--sim2real` added is not.

Check to run on the PPO checkpoints (this is the missing half of section 3.2):

    python -m training.eval_ghost --weights <ckpt> --temp 1.0 --when-bias 0.4 --act-delay 3,6,1 --no-human --n 1000
    ... the same with  --sim2real v1                        (live level)
    ... the same with  --sim2real v1 --sim2real-scale 2     (beyond it)

Pass: 1x within about 3 points of clean (se of the paired difference ≈ 1.8), 2x still
smooth, clean no worse than a run trained without the layer. If 1x stays far below clean,
turn randomizations off one at a time with `--sim2real-off` to find the one that is too
strong.

### What to measure on the phone to tighten the ranges

Ordered by how much each would change the preset.

1. **Tap registration.** Taps issued against cards that left the hand, over 300+ plays
   with the current driver. Now 0 of 32; the preset assumes up to 6 %.
2. **Elixir digit against the bar.** A full-frame-rate recording with the bar visible:
   when the digit changes relative to the bar filling, and how long it is wrong after a
   play. This fixes the offset range and the transient, the largest single cost.
3. **Opponent-play detection against the bot's real opponents**, hand-labelled for one or
   two matches: recall, wrong-card share, delay. Now from 89 gold plays on other videos.
4. **Decision-to-play latency** over 100+ plays, to narrow the existing 0.3-0.6 s delay
   (recordings suggest 0.45-0.54 s).
5. **Placement.** Commanded tile against detected tile for 100+ stationary troops and
   buildings. Now 10 of 12 with 2 doubtful.
6. **Held decisions.** Count of decisions skipped on stale frames over several full
   matches. Now 0 of 645.
7. **Unit-missing run lengths with ground truth** (two or three hand-labelled minutes),
   for the existing board noise.
8. **Levels and unit stats.** Confirm both sides' levels in the format played and time a
   few known interactions against the simulator. If they agree, drop `levels`.
9. **Tower numerals** with the current driver while towers take damage. Low priority: the
   policy barely uses them.

## 6. Changes proposed, not made

Live driver (closes a gap instead of randomizing over it):

1. **Estimate fractional elixir between digit changes.** The integer digit is the largest
   single cost (section 3). The driver knows the regeneration rate and its own plays; it
   could dead-reckon the fraction and correct on each digit change. Training would then
   need only a small error on the fraction. Done as a pending patch and an opt-in layer
   mode: section 9.
2. While a slot is refilling the driver still writes the next card's cost into the slot
   and relies on the mask to block it. The layer shows cost 0 there. One of the two
   should change so they agree; zeroing the cost in the driver is the simpler one.

Engine (`engine/`, not modified):

3. Per-episode unit-stat multipliers (hit points, damage, speed, hit speed per card,
   ±x %). Today the only handle is `set_team_level`, a ±1 level step of about 10 %, which
   is what `levels` uses.
4. A "wrong card" outcome in the opponent detector (a confusion draw), next to "unknown".
5. A share of opponent reports emitted at once (spell casts), to match the live p5.

Protected files (not modified):

6. `training/ghost.py`: with `ghost_tower_troops` on, the engine match is rebuilt inside
   `reset`, after the layer has set the detector rates, so `oppdet` falls back to the
   engine defaults there. A hook after the rebuild would fix it. The default eval
   convention and `PoolEnv` do not rebuild (100 of 100 traced matches had the rates set).
7. `training/obs_noise.py` (not protected, left alone): tail noise and the cycle /
   opponent-state blocks come from two independent simulated detectors, live from one.
   A play missed in one block can be present in the other.

Trainer (could be added later, not done):

8. The behaviour-cloning auxiliary loss in `ppo_pool` uses clean dataset rows. The
   integer-elixir and whole-second-clock transforms are stateless and could be applied to
   those rows as the scratch fine-tune did.

## 7. Facts and judgment calls

Verified:

- With no new flag, `ppo_pool` and `eval_ghost` build the plain env classes; the preset at
  scale 0 reproduces the clean eval row for row on 1,000 matches; the existing test files
  for the three edited modules pass.
- The policy never receives privileged information: with `--priv-critic` the rollout
  buffer and the first policy update are identical to a run without it, and the net's
  state dict has no new entry (test).
- The recommended flag set runs: four rollouts and updates with all five flags and the
  async vector env, and at scale 0 the clean rows equal the rollout's own observations
  over 1,100 steps per env including episode ends (scratch check, not in the test suite).
- Every live figure in sections 1 and 4 was computed from the named recordings and logs.
- Every number in section 3.

Judgment calls:

- The elixir offset range (-0.25 to +0.05) and the transient probability (0 to 0.3):
  chosen to sit inside upper bounds, because the truth they were measured against drifts.
- Halving the labelled hand-slot error rate for the bot (it never drags cards).
- The ranges for `tapfail`, `place` and `hold`: bounded by the recordings, not shown by
  them.
- `levels` at 30 % of matches, and using a ±1 level step as a stand-in for unit-stat error.
- The "late" range of `oppdet` (unmeasured).
- Switching structural differences on with probability min(1, s), and scaling the refill
  lag with s, so that scale 0 is exactly clean.
- The 0-1.5 scale range, the 10,000-episode widening, and recommending the clean anchor
  and the privileged critic. Their benefit in training is not measured: the trainer path
  with these flags was exercised in tests (a few rollouts and updates), not in a real run.

Where the brief's principles could not be held:

- Principle 6, first half: not shown that a policy trained with the layer loses little at
  the live level (section 3.2).
- Principle 2 for `tapfail`, `place`, `hold`, `levels` and the `oppdet` late rate: no
  measurement behind the centre of the range, only bounds or none.

## 8. Files

New: `training/sim2real.py`, `training/priv_critic.py`,
`training/tests/test_sim2real.py`, `training/tests/test_sim2real_trainer.py`,
`training/SIM2REAL_REPORT.md`.

Edited (opt-in additions only): `training/pool_env.py`, `training/ppo_pool.py`,
`training/eval_ghost.py`.

Outputs: `evals/s2r/` (eval JSON and logs; `ep4_*` = the clone, `ft_*` = the fine-tuned
checkpoint; `ep4_v1_1x_olddet.*` and `ep4_v1_0.5x_oldel.*` are superseded runs with
earlier parameters, kept).

Not touched: the six protected files, `engine/`, `build/`, the phone, git history.

## 9. Fractional elixir estimate (2026-11-23)

Section 6 item 1, done: the driver can feed a fractional elixir estimate instead of the
integer, and the layer can run the same estimator on its own integer so the simulator
shows what that gets. Both are off unless asked for. The driver side is a patch, not
applied.

### 9.1 The estimator

`live_ppo/elixir_estimate.py`, class `ElixirEstimator`: plain Python, no engine or driver
import. It keeps the range of bar values the readings so far allow, [lo, hi].

- Between readings both ends grow at the regeneration rate (cap 10).
- An integer reading d cuts the range to the digit's cell. A tick-over therefore pins the
  fraction; a tick-over that was due but not shown holds the estimate just under d + 1.
- A reading the range cannot explain, while one of our plays is settling (the caller
  announces it, or the reading dropped), moves the range by whole elixir and keeps the
  fraction: cards cost whole numbers. It moves to the lowest whole number that fits.
- A reading that jumps up with no play is believed after 0.3 s and then taken at the
  bottom of its cell.
- Output: the middle of the range, held inside [d, d + 0.99]. Before the first tick-over
  of a match the fraction is unknown and the output sits at d + 0.2.

### 9.2 The game against the engine

Engine (`build/gen2_c/engine/include/core/GameManager.h`): 1/28 elixir per tick (one per
2.8 s), twice that from tick 1200, three times from tick 2400, cap 10, start 5.

Measured from the purple bar's fill on six recorded matches (9.4 names them):

| | engine | recordings |
|---|---|---|
| rate at 1x and 2x | 1 per 2.8 s, 1 per 1.4 s | fill within 0.05 elixir of that model on 95-97 % of frames read |
| 2x starts | clock 120.0 s | 120.0, 120.0, 119.5, 120.0, 119.5 s (the five matches that reached it) |
| 3x | from 240 s | not in any recording |
| after a play from a full bar | refills at once | bar sits on the whole number 0.99-1.10 s first (14 of 14 such plays) |
| digit against the bar | `v1`: from 0.25 late to 0.05 early | digit turns over 0.024 ± 0.025 elixir before the bar gets there (118 tick-overs, range -0.08 to +0.06) |
| digit one below the bar's floor, away from our plays | `v1`: about 10 % at scale 1 | 0.7 % of frames (first three matches), 0.3 % (last three) |

- The wait after a play from 10 is in the estimator (`cap_hold_s`, 1.0 live). The engine
  does not have it, so the layer passes 0. A policy trained in the engine expects 0.36
  elixir more than the game gives in the second after such a play. Not fixable outside
  `engine/`.
- The last two rows contradict `v1`'s `elixir_offset` (-0.25 to +0.05), which section 7
  lists as a judgment call made against a truth that drifts. The bar is a better truth.
  `v1` itself is not changed (9.7).

### 9.3 The bar's fill as an input

- `bar_fill(image)` reads the fill to a fraction on 95-97 % of in-match frames. Unread
  frames: the numeral covering the edge, the pale flash after a play, the bar's tail.
- It is the truth used in 9.4, after a fit that looks at the whole match (not available
  live).
- As a live input next to the digit it adds little: spread away from our plays 0.055 ->
  0.050 elixir on the first three matches, 0.093 -> 0.072 on the last three. Near plays
  no gain.
- So: readable and useful for checking, optional as an input (`--elixir-est bar`). The
  default is `digit`, which is what the simulator models.

### 9.4 Residual on recordings

Estimate minus bar-fit truth, elixir, every recorded frame (9-11 per second). "Near" is
from 0.2 s before to 1.0 s after a drop of the bar (one of our plays); "else" is the
rest. "Floor differs" is the share of frames where the estimate's whole part is not the
truth's (above / below).

Three matches the constants were measured on (in sample): `runs/live/20261120-185026`,
`20261120-093139`, `20261122-213216`. 5,371 frames, 103 plays.

| fed value | where | n | bias | sd | mean abs | floor differs % (above / below) |
|---|---|---|---|---|---|---|
| integer (now) | else | 4,182 | -0.469 | 0.293 | 0.470 | 3.2 (2.5 / 0.7) |
| integer (now) | near | 1,189 | -0.372 | 0.794 | 0.571 | 16.2 (7.2 / 9.0) |
| integer + 0.5 | else | 4,182 | +0.023 | 0.287 | 0.246 | same |
| estimate, digit | else | 4,182 | +0.016 | 0.055 | 0.028 | 3.2 (2.5 / 0.7) |
| estimate, digit | near | 1,189 | +0.070 | 0.745 | 0.266 | 16.2 (7.2 / 9.0) |
| estimate, digit | all | 5,371 | +0.028 | 0.354 | 0.081 | 6.1 (3.6 / 2.5) |
| estimate, digit + bar | else | 4,182 | +0.005 | 0.050 | 0.016 | same |
| estimate, digit, fed every 0.55 s | all | 863 | +0.053 | 0.370 | 0.127 | 6.3 (3.9 / 2.3) |

Three matches recorded after the estimator was fixed (out of sample):
`runs/live/20261123-004549`, a live session with generation-2 checkpoint `ep0030306`,
split into its matches. 3,281 frames, 75 plays. Nothing was changed after seeing them.

| fed value | where | n | bias | sd | mean abs | floor differs % (above / below) |
|---|---|---|---|---|---|---|
| integer (now) | else | 2,612 | -0.454 | 0.301 | 0.455 | 2.6 (2.3 / 0.3) |
| integer (now) | near | 669 | -0.377 | 0.734 | 0.581 | 14.1 (8.2 / 5.8) |
| estimate, digit | else | 2,612 | +0.023 | 0.093 | 0.043 | 2.6 (2.3 / 0.3) |
| estimate, digit | near | 669 | +0.137 | 0.709 | 0.296 | 14.1 (8.2 / 5.8) |
| estimate, digit | all | 3,281 | +0.046 | 0.334 | 0.094 | 4.9 (3.5 / 1.4) |
| estimate, digit + bar | else | 2,612 | +0.009 | 0.072 | 0.020 | same |
| estimate, digit, fed every 0.55 s | all | 699 | +0.045 | 0.296 | 0.121 | 3.6 (2.3 / 1.3) |

- Away from our plays the error falls from half an elixir to under 0.05 (mean abs 0.47 ->
  0.03-0.04).
- Within a second of a play the estimate is no better than the digit in spread (0.7-0.8):
  the digit and the bar animate down over that second and the truth is the bar. The
  whole part matches the digit's by construction (the clamp), so the floor shares equal
  the integer's.
- What the policy is fed has our pending plays taken off (`PlayWatch.owed()`), as now.
  With the costs known (first three matches): integer -0.543 sd 0.680, estimate -0.068
  sd 0.613, against the true bar net of the same plays.
- Which recordings have what: the four sessions above have raw frames, so digit and
  bar. Other sessions under `runs/live` (`20261121-*`, `20261122-174340`,
  `20261123-003323`, `20261123-003509`) logged the integer only, no frames, so no truth.
  The `vid_*` folders have neither. The held-out videos and the two test videos were not
  opened.
- Tool: `live_ppo/eval/elixir_estimate_replay.py`; results in
  `evals/elixir_estimate_replay_20261123.json` and `..._oos_20261123.json`.

### 9.5 In the layer

`--sim2real-elixir-est` on `training.eval_ghost` and `training.ppo_pool`, with
`--sim2real`; config field `elixir_est`. Needs `elixir` among the randomizations.

- The layer's integer (its timing offset, its transients after our plays) goes to the
  same class, once per view (0.5 s at the eval's decision rate), and the fraction is
  written to the elixir scalar. Nothing else in the observation changes (test).
- Our plays are announced from the env's own queue of delayed picks, which the driver
  has too. No new engine read, no new random draw (tests).
- A pick the true bar cannot pay for is a failed tap: its cost is owed and its slot is
  held back, as `tapfail` does (`elixir_est_fail`, on). With the clamp this is rare:
  0.015 per match.
- In this mode the digit's timing is `elixir_est_offset` = (0, 0.05), the measured range,
  not `elixir_offset`. Without the flag nothing changes: same draws, same rows (9.8).
- Residual in the simulator against the recordings, estimate fed about twice a second:

| | bias | sd | floor above % | floor below % |
|---|---|---|---|---|
| simulator, `v1` + flag (1,000 matches) | +0.004 | 0.232 | 4.3 | 1.7 |
| recordings, in sample (863 samples) | +0.053 | 0.370 | 3.9 | 2.3 |
| recordings, out of sample (699 samples) | +0.045 | 0.296 | 2.3 | 1.3 |

  The simulator is a little kinder: no wait after a play from 10, and its transient is
  one step where the game's digit animates.

### 9.6 Evaluation

`ep0020206`, 1,000 matches against the 421 held-out replays, `CR_ENGINE=build/gen2_c/engine`,
same seeds as the references. Differences are paired on (replay, side, seed); standard
errors are clustered on the replay. References: clean 47.8 % wins, leak 2.01 per match;
all randomizations on (`s2r1`) 40.8 %, 4.33; all on except `elixir` (the ceiling, true
fractional elixir) 46.3 %, 2.95.

| variant | win % | leak | win - all on | win - ceiling | leak - all on | leak - ceiling | est. bias / sd | floor above / below % |
|---|---|---|---|---|---|---|---|---|
| **`v1` + flag** (digit timing as measured) | 44.7 | 2.79 | +3.9 ± 1.6 | -1.6 ± 1.5 | -1.54 ± 0.16 | -0.16 ± 0.16 | +0.004 / 0.232 | 4.3 / 1.7 |
| digit timing as `v1` has it (-0.25..0.05) | 43.7 | 3.12 | +2.9 ± 1.6 | -2.6 ± 1.6 | -1.22 ± 0.17 | +0.17 ± 0.16 | -0.075 / 0.265 | 3.5 / 9.6 |
| same, estimator told that timing (`lead`) | 45.6 | 3.00 | +4.8 ± 1.5 | -0.7 ± 1.6 | -1.33 ± 0.16 | +0.05 ± 0.16 | -0.002 / 0.296 | 3.5 / 9.7 |
| `v1` timing, bias -0.15 | 42.5 | 3.50 | +1.7 ± 1.5 | -3.8 ± 1.6 | -0.84 ± 0.15 | +0.55 ± 0.15 | -0.204 / 0.261 | 3.4 / 9.5 |
| `v1` timing, bias +0.15 | 44.1 | 2.78 | +3.3 ± 1.6 | -2.2 ± 1.4 | -1.55 ± 0.16 | -0.16 ± 0.16 | +0.055 / 0.262 | 3.5 / 9.6 |
| `v1` timing, bias +0.30 | 46.0 | 2.70 | +5.2 ± 1.6 | -0.3 ± 1.6 | -1.64 ± 0.16 | -0.25 ± 0.17 | +0.163 / 0.267 | 3.4 / 9.7 |
| `v1` timing, no clamp | 43.7 | 3.14 | +2.9 ± 1.6 | -2.6 ± 1.6 | -1.19 ± 0.17 | +0.19 ± 0.16 | -0.076 / 0.268 | 2.9 / 10.7 |
| `v1` timing, no clamp, bias +0.30 | 45.5 | 2.55 | +4.7 ± 1.6 | -0.8 ± 1.6 | -1.79 ± 0.16 | -0.40 ± 0.16 | +0.214 / 0.272 | 26.1 / 3.0 |
| `v1` timing, failed-tap rule off | 43.7 | 3.12 | +2.9 ± 1.6 | -2.6 ± 1.6 | -1.21 ± 0.17 | +0.17 ± 0.16 | -0.075 / 0.264 | 3.4 / 9.6 |
| measured timing, no clamp | 45.2 | 2.80 | +4.4 ± 1.6 | -1.1 ± 1.5 | -1.53 ± 0.17 | -0.15 ± 0.16 | +0.003 / 0.235 | 4.0 / 3.2 |
| measured timing, bias +0.15 | 45.4 | 2.65 | +4.6 ± 1.6 | -0.9 ± 1.5 | -1.69 ± 0.16 | -0.30 ± 0.15 | +0.137 / 0.227 | 4.3 / 1.6 |
| measured timing, bias +0.30 | 45.9 | 2.38 | +5.1 ± 1.5 | -0.4 ± 1.6 | -1.96 ± 0.16 | -0.57 ± 0.15 | +0.249 / 0.236 | 4.4 / 1.6 |

- The integer cost 5.5 ± 1.5 points inside the full set (ceiling minus all on). The
  estimator gets back 3.9 ± 1.6 of them and all of the leak (2.79 against the ceiling's
  2.95). The remaining -1.6 ± 1.5 is within noise of zero and of the whole gap alike.
- Against clean, `v1` + flag is -3.1 ± 1.8 (all on was -7.0 ± 1.6).
- Digit timing: with the digit lagging as `v1` has it and the estimator expecting the
  measured timing, the gain is 2.9 ± 1.6; told the lag, 4.8 ± 1.5. Paired, the three
  top rows differ by 1 to 2 points (± 1.2-1.3): not separable at this size, and with
  `elixir` alone the order reverses (next item). On leak the measured timing is lower
  both times (-0.32 ± 0.10 all on; 1.93 against 2.11 alone).
- `elixir` alone, no other randomization: integer 45.7 %, leak 3.31; with the flag
  48.6 %, leak 1.93. Against the integer +2.9 ± 1.7 points and -1.38 ± 0.15 leak;
  against clean (47.8 %, 2.01) +0.8 ± 1.0 and -0.08 ± 0.07. The integer alone cost
  2.1 ± 1.7 and the estimate takes it back to clean.
- The same run with `v1`'s digit timing, made earlier, scored 51.3 %: +3.5 ± 1.2 over
  clean with clean's plays and leak. No mechanism was found for beating the true value
  and the run above does not repeat it (-2.7 ± 1.1 paired). Taken as noise; it is the
  largest unexplained number here.

### 9.7 Decisions, and the evidence

- **Under- against over-estimate.** Low costs: bias -0.15 leaks 0.38 ± 0.14 more than
  no bias and wins 1.2 ± 1.3 fewer. High does not cost in the simulator: +0.30 wins
  2.3 ± 1.5 more (`v1` timing), 1.2 ± 1.4 more (measured timing). But with the estimate
  already unbiased, +0.30 takes the leak to 2.38, below what the true value gives
  (2.95): the policy is spending sooner than it would on the truth, not seeing better.
  This policy leaks, so spending sooner helps it. **Default bias 0**: feed what it was
  trained on. The bias is a constructor argument and an `elixir_est_kw` entry for
  experiments, not a driver flag.
- **Clamp to [digit, digit + 0.99]: kept.** Removing it changes 0.0 ± 0.3 points (`v1`
  timing) and +0.5 ± 0.3 (measured timing, few matches differ). With a +0.30 bias and
  no clamp, picks the bar cannot pay for go from 0.025 to 0.119 per match and the
  estimate's whole part is above the truth's on 26 % of samples, for -0.5 ± 1.1 points.
  Live a failed tap also costs the tap time, which the simulator's rule understates.
- **Whole-elixir moves go to the lowest number that fits.** The alternative (largest
  overlap with the predicted range) was tried and was worse on the recordings at frame
  rate; dropped.
- **Digit only by default** (9.3).
- **`elixir_est_offset` = (0, 0.05) in this mode.** With the estimator on, the layer
  should make the residual the recordings show, and with `v1`'s lag it made a different
  one (bias -0.075, floor below 9.6 % against 1.3-2.3 % recorded).

### 9.8 Driver patch

`patches/pending/live_elixir_estimate.diff`: adds `--elixir-est {off,digit,bar}` to
`live_ppo/driver.py`, default off. Rebased at 01:07 PST on the driver as edited at 01:05
(`--keep-spent`); it will need the same again if the driver moves.

- One estimator per launch, reset at each match start. Every frame it gets the integer
  reading, the capture time, the regime from the match clock, and with `bar` the fill.
  While PlayWatch has a play unresolved or an ability press is owed, our play is
  announced. The result replaces `my_elixir`; `mvp_loop` takes `PlayWatch.owed()` off it
  as it does now. The integer still goes to the ledger, PlayWatch and the logs.
- Logged: `elixir_est` on the `board` rows of `latency.jsonl`, and in the HUD line.
- Offline check, no phone: one recorded match (`runs/live/20261122-213216/raw`, 185 s,
  into 2x) replayed at its recorded times with checkpoint `ep0030306`, through the
  driver as it is and through a patched copy with the flag off, `digit` and `bar`; off
  and `digit` a second time, started together. A replay is not frame-exact (each run
  serves a slightly different set of frames and samples its own picks), so runs are
  compared on the frames they share.
  - Flag off against the unpatched driver: hand and elixir equal on all 1,004 shared
    frames, elixir an integer on all 1,138, the ledger's total equal on all 83 shared
    decisions.
  - `digit`: the logged estimate equals the estimator run standalone on the logged
    readings on 1,183 of 1,183 frames and 1,221 of 1,221 (largest difference 0.0005,
    rounding), over 400 of them in 2x. Always inside [digit, digit + 0.99].
  - What the net was shown (`--dump-obs`): the elixir scalar is a whole number on every
    decision with the flag off (290 and 296) and the estimate minus a whole number owed
    with `digit` (292 of 292; 296 of 297, the other held at 0). With nothing owed it is
    inside the digit's cell on 202 of 202 and 216 of 216.
  - Hand and integer reading equal between flag off and `digit` on shared frames (879
    and 1,142). `bar` minus `digit`: mean 0.000, sd 0.057.
  - PlayWatch held plays owed on 80-90 of about 295 decisions (the replay's own picks,
    which never land), so the branch that announces our plays ran. No ability press
    was issued.
  - Not settled: the ledger's spent total differed by 1 between flag off and `digit`
    from frame 550 on, +1 in one pair of runs and -1 in the other. The ledger is fed
    the integer before the hook (`mvp_loop.py:772`), and its end total also differed
    between two flag-off runs (75 and 79), so this reads as replay noise. Not proven.

To turn it on (after reading the patch):

    git apply --check patches/pending/live_elixir_estimate.diff
    git apply patches/pending/live_elixir_estimate.diff
    # recorded frames first, never taps
    CR_ENGINE=build/gen2_c/engine .venv/bin/python play_live.py \
        --checkpoint runs/g2_ppo_s2r/snapshots/ep0030306.pth --temp 1.0 \
        --replay runs/live/20261123-004549/raw --seconds 200 --no-record \
        --driver-flags="--loop-timestamps --ability tap --elixir-est digit"
    # live: a dry run as written; --act added places cards (it asks first)
    CR_ENGINE=build/gen2_c/engine .venv/bin/python play_live.py \
        --checkpoint runs/g2_ppo_s2r/snapshots/ep0030306.pth --temp 1.0 \
        --driver-flags="--ability tap --elixir-est digit"

`--elixir-est bar` adds the fill. The simulator's twin:
`python -m training.eval_ghost ... --sim2real v1 --sim2real-elixir-est`, the same two
flags on `training.ppo_pool`.

### 9.9 Facts and judgment calls

Verified:

- Every number in 9.2 to 9.6 comes from the named recordings and eval files.
- Without the flag the layer is unchanged: `--sim2real v1` rerun on the final code
  gives the 1,000 rows of `..._s2r1_ec_20261122.json` exactly, run time aside
  (`..._s2r1_recheck_ec_20261123.json`).
- `v1` + flag equals the run made from a config file before the measured timing became
  the mode's default (`_liveoff`): 1,000 of 1,000 rows.
- `--sim2real evals/elest_cfg/v1off.json` (the flag with `v1`'s digit timing) reproduces
  the run made before `elixir_est_offset` existed: 1,000 of 1,000 rows.
- Tests: 17 for the estimator (`live_ppo/eval/test_elixir_estimate.py`), 6 for the
  layer mode (`training/tests/test_sim2real_elixir_est.py`); `test_sim2real.py`,
  `test_sim2real_trainer.py`, `test_ppo_pool.py`, `test_pool_env.py` pass (92 in all).
- The patch applies to the driver as of 01:05 PST and the patched file compiles.

Judgment calls:

- The digit's cell, (0, 0.05) early: from 118 tick-overs in three matches on one phone.
  It is in the estimator (`lead`) and in the layer (`elixir_est_offset`).
- Baking the measured timing into the mode instead of keeping `v1`'s. Both results are
  in 9.6.
- Bias 0, clamp on, lowest-whole-number rule, digit only (9.7).
- The failed-tap rule for a pick the bar cannot pay for borrows `tapfail`'s hold time.

Not verified:

- No live run with the patch. Phone access is withheld; nothing was tapped or captured.
- 3x elixir: in no recording. The estimator takes the engine's rate there.
- The patch with real taps: PlayWatch's plays in a replay never land, and the
  ability-owed branch did not run at all. The estimator's own handling of announced
  plays is unit-tested.
- The ledger total's difference of 1 in the replays (9.8).
- The wait after a play from 10 does not exist in the engine, so the simulator result
  does not include its cost live.
- The simulator feeds the estimator twice a second, the driver every frame. Fed at the
  simulator's rate the recorded residual is the "every 0.55 s" rows of 9.4.
- Fed-value residual (net of pending plays) for the out-of-sample session: its play
  costs are not written yet (`plays.json` missing).

Failed or skipped:

- The Mac restarted at 00:43 PST (clean shutdown, cause not known to this task). It
  killed two eval queues and the PPO run, and emptied the scratch folder: `paired.py`,
  the first no-flag recheck and the first wiring-check outputs were lost. The evals,
  the recheck and the wiring check were rerun; pairing was redone with the same rule (replay, side, seed; clustered on
  replay), which had matched `paired.py` on the pairs both computed. The PPO run was
  not restarted by this task; it was running again from 01:00:10.
- `training/tests` in full: 267 pass, 2 fail (`test_obs_compat.py::
  test_adapted_observation_equals_the_gen1_engines`, `test_obs_v2.py::
  test_env_tail_holds_every_play_the_engine_recorded`). Both fail the same way run
  alone with none of the files this task edited loaded.
- Using the play's cost to pick the whole-elixir move (`own_play(t, cost)` takes the
  cost and ignores it). Not done.

Proposed, not made:

- Narrow `v1`'s own `elixir_offset` to the measured range. It changes what the running
  PPO trains on, so it is Sam's call.
- Train the next generation with `--sim2real-elixir-est`, once the patch has had a live
  check. `ep0020206` was trained on integers and still gained.
- A driver flag for the bias, if a live A/B is wanted.
- Apply the patch.

Files:

- New: `live_ppo/elixir_estimate.py`, `live_ppo/eval/test_elixir_estimate.py`,
  `live_ppo/eval/elixir_estimate_replay.py`,
  `live_ppo/eval/elixir_estimate_split_session.py`,
  `training/tests/test_sim2real_elixir_est.py`,
  `patches/pending/live_elixir_estimate.diff`, `evals/elest_cfg/` (variant configs).
- Edited (opt-in additions only): `training/sim2real.py`, `training/eval_ghost.py`,
  `training/ppo_pool.py`. `training/pool_env.py` not edited.
- Outputs: `evals/elixir_estimate_replay_20261123.json`,
  `evals/elixir_estimate_replay_oos_20261123.json`,
  `evals/ghost_g2_ppo_s2r_ep0020206_*elest*_ec_20261123.json` (`s2r1_elest` = `v1` +
  flag; `_v1off` = `v1`'s digit timing; `_liveoff*` = measured timing by config, run
  before it became the default), `..._s2r1_recheck_ec_20261123.json`, logs in
  `runs/logs/eval_*_20261123.log`.
- Not touched: the six protected files, `engine/`, `build/`, `live_ppo/driver.py`,
  `runs/g2_ppo_s2r`, the phone, git history.
