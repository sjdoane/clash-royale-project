# Live latency experiment, 2026-11-24 (written before any data was collected)

Checkpoint `runs/g2_ppo_s2r/snapshots/ep0327293.pth`, engine `build/gen2_c` (as in the
earlier live sessions), detector r5, `--temp 0.5` default, flags
`--loop-timestamps --ability tap --dump-obs --leak-floor 9.5 --elixir-est digit`.
One launch = one block = one match. Conditions are read back from `run.json` (tap mode)
and `runs/latency_exp/load.jsonl` (load average, trainer state R/S/T, sampled every second).

## Clocks and events, per play (all offline, nothing added to the hot path)
- `latency.jsonl`: board `cap` (capture start) and `proc`, `decide.t`, `tap.t0/t1`, `post` internals.
- `raw/frames.jsonl` + jpgs (every frame, about 12-15 fps, `t_perf` on the same clock):
  E1 the tapped card rises in the hand (click mode: card selected, local UI only);
  E2 the card leaves its slot / elixir drops (local client);
  E3 our deploy stopwatch first visible on the target tile (server-registered placement);
  E4 the unit's first detection on the arena; for spells, first visible at our King Tower.
- Frame resolution is about 70 ms; means over 30+ plays resolve about 15 ms.

## Decision numbers
| H | number | comparison | n |
|---|---|---|---|
| H1 | median and p10/p90 of each stage: cap->decide, decide->t0, t0->t1, t1->E2, E2->E3, E3->E4 | budget sums to decide-frame->E3 | all plays (~35 a match) |
| H2 | t0->E1 minus the 20 ms pre-press wait (input + display round trip through Mirroring) | share of decide-frame->E3 | click plays only; cannot split input lag from display lag (no external camera) |
| H3 | t0->E2 and t0->E3, drag vs click | difference of medians, 95 % bootstrap CI over matches | >=2 matches per mode (~70 plays) |
| H4 | cap->decide and detector ms, trainer paused vs running | difference of medians | 1-2 matches per state (thousands of boards) |
| H5 | sustained board rate and decision cost; mean wait to next 0.5 s step | computed from boards | all |
| H6 | distribution of decide-frame->E3 (mean, sd, p10, p50, p90, p99) | -> `--act-delay` ticks and `OWN_LAND_S` | all plays, Training Camp and real separately |
| H7 | taps without E2 within 3 s; E2 without E3 (refused by the game) | rate per mode with Wilson CI | all; ~100 plays per arm only detects rates differing by >~8 points |
| H8 (added) | E2->E3 in Training Camp vs real matches (is the ~1.2 s network/server or fixed game delay) | difference of medians | >=2 matches each |

Power: per-play sd of decide->stopwatch was about 0.10 s (p10-p90 1.58-1.84 s, 330 plays,
2026-11-23). With ~70 plays per arm, SE of a difference of means is about 0.017 s, so 0.1 s
differences are decided many times over; the limit is match-to-match drift (network), handled
by ABBA ordering and a per-match bootstrap.

## Order (blocks)
Training Camp: 1 click/run, 2 drag/run, 3 drag/paused, 4 click/paused, 5 drag/run, 6 click/run.
Real matches (mode Sam names): alternate click, drag, drag, click... only if drag's Training
Camp failure rate is no worse than click's.

## What this cannot decide
- Mirroring input lag vs display lag separately (only their sum).
- Server vs game-side share inside E2->E3 except through the Training Camp vs real comparison.
- Whether a faster decision cadence helps play: that needs retraining (main thread owns it).

## Amendment 2026-11-24 15:20 (after block 1, before any comparison was looked at)
Sam does not want Training Camp: all blocks are real matches. Order: 1 click (done,
runs/live/20261124-150853), 2 drag, 3 drag, 4 click, 5 drag, 6 click. Trainer pause only
if Sam keeps it in block 4/5. H8 (Training Camp vs real) is dropped: undecided.
Drag has never run in a real match; if a drag block shows cards not leaving the hand,
stop the block and fall back to click.
