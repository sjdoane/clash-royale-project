# Why C5 doesn't Log Goblin Barrels (2026-11-27)

Written by the coordinator from the analysis agent's final message (the agent could not write this file). The scripts and outputs are in this folder:
- `timing_summary.txt`, `evalp_summary.txt`, `evalp_summary.json`
- `timing.py`, `evalp.py`, `run_evalp.sh`, `tile_check.py`, `renoise_speed.py`

## Verdict: the delay-shift hypothesis is NOT confirmed. The cause is a reader mismatch.

The L13 shift puts few barrel Log labels before the barrel is visible: 3.1 % for the plain barrel and 3.7 % for the evo [measured].

The real cause is a mismatch between two readers:
- C5 and main's start were trained on imitation data noised with the OLD tail reader (2026-11-20). That reader never shows a young barrel's landing tile. Across 762 barrel casts the tile shows at a median of +4.2 s, never within +1.5 s, and OPP_SPELLS is always empty [measured].
- The sim, the PPO run, the defence probe and live all use reader2, which shows the tile at +0.4 to +0.9 s [measured].

So C5 reads a young barrel *with* a tile as "nothing to Log".

## Log probability at people's barrel casts

Each cell is P(at least one Log decision in the window), for the windows 0–1.5 s / 1.5–3 s after the cast, on 2,000 of 5,612 casts [measured].

| Model | Training view | Sim view (reader2 tail) | Sim view, barrel tile removed |
|---|---|---|---|
| C5 | 0.372 / 0.339 | **0.177 / 0.080** | 0.365 / 0.330 |
| L0-pocket (hog_g3j, lead 0) | 0.153 / 0.643 | 0.180 / 0.407 | 0.149 / 0.631 |
| Main's start | 0.106 / 0.530 | 0.284 / 0.572 | 0.103 / 0.524 |
| Restart ep 133,387 | 0.303 / 0.401 | 0.253 / **0.115** | 0.347 / 0.435 |

What this shows:
- **The tile is the whole effect.** Removing it brings C5 back to its training view [measured].
- **The L13 recipe itself is fine.** In its own view, C5 Logs as often as main's start (0.58 for any Log in 0–3 s), only 1 to 1.5 s earlier [measured].
- **Why the shift makes it worse.** C5's Log comes before the goblins land, so it depends on the tail entry alone. The lead-0 models Log once the goblins are on the board, which looks the same in both readers [inferred].
- **Why the restart is stuck** [inferred]:
  - its BC rows come from the old-reader cache (`ppo_pool.py` around line 837);
  - its KL anchor is C5, evaluated on reader2 rollouts, where C5 is suppressed.

## A related effect for troops

With L13, many answer labels land before the threat is visible. This is real, but it affects troops, not barrels [measured]:

| Threat | C5 Log labels before visible | L0 before visible | C5 "any answer" labels before visible |
|---|---|---|---|
| Hog Rider | 25.1 % | 7.5 % | 50.9 % |
| Royal Ghost | 30.2 % | 9.9 % | 49.7 % |
| Mini PEKKA | 28.7 % | 9.3 % | 58.3 % |
| Bush goblins | 11.1 % | 5.8 % | 42.4 % |

Other timing facts [measured]:
- People Log barrels at +2.8 s (p10 +2.4 s, p90 +3.5 s).
- C5's label row for those Logs falls at +1.0 s (45 %), +1.5 s (25 %) or +0.5 s (15 %).

## Fix
1. **Re-noise the imitation tail with reader2.**
   - Keep the board entries and redo only the tail, with `noise_tail_rows(..., cfg={"names":"spells","preset":"reader2"}, seed=_episode_seed(0, ep)+9974)`.
   - Write it to a NEW cache root (e.g. `runs/bc3/_cache_r2/`). The cache directory name doesn't include the preset, so writing to the old root would overwrite the old cache.
   - Cost: about 12 min of compute, 20–30 min including writing about 10 GB.
2. **Fine-tune C5 on it → "C5r2"**: 1–2 epochs, about 15–25 min with the GPU shared with PPO.
3. **Gate C5r2 before use:**
   - `evalp.py`: the sim view should come back to about 0.37 / 0.34;
   - defence probe: barrels at least 57 % answered and 19 % Logged (main's start), and general defence not below 46 %.
4. **Restart run: swap the anchor and the BC cache** (C5 → C5r2, old cache → reader2 cache) at a snapshot boundary, with a revert copy. This needs default-off `ppo_pool` options. Measure the KL between the restart and C5r2 first; if it is far above `--kl-max`, start fresh from C5r2 instead, which loses about 144k episodes.
5. **Separate experiment, troops only:** clamp each label row to `max(row at tau-L, first row where the trigger is visible)`. This is a load-time change in `build_targets_fixed` and needs no data rebuild.
6. **Next gen2_k data rebuild:** use reader2 from the start.
