# net_v2 PPO, FLZ ports, dead spells, final checkpoint — 2026-11-18

Training box balerion (RTX 5090). Deck: Hog 2.6, all level 11.
All live checks here are dry runs. Nothing in this work was run with `--act`.

## TL;DR — the checkpoint to run live

**`remote_results/ppo_league26_ep138708.pth`** (v1 net, league run, episode 138,708).

```
.venv/bin/python live_ppo/driver.py --run-dir runs/live/<name> --temp 0.5 \
    -- --capture mirror --policy neural \
       --checkpoint remote_results/ppo_league26_ep138708.pth \
       --ios-geometry ios/geometry.json --seconds 300
# add --act (after the --) only in Training Camp, once phone access is re-granted
```

- **`--temp 0.5`** is the only non-default flag.
- **Decision rate.** It is 1 Hz, and the driver picks that up with no flag: `--decision-hz 0` means "read the checkpoint's `decision_hz`, else 1.0". A net_v2 checkpoint loads through the same command (see the driver section).
- **How it was picked.** It won the round-robin Bradley–Terry among every checkpoint available on the Mac (table below).
- **What the selection did not include.** Two candidates weren't in the round-robin, because the SSH master socket to balerion dropped at ~12:03:
  - the newer league checkpoint (ep 141,727);
  - the two net_v2 PPO arms.

  "Still to do" below has the commands to finish.

## 1. BC: v1 vs net_v2 at 1 Hz vs net_v2 at 5 Hz (held out)

Held-out hog26 validation set: 90,253 decision rows, the same split as `runs/bc_v2_compare/v1_hog26_val.json`.

| metric | v1 (`bc_hog26`) | net_v2 1 Hz (final, step 54.8k) |
|---|---|---|
| play AUC | 0.771 | **0.783** |
| play ECE | 0.015 | **0.004** |
| card \| play | 0.519 | **0.578** |
| card NLL \| play | 1.038 | **0.941** |
| cell exact | 0.224 | **0.252** |
| cell within 1 | 0.406 | **0.446** |
| card+cell NLL per play | 4.43 | **4.10** |

- **net_v2 at 5 Hz** (step 19k of a 120-minute run, stopped at 11:25):
  - card \| play 0.580, cell 0.244, within 1 0.426: about the same as 1 Hz.
  - But its per-second play timing is poor: sec AUC 0.58, and it under-predicts plays (0.195/s against a human 0.232/s).
  - It also costs about 5× more per step.
- **1 Hz was used to seed PPO.**
- I stopped the 5 Hz run because the GPU ran out of memory (see section 2). Its last checkpoint is still at `checkpoints/bc_v2_5hz.pth` on balerion.

## 2. PPO from net_v2 BC, with the FLZ ports

All the FLZ ports are flags that default to the old behaviour:

| port | where | flag |
|---|---|---|
| actuation delay (5,13,3 ticks) plus in-flight card features in the observation | `training/reward.py` (env wrapper; no engine change) | `--act-delay 5,13,3` (ppo_bc, league, eval_panel, h2h, roundrobin) |
| temperature at eval/h2h; training stays at T=1 | `net_v2.sample_logits` | `--temp` (eval_panel, h2h, roundrobin, driver) |
| round-robin Bradley–Terry Elo with bootstrap CIs and opponent activity | `training/roundrobin.py` (new) | — |
| asymmetric (unilateral) elixir-overflow penalty | `reward.py`, `league_env.py` | `--w-unilateral` (default 0; FLZ value 0.075) |
| King chip ×0.2 while both Princess Towers stand | `reward.py` | `--king-chip` (default 1.0) |
| PFSP win-rate smoothing (n·s+1)/(n+2) over 300 games | `league.SnapshotPool` | `--pfsp-window 300` (0 = old EMA) |
| larger mirror share | `league.py` | `--p-mirror 0.4` (default 0.25) |
| net_v2 in PPO | `ppo_bc._install_arch` | `--arch net_v2` |

**Runs launched at 11:15.** Two arms:

- **Command:**
  ```
  league --arch net_v2 --fork-from "" --init checkpoints/bc_v2_1hz.pth --lr 1e-4 --p-mirror 0.4 --pfsp-window 300
  ```
  - `--fork-from ""` means start from the BC model, with the BC-KL anchor on.
  - lr 1e-4, against 5e-5 for v1.
- **The only difference between the arms:**
  - **v2_league_a:** no delay.
  - **v2_league_b:** trains with `--act-delay 5,13,3` and the in-flight features.
- **How they're evaluated.** Both have eval and h2h watchers at T=0.5 under the live-like delay 5,13,3. The A/B therefore asks which arm plays better under real latency.
- **Not tested:** the unilateral-overflow penalty and King chip. They're implemented and smoke-tested, but there was no budget to run them.

**Last readings, 12:01.** The run started from BC at 1–2% on the panel.

| run | episodes | panel win (T=0.5, delay) | h2h vs v1 ep 60.7k | h2h vs league26 |
|---|---|---|---|---|
| v2_league_a (no delay) | 2,154 | 0.109 | 0.417 | 0.125 |
| v2_league_b (delay) | 2,001 | 0.062 | 0.146 | 0.146 |

- **Throughput.** Each arm runs about 5k episodes per hour. The GPU is the bottleneck: 98% utilization, with env workers at about 4% CPU.
- **Timescale.** v1 needed about 25k episodes to reach 0.45 on the panel and about 130k for 0.6. net_v2 is not going to pass league26 today.
- **Delay A/B: no conclusion yet.** At 2k episodes it is noise.

**Runs stopped.** Checkpoints were copied to the Mac before each stop:

- **ppo_bc26** at ep ≈145.9k (plateaued at 0.54–0.58 on the panel) → `remote_results/ppo_bc26_final.pth`. Also copied: `ppo_bc26_ep144920.pth`.
- **bc_v2_5hz** (GPU out of memory, see section 1).
  - The first launch of v2_league_b died with CUDA OOM at update 6.
  - It was relaunched at 11:25 with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and a fresh start.
  - The crashed files are kept as `*.crashed_oom`.
- **ppo_league26: not stopped.** I meant to stop it (plateaued at 0.59–0.65 since ep ~132k) to give the GPU to net_v2.
  - The SSH socket dropped first, so it is still running.
  - Its ep 141,727 checkpoint is saved on balerion at `runs/sweep/ppo_league26_final.pth` but is **not on the Mac yet**.

## 3. Temperature (sim panel, 192 matches per arm, no delay)

| checkpoint | T=1 | **T=0.5** | argmax |
|---|---|---|---|
| ppo_league26 ep 138,708 | 0.589 | **0.693** | 0.635 |
| ppo_bc26 ep 144,920 | 0.547 | 0.630 | 0.651 |

- No passive wins at any setting: opponents play 27–30 cards per game.
- Raw rows: `remote_results/temp_eval.jsonl`.
- **Caveat for net_v2 BC:** T=0.5 sharpens its no-op gate. BC at T=0.5 plays about 7 cards per match and leaks about 19 elixir. Choose the temperature per checkpoint.

## 4. Fireball and Log ("dead cards") — win rate against a control

**P(play | in hand, affordable, good opportunity):**

| | league26 | ppo_bc26 | league26 at T=1 |
|---|---|---|---|
| Fireball, at T=0.5 | 0.0045 | 0.0034 | 0.006 |
| Log, at T=0.5 | 0.039 | 0.014 | 0.065 |

- An opportunity means:
  - **Fireball:** the best cell catches ≥ 1 Fireball's damage (689).
  - **Log:** ≥ 2 enemy ground units on our side of the river.
- **net_v2 so far does not fix it:**
  - BC: Fireball 0.005, Log 0.01.
  - v2 PPO at 2k episodes: Fireball 0.02, Log 0.05.
- **The league did not fix it either.**

**The intervention.** `eval_panel --spell-mode {ban,force,force_aim} --spell-cards`:

- **ban:** mask the card out of the card head.
- **force:** play the card whenever there is an opportunity; the net's own placement head aims it.
- **force_aim:** as force, but Fireball goes to `tactics.best_spell_cell`.

Setup: league26 ep 138,708, T=0.5, 384 matches per arm, seeds paired with the control. Raw rows are in `remote_results/spells/`.

| arm | win | Δ vs control (paired, 95% bootstrap CI) |
|---|---|---|
| control (net as is) | 0.648 | — |
| ban Fireball + Log | 0.648 | +0.000 [−0.052, +0.049] |
| ban Fireball | 0.651 | ≈ 0 |
| ban Log | 0.641 | ≈ 0 |
| force Log | 0.646 | ≈ 0 |
| force Fireball (net aims) | 0.461 | about −0.19 |
| force Fireball (tactics aims) | 0.508 | about −0.14 |
| force both (net aims) | 0.385 | −0.263 [−0.315, −0.214] |
| force both (Fireball by tactics) | 0.490 | −0.159 [−0.214, −0.107] |

**What this means:**

- In this simulator, against this opponent panel, neither spell is worth anything as the net would play it. Banning them costs nothing.
- Forcing Fireball on every "opportunity" costs 14–19 points of win rate, even when aimed by the tactics helper.
- Not playing Fireball is the policy being right about this environment (as in the 2026-08-29 autopsy in `engine/.claude/CLAUDE.md`), not a training failure.
- The sim *overvalues* Fireball against towers (100% Crown Tower damage, against 159/688 in the real game, UPSTREAM item 29), so live it's worth even less.
- Forcing Log is neutral. So "Log never played live" costs no measurable win rate in sim.
- **Not established:**
  - whether a stricter opportunity (for example, a Musketeer or a cluster ≥ 2× damage) is positive;
  - how net_v2 behaves once trained.

## 5. Final selection: round-robin Bradley–Terry

Setup: `training/roundrobin.py`, run on the Mac. Every pair plays 96 seeds per seat with the seats swapped, under the live-like delay 5,13,3. Elo is anchored on the winner. P(best) comes from 1,000 bootstrap resamples. Raw output: `remote_results/rr/`.

**At T=0.5 (1,152 games):**

| checkpoint | Elo [95% CI] | score | plays/game | beaten-opponent plays/game | P(best) |
|---|---|---|---|---|---|
| **league26 ep 138,708** | 0 | 0.628 | 51.0 | 50.7 | **1.000** |
| ppo_bc26 ep 145,945 (final) | −65 [−102, −31] | 0.507 | 49.1 | 46.7 | 0 |
| ppo_bc26 ep 144,920 | −83 [−119, −46] | 0.474 | 48.7 | 46.2 | 0 |
| ppo_bc26 ep 132,688 (the one used live before) | −127 [−166, −91] | 0.391 | 44.8 | 51.6 | 0 |

- **At T=1:** same order; league26 P(best) 0.997.
- **Opponent activity:** beaten opponents play about 46–51 cards per game, so no rating comes from beating a passive net.
- league26 is also the panel winner at T=0.5 (0.693).

## 6. Live driver (`live_ppo/driver.py`)

**net_v2 support:**
- `PPOPolicy` detects net_v2 (from the `arch` key, or from its state-dict keys) and builds `NetV2` from the checkpoint's cfg.
- It slots the net in through a temporary `_Prebuilt` class shim, because `MicroRoyaleNet.__init__` calls `super(MicroRoyaleNet, self)`.
- The adapter line now prints the arch and the decision rate.

**Decision rate:** `--decision-hz` (0 = the checkpoint's `decision_hz`, else 1.0). It sets `L.DECISION_HZ`. Every PPO checkpoint here trains at 1 Hz (SKIP_TICKS=10).

**Offline check.** Replayed `runs/live/rec_dry1` from frame 650, dry run:
- **net_v2:** adapter "arch net_v2 (NetV2), missing [], unexpected []; decision rate 1.0 Hz"; it decides every second.
- **The chosen v1 checkpoint at `--temp 0.5`:** decides and plays.
- **0 taps** in both runs.
- That recording is a ladder match, so the Training Camp guard correctly refuses every tap.
- Note: replays need `--loop-from` ≳ 650. Earlier frames are pre-game and the screen reads `unknown`.

## Still to do (blocked on the SSH socket)

The control socket `~/.ssh/cm-balerion` is gone, and I may not open a new connection. Once Sam re-opens the master connection:

1. **Copy the newer league checkpoint:**
   ```
   rsync -a -e "ssh -o ControlPath=~/.ssh/cm-balerion -o BatchMode=yes" user@balerion:cr-robot-remote/train/runs/sweep/ppo_league26_final.pth remote_results/
   ```
   Then copy the current `checkpoints/v2_league_{a,b}.pth` the same way.
2. **Re-run the round-robin** with the newer checkpoints added:
   ```
   python -m training.roundrobin \
     league26_ep138708=remote_results/ppo_league26_ep138708.pth \
     league26_ep141727=remote_results/ppo_league26_final.pth \
     v2a=remote_results/v2_league_a.pth \
     v2b=remote_results/v2_league_b.pth \
     bc26_final=remote_results/ppo_bc26_final.pth \
     --n 96 --temp 0.5 --act-delay 5,13,3 --procs 12 --out remote_results/rr/final.json
   ```
   If something else wins, swap the `--checkpoint` path above. For a net_v2 winner, check its temperature at 0.5 vs 1, because of the no-op-gate caveat in section 3.
3. **Decide whether to stop ppo_league26** (pgid 3634909; watchers 3635752, 3635753) to give the GPU to the net_v2 arms.
4. **Process groups** on balerion:
   - v2_league_a trainer: pgid of `training.league --run v2_league_a`.
   - v2_league_b trainer: relaunched at 11:25.
   - Status script: `sh ~/cr-robot-remote/train/runs/v2status.sh`.

Nothing was committed. Nothing under `engine/` was modified.
