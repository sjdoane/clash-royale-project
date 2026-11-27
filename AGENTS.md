# AGENTS.md: start here

This is a guide for agents (and people) picking up this project. State as of **2026-11-27**.

## What this is
A bot that plays Clash Royale (the Hog 2.6 deck) against real people on an iPhone.

The pipeline:
1. **Imitation learning (IL):** a policy is trained on top players' replays.
2. **PPO** in a C++ simulator (the `engine/` submodule, our fork of ClashRoyaleAi, with patches 0083–0091).
3. **Live play:** a YOLO detector reads the mirrored iPhone screen, and the policy plays through iPhone Mirroring.

**The target is win rate against real people.** Simulator scores are only a guide: they have overstated live strength before.

## Where things are
| Path | What |
|---|---|
| `engine/` | Simulator submodule: [sjdoane/ClashRoyaleAi](https://github.com/sjdoane/ClashRoyaleAi), branch `gen2-patches-0088-0091`. Mechanics fixes are runtime switches chosen with `CR_P0088=<comma list>`; see `patches/README.md` and `patches/pending/`. |
| `training/` | IL (`replay_bc.py`, `net_v3.py`, `obs_v2.py`), PPO (`ppo_pool.py` is the current trainer; opponent pool: replays ("ghosts"), imitation clone, scripted teacher, building-heavy teacher, self-play), reward, observation noise. |
| `evals/` | Evaluation tools: `evals/tools/` (threat rate, opponent scorecard, building-pull probe and more). Earlier probes are in the subfolders. |
| `live_ppo/` | Live driver (`driver.py`), standard session wrapper (`live_session.sh`), engine pick per checkpoint (`engine_pick.py`), HUD and tower readers, offline re-read of recorded matches (`reread.py`), tests in `live_ppo/eval/`. |
| `video_dataset/` | Detector data tools and the live fine-tune tools (`detector/live_ft/`). |
| `card_audit/` | Card stats checked against official data. |
| `docs/` | Official Clash Royale API docs; **`docs/reports/`** holds the most recent findings (read these first, list below). |
| `ios/` | iPhone setup (`ios/SETUP.md`). |

Large files are not in git. Checkpoints, the detector and the IL dataset are release assets in the private repo `sjdoane/clash-royale-project-assets`: release **v2** for 2026-11-27, v1 for older ones. `scripts/fetch_assets.sh` fetches v1; download v2 with `gh release download v2 -R sjdoane/clash-royale-project-assets`.

## Checkpoints (release v2)
| Asset | Role | Common panel (sim) | Live |
|---|---|---|---|
| `best_old_main_ep0496977.pth` | Best so far (old main PPO run `g2_ppo_s2r`) | 67.7 % | 3 of 12 (11-25) |
| `current_restart_ep0210132.pth` | Current PPO run (`g2_pocket_fresh`, from C5, engine gen2_l), still training | 65.6 % at 200k | 0 of 10 at 204k (11-27): least tower damage taken yet; 7 lost by one crown |
| `il_start_C5_ep4.pth` | IL start of the current run (delay-labelled lead 13) | 31 % | |
| `clone_opponent_all3j_ep5.pth` | IL "people clone" used as an opponent | | |
| `kl_anchor_C5r2_ep1.pth` | KL anchor of the current run since episode 152,502 | | |
| `detector_r5.pt` | Live detector (round 5) | | |

The common panel is 1,000 held-out replay matches, paired, with engine gen2_l and the 20 switches.

**Live use:** checkpoints of the C5 lineage need `--pocket-mask --afford-project` in `--driver-flags`. `live_ppo/engine_pick.py` picks the engine and switches from each run's `launch.txt`.

## Read these first (`docs/reports/`)
1. `live_20261127pm_audit.md`: latest live session.
   - Big tower damage is **75 % policy choice** (held, misplaced or sent a Hog instead), **23 % elixir** and **3 % perception**.
   - The simulator overstates defence: threat answer rate 0.26 in the sim vs 0.13 live.
2. `live_20261127pm_data_plan.md`: how live sessions become training and eval data (deck pool, ghost store, held-out split).
3. `engine_patch_0091.md` and `engine_patch_0091_switch_plan.md`: latest simulator patch (Royal Ghost and Suspicious Bush on the board, Bush pops, evo Goblin Barrel decoy) and how a mid-run engine switch with revert is done.
4. `imitation_barrel_reader_mismatch.md`: why the IL model ignored Goblin Barrels. Old and new observation readers disagreed; the fix is to give the IL data reader2 from the start.
5. `defence_probe_20261127.md`, `training_side_20261127.md`, `barrel_predictor_verdict.md`, `live_20261127am_decision_audit.md`.
6. `training_plan.md`, `live.md`: older plans and live notes.

## Open work (2026-11-27)
- Simulator fidelity for misplaced defence: Electro Giant reflect, Cannon against air, wrong-lane answers (planned patch 0092 / `gen2_m`).
- Live fixes: overtime-clock misread, an elixir read of 0, and a lane-aware option for the leak-floor forced play.
- Fold the 11-27 decks and replays into training at about 250k episodes, keeping the afternoon session held out.

## House rules that matter
- **Judging:** win rate against real people is what counts.
  - Fireball and pocket stats are observe-only.
  - No convoluted rewards and no extra bait weighting.
  - The simulator must not reward plays that fail against people.
- **Training runs:** never run two trainers on the same run folder. Change engine or data mid-run only at a snapshot boundary, with a revert copy, as in the switch plans.
- **Simulator code:** new simulator behaviour goes behind a default-off switch, and with it off the result must be bit-identical.
- **Privacy:** no opponent names or player tags in files; no credentials in the repo.
