# Training status (Hog 2.6, BC -> PPO)

_Last updated: 2026-11-18 09:55 PST (ppo_league26 added). The numbers below are refreshed at the bottom._

## Where it runs

| | |
|---|---|
| host | **balerion** (RTX 5090, 24 cores). Reach it only through the existing socket: `training/rsh '<cmd>'` |
| path | `~/cr-robot-remote/train/`, holding `engine/` (Linux build in `engine/build_linux`), `training/`, `data/bc/hog26`, `data/decks`, a venv with py3.12 and torch 2.11 cu128 |
| trainer | `training.ppo_bc --run ppo_bc26`, PID **3434021** (nohup + setsid), 16 env workers, GPU |
| eval panel | `training.eval_panel --watch ppo_bc26`, PID **3434022**, CPU only, every 2,000 episodes |
| B arm | `training.league --run ppo_league26`, PID **3634909**, plus its CPU eval panel **3635752** and h2h watcher **3635753**. See [ppo_league26](#ppo_league26-self-play-league-b-arm) |
| GPU rule | `training/launch_remote.sh` checks `nvidia-smi --query-compute-apps` and falls back to CPU if any process's cwd is outside `~/cr-robot-remote/`. At launch only our own and the detector agent's processes were present. |

**Tail:**

    training/rsh 'tail -f ~/cr-robot-remote/train/runs/ppo_bc26.log | grep --line-buffered BCPPO'
    training/rsh 'tail -n 3 ~/cr-robot-remote/train/runs/ppo_bc26/progress.jsonl'
    training/rsh 'tail -n 2 ~/cr-robot-remote/train/runs/ppo_bc26/eval.jsonl'
    training/rsh 'tail -n 5 ~/cr-robot-remote/train/runs/ppo_bc26_eval.log'

**Stop:** `training/rsh 'kill 3434021 3434022'`. SIGTERM loses at most 200
episodes, because a checkpoint is saved every 200 episodes. Send SIGINT
(`kill -INT 3434021`) to save first.

**Resume:** `training/rsh 'cd ~/cr-robot-remote/train && RUN=ppo_bc26 NUM_ENVS=16 bash training/launch_remote.sh'`.
This picks up `checkpoints/ppo_bc26.pth` automatically, including the update
count, the curriculum rung and the PFSP state.

**Fetch results to the Mac** (weights and logs only, no data):

    rsync -a -e "ssh -o ControlPath=$HOME/.ssh/cm-balerion" user@balerion:cr-robot-remote/train/{checkpoints,runs} remote_results/

**Weights & Biases.** The project is https://wandb.ai/clash-royale/clash-royale-ai.
- The sidecar is `training/wandb_sidecar.py`. On balerion it lives at `~/cr-robot-remote/train/wandb_sidecar.py`, uses the `~/cr-robot-remote/wbvenv` venv, and writes its log to `~/cr-robot-remote/wandb_sidecar.log`.
- It polls every directory under `runs/` every 60 s. For each run it reads `progress.jsonl`, `eval.jsonl`, `bc_metrics.jsonl` and every TensorBoard scalar, and creates one W&B run named after the run.
- Any new run that writes those files into `runs/<run>/` appears in W&B automatically. No code change is needed in the trainers.
- The read offsets are stored in `~/cr-robot-remote/wandb_sidecar_state.json`.
- The key is read from `~/.config/wandb/key` (mode 600). Never put the key in the repo.
- Restart command:
  `training/rsh 'cd ~/cr-robot-remote && setsid nohup wbvenv/bin/python -W ignore train/wandb_sidecar.py > wandb_sidecar.log 2>&1 < /dev/null &'`

**Mac launch path** (same code, CPU, 8 workers):

    .venv/bin/python -m training.ppo_bc --run ppo_bc26_mac --init checkpoints/bc_hog26.pth --num-envs 8 --lr 5e-5

On the Mac this needs `checkpoints/bc_hog26.pth` copied back first. It is
slow there: each CPU update with the BC terms takes minutes.

## BC (`checkpoints/bc_hog26.pth` on balerion)

* Training ran for 35 min on the 5090 at about 24k rows/s, 36 epochs over
  8,128 matches / 1.89M rows.
* The no-op weight was 0.286, with a post-hoc no-op logit shift of +1.25.
* `--lost-weight 0.5` was chosen by validation. In the sweep on the first 19
  shards it gave a card+place NLL of 4.078, against 4.090 for 1.0.

Held-out results: 424 matches, 90,253 decision rows, 21,370 play rows.

| metric | BC | baseline |
|---|---|---|
| card match, all decision rows | 0.766 | always no-op 0.763 |
| card match given the human played (which card) | **0.519** | uniform over legal cards is about 0.3 |
| cell exact | **0.224** | per-card modal cell 0.112 (the lead's modal-cell figure of 0.273 uses a different definition) |
| cell within one tile | **0.405** | |
| mean P(play) per decision vs human play rate | 0.221 vs 0.237 | calibration ECE 0.016 |
| greedy (argmax) play rate | 0.015 | a timing policy has to *sample* |
| val NLL card / place | 0.671 / 3.176 | |

* The ability head was not trained by BC. The dataset only flags
  `ability_ready` on rows where the human activated, so 100% of ready labels
  are positive.

**BC policy in the simulator:**

* The policy samples, with fixed seeds. There are 12 matches per opponent and
  192 in total.
* Overall it won **8.9%**, made 25.4 plays per match and leaked 5.3 elixir per
  match.

| opponent | win rate | plays/match | leak/match |
|---|---|---|---|
| teacher rung 0 (mirror deck) | **0.67** | 33 | 3.7 |
| teacher rung 2 | 0.25 | 42 | 7.5 |
| teacher rungs 4–10 | 0.00 | 18–35 | 7–11 |
| top-10 real decks at rung 2 | 0.05 on average (r0001 0.25, r0009 0.17, r0007 0.08, the rest 0) | 17–36 | 0–9 |

## PPO config (`ppo_bc26`)

* **Init:** BC weights. The first 6 updates were critic-only warm-up; at the
  start the explained variance was already 0.8 from the BC value head.
* **Reward:** outcome ±1, PBRS tower term (k=0, γ=0.999), leak worst case 0.2,
  draw 0.
* **Optimisation:**
  * lr 5e-5, eps_clip 0.2, 4 epochs, 8 minibatches, BPTT 50, GAE λ 0.9.
  * Each rollout is 500 steps × 16 envs.
  * Entropy coefficients are fixed: card 0.01, placement 0.005.
* **Anchoring to BC:**
  * KL(π‖π_BC) coefficient 0.1 → 0.01.
  * BC auxiliary CE coefficient 0.1 → 0.01.
  * Both decay log-linearly over 30k episodes, then hold at the floor.
* **Opponents:** the real-deck pool of 4,221 decks. Half the episodes draw by
  real frequency, the other half use PFSP over the top-24 decks plus a tail
  bucket. The teacher pilots on the 11-rung ladder from rung 0, with gate 0.65.
* **Off:** advisor, placement and deck coverage; scenarios; flawless bonus;
  engine shaping; replays.
* **Snapshots:** `checkpoints/ppo_bc26_snapshots/` every 2,000 episodes, and
  stage snapshots in `checkpoints/ppo_bc26_stages/`.
* **Throughput:** about 25 s per rollout+update, roughly 50 episodes per update
  and about 7k episodes per hour. The balerion CPU is shared with the
  detector agent, which uses about 20 cores. With 20 envs alone it measured
  30 s per update of 10k steps.

## ppo_league26 (self-play league, B arm)

Launched 2026-11-18 09:36 PST next to ppo_bc26, which keeps running unchanged.
It is forked from `checkpoints/ppo_bc26.pth` at **episode 134,601** (curriculum
rung 4). The fork keeps the weights, the optimizer state and
the update count, so the KL and BC coefficients are already at their 0.01
floors.

| process | PID (= its own process group, setsid) | log |
|---|---|---|
| trainer, GPU, 16 env workers | **3634909** | `runs/ppo_league26.log`, `runs/ppo_league26/{league.jsonl,progress.jsonl,events*}` |
| eval panel, CPU, `--procs 2`, every 2,000 episodes | **3635752** | `runs/ppo_league26_eval.log`, `runs/ppo_league26/eval.jsonl` |
| h2h watcher, CPU, `--procs 2`, every 2,000 episodes | **3635753** | `runs/ppo_league26_h2h.log`, `runs/ppo_league26/h2h.jsonl` |

The one-off ppo_bc26 reference h2h is in
`runs/ppo_league26/ref_ppo_bc26_h2h.jsonl`. It was computed from a copy, so no
ppo_bc26 file is written.

**Code.** All of it is in `training/`. The engine is untouched.
- `league.py` is the trainer. It subclasses `BCPPOTrainer` and `AnchoredUpdater`, and wraps the vector env.
- `league_env.py` is the per-env opponent: teacher, mirror or snapshot.
- `h2h.py` runs the head-to-head matches.
- `reward.py` gains `w_king`. It defaults to 0, which leaves ppo_bc26 bit-identical.

**Command:**

    training/rsh 'cd ~/cr-robot-remote/train && nohup setsid .venv/bin/python -u -m training.league --run ppo_league26 \
        --fork-from checkpoints/ppo_bc26.pth --init checkpoints/bc_hog26.pth --num-envs 16 --lr 5e-5 --seed 26 \
        --save-every 200 --torch-threads 4 >> runs/ppo_league26.log 2>&1 < /dev/null &'
    training/rsh 'cd ~/cr-robot-remote/train && CUDA_VISIBLE_DEVICES="" nohup setsid .venv/bin/python -u -m training.eval_panel \
        --watch ppo_league26 --bc checkpoints/bc_hog26.pth --every 2000 --procs 2 >> runs/ppo_league26_eval.log 2>&1 < /dev/null &'
    training/rsh 'cd ~/cr-robot-remote/train && CUDA_VISIBLE_DEVICES="" nohup setsid .venv/bin/python -u -m training.h2h \
        --watch ppo_league26 --every 2000 --procs 2 >> runs/ppo_league26_h2h.log 2>&1 < /dev/null &'

The league defaults are `--p-mirror 0.25 --p-snapshot 0.25` (teacher = the
rest), `--mirror-both-seats 1`, `--snap-dirs checkpoints/ppo_bc26_snapshots`,
`--snap-min-age 6000 --snap-uniform 0.10`, `--rung-bias-p 0.5 --rung-bias-lo 4
--rung-bias-hi 10` and `--w-king 0.1`. The trainer refuses `--run ppo_bc26`.

**Stop.** Use SIGTERM on the process group:

    training/rsh 'kill -TERM -3634909 -3635752 -3635753'

Then wait until `pgrep -g 3634909` is empty before relaunching. SIGINT does
**not** work: the AsyncVectorEnv workers take the KeyboardInterrupt and the
main process hangs, which was checked on the smoke run. SIGTERM loses at most
200 episodes.

**Resume.** Rerun the same trainer command. When
`checkpoints/ppo_league26.pth` exists, `--fork-from` is ignored, and the resume
restores the weights, the optimizer, the episode and update counts, the rung,
and the PFSP pool statistics (`ck["league"]`). This was checked on the smoke
run. `--num-envs` may differ on resume.

**Differences from ppo_bc26:**
- **Opponent mix per episode** (measured over the first 1,939 episodes: 0.502 / 0.252 / 0.246):
  - 50% the real-deck teacher, exactly as in ppo_bc26 (real-deck PFSP plus the rung ladder). When the opponent is off the curriculum (`--rung-bias-p 0.5`), the rung is drawn uniformly from 4–10. Only on-curriculum teacher episodes feed the gate and the rung ladder.
  - 25% mirror: the current policy plays both seats, and **both seats' trajectories go into the PPO batch**. Seat 1 gets its own mirrored observation, LSTM state and Hero abilities.
  - 25% frozen snapshots by PFSP, (1−p)^2 with a 10% uniform share. The pool is `ppo_bc26_snapshots` up to the fork episode plus this run's own `ppo_league26_snapshots`, each snapshot at least 6,000 episodes old (63 eligible at launch).
- **Opponent inference:** neural opponents run on the GPU under `no_grad`, batched per network across envs, with per-env LSTM state.
- **King-activation penalty:** 0.1 of a win, zero-sum. It applies only when a King wakes from damage, not when it wakes because a Princess Tower fell. We get −0.1 when we wake theirs and +0.1 when they wake ours.
- **Batch:** the PPO batch is 500 × 16 seat-0 columns plus up to 16 seat-1 columns (about 2.2k seat-1 rows per update so far). The minibatch segmentation drops all-padding segments. lr, clip, epochs, the entropy terms, the KL and BC floors (0.01) and γ are all unchanged.

**Smoke evidence** (run `ppo_league26_smoke`: 4 envs, then a 16-env resume at 90% mirror; its logs and checkpoints are kept in `runs/ppo_league26_smoke*` and `checkpoints/ppo_league26_smoke*`, and it also shows up in W&B):
- Every opponent type occurs. Typical updates mixed T8/M2/S1 and T3/M1/S5.
- Neural opponents play: 50–300 plays per update per kind, and abilities are used.
- The zero-sum residual |r0 + r1 + leak terms − 2·draw| is ≤ 3e-18 on every neural step.
- The King term fires only on damage wakes: `king_r_on_tower` is always 0.0.
- All losses are finite.
- Ratio dev (first minibatch) is 0.008–0.015, against 0.009–0.017 for ppo_bc26.
- Clip fraction is about 0.10 at 16 envs.

**Throughput** (first 14 min, load 15–17 with both runs and both eval watchers):

| | episodes/h | updates/h | learner rows/h |
|---|---|---|---|
| ppo_league26 | 8.4k | 195 | about 2.0M (8,000 + 2,158 seat-1 per update) |
| ppo_bc26, same window | 16.1k | 341 | about 2.7M |

Neural-opponent episodes are longer and run a second forward pass in the main
process.

**First rows.**

eval_panel (48 vs teacher rungs, 144 vs real decks):

| | episode | win rate | rungs | real decks | plays/match |
|---|---|---|---|---|---|
| ppo_league26 | 134,811 (fork + 210) | 0.573 | 0.486 | 0.625 | 39.6 |
| ppo_bc26 | 137,012 | 0.583 | | | 40.4 |

h2h (48 matches per reference, seats swapped):

| | episode | vs 60k snapshot | vs 119k snapshot |
|---|---|---|---|
| ppo_league26 | 134,811 | 0.604 | 0.458 |
| ppo_bc26 | 135,081 | 0.688 | 0.479 |

These are still the fork's own numbers. The standard error is about 0.07 at
n = 48.

**Where it differs from the spec** (closest feasible version without engine edits):
- **King wake:** the engine's `Tower::awake` is not exposed to Python. The wake is reconstructed from tower HP each step: the King woke if King HP < max, and the cause is "tower" if a Princess Tower is already down.
- **Elixir leak:** the leak term stays per seat, not zero-sum, as in ppo_bc26. The residual check subtracts it.
- **Decks:** in neural episodes team 1 plays our own deck (CLASH_DECK).
- **No eligible snapshot:** a snapshot episode falls back to mirror. This never happened here (63 eligible).
- **Head-to-heads:** both policies sample, as eval_panel's does.

## Engine patches

None. Everything is outside `engine/`. The reward, the opponent sampler and
the KL/BC hook go in through subclassing and name rebinding; see
training/README.md.

## Learning curve
In the console, "wr200" is the engine's rolling outcome window, which holds 100
matches. All of these episodes are at rung 0 against real decks. Full per-update
data is in `runs/ppo_bc26/progress.jsonl`.

| update | episodes | rolling win rate | KL(π‖π_BC) | BC aux CE | EV |
|---|---|---|---|---|---|
| 1–3 (critic warm-up) | 44–143 | 0.18–0.26 | – | – | 0.79–0.86 |
| 7 | 348 | 0.22 | 0.004 | 4.14 | 0.80 |
| 11 | 538 | 0.30 | 0.034 | 3.97 | 0.84 |
| 13 | 632 | 0.43 | 0.041 | 4.16 | 0.79 |
| 17 | 823 | 0.38 | 0.050 | 4.07 | 0.83 |
| 22 | 1060 | 0.37 | 0.070 | 3.99 | 0.76 |

Held-out human metrics of the current policy:

* At update 1 (BC): card-given-play 0.519, cell 0.220, within-1 0.397,
  P(play) 0.223.
* At update 11: card-given-play 0.519, cell 0.214, within-1 0.393,
  P(play) 0.299.

The policy keeps what it learned from the data and plays more often. The
real-deck PFSP buckets now range from 0.16 (r0000, the most common deck) to
0.69 (r0020).

Eval panel at episode 0: BC 8.9%, PPO 9.9%. The next panel row comes at about
episode 2,000; see `runs/ppo_bc26/eval.jsonl`.

## Known limitations / next steps
* ppo_bc26 only: frozen self-snapshots are saved but are not in its opponent
  mix. ppo_league26 adds them (see above).
* The secondary dataset (`data/bc/hog26_secondary`, variant 2.6 decks) is not
  used yet.
* ppo_bc26 has no network-piloted opponent, so its "vs the BC policy" means the
  same eval panel scored for both BC and the current policy. `training/h2h.py` now
  gives real network-vs-network head-to-heads.
* Sim fidelity: replayed human games kill about 3x fewer towers than the real
  ones did, so BC critic targets use the real outcome only.
