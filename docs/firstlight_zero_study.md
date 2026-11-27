# FirstLight-Zero: technical study and what it means for cr-robot

_2026-11-18. Source: shallow clone of https://github.com/Jaasssoooonnnnn/FirstLight-Zero at `4fc9ebe` (2026-11-04), read only and nothing executed. The only exception was loading `checkpoints/firstlight-zero-v1450.pt` with `torch.load(weights_only=True)` to count parameters. The clone is in `/tmp/flz/repo`. I also read the README, `docs/training_history.md` and `docs/architecture.md` of the parent project FirstLight_CR (same author) through `gh api`. I did not watch the video; everything here comes from the code and docs. Paths of the form `firstlight_zero/…` and `native_runner/…` are in their repo; `training/…`, `live_ppo/…` and `engine/…` are in ours._

---

## TL;DR

* **FLZ is a sim-only project.** It never plays the real game. It has no screen reading, detector, phone or tap code, and its NOTICE says it "doesn't connect to official servers".
* **The engine is the real Clash Royale battle engine (`libg.so`), lifted from Null's Royale**, an unauthorized private-server client (`package nullsroyale.rel.free`).
  * It is driven through an injected ARM64 probe that inline-hooks `GameApp.nOnTouchEvent`.
  * This makes it extremely faithful.
  * **We cannot use it.** It needs a modified client, client reverse engineering and a private-server APK. All three are against Supercell's ToS. See §1.4.
* **The model and RL recipe are clean Apache-2.0 Python and transfer directly:**
  * a 1.17M-param entity-scatter CNN + LSTM with a gate / card-pointer / FiLM-location head;
  * PPO with whole-episode BPTT;
  * 50% self-play (both seats train) and 50% PFSP snapshots;
  * a reward of ±1, tower-HP PBRS, two elixir-overflow penalties and a King-activation penalty.
* **It solves a much narrower problem than ours:** one deck (3.0 X-Bow), a mirror match, level 11, perfect state from the engine.
  * The "17M boss" is their own imitation model on the same engine, also playing the mirror.
* **Biggest cheap takeaways for us:**
  1. Evaluate and deploy at a **sampling temperature of about 0.5**, not argmax and not T=1. That was worth ~66 Elo over T=1 for them.
  2. Pick checkpoints by **round-robin Bradley–Terry**, not by a single panel number.
  3. Add a **unilateral overflow** term to the reward.
  4. **Model the real touch-to-execution latency** in training.
  5. Raise the decision rate with a learned wait gate.
  6. Build the **opponent cycle and elixir tracker** in live perception.

---

## 1. Environment and simulator

### 1.1 What it is

* `native_runner/` is a trimmed copy of FirstLight_CR's environment layer (`NOTICE`, `docs/architecture.md`).
* The docstring at `native_runner/cr_native_env.py:2-9` says the env "run[s] the real ARM64 battle engine inside the isolated stock client… Actions and structured observations use the native command/state APIs… without screenshot interpretation or ADB touch injection."
* **Engine source:**
  * `docs/getting_started.md` requires "Game version 15.535.13, content version 15.535.86" plus a user-supplied client.
  * The parent README names the supported engine as **"Null's Royale 15.535.13 / arm64-v8a"**.
  * The probe hard-codes `kPackageName = "nullsroyale.rel.free"` (`native_runner/probe/cr_replay_probe.cpp:49`) and fixed `libg` offsets such as `kReplayJsonSetterOffset = 0x00c8de8c` (line 60).
* **Probe:** 21.9k lines of C++/ARM64 assembly in `native_runner/probe/`.
  * It is built with the Android NDK (`build_probe.ps1`) into `libcrprobe.so` and injected into the APK.
  * `native_touch_interceptor.h:31-33` "Installs an inline hook at the v15.535.13 GameApp.nOnTouchEvent JNI entry."
  * It exposes a TCP control port, 26789 (`cr_replay_probe.cpp:44`), and runs 8 resident matches per engine process (`kEngineInstanceSlots = 8`, line 46).
* **Hosting:**
  * FLZ results were produced "on Linux with an extracted engine" (`docs/getting_started.md`, §1).
  * FirstLight_CR's quick start uses a rooted MuMu Player Android-12 emulator or a Linux AVD.

### 1.2 Fidelity

Because it *is* the game's engine, all card stats, levels, evolutions, Heroes, targeting and projectiles are exact for that client version.

* **Card data:** reviewed JSON exports of the release are in `native_runner/data/competitive/` (`card_specs.json`, `card_logic.json`, `effects.json`, `projectiles.json`, …). FirstLight_CR claims "122 supported cards".
* **Elixir schedule:** `native_runner/battle_env.py:133-137`. These are 50 ms ticks (`TICK_MS = 50`, line 118):
  * ticks 0–2400 at 28 s per full bar;
  * ticks 2400–4800 at 14 s;
  * ticks 4800–6000 at 9.3 s.
  * `GAMEPLAY_END_TICK = 6000`, i.e. 300 s (`firstlight_zero/spec.py:53`).
* **Command latency:** `cr_native_env.py:34-38` says "A stock client touch at tick t is executed by the battle at t + 20 … Online play gets this delay from the server; simulated matches must add it explicitly."
  * `battle_env.py:2713-2733` models this as 1 touch tick + 20 server ticks + uniform 0–3 ticks of jitter.
  * Training turns the jitter on with `jitter_ticks=3` (`firstlight_zero/launch.py:24`, `collector.py:111`).
* **Contrast with ours:**
  * Our engine has known gaps: triple elixir at 3:00, no pocket placement, 7 missing Hero/Evo forms, a 20-tick hand cooldown, and 100% spell crown-tower damage (`MORNING_REPORT.md` §1).
  * Theirs has none of these by construction.
  * Their "20-tick execution delay" is the real-game analogue of what we model as a hand cooldown. That is worth checking against our iPhone recordings (§8, R4).

### 1.3 Build requirements

* **Python side:** Python 3.12, `torch>=2.4`, `numpy`, `orjson`, `pyarrow` and `sc-compression` (`requirements.txt`). It needs a CUDA GPU because the learner calls `torch.cuda.set_device` unconditionally (`learner.py:83`).
* **Engine side:** the Android NDK (aarch64 clang), the Null's Royale APK and its downloaded resources under `$CR_WORKSPACE_ROOT`, and either an extracted-engine Linux runner or an ARM64 Android emulator.
* The only test that needs no engine is `tests/test_consistency.py`. It checks that step-wise `act` and whole-episode `evaluate` agree to within 5e-2 in log-prob and value (line 57).

### 1.4 License, and can we use it?

* **Code:** Apache-2.0 (`LICENSE`, `NOTICE`), so the Python can be ported freely with attribution.
* The NOTICE explicitly excludes the game binaries: "User-supplied APKs… remain subject to their respective owners' rights."
* **⚠️ ToS flag. Do not adopt the engine, or any part of the probe or APK tooling.** Running it requires:
  1. obtaining and running Null's Royale, an unauthorized private-server mod of Clash Royale;
  2. patching and re-signing the APK with an injected native library;
  3. reverse-engineered offsets and inline hooks into Supercell's binary.

  Supercell's ToS and Fair Play policy prohibit modified clients, private servers, and reverse engineering or decompiling the client. This sits squarely outside our rules: no anti-cheat bypass, no blocked emulators, live play in Training Camp only. It would also taint any result we publish for the course.
* **What *is* usable:**
  * the PPO, league and inference code;
  * the model architecture;
  * the reward formulas;
  * the evaluation protocol (Bradley–Terry round robin, temperature sweep).

  None of these depend on the engine. The released weights are X-Bow-mirror-only, use their observation schema and are useless to us (§7).

---

## 2. Observation encoding

The schema is defined in `firstlight_zero/spec.py`; featurization is `Featurizer2.__call__` in `firstlight_zero/features.py:53-179`. Everything is in the acting seat's perspective, with owner 1 rotated 180° (`legality.py:43-53`, `features.py:87,110`).

| Input | Shape | Contents (code) |
|---|---|---|
| `unit_type` | 40 (`MAX_UNITS`, spec.py:79) | Index into `UNIT_VOCAB` = 22 types (spec.py:59-75): 8 forms (Hero Knight, Archer, Evo Archer, Skeleton, E-Spirit, X-Bow, Tesla, Evo Tesla), 10 projectile types, princess, king, other. Towers are included as units (features.py:65-71). Area effects are skipped (line 75) |
| `unit_feat` | 40 × **17** (spec.py:82-86) | `mine, x, y` (normalized 0–1 over 18000×32000 world units), `hp_ratio, hp/5000, age/60s, has_target, vx, vy` (tiles per tick), `attack_cooldown_remaining/5s, attack_windup, attack_cooldown_phase, deploying, deploy_remaining/5s, projectile_in_flight, projectile_target_x, projectile_target_y` (features.py:80-100) |
| `scal` | **31** (spec.py:88-99) | Time remaining/300 s, multiplier/3, overtime, tiebreak, **my elixir/10, reserved elixir/10, opponent elixir estimate/10, opponent elixir upper bound/10**, both sides' crowns, 6 tower-HP ratios, both King-activated flags, my and opponent evo-ready and evo-cycles-remaining for Archers and Tesla, my Hero-ability-ready, opponent Hero-on-board-and-ready, `i_have_played`, both sides' Hero-ability remaining window (features.py:134-175) |
| `cards` | 8 × **5** (spec.py:101) | For each card of the deck in fixed `CARDS` order: `in_hand, is_next, pending, pending_x, pending_y`. "Pending" is an issued but not yet executed touch, which makes the 1 s latency visible to the net (features.py:105-115) |
| `opp_cards` | 8 × **3** (spec.py:103) | **Opponent-card memory** from FirstLight's public `DeterministicPublicTracker`: `available` (in the opponent's hand), `cycling`, `cycle_distance/4` (features.py:116-124). The opponent's deck is known (mirror), so the tracker is a deterministic 8-card cycle model updated from observed plays |
| `option_card` | 5 (int) | The card index in each of the 4 hand slots, plus the next card in slot 4 (`legality.py:105-112`) |
| `opt_feat` | 5 × **3** (spec.py:105) | Per option (4 slots + Hero ability): `cost/10, legal_cell_fraction, earliest_delay_bin/4` (features.py:127-132) |

Masks (legality.py:91-226):

* **`opt_mask[5]`** covers affordable or legal options, including "affordable within the 0–200 ms delay window" via `touch_window_elixir` (`battle_env.py:388-409`, `legality.py:242`).
* **`place[4,32,18]`** holds per-slot legal cells, bit-packed for storage (`store.py`, `batch.py:22`).
* **`force`** is set when tick ≥ 300 (15 s) and the seat has not yet played (`legality.py:223`).

Opponent elixir is tracked two ways (`features.py:138-141`, `legality.py:55-89`):

* a deterministic estimate: regen, minus the `effective_cost` of each executed enemy play;
* the public tracker's upper bound.

**Board grid:** 18 × 32 tiles (`spec.py:46`). Units are not pre-rasterized. The *network* scatters them onto the grid (§3).

**Contrast with ours:** we feed a dense raster plus scalars of 13,977 floats (`training/net_v2.py:8`) on a 34 × 18 board.
* Their observation is about 40×17 + 31 + 40 + 24 + 15 + 5 ≈ 800 numbers.
* It also carries **dynamics features we lack**: velocity, attack phase, deploy timer and projectile target.
* It has an explicit **opponent cycle model**. Ours exists in the encoding, but live it is always empty (`runs/live/REPORT_2026-11-18.md` §3).

---

## 3. Architecture (`firstlight_zero/model.py`, `XbowNet`)

The trained config is `channels=64, dilations=(1,2,4,8), scatter_dim=16, hidden=256` (`launch.py:26`; the class defaults to 96 channels, model.py:44). I counted the parameters from the v1450 state_dict: **1,168,723 total**.

| Block | Code | Layers | Params |
|---|---|---|---|
| Unit encoder | model.py:47-48 | `Embedding(22,16)` + MLP(16+17 → 64 → 16) | 3.6k |
| Scatter | model.py:80-108 | Each unit's 16-d vector is **bilinearly splatted** onto its 4 neighbouring cells of the 18×32 grid (`index_add_`, fp32) | – |
| Explicit planes | model.py:97-107 | 8 count planes: own/enemy × (characters, buildings incl. towers, projectiles, Σ hp/5000), nearest cell | – |
| Positional | model.py:50-52 | 8 **learned** position planes `pos[8,32,18]` + 2 fixed coordinate planes | 4.6k |
| Stem | model.py:54 | `Conv2d(34→64, 3×3)` (34 = 16 + 8 + 8 + 2) | 19.6k |
| CNN | model.py:29-40, 55 | 4 pre-activation residual blocks, GroupNorm(8) → SiLU → 3×3 dilated conv, twice; dilations 1/2/4/8; second conv zero-initialized; bf16 autocast on CUDA (line 122) | 296k |
| Board summary | model.py:128 | Global mean ‖ max over the grid → 128 | – |
| Vector encoder | model.py:58, 130-131 | concat(scal 31, cards 40, opp_cards 24, option-card embeddings 5×32, opt_feat 15) = 270 → MLP 256 → 128. Card embedding `Embedding(9,32)` | 102k |
| Trunk | model.py:59 | Linear(128 + 128 → 256) + ReLU | 66k |
| LSTM | model.py:60, 134-138 | `nn.LSTM(256,256)`, single layer, then **residual + LayerNorm**: `norm(lstm_out + z)` | 526k |
| Gate head | model.py:62, 70-71 | Linear(256 → 2): WAIT / ACT. **Bias initialized to log(0.85), log(0.15)** and weights ×0.01, so the policy starts out acting rarely | 0.5k |
| Option (card) head | model.py:63-64, 141-145 | **Pointer**: query = Linear(h → 64); key = MLP(card_emb 32 + opt_feat 3 + is_ability 1 → 64 → 64); logits = q·k / 8 over the 5 options; masked with −1e9 | 23k |
| Location head | model.py:65-68, 147-153 | `cond = MLP(h 256 + chosen-card emb 32 + is_ability → 128 → 128)`; **FiLM**: `γ, β = Linear(128 → 2·64)`, applied to the CNN map as `g·(1+γ) + β`; keys = 1×1 conv (64 → 32) per cell; query = Linear(128 → 32); logits = q·k/√32 over 576 cells; masked by `place`. FiLM and query are initialized ×0.01, so cells start near-uniform | 76k |
| Value | model.py:69, 186 | MLP(h 256 + board summary 128 → 128 → 1) | 49k |

**Action semantics** (model.py:1-5, `actions.py`):
* One decision every 5 ticks (250 ms, `spec.py:49`).
* The gate decides whether to act. If it acts, it picks one option. If the option is a card, it picks one of 576 cells.
* There is **no delay head**: the earliest legal delay bin of 0–200 ms is used (`actions.py:14`).
* There is at most one card per decision.

**Masking and log-probabilities:**
* The gate log-prob only counts when `gate_free`, meaning some option is legal and no open is forced (model.py:165-170, 259-261).
* Option and location log-probs are added only when acting, and only for card plays (model.py:262-272).
* Entropy is the sum of the active heads' entropies.
* For CUDA-graph capture, `heads()` computes location logits for all four slots with static shapes (model.py:190-205).

**Recurrence and BPTT:**
* **Whole-episode BPTT** from a zero initial state (`evaluate`, model.py:255-258).
* Minibatches are 8 complete seat-episodes, padded time-major (`learner.py:40-59`), sorted by length (`learner.py:104`). That is about 1,200 decisions per episode and ~9.4k rows per minibatch (model.py:113).
* During collection, the inference server keeps LSTM state per (client, seat) (`server.py` docstring).

**Design note from the authors** (`docs/architecture.md`): "an earlier design without a board view learned the X-Bow but never learned Fireball or Tesla placement". The FiLM-on-CNN location head is what fixed it. Our `training/net_v2.py` already has the same idea: FiLM on a full-resolution, dilated 64-wide board. But **ppo_bc26 runs the v1 `MicroRoyaleNet`** (live report: "loads with `MicroRoyaleNet(num_ability_slots=1)`").

---

## 4. Training

### 4.1 PPO hyperparameters (`firstlight_zero/launch.py:18-28`; the docs table is in `docs/training_history.md`)

| | FLZ | ours (ppo_bc26, `training/STATUS.md`) |
|---|---|---|
| decision interval | 250 ms (4 Hz) | 1 s (1 Hz) |
| γ / λ | **0.9993 / 0.95** (horizon ≈ 1,430 decisions ≈ 6 min, i.e. the whole match) | 0.999 / 0.9 |
| lr | 3e-4, Adam eps 1e-5 (`learner.py:151`), **no annealing** | 5e-5 |
| clip / value clip | 0.2 / **0.5 absolute** (`learner.py:125-126`) | 0.2 / std-relative |
| entropy coefficient | 0.003 over the summed heads, constant | card 0.01, placement 0.005 |
| vf_coef / max grad norm | 0.5 / 1.0 | – |
| epochs | **2** | 4 |
| batch | **1,024 seat-episodes per update** ("round") | 500 steps × 16 envs ≈ 50 episodes |
| minibatch | 8 whole episodes | 8 minibatches, BPTT 50 |
| advantages | GAE per episode, normalized over the whole batch (`learner.py:101-103`) | |
| staleness | async; inbox read newest-first, backlog beyond 2 batches deleted, `max_lag=8` versions (`training.py:26-50`) | synchronous |
| sampling temperature | 1.0 in training | 1.0 |
| imitation / KL anchor | **none** (random init) | BC init + KL-to-BC 0.1→0.01 + BC aux CE 0.1→0.01 |

Their logged health metrics (`docs/training_metrics.md`):
* entropy 0.46 → 0.18;
* explained variance > 0.9 by v80, about 0.977 late;
* approx-KL 0.004 → 0.017 → 0.010;
* clip fraction 3–5%;
* grad norm ≈ 0.13–0.17, never clipped.

### 4.2 Self-play and PFSP

All of this is in `firstlight_zero/collector.py` and `training.py`.

* **50/50 split** (`collector.py:133-138`):
  * with probability `self_play_fraction = 0.5`, it is current vs current and **both seats** write trajectories (`seats={o: SeatStore() for o … if roles[o]=='current'}`, line 146);
  * otherwise the current policy takes a random seat against a history snapshot, and only the current seat trains.
* **PFSP weight** (`collector.py:57-62`): `(1 − (n·score + 1)/(n + 2))^2`.
  * This is a Laplace-smoothed win score, so unseen snapshots count as 0.5.
  * `score` is the mean over the last `pfsp_window = 300` results against that snapshot (`training.py:53-82`).
  * A share `pfsp_uniform = 0.1` is drawn uniformly (`collector.py:135`).
* **Snapshots:** every `snapshot_every = 10` versions (`training.py:20-21`). The whole history stays eligible, about 150 snapshots by v1500. The inference server caches up to 256 models (`server.py:78-80`).
* **Opponent sampling** is decided per game at reset. The roster is refreshed every 20 frames (`collector.py:181`).
* **Effect:** late in the run the current policy scores about 53% against snapshots ≤20 versions back and about 71% against those >60 back (`docs/training_metrics.md`, "Older versions").

### 4.3 Reward (`firstlight_zero/reward.py:21-29`, recipe `hp02-win1.v1`)

| Term | Coefficient | Definition |
|---|---|---|
| Terminal | ±1 (`terminal_reward_scale = 1`), draw 0 | |
| Tower potential | **β = 0.2**, `shaping_tau_ticks = 5/−ln(0.9997)` | PBRS on Φ = crown difference + Σ tower-HP loss in units of mean princess HP, from the opponent's towers minus your own (`battle_env.py:4468-4510`). The King's loss while its Princesses stand is **weighted 0.2** (`king_shaping_mode='hp_point_weighted'`, `king_potential_weight=1` after that). Φ(terminal) = 0, so the term is policy-invariant (comment at `battle_env.py:4511-4514`) |
| Personal overflow | **0.0225 per elixir wasted beyond a 4-elixir grace**, cumulative over the episode | `reward.py:99-104` |
| Unilateral overflow | **0.075 · w²**, where w = elixir wasted *while the opponent is not also full*; resets whenever both are full | `reward.py:105-112`. This punishes leaking only when it is a real disadvantage |
| Per-step cap | 0.1 combined overflow penalty per transition | `reward.py:113-116` |
| King activation | **0.1**, zero-sum, once per King, `scope='any'` | `native_runner/training/v4/king_activation_reward.py:242-` |

Every term is zero-sum, and none is card-specific (`docs/architecture.md`).

**Compared with ours** (`training/reward.py`):
* Our PBRS uses mean tower damage fraction × w_tower 1.0, with no King discount.
* Our leak penalty is linear and tiny: 0.2 of a win for leaking an entire match, about 7e-4 per elixir, versus their 0.0225 per elixir after grace.
* We have no unilateral term.
* Our `w_king = 0.1` (ppo_league26) is the same idea as their King-activation term. FLZ's own docs are the reference our `league.py` docstring cites.

### 4.4 Curriculum, pretraining and tricks

* **No imitation and no curriculum.** Weights start random. The deck is fixed.
  * FirstLight_CR (the parent) did use IL → PPO, and logged a cautionary result: a fixed-IL-opponent run hit 80% wins, but 296 of 306 wins came against an IL opponent that made ≤3 actions. They were exploiting passivity (FirstLight_CR `docs/training_history.md` §2).
* **Forced opening:** if a seat has not played by 15 s, the gate is forced to ACT while the policy still picks card and tile (`legality.py:223`, `model.py:166-169`). The forced gate gets no log-prob (`gate_free = False`).
* **Gate prior:** P(ACT) = 0.15 at init (model.py:70-71).
* **Execution jitter:** 0–3 ticks in training only (`launch.py:24`).
* **Value clipping:** absolute 0.5 (`learner.py:125`).
* **Advantage normalization:** per batch.
* **No reward normalization, no annealing.**
* **Episode validity:** games with any rejected action are dropped from training (`collector.py:155`, `valid = … not any(g['rejected'])`).
* **Throughput tricks:**
  * CUDA graphs per (model, batch-size bucket) (`graph.py`), with coarse buckets for history models;
  * server-side LSTM state;
  * float16 wire format (`batch.py:9`);
  * trajectory writes moved off the game loop to dodge GPFS stalls (`collector.py:114`);
  * selective checkpointing of the CNN for big minibatches (model.py:110-119).
* **Checkpoint selection** (`docs/training_history.md`, "Choosing the final model"):
  * the 10 best Boss-sweep snapshots × 5 temperatures, a full round robin of 2,450 games;
  * Bradley–Terry Elo with bootstrap CIs and P(best);
  * **T = 0.5 beat both T = 1.0 (+66 Elo mean) and argmax (+18).**

---

## 5. Real-game deployment

* **FLZ does not play the real game.** It has no capture, no detector and no actuation code. The "native-render" mode of `cr_native_env.py` draws the *offline* engine's own scene; it does not read pixels.
* Their only "perception" is the engine's structured telemetry. That has very wide coverage (rich telemetry, phase and visibility runtimes; `native_runner/rich_telemetry_adapter.py`, 4.6k lines). **None of it is available to a screen-reading agent.**
* **Action delay** is handled *inside the model of the game*, not by a live controller:
  * the 20-tick server execution delay, plus the 0–200 ms "earliest legal delay" bins, plus jitter;
  * `pending` card features expose touches that are in flight;
  * affordability is judged on the elixir that *will* exist at execution (`touch_window_elixir`, `battle_env.py:395-409`).
* **The parent project's "Hall of Fame" claim** (FirstLight_CR `docs/training_history.md`, line 88) is an "author-reported account achievement" with the Hog 2.6 specialist. The repo does not document *how* it was played.
  * We should not infer or replicate it.
  * Any bot play on ladder would breach ToS and our own rules.

---

## 6. Throughput

* **Figures:** "1,500 rounds took 43.3 hours, about 104 seconds per round" (`docs/training_history.md`, Cost).
* **What a round is:** each round consumes `batch_episodes = 1024` *seat-episodes*.
  * Self-play games yield 2 seat-episodes and history games yield 1.
  * So one round ≈ 1024 / 1.5 ≈ **680 games**.
  * The whole run is **≈ 1.54M seat-episodes ≈ 1.0M games**, not counting stale games discarded by the reader.
  * **The "~1.5M games" figure is very likely the seat-episode count (1,500 × 1,024).**
* **Rate:** ≈ **23–24k games/hour** (≈ 35k seat-episodes/h), ≈ 6.6 games per second of simulated 300 s matches. That is about 2,000× real time in aggregate.
* **Hardware:** not stated explicitly. Clues:
  * "80 GB card" (`model.py:114`), so an A100 or H100 learner;
  * GPFS, `ssh`-spawned remote engine hosts and CPU pinning (`hosts.py`, `launch.py`), so an HPC cluster;
  * 8 resident matches per engine process; `engines.example.json` shows several engines per host;
  * the parent FirstLight_CR PPO used "1,528 concurrent matches … eight GPUs".
  * At 104 s per round and ~680 games per round, FLZ needed on the order of **~2,000 concurrent games**, so likely hundreds of engine processes across many CPU nodes.
* **Speed-ups:**
  * the asynchronous actor/learner split credited to daybreak-cr ("1.5× faster");
  * CUDA graphs;
  * centralized batched GPU inference;
  * dropping stale data instead of queueing it.
* **Ours:** about 7k *episodes*/h on one shared 5090 box with 16 env workers (`training/STATUS.md`, Throughput), so roughly 3–5× fewer games per hour. The gap is mostly CPU and engine bound, not algorithmic.

---

## 7. Released assets

| Asset | Where | Use to us |
|---|---|---|
| 30 checkpoints, v50…v1500 every 50, ~4.7 MB each, SHA-256 in `checkpoints/manifest.json`. **v1450 at T = 0.5 recommended** | `checkpoints/` (in-repo, not LFS) | X-Bow mirror only, their 31/40×17/… schema, their engine's ids. **Not usable** as a policy or opponent for us. Useful only as a reference implementation and for parameter-count sanity |
| IL "Boss" (Everlight, 16.8M) used for eval | **not included** (`checkpoints/README.md`) | – |
| Datasets | None in FLZ. The parent links **IL_Replay** on HF (`VanguardX101/IL_Replay`, 252,238 replays, 17.8M actions), which we already use | Already in our pipeline |
| Detector or perception models | **None.** No screen perception exists | – |
| Parent FirstLight_CR checkpoints | General, IL, Active IL, **two Hog 2.6 specialists, on our exact deck** (Evo Cannon, Evo Skeletons, Hero Musketeer, Hog, Ice Golem, Ice Spirit, Fireball, Log) | Tempting as a strong reference opponent. But their V4 inputs come from native-engine telemetry, so running them needs their engine. **Same ToS problem as §1.4. Not usable** |
| Tooling | `evaluate.py` (sweeps, round robin, Bradley–Terry), `tools/eval_analysis.py` (tactics counts), `tests/test_consistency.py` | Portable ideas, Apache-2.0 |

---

## 8. Recommendations for cr-robot, ranked

Ranking weighs expected win-rate impact, in sim and especially live, against effort. Nothing here requires the FirstLight engine or any ToS-violating component.

| # | Change | Why (FLZ evidence) | Expected impact | Effort | Our files |
|---|---|---|---|---|---|
| **R1** | **Temperature sweep at evaluation and deployment.** Add `--temp` (gate, card and placement separately) to the eval panel, h2h and the live driver. Sweep T ∈ {0.3, 0.5, 0.7, 1.0}, plus argmax with a gate-only temperature | FLZ: T=0.5 is +66 Elo over T=1 and beats argmax (§4.4). Our BC argmax play rate is 0.015 (STATUS "greedy play rate"), so pure argmax is broken for timing. Temperature lets us keep stochastic timing while sharpening placement | **High, likely several points of win rate**, essentially free | **Low** (½ day) | `training/eval_panel.py`, `training/h2h.py`, `live_ppo/driver.py` (beside `--greedy`) |
| **R2** | **Select checkpoints by round robin with Bradley–Terry**, not by one eval_panel number. Take the top ~8 snapshots × 2–3 temperatures, play each pair both seats, and report Elo with bootstrap CIs | FLZ's selection protocol. FirstLight_CR's lesson that a high win rate against one opponent can be exploitation (296 of 306 wins vs a near-passive IL) | Medium: avoids shipping a lucky or exploitative checkpoint to the phone | Low–medium | `training/h2h.py`, a new `training/roundrobin.py` |
| **R3** | **Log opponent activity in eval.** Record opponent plays per match and first-play tick, and flag wins where the teacher made ≤3 plays | Same FirstLight_CR finding | Medium. It guards against a fake 60% | Low | `training/eval_panel.py`, `training/league.py` logging |
| **R4** | **Model real touch→execution latency in training.** Measure it on our iPhone recordings: our perception lag (≈50 ms capture + 50–190 ms YOLO, live report §2) + tap + the game's own deploy delay (FLZ: 20 ticks = 1 s, `cr_native_env.py:34-38`). Add a per-episode random action delay of 0–N engine ticks in our env wrapper, plus a "pending play" feature so the LSTM knows a card is in flight | FLZ trains with exact delay + 0–3 tick jitter (§1.2, §4.4) and exposes `pending` card features (§2). Our sim currently acts with zero latency, while live adds ~0.3–1.3 s | **High for live**, low in sim | Medium. Wrapper-only if the delay is applied in `RewardEnv.step` by queueing the action; the pending feature needs an obs change | `training/reward.py` (`RewardEnv.step`), `training/league_env.py`; obs feature via an engine upstream request |
| **R5** | **Opponent cycle and elixir tracker in live perception**, feeding the cycle blocks and `e+2`, which are always empty live. From YOLO enemy *new-unit* events, infer the card played, keep a deterministic 8-card cycle (available / cycling / distance) once the deck is seen, and estimate opponent elixir as regen minus costs, with an upper bound | FLZ's `opp_cards` (8×3) + two opponent-elixir scalars (§2) are core inputs. Our net was trained with these populated, so live they are out of distribution (live report §3 rows `e+2`, cycle) | **High for live** | Medium (2–4 days) | `live_ppo/driver.py`, `video_dataset/detector` mapping, `engine/perception` (upstream request #36) |
| **R6** | **Add the unilateral-overflow penalty** (0.075·w², reset when both sides are full, step cap 0.1). Optionally a personal grace term of 0.0225 per elixir after 4 elixir | FLZ `reward.py:99-116`. It punishes leaking only when the opponent is not also leaking, which is the strategically meaningful case. Our current leak term is ~30× weaker per elixir; leak is already down to 0.2 per match, so the gain is in *when* it holds elixir, not how much | Low–medium | **Low**: same pattern as `w_king`, default 0 keeps ppo_bc26 identical. Needs the opponent's elixir, which the env has | `training/reward.py`, `training/league.py` flags |
| **R7** | **Discount King chip in the PBRS potential** (weight 0.2 on King damage while both Princesses stand) | FLZ `battle_env.py:4477-4485`. Removes the incentive to chip a sleeping King, complementing `w_king` | Low–medium | Low | `training/reward.py` (`potential`) |
| **R8** | **Switch the next BC→PPO run to `net_v2`**, adopting FLZ details: gate bias at P(act) ≈ the human play rate; LSTM residual + LayerNorm (`norm(lstm(z) + z)`); a value head that also reads the board's mean/max summary; a pointer-style card head (query·key over card embeddings + per-option cost and legal-cell fraction) | FLZ's ablation: without a board-conditioned location head it never learned spell or building placement. ppo_bc26 still runs v1 `MicroRoyaleNet`, where the LSTM input is 96% of params (net_v2 docstring) | Medium–high in sim (placement quality: Fireball, Log, Cannon) | Medium–high (new BC run of ~35 min on the 5090, then PPO from scratch with our anchors) | `training/net_v2.py`, `training/ppo_bc.py`, BC script |
| **R9** | **Raise the decision rate from 1 Hz to 2–4 Hz** with an explicit wait gate (net_v2 already supports one) | FLZ decides at 4 Hz with one card per decision, which is critical for reactive defense such as Log timing and Ice Spirit on Hog. Our live perception runs at about 16 Hz (live report §1), so 2–4 Hz is feasible live. BC labels must be re-binned and γ adjusted (γ_new = γ^(1/k)) | Medium–high | **High**: env `skip_frames`, BC data rebuild, retraining, and live loop timing | `training/reward.py` (`SKIP_TICKS`), `training/ppo_bc.py`, BC builder, `live_ppo/driver.py` |
| **R10** | **League mix toward FLZ's recipe once ppo_league26 is stable.** Raise mirror (both seats) to about 40–50%. Use **Laplace-smoothed PFSP** `(1-(n·s+1)/(n+2))^2` over a 300-game window instead of an EMA with p₀ = 0.5. Keep 10% uniform. Snapshot more often | FLZ §4.2. Smoothing stops new snapshots from being ignored or over-sampled from 1–2 games. Snapshots ≤20 versions back stayed near 53%, so the window is informative | Medium (robustness, less forgetting) | Low | `training/league.py` (`weights()` ~line 165, `--p-mirror`, `--p-snapshot`), `training/league_env.py` |
| **R11** | **Add dynamics features we can perceive live:** unit velocity (from YOLO tracking across frames), HP ratio, and "deploying" (spawn animation). Do **not** add attack-phase or projectile-target features we cannot see on screen | FLZ unit features (§2) | Medium | Medium–high (obs schema change = new BC) | engine obs (upstream), `live_ppo/driver.py` tracking |
| **R12** | **Act/train consistency test.** Like `tests/test_consistency.py`: run recorded observations through the live `PPOPolicy` step by step and through the trainer's `forward_sequence`, and assert that log-probs and values match | It would have caught the hand base→Evo remap and the stale-LSTM issues earlier (live report §1, §3) | Medium (bug prevention) | Low | new `training/test_live_consistency.py`, `live_ppo/driver.py` |
| **R13** | **Throughput:** a centralized batched GPU inference server for env workers and snapshot opponents, async rollouts, and dropping stale batches | FLZ ~24k games/h vs our ~7k episodes/h; daybreak-cr's async design gave 1.5× | Medium (more PPO samples per wall-clock hour) | High; our trainer lives in `engine/python_ai/rl` | `training/league.py`; most changes upstream in `engine/` |
| **R14** | **Whole-episode BPTT** (or BPTT 50 → 200+) at our 1 Hz rate. An episode is only about 180–300 decisions, so it fits easily | FLZ trains on full episodes from a zero state. Truncation at 50 s hides elixir-saving and cycle plans | Low–medium | Medium (engine trainer config) | `engine/python_ai/rl/config.py` (upstream), `training/ppo_bc.py` |
| **R15** | **Lower PPO epochs (4 → 2) and use a larger batch** if KL/clip fraction run hot | FLZ: 2 epochs on 1,024 episodes; clip fraction 3–5%, KL ~0.01 | Low | Low | `training/ppo_bc.py` |

**Explicitly not recommended (ToS):**
* the FirstLight / Null's Royale engine;
* the probe, APK build tools and inline hooks;
* emulator-hosted clients;
* FirstLight_CR's Hog 2.6 specialist as an opponent (it needs that engine).

Live play stays on the iPhone through macOS iPhone Mirroring, in Training Camp only, behind the existing `FreshGate` and Training Camp guard (`live_ppo/driver.py`).

### Suggested order
1. **R1 + R3 + R12 this week.** They are cheap and de-risk the first Training Camp run.
2. **R4 + R5** before trusting any live win rate.
3. **R6 / R7 / R10** as flags on the next league fork.
4. **R8 → R9** as the next major BC→PPO generation.
5. **R13 / R14** if compute becomes the bottleneck.
