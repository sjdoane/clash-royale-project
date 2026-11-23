# training/ - BC -> PPO for the Hog 2.6 bot

Everything here lives outside `engine/` (read-only submodule). The engine's
modules are imported through `training/common.py`, which also sets
`CLASH_DECK=125,168,129,15,40,72,33,7` (evo Skeletons, Hero Musketeer, evo Cannon,
Hog, Ice Golem, Ice Spirit, Log, Fireball). That deck id order matters: the Hero
ability head drives deck slot 1. No engine patch was needed (no `patches/0004`).

| file | what |
|---|---|
| `common.py` | paths, deck, engine imports, `new_net()` (the exact PPO network), `load_weights()` |
| `reward.py` | the minimal reward + `RewardEnv` + `install_reward()` |
| `opponents.py` | real-deck opponent pool (`data/decks/train_pool.json`) + sampler |
| `bcdata.py` | human dataset loader (sparse CSR cache in `data/bc_csr/`) |
| `bc.py` | recurrent behaviour cloning |
| `ppo_bc.py` | PPO from BC (engine Phase1Trainer subclass) |
| `eval_panel.py` | fixed-seed evaluation panel, `runs/<run>/eval.jsonl` |
| `tests/test_reward.py` | telescoping / leak / env-wiring tests (`.venv/bin/python -m pytest training/tests -q`) |
| `rsh` | ssh to balerion through the existing ControlMaster socket only |

## Reward (`reward.py`)

    r_t = outcome (+1 win / -1 loss / --draw, terminal only)
        + gamma * Phi(s_{t+1}) * (1 - done) - Phi(s_t)
        - c * leak_t

* `Phi(s) = w * (mean_i g(d_enemy_i) - mean_i g(d_own_i))`, where `d` is each
  tower's damage fraction and `g(d) = d`. With `--k > 0`, `g` is the convex
  `(e^{kd}-1)/(e^k-1)` instead.
* `Phi(terminal) = 0`, so `sum_t gamma^t F_t = -Phi(s_0) = 0`. The shaping is
  exactly policy-invariant, and the unit test checks this on real episodes.
* `leak_t` is the elixir generated while the bar sat at 10. `c` is set so that
  leaking for an entire 3600-tick match (about 273 elixir) costs 0.2 of a win.
* Nothing else counts. The engine's shaping, flawless-defense bonus and draw
  penalty are switched off by rebinding names in `base_trainer`.
* The env's returned reward stays the raw outcome, so the engine's win/draw
  bookkeeping (`|r| > 0.5`) is untouched. The shaping travels in
  `info["r_shape"]`.

## Opponents (`opponents.py`)

* The pool is all 4,221 real opponent decks that 2.6 players faced, covering
  14,305 sides. It comes from `data/decks/il_replay_pool.json`.
* Every deck is validated against the engine and the UtilityTeacher, and none
  were skipped. 2,520 decks contain a substituted base form, for example
  berserker-hero becomes Berserker.
* The opponent for each episode is picked one of two ways:
  * With probability `--p-freq` (0.5), a deck is drawn by its real frequency,
    long tail included.
  * Otherwise PFSP runs over 25 buckets: the top 24 decks, plus a "tail"
    bucket drawn by frequency. It uses the engine's `(1-wr)^2` weights with a
    0.05 floor.
* The UtilityTeacher pilots the drawn deck at the current curriculum rung
  (11-rung ladder, gate 0.65).
* Frozen self-snapshots are saved to `checkpoints/<run>_snapshots/` every 2,000
  episodes. They are not yet used as opponents; that is the next step, via
  pipeline 2.

## BC (`bc.py`)

* The data is `data/bc/hog26`: 43 shards, 8,552 matches and 1.99M one-second
  decision rows. 424 matches are held out for validation, using meta.json's
  `val_episodes`.
* Whole matches run in order as TBPTT chunks of 50 rows, with the LSTM state
  carried across chunks. The network is the exact PPO network, and the masks
  are the same as PPO's: `affordability_mask` and `placement_mask`.
* **Card loss:** cross-entropy on decision rows, with 0.05 label smoothing
  spread over legal arms only.
  * No-op rows (77% of rows) get weight `w_noop = p_play/p_noop ≈ 0.29`.
  * The prior-shift correction `-log(w_noop)` is added back to the no-op bias
    at save time, so the saved policy's P(play) is calibrated to the human
    rate. Both rates are reported.
* **Placement loss:** teacher-forced on the human card, with 0.1 of the mass
  spread over legal 8-neighbours. `relocated` labels are weighted 0.3.
* **Value loss:** MSE to the discounted *real* outcome, not the sim tower
  return. The replayed sim destroys about 3x fewer towers than the real
  matches did.
* **Ability head: not trained by BC** (`--ability-coef 0`). The dataset only
  marks `ability_ready` on rows where the human activated the ability, so every
  ready label is positive and the head would learn "always fire". PPO learns
  ability timing.
* **Win-weighting** (`--lost-weight`) was picked by validation from a sweep on
  the first 19 shards. 0.5 gave a card+place validation NLL of 4.078 against
  4.090 for 1.0, so the final model uses 0.5.

## PPO from BC (`ppo_bc.py`)

* The engine's Phase1Trainer handles rollouts, GAE, the PPO clip update, the
  teacher curriculum and checkpointing. The subclass changes the following:
  * **Critic warm-up:** for the first `--critic-warmup` (6) updates, only
    `value_head` has gradients and the actor is frozen.
  * **KL to BC:** a `KL(pi || pi_BC)` term on rollout states, covering the card,
    the placement of the taken card, and the ability when ready. Its
    coefficient decays log-linearly from 0.1 to 0.01 over 30k episodes, then
    holds at 0.01.
  * **BC auxiliary loss:** BC cross-entropy on a human-data chunk every
    minibatch, decaying from 0.1 to 0.01 on the same schedule. The human-data
    stream carries its own LSTM state through whole matches.
  * **Fixed entropy coefficients:** card 0.01 and placement 0.005, replacing
    the adaptive controller, which would push placement entropy back up and
    dissolve the human placements.
  * **Regularizers off:** advisor, placement and deck coverage; defensive and
    offensive scenarios.
  * **Learning rate:** `--lr` 5e-5 for the BC-initialised run, below the
    engine's 3e-4, because the first test update clipped about 20% of samples
    at 1e-4.
* The KL and BC terms are added by re-using the engine's
  `PPOUpdater.update` source with one injected line before `zero_grad()`. The
  anchor is asserted, so an engine change cannot silently drop the term.
* Logging:
  * TensorBoard `runs/<run>/`: the engine series plus `BC/kl_to_bc`,
    `BC/aux_bc_loss`, `BC/val_*` (the held-out human metrics of the *current*
    policy, every 10 updates) and `Opp/<bucket>` (per-deck win rate).
  * `runs/<run>/progress.jsonl`: one line per update.
* The run is resumable: re-run the same command and it continues from
  `checkpoints/<run>.pth`. `--init` is only used when that file does not exist
  yet.

## Commands

    # BC (GPU if present)
    .venv/bin/python -m training.bc --run bc_hog26 --lost-weight 0.5 --minutes 35 --batch 64
    # PPO from BC (resumable)
    .venv/bin/python -m training.ppo_bc --run ppo_bc26 --init checkpoints/bc_hog26.pth --num-envs 20 --lr 5e-5
    # eval panel alongside
    .venv/bin/python -m training.eval_panel --watch ppo_bc26 --bc checkpoints/bc_hog26.pth --every 2000

On the Mac, use `--num-envs 8`. The trainer uses CUDA if it is available and
the CPU otherwise; the engine trainer never uses MPS.
