# Perception-noise model for the simulator (gen-2 actor observation)

Status: code and tests are done; no training was launched and `engine/` is untouched (no patch 0020 was needed).
All numbers below come from the local gen-1 engine build on CPU, 2026-11-19.

## 1. How the two observations are built

**Simulator.** `ClashEnv::extractObservationForTeam` (C++) writes the whole vector (13977 floats on the local build).
The first 21 x 34 x 18 = 12852 floats are spatial planes, index `ch*612 + y*18 + x`:

| channels | content | write rule |
|---|---|---|
| 0-3 ally, 4-7 enemy | unit HP by type (melee, ranged, building-targeter, building), `hp / MAX_TROOP_HP` or `MAX_BUILDING_HP` | assign (last unit in the cell wins) |
| 8 | river | constant |
| 9, 10 | unit count per side, `1/5` per body | add, capped at 1 |
| 11-20 | flying, anti-air, DPS, range, speed per side | max |

The rest is scalars: elixir, hand, towers, clock, spend, card cycle.
The binding exposes no unit list, only this vector.

**Live.** Detector -> tracker -> `contracts.UnitObservation` -> `GameState` -> `perception_encoder.encode` (through `live_ppo/encoder_fixes.encode`).
`encode` starts from the engine's fresh board and writes each unit with the same three rules, taking its attributes from `perception_encoder.card_table()` (one fixed row per card, probed from the engine).

## 2. Where the noise is applied, and why

The preferred design in the brief (perturb a unit list, encode with the live encoder's rules) is what was built, with one extra step because the engine gives no unit list:

1. **Decode** the clean observation into unit records (cell, side, type, HP, attribute vector, card), in `training/obs_noise.py::decode_units`. Cards are resolved by matching each cell's attribute vector against the card table.
2. **Track** the records across decisions (nearest neighbour per side and type) so each unit keeps one identity and one set of errors for life.
3. **Perturb** the unit list in perception-frame time (default 4 Hz), independent of the decision interval.
4. **Re-render** only the cells that changed, with `encode`'s exact write expressions.

Everything lives in `training/`. The wrapper is `RewardEnv` (`training/reward.py`), which both `ppo_bc.py` and `league.py` use.
The clean observation is kept for the reward shaping; the noised one is what `step`/`reset` return.

Proof that this is the live encoding: `test_noised_obs_equals_encoder_of_noised_units` runs a real episode and checks the noised observation equals `perception_encoder.encode(GameState(model.units()))` on every non-tower cell, for both HP modes.
A second test checks decode -> clear -> write reproduces the clean observation bit for bit.

What the model draws (defaults in `training/obs_noise_default.json`, every value configurable):

| error | default | persistence |
|---|---|---|
| missing | isolated 3%, deploy window 15%, overlapped 16%, under a spell 13% (the highest applicable rate) | two-state chain, mean run 3 frames |
| ghost after death | every death, held 1-2 frames at the last position | per death |
| spurious box | 0.02 per unit-frame, 1-3 frames, 70% next to a real unit | per box |
| wrong side | 5% | 80% for life, 20% in runs of about 4 frames |
| wrong card | form 3.75%, lookalike 3%, other 0.75% | same split as wrong side |
| position | per-unit bias sigma 0.15 tile + per-frame jitter 0.17, 20% tail at 0.45 | bias for life |

Form pairs (evolution or hero versus plain) are read from the engine's card data, not typed in.
The three lookalike pairs that are not form pairs are listed by card name in the config.
`scale` multiplies every rate and position sigma: 0 returns the clean array object itself, 1 is the measured level. `scale_range` redraws it each episode.

### Rates measured on real episodes at the defaults

12 episodes, Hog 2.6 against the teacher, random actions.

| | live (966 units) | sim, 1-s decisions (11,203 unit-steps) | sim, 0.2-s decisions (61,241) |
|---|---|---|---|
| missing | ~7% | 9.0% | 9.2% |
| ghosts, share of reported | ~8% | 5.9% (death 4.2, spurious 1.7) | 6.8% |
| wrong side | ~5% | 5.7% | 4.1% |
| wrong card | ~7-8% | 4.7% (form 4.2, lookalike 0, other 0.5) | 3.7% |
| in the correct tile | ~63% | 61.0% | 61.6% |

Per condition the miss rates land on the configured values (isolated 3.8%, deploy 15.1%, overlap 15.2%, spell 10.2%).
The aggregates differ from live for reasons that are about the unit mix, not the per-unit rates:

- **Missing is higher** because 39% of sim unit-steps are overlapped, more than in the labelled live frames.
- **Ghosts are lower at 1-s decisions** because a 1-2 frame ghost only shows if it is still held at the decision instant (37.5% of deaths).
- **Wrong card is lower** because no card in these matches has a lookalike partner, and form errors only apply to cards that have a second form. With the deck pool the lookalike share will rise.

I left the defaults at the measured per-condition values rather than tuning them to hit the aggregates.

## 3. HP channels: sim and live disagree

The sim writes true troop HP. Live cannot read troop HP, so the driver sets `hp_fraction=1.0`, and every unit is written at its card's full level-11 HP.

Over 9,932 resolved unit-steps in the sim: 45.8% show less than 98% of full card HP, 19.6% less than half, mean fraction 0.79.
So the policy trains on a damage signal it never gets live.

`hp_as_live` (off by default) renders every resolved unit as live would: full card HP and table attributes. It is independent of `scale`.
Tower HP is read live and is left alone.

## 4. Actor only

- **Reward**: computed from the clean observation (test: shaping terms identical with noise on and off for the same state).
- **Opponent**: `info["obs1"]` comes straight from the engine and is untouched (test on a `LeagueEnv` mirror episode).
- **Critic**: there is no privileged critic. `value_head` reads the same LSTM features as the policy head, so with noise on the value function also sees the noised observation. Patch 0012's opponent-state block is an observation section, not a critic input. `publish_clean: true` puts the clean observation in `info["obs_clean"]` for a future asymmetric critic; nothing consumes it yet.

## 5. How to enable

Off by default everywhere.

```bash
# ppo_bc.py and league.py take the same flags
python -m training.ppo_bc ... --obs-noise default
python -m training.league ... --obs-noise default --obs-noise-scale 0.2:1.0 --obs-noise-hp-live
python -m training.ppo_bc ... --obs-noise path/to/config.json --obs-noise-scale 0.5
```

In code: `env_config["obs_noise"] = True`, a JSON path, or a dict of overrides. `make_league_env(..., obs_noise=...)` passes it through.

**BC data generation is not wired.** The stored BC observations are float16 rows, one per 1-s decision, with no env to wrap.
`obs_noise.noise_trajectory(rows, dt_ticks=10, cfg, seed, match_tol=2e-3)` noises one stored match in order; calling it per match in the BC loader is the whole change. I did not test it on real BC data.

## 6. Overhead

| | env.step, microseconds |
|---|---|
| noise off | 607 |
| scale 0 | 590 |
| scale 1 | 672 |
| scale 1 + hp_as_live | 670 |

About 65 microseconds per step (11%) at 4.1-4.8 units on the board; the model alone measures 60. Mac CPU, 1-s decisions.

## 7. Other sim-versus-live mismatches (listed, not fixed)

Checked:

1. **A played unit is in the sim observation the next tick**, at full HP in its target cell, through its whole deploy time. Live sees it after capture latency and misses it 13-17% of the time in that window. The noise covers the misses, not the latency.
2. **Evolution ids are not in `perception_encoder.card_table()`** (125 and 129 return nothing; hero 168 is there). A live unit reported with an evolution id is dropped by `encode` unless the driver maps it to the base card first.
3. **Form confusions barely change the planes.** An evolved or hero unit has the same table row as its plain form, so the 3.75% form error is nearly invisible to the policy in the spatial channels.
4. **Hero Musketeer's turret has no card id.** The sim shows it (0.6-0.8% of unit-steps); the live encoder cannot represent it. The noise model leaves such cells exactly as the sim wrote them.
5. **Dead tower footprint**: stock `encode` leaves a destroyed tower's attribute cells set. `live_ppo/encoder_fixes.py` already corrects this live.

Read from the code, not measured:

6. Sim attribute channels use the unit's actual DPS, range and speed (levels, buffs, rage, slow). Live writes fixed table rows.
7. Opponent elixir spent is exact in the sim; live writes a constant unless it is estimated.
8. Sim cell is the truncated true position; live cell comes from the tracker's ground-point estimate.
9. When two units of one type share a cell, the HP shown depends on entity order in the sim and on list order live.
10. Gen-2 engine sections (per-team levels, opponent-state block, tower HP over own maximum) need their live counterparts.

Not modelled by this work: elixir off-by-one reads, hand and next-card misreads, tower-HP read noise, clock error.

## 8. Limits and assumptions

- **Six tower cells are never noised**, and noise never moves a unit onto one. A unit walking onto a tower cell looks like a death to the tracker and can leave a ghost (0.1-0.3% of unit-steps).
- **Opponent spell position is unknown** (the env reports the card, not where it landed). The spell condition is applied around own units that lost HP that step.
- **Assumed, not measured**: mean miss run (3 frames), lifelong share of side and card errors (80%), transient run length (4 frames), spurious placement (70% near a unit), spell window (1 s). The live labels gave totals, not these run lengths.
- **Identity** is by nearest-neighbour tracking at decision steps; two same-type units crossing in one cell can swap error sets.
- **Not tested on the gen-2 engine.** Constants are read from the bindings and levelled units are handled, but only the gen-1 build exists locally.

## 9. Tests

`.venv/bin/python -m pytest training/tests/test_obs_noise.py -q`: 27 passed.

Full `training/tests`: 28 passed, 2 skipped, 4 failed. The 4 failures are in `test_reward.py` and are not from this change: `data/decks/train_pool.json` holds gen-2 card ids (181, 183) that the local gen-1 engine rejects in `set_opponent_deck`.
