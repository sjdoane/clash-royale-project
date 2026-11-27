# Patch 0091 report (2026-11-27)

Saved by the coordinator from the engine agent's final message (the agent could not write this file).

Patch 0091 is built in `build/gen2_l` with seven default-off switches. With all seven off it is identical to gen2_k [measured]. Recommendation: move training to gen2_l with the four priority-1 switches plus the two sourced Ghost switches.

I could not write `runs/sim_0091/REPORT.md`: the harness blocks report files from subagents. P0088.h, the patch header and the test docstring all point at that path, so this message is the report and can be saved there.

## Switches

All seven are matched by exact name in CR_P0088, and "all" does not turn them on. "Hash" below means per-tick hashes over 30 seeded matches, counted as matches that differ in obs / units / pending spells / length against everything off [measured].

**Priority 1**
- **cloakboard**: a cloaked Royal Ghost (card 47, which also covers the Evo Ghost's bodies) is drawn on the board. Who can target it is unchanged.
  - Source: the r5 detector boxes the cloaked Ghost live, and `live_ppo/cloaked_rows.py` already feeds it to the live policy.
  - Hash: 28/0/0/0, so it changes only what the policy sees.
- **bushboard**: the walking Suspicious Bush (card 85) is drawn on the board. Live detector confidence while it walks is 0.58–0.96. Hash: 26/0/0/0.
- **bushpop**: the Bush never attacks.
  - With range 1.6 (was 0.5), it dies with no damage when a building is in reach. It lets out 2 Bush Goblins that stay hidden for 7 and 6 ticks, then do their own 1 s deploy.
  - Bush Goblin speed is Medium, 1.2 tiles/s (was 0.5).
  - Sources: royalesim SuspiciousBush (Damage 0, Kamikaze, Range 1600), SuspiciousBush_DummyAEO (675/625 ms) and BushGoblin Speed 60.
  - Live: the goblins first appear 0.77–1.03 s after the bush stopped (3 pops) [measured]. Those goblin box times were not saved; only the bush boxes are, in `runs/sim_0091/live/ghost_bush_dets.json`.
  - Hash: 24/24/14/14.
- **evogb**: the evolved card-132 cast is the plain barrel plus a decoy barrel at the mirrored x. The decoy has the same flight and lands 3 GoblinDummy (81 hp, 66 damage, hit 1.1 s).
  - Sources: royalesim GoblinBarrel_EV1 evo_mirror and units.GoblinDummy; live frames f010151–f010179 of `runs/live/20261124-094533` show two barrels with goblins at both towers.
  - The old behaviour (6 Goblins at one point) had no source.
  - Also adds a `get_evolution_state` binding. Hash: 6/6/6/5.

**Priority 2 (only the sourced items)**
- **gangair**: Goblin Gang's Spear Goblins hit air (royalesim SpearGoblin attacks_air). Hash: 29/0/0/0.
  - This only changes what the policy sees for our deck, since 2.6 has no air unit: card-table row 112's anti-air flag goes 0→1 [measured].
- **ghostsplash**: Ghost hits splash 1.0 (royalesim Ghost / Ghost_EV1 area_damage_radius 1000). Hash: 3/3/3/3.
- **ghostcloak**: the Ghost hides again 23 ticks after a hit instead of 18 (royalesim idle_invisibility 2000 ms; 18 of 26 calibration cases). Hash: 21/21/3/3.

**Left out, no usable source**
- Ghost HP 1210: the 6 Oct 2026 note says 1152, so 1152 stays.
- Evo Souldiers: conflicting stats, and the trigger has no source.
- Hero Mini PEKKA: its behaviour is unverified.
- Evo PEKKA heal: the threshold quantity is unnamed and the engine has no on-kill hook.
- Three Musketeers: the triangle's y direction and the deploy sequence are unknown, and the melee scaling is unstated.
- Still open, not changed: Bush Goblin HP is 304 in the engine against 337 in royalesim.

**READER2 flag** (`training/obs_noise.py`, `evo_no_tile`, default `[]`; backup `obs_noise.py.20261127.bak`)
- Default output is identical to the backup: 789,878 / 789,878 / 791,634 viewed plays across the three configs [measured].
- With `[132]`, the evolved cast never gets a tile while in the air (0 of 20, against 20 of 20 without the flag). Plain casts are unchanged.
- The dataset path refuses the flag, and it raises when there is no game.
- Live: the evo tile was read 0 of 9 times, while the sim gives one 93 % of the time.
- If you turn it on, also set the Goblin Barrel fail rate from 0.07 to 0 [inferred].

## Identity and tests

- **Switches off [measured]:**
  - hashes match on 30/30 matches for CR_P0088 empty, BASE14 and BASE14,all;
  - 150/150 replays are byte-identical;
  - 1000/1000 panel rows match (g2k_off vs g2l_off on ep0147485).
- **Tests:** `test_sim_real_0091.py` has 25 tests, which pass on gen2_l; on gen2_k 15 fail and 10 pass [measured].
- **Patch:** it dry-ran cleanly onto a copy of gen2_k.

## Panel

1000 held-out ghost matches, paired against all switches off, win diff ± SE [measured]. P1 = BASE14 plus the four priority-1 switches; P2 = P1 plus gangair, ghostsplash and ghostcloak. "Decks with" = decks holding card 47, 136, 85 or 132 (n 169).

| Snapshot | Arm | All 1000 | Decks with | Decks without (n 831) | Crowns against, decks with |
|---|---|---|---|---|---|
| ep0147485 (off: 0.601) | P1 | +0.015±0.007 | +0.089±0.042 | 0.000±0.002 | (all decks: −0.040±0.015) |
| ep0147485 | P2 | +0.015±0.009 | +0.071±0.042 | +0.004±0.006 | |
| ep0153505, newest (off: 0.623) | P1 | +0.006±0.006 | +0.030±0.037 | +0.001±0.001 | −0.160±0.087 |
| ep0153505 | P2 | −0.007±0.008 | −0.024±0.040 | −0.004±0.004 | −0.071±0.089 |

Other results:
- **P2 vs P1 on ep0153505:**
  - all 1000: −0.013±0.006;
  - decks with: −0.053±0.032;
  - Evo Ghost (136, n 81): 0.568→0.420, −0.148±0.047.
  - On ep0147485 the same comparison was 0.000±0.007.
- **P1 per card on ep0153505:**
  - Ghost 47: 0.270→0.324;
  - Evo Ghost 136: 0.531→0.568;
  - Bush 85: 0.545→0.500;
  - Evo GB 132: 0.515→0.545;
  - Gang 112: 0.569→0.554.
  - Each is within 1–2 SE.
- **Tower share taken on decks with:** −0.034±0.019 on ep0153505.
- **Unexplained rows:** on ep0153505, 15 of the 831 decks without these cards still changed under P1, net +0.001. I did not trace why.

Reading [inferred]: these numbers mix two effects — the policy gains from seeing the Ghost and Bush, and the opponents change strength. ghostsplash and ghostcloak make the Ghost stronger, so lower sim wins against Evo Ghost decks are the expected direction. They are not evidence the switches are wrong, and the panel cannot judge whether a change is realistic.

**Live card table [measured]:**
- Row 47 is identical between training with cloakboard and live `cloaked_rows`.
- Row 85 does not exist live unless the live engine has bushboard, so live currently drops bush boxes.
- Live must therefore run gen2_l with the same CR_P0088 [inferred].

## Proposed switch

1. **Engine:** move the trainer to `build/gen2_l/engine`, with CR_P0088 = BASE14 + `cloakboard,bushboard,bushpop,evogb,ghostsplash,ghostcloak`. Ghost splash and re-cloak go in on their sources. If you want a live result you can attribute to each part, keep them for a second step.
2. **gangair:** add it only if live uses the same switches.
3. **evo_no_tile:** optional, `[132]` with Goblin Barrel fail set to 0.
4. **Revert point:** `state_ep<N>_pre_gen2l1127.pth`, `latest_ep<N>_pre_gen2l1127.pth` and `launch.txt.bak_pre_gen2l`. To revert, relaunch on `build/gen2_k` with BASE14.
5. **Live:** `CR_LIVE_ENGINE=build/gen2_l/engine` with the same CR_P0088. I did not touch the live scripts.

Decisions for you:
- Is the imitation anchor data acceptable on gen2_l? It has no Ghost or Bush rows.
- `bld_pull_gen2k.json` can stay [inferred]: 0091 changes nothing about building pulls.
- Judge the switch by live win rate.

Files are in `~/cr-robot`:
- patch: `patches/pending/0091-ghost-and-bush-on-the-board-bush-pops-evo-barrel-mirrors.patch`
- engine: `build/gen2_l/engine` (tests in `python_ai/tests/test_sim_real_0091.py`)
- identity checks: `runs/sim_0091/hash/COMPARE.txt` and `runs/sim_0091/replay`
- panel: `runs/sim_0091/panel/` (`paired_p1_ep0153505.json`, `paired_p2_ep0153505.json`, `paired_p2_vs_p1_ep0153505.json` and the ep0147485 pair files)
- READER2 flag: `training/obs_noise.py` (backup `.20261127.bak`) and `runs/sim_0091/test_tail_evo.py`
