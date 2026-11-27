# Training side: sim fidelity and exposure for the cards the bot fails against (2026-11-27)

Policy under study: ep0101074 of runs/pocket_1126/g2_pocket_fresh (anchor runs/bc3/g3j_L13_Wnone_pk_fxafbw/ep4.pth),
engine build/gen2_k with CR_P0088=tornado,jump,log,turret,king,speed,bankrows,stomp,jumpcol,pocketcells,kingpath,jumpedge,walklegs,hogspeed, CR_POCKET_TABLE=verified.

Labels:
- [measured, sim]: probe on gen2_k with the trainer's switches.
- [measured, code]: read from CardRegistry.h, the obs code or the encoder.
- [measured, data]: counted from the pool files, ghost stores, battle logs or eval rows.
- [inferred]: reasoning, not measured.

Nothing in training/, build/ or the pool was changed. Scripts and outputs are in the session scratchpad:
- exposure2.py / exposure2.json
- probe_bush_barrel.py / probe_bush_barrel.json

## 0. Today's live session (battle log, 12 matches, 3 wins) [measured, data]

The battle log now holds 107 battles, the last at 2026-11-27 09:31 UTC.

Six of today's 9 losses were against decks holding a card from this audit:

| Opponent card | Our crowns – their crowns |
|---|---|
| Hero Mini PEKKA | 0-3 |
| Three Musketeers | 0-1 |
| Evo PEKKA + Evo Royal Ghost | 0-3 |
| Evo Goblin Barrel | 0-3 |
| Evo Royal Ghost + PEKKA | 0-3 |
| Suspicious Bush | 0-3 |

## 1. Per-card table

How the exposure columns were computed:
- **Training %**: share of all training episodes whose opponent deck holds the card, using the pool weights of the running trainer. Self-play is 20 % and never holds these cards.
- **Real %**: share of the 107 battle-log opponents whose deck holds the card. The form is exact: evo, hero and base are counted separately.
- **Plays /1000 ep**: opponent plays of the card per 1,000 training episodes. Ghost kinds are measured from the stores. Teacher and clone are estimated as deck share × plays per match by ghost players holding the card [inferred].

| Card | Fidelity in the sim | Training % | Real % (n/107) | Ratio | Plays /1000 ep |
|---|---|---|---|---|---|
| Mini PEKKA (5) | OK [code] | 2.10 | 1.9 (2) | 1.13 | ~64 |
| **Hero Mini PEKKA (170)** | WRONG, too weak. The ability is a flat +210 HP and ×1.15 damage. Real: Breakfast Boost heals 30 % and adds +1 to +5 levels from meters (22 s each, 10 s per attack) [code + wiki]. The live detector gets 0 of 6 right (perception agent). | 3.53 | 3.7 (4) | 0.94 | ~106 |
| PEKKA (13) | OK [code] | 1.99 | 3.7 (4) | **0.53** | ~48 |
| **Evo PEKKA (148)** | APPROX. Heals 40 per hit up to 6,242 HP, also when hitting buildings and towers. Real: heals on kills (160 / 305 / 577) up to 5,640 HP [code + wiki]. | 1.36 | 2.8 (3) | **0.48** | ~33 |
| Royal Ghost (47) | WRONG in detail. No splash (real radius 1.0). HP 1,152 vs 1,210. Re-cloaks after 1.8 s vs 2.0 s. Untargetable while cloaked is correct. With no defence it reaches the tower 5.8 s after a bridge drop, stays visible while it keeps hitting, and deals 6 × 263 = 1,578 before the tower kills it [sim]. | 1.90 | 0.0 (0) | — | ~85 |
| **Evo Royal Ghost (136)** | APPROX. Spawns 2 Souldiers on every landed hit, with no "only while invisible" check. Souldier damage is 133, which is outdated [code]. This makes the sim's evo stronger than the real one. | 5.49 | **7.5 (8)** | 0.73 | ~278 |
| **Royal Ghost (both forms), observation** | MISMATCH. The sim board omits a cloaked unit (`extractObservationForTeam` skips `!isSeenOnBoard`) [code]. Live now encodes the walking ghost (live_ppo/cloaked_rows.py, added today), so the policy sees an input it never saw in training [inferred]. | | | | |
| **Three Musketeers (80)** | WRONG. No melee mode: since the 3/11/2025 Elite rework the real one hits 314 up close and 204 at range. The three bodies spawn in a straight line 2 tiles apart, at x = 1, 3, 5 for a lane drop [sim]; the real formation is a triangle. HP 906 vs 883. Cost 9 and the stagger are OK [code + wiki]. The detector maps the bodies to Musketeer (perception agent). | 1.73 | 2.8 (3) | 0.62 | ~41 |
| Goblin Barrel (109) | OK. Lands 2.7 s after the cast (real 2.76 s). 3 goblins with 1.0 s deploy. First tower hits 4.2 s after the cast (real about 4.26 s). Damage 125 vs 120 [sim + wiki]. | 2.21 | 3.7 (4) | 0.59 | ~107 |
| **Evo Goblin Barrel (132)** | WRONG. The sim drops 6 goblins at one point. The real evo drops 3 plus a decoy barrel of 3 weak decoys in the mirrored lane [code + wiki]. | 5.09 | **9.3 (10)** | **0.54** | ~262 |
| Goblin Barrel, any form | | 7.3 | **13.1 (14)** | **0.56** | ~369 |
| The Log (33) | OK [code + wiki]. | | | | |
| **Suspicious Bush (85)** | Present. Correct: 81 HP, Medium speed, untargetable while walking, killable by spells. WRONG on arrival at a building: the Bush reveals, **hits the tower for 256 itself**, and waits for the tower to kill it (1.1 s) before the goblins spawn [sim]. Real: on reaching a building it turns straight into the 2 goblins and never attacks. Also wrong: the Bush's range is 0.5 (real 1.6 since 6/4/2026) [code + wiki]. | 1.48 | 2.8 (3) | **0.53** | ~60 |
| Bush Goblins (child −30) | OK: 304 HP, 256 damage, 1.4 s hit, range 0.8, 0.9 s deploy [code + sim]. Their speed constant is 0.5, while Medium is 0.66 [code]. That matters only if the Bush pops early. | | | | |
| **Suspicious Bush / Bush Goblins, live observation** | The live encoder's `card_table()` has no row for 85 or −30 [measured, code]. The walking Bush is therefore dropped live, which matches the sim. The detector has no Bush-Goblin class, so popped goblins should box as `goblin` and encode and log as **Goblins**. `SPAWNED_BY['goblin']` does not list the Bush, so they would count as a new Goblins play [inferred from live_ppo/opp_plays.py]. | | | | |

Other exposure facts [measured, data]:
- No meta form is missing or modelled as its base form in the battle log.
- The evo/hero forms the engine lacks entirely are evo Royal Hogs (162) and evo Mega Knight (160).
- The cards most under-exposed relative to real decks (at least 3.5 % real):

| Card | Training % | Real % |
|---|---|---|
| Witch | 0.4 | 4.7 |
| Evo Ice Spirit | 0.5 | 3.7 |
| Ronin | 1.5 | 7.5 |
| Clone | 1.0 | 4.7 |
| Little Prince | 0.8 | 3.7 |
| Evo Mirror | 1.4 | 5.6 |
| Skeleton Army | 1.2 | 4.7 |
| Tesla | 3.3 | 11.2 |

## 2. Probes

### Bush: answers vs our left Princess Tower (single seed) [measured, sim]

The Bush is dropped at their bridge.

| Our answer | Tower damage taken |
|---|---|
| None | 1,280 (256 from the Bush itself + 4 goblin hits) |
| Log in front of the goblins as they pop | 256 |
| Same Log, 1 s late | 768 |
| Log on the invisible Bush mid-lane (y ≈ 12) | **0**: the Bush dies, and the rolling Log hits the goblins as they spawn (304 → 35 HP) |
| Cannon in the pull spot (8.5, 9) when the Bush crosses | **0**: it pops at the Cannon |
| Ice Spirit or Ice Golem on the popped goblins | 1,024 |

### Goblin Barrel on our Princess Tower vs Log cast time after the barrel is cast [measured, sim]

| Log cast | Tower damage taken |
|---|---|
| No Log | 875 |
| From behind the tower (y 4), 1.0–2.6 s | **0** |
| From behind the tower, 2.8–3.6 s | 375: all 3 die, but each hits once first |
| From behind the tower, 4.0 s | 625 |
| From behind the tower, 5.0 s | 750 |
| At the tower, ≤ 3.0 s | 0 |
| At the tower, 3.6 s | 375 |
| At 0.5 s | Misses (the Log has rolled through before the landing) |

The sim rewards the correct Log answer, so the sim is not what stops the policy from learning it.

### Policy in sim, 1,000 common ghost matches (runs/pocket_1126/fix/evals) [measured, data]

| Metric | ep0101074 | People in the same replays |
|---|---|---|
| Barrel answered with Log ≤ 3 s | **18 %** (27 / 148) | **59 %** (91 / 155) |
| Barrel answered with Log ≤ 5 s | 29 % | 87 % |
| Fireballs per match | **0.50** | 1.94 |
| Logs per match | 2.57 | 4.14 |

**The barrel failure and the Fireball shortfall already exist in the sim. They are policy and training gaps, not only sim-to-real gaps.**

## 3. Gaps ranked by likely effect on live wins [inferred ranking from the numbers above]

1. **Goblin Barrel answer (policy).**
   - Any-form barrel is in 13 % of real decks, and holders play it about 5.6 times a match.
   - The policy Logs only 18 % of barrels in time, vs 59 % for people.
   - An unanswered barrel costs 875 HP, 29 % of a Princess Tower.
   - Barrel decks are under-exposed ×0.56.
   - The evo's decoy lane is not modelled, so the evo plays out wrong as well.
2. **Evo Royal Ghost.**
   - It is the most common target card: 7.5 % of real decks, about 5.6 plays a match.
   - Since today's cloaked_rows fix, live shows the walking ghost and the sim never does, so this input is out of distribution for the policy.
   - The missing splash makes Skeletons and Ice Spirit look like good answers in the sim when they are not.
3. **Fireball shortfall (policy).** 0.5 vs 1.94 per match, against every deck. It sits in the imitation model (11-25 findings).
4. **Suspicious Bush.**
   - Arrival timing is wrong, and the sim adds an extra 256 hit.
   - Live encodes the popped goblins as a Goblins play.
   - Bush decks are under-exposed ×0.53. The cheapest answers in the sim are a Cannon pull or a Log mid-lane.
5. **Hero Mini PEKKA.** The sim's ability is too weak, but the detector failure (0 of 6) dominates on the live side.
6. **Evo PEKKA / PEKKA.** Exposure ×0.5; the heal model is wrong.
7. **Three Musketeers.** Wrong formation and no melee mode; only 2.8 % of real decks.

## 4. Fixes and cost

**A. Pool calibration at a resume.** No restart and no engine change.
- Build a new pool file, data/decks/meta_api_pool_live5.json.
  - Add today's 12 decks; this was due anyway, since the pool is rebuilt after each session.
  - Re-weight decks so that per-card shares match the battle-log shares, for barrel (any form), Bush, PEKKA (both forms), Evo Royal Ghost, Tesla, Witch and Ronin.
- Load it at the next resume, with the dated revert point `state_ep<N>_pre_pool1127.pth`.
- Cost: about 1 h of CPU and scripting, no GPU.
- **Conflict to settle first:** the 2026-11-20 ruling says "no extra bait weighting". This change is framed as matching real shares, not extra weighting, but the owner should confirm.

**B. Patch 0091 in a new build/gen2_l.** One default-off switch per item, like 0088–0090. Mechanics only; the observation shape does not change.

| Switch | Change |
|---|---|
| `bushpop` | On reaching a building the Bush dies at once and spawns its goblins. The Bush never attacks. Range 1.6. Goblin speed Medium. |
| `ghostfix` | Royal Ghost splash 1.0, HP 1,210, re-cloak at 20 ticks. The evo spawns Souldiers only on a hit made while cloaked. |
| `cloakboard` | Cloaked units drawn on the board planes, to match live. This changes the observation's content but not its shape. **Alternative:** take the live ghost off the board until it hits. That is a perception-side change and needs no engine work. |
| `evogb` | Evo barrel: 3 goblins plus 3 decoys in the mirrored lane. |
| `trimusk` | Triangle formation, melee 314 mode, 883 HP. |
| `heromp` | Hero Mini PEKKA meter-level boost and 30 % heal. |
| `evopekka` | Heal on kill, cap 5,640. |

- Cost:
  - Engine work: about 0.5–1 day for an Opus agent per two or three switches, plus tests.
  - Build: a fresh copy, as for gen2_k.
  - Re-run evals on gen2_l.
- Restart need: resume the trainer on gen2_l with the switches on, as was done going from gen2_j to gen2_k. That is not a restart from zero, but it needs a revert point.
- `cloakboard` changes what the board shows, so expect a short adaptation period [inferred].

**C. Barrel and Fireball behaviour.** This is a training-method fix, not a sim fix.
- The sim already rewards the right Log, so more exposure (fix A) helps but is unlikely to close a 59 % vs 18 % gap alone [inferred].
- Options:
  - Re-weight human barrel-answer and Fireball states in the BC data, which means a new imitation anchor. Cost: a BC retrain and a new KL anchor, close to a restart.
  - A targeted KL-anchor or trigger auxiliary loss inside the current run. Cost: a code change in new files, then a resume.

**D. Perception side** (hand to the perception agent; no training change):
- Add `suspicious-bush` to `SPAWNED_BY['goblin']`, and map those goblins to the Bush Goblins row.
- Decide whether `cloaked_rows` should encode the ghost while it walks, or only once it is revealed, to match the sim until `cloakboard` exists.
