# Live session 20261127-115232: post-session audit

- Session: 10 matches against people, 2026-11-27 11:52-12:24 PST.
- Checkpoint: `runs/pocket_1126/g2_pocket_fresh/snapshots/ep0204071.pth`, the restart PPO from C5 on `build/gen2_l` since 08:35:40.
- Detector r5, with `--dump-obs` on.

The audit is offline and read-only with respect to anything live. Every number is labelled [measured] (computed from the recorded data) or [inferred] (a judgement, or arithmetic on measured numbers with an assumption).

## 1. Short answer

**The damage that decides these matches is mostly policy, not perception** [inferred from the measurements below].

Take the 19 pushes that did 1000 or more tower damage, leaving out the King-falls-at-the-end rows. They total 37,735 damage [measured]. By cause:

| Cause group | Events | Damage | Share |
|---|---|---|---|
| Policy choice | 14 | 28,114 | 74.5 % |
| Economy (elixir spent elsewhere) | 4 | 8,565 | 22.7 % |
| Perception | 1 (E30, Cannon Cart track lost) | 1,056 | 2.8 % |

All [measured] by classification.

In every big policy event the attacker was in the observation with roughly the right HP. In most of them the bot had the elixir and an answer card in hand. It then either held (p(play) 0.05-0.37), played into the wrong place (Cannon against air, the wrong lane, an Electro Giant reflect), or sent a Hog instead.

Sam's "out of nowhere it just lets a troop do full damage" is right about what happens: single unanswered or misanswered pushes take whole towers (3,020, 3,052, 3,052, 3,068, 3,157, 3,395). It is not right about why: those troops were seen.

The one Hero Mini PEKKA misread (m2) was tested by injection. Naming it correctly barely moves the policy (section 5).

Sam's "much better" is partly supported:
- 1.73 tower-health share taken per match is the lowest of the four sessions compared [measured].
- 6 of 10 losses were 0-1 [measured].
- But it won 0 of 10, and offence destroyed 0 enemy towers [measured].

## 2. Per-match results

The battle log is available for all 10 matches.
- "Dealt" and "taken" are tower-health shares, where 1.0 = one Princess Tower.
- "Hog plays / hits" counts our Hog Rider plays and the Hog hits on towers.

All [measured].

| Match | Result | Crowns | Dealt | Taken | Hog plays / hits | Opponent deck (battle log) |
|---|---|---|---|---|---|---|
| m1 | L | 0-3 | 1.06 | 3.00 | 3 / 6 | Skeletons evo, Electro Wizard hero, Wizard evo, Arrows, The Log, Prince, Bandit, Goblin Barrel |
| m2 | L | 0-1 | 1.04 | 1.35 | 4 / 5 | Electro Giant evo, Mini P.E.K.K.A hero, Ice Wizard hero, Skeleton Barrel, The Log, Zap, Cannon, Spear Goblins |
| m3 | L | 0-1 | 0.23 | 1.02 | 1 / 2 | Electro Giant evo, Bowler hero, Baby Dragon evo, Mother Witch, Lightning, Goblin Hut, Tornado, Barbarian Barrel |
| m4 | L | 0-1 | 0.80 | 1.54 | 6 / 6 | Baby Dragon evo, Tombstone hero, Ice Wizard hero, Knight, Poison, Tornado, Graveyard, Barbarian Barrel |
| m5 | L (OT) | 0-1 | 0.87 | 1.34 | 6 / 7 | Battle Ram evo, Electro Wizard hero, Royal Ghost evo, Magic Archer, P.E.K.K.A, Bandit, Zap, Fireball |
| m6 | L (1.46 min) | 0-3 | 0.49 | 3.00 | 1 / 2 | Cannon evo, Ice Golem hero, Electro Giant evo, Tornado, Earthquake, Ice Spirit, Skeletons, Sparky |
| m7 | L | 0-1 | 0.76 | 1.00 | 5 / 6 | Mortar evo, Goblinstein, Ice Wizard hero, Cannon Cart, Barbarian Barrel, Skeletons, Rascals, Fireball |
| m8 | L | 0-1 | 0.92 | 1.05 | 4 / 8 | Lumberjack evo, Musketeer hero, Giant Snowball evo, Ice Golem, Skeleton Barrel, Clone, Phoenix, Cannon Cart |
| m9 | L (OT, about 293 s) | 0-1 | 0.92 | 1.00 | 12 / 1 | Executioner evo, Golden Knight, Giant Snowball evo, Skeletons, Cannon, Fire Spirit, Tornado, Rocket |
| m10 | L | 0-3 | 0.54 | 3.00 | 3 / 5 | Lumberjack evo, Electro Wizard hero, Battle Ram evo, Giant Skeleton, Mother Witch, Heal Spirit, Furnace, Arrows |

Lowest enemy Princess Tower left at the end [measured]:
- under 15 %: m1 at 173 HP and m2 at 321;
- under 30 %: m7 at 727 and m9 at 668.

In m9, 12 Hog plays gave 1 hit: the opponent's Cannon pulled them [measured].

### Scoreboard against the earlier sessions

| Session | Checkpoint | Wins | Crowns for / against | Tower share for / against |
|---|---|---|---|---|
| **1127pm** | ep0204071 (gen2_l) | **0 of 10** | **0.00 / 1.60** | **0.76 / 1.73** |
| 1127am | ep0101074 (gen2_k) | 2 of 11 (0 of 9 against active opponents) | 0.11 / 2.44 | 0.58 / 2.65 |
| 1125pm | ep0496977 (old main) | 3 of 12 | 0.50 / 1.50 | 1.07 / 1.93 |
| 1124am | ep0284847 | 2 of 11 | 0.55 / 1.82 | 1.14 / 2.07 |

All [measured].

Other session numbers for 1127pm [measured]:
- 4.5 Hog plays a match;
- 1.07 hits per Hog;
- 12.6 plays a minute.

Real opponents destroyed about 0.67 of our towers per match in the human reference set (`audit/real_final_humans.txt`) [measured].

Reading [inferred]:
- Defence is better than 1127am and slightly better than 1125pm on tower share taken (1.73 against 1.93 and 2.65).
- Offence is the weakest of the four sessions: 0.76 dealt, and no tower taken.
- Against people, 0 of 10 is a regression from old main's 3 of 12.
- With these sample sizes the win counts are not significantly different [inferred]: Fisher's exact test, 0/10 against 3/12, gives p of about 0.22.

## 3. Tower-damage audit (46 events, `decision_rows.csv`)

Method:
1. Every tower-HP drop segment is one event, read from the de-spiked tower read.
2. For each event:
   - the attackers come from the battle-log plays and the observation at onset;
   - elixir and hand come from the HUD;
   - the policy is recomputed on the dumped observation (`obs_compat.load_policy`, sampling temperature 1, zero LSTM state). The recompute reproduces the logged p(play) exactly.
3. The cause is then classified (`audit/classify.py`, which writes `audit/cause_table.txt`).
4. Five events were checked by eye on contact sheets in `audit/sheets/`.

### Cause table [measured, by classification]

| Cause | Events | Damage | Share of all | Share excl. KING FELL | Morning 1127am share |
|---|---|---|---|---|---|
| CHOSE OTHER: misplaced | 9 | 14,377 | 24.8 % | **31.3 %** | 32.6 % |
| KING FELL (end-of-match rows) | 6 | 11,974 | 20.7 % | n/a | |
| NO ELIXIR | 6 | 8,641 | 14.9 % | 18.8 % | 6.0 % |
| CHOSE OTHER: held | 5 | 8,080 | 13.9 % | **17.6 %** | 33.1 % |
| CHOSE OTHER: offence | 4 | 5,402 | 9.3 % | 11.8 % | 7.2 % |
| SPELL (chip) | 7 | 2,862 | 4.9 % | 6.2 % | |
| MASKED/FORCED (leak floor) | 2 | 2,592 | 4.5 % | 5.6 % | |
| ANSWER NOT IN HAND | 1 | 1,401 | 2.4 % | 3.0 % | |
| NOT SEEN | 1 | 1,056 | 1.8 % | 2.3 % | 6.5 % |
| SEEN AS WEAKER | 0 | 0 | 0 | 0 | 4.3 % |
| GOOD (defended, chip only) | 4 | 982 | 1.7 % | 2.1 % | |
| LATE | 1 | 558 | 1.0 % | 1.2 % | |
| **Total** | 46 | 57,925 | | 45,951 | 83,832 over 59 events |

Groups, excluding KING FELL [measured]:

| Group | Events | Damage | Share |
|---|---|---|---|
| Policy (CHOSE OTHER + MASKED) | 20 | 30,451 | **66.3 %** |
| Economy / timing | 8 | 10,600 | 23.1 % |
| Spells / chip | 11 | 3,844 | 8.4 % |
| Perception | 1 | 1,056 | **2.3 %** |

Perception was also a secondary factor in E6 and E7 (the Hero Mini PEKKA read as miner or knight-hero). Counting those generously with E30, perception reaches 5,111 damage, or 11.1 % [inferred upper bound].

Against the morning, total damage taken fell from 83,832 to 57,925 [measured], with one match fewer. "Held" fell from 33 % to 18 %. "Misplaced" stayed at about 31-33 %. "No elixir" rose from 6 % to 19 % [measured].

Reading [inferred]: this policy plays more often under threat than ep0101074, but often in the wrong place, and it then has nothing left.

### The 19 big events (1000 or more damage, KING FELL excluded)

"Elixir" is the HUD reading at onset. The "what happened" column is [measured] from the timeline and the recompute; the cause is a [measured] classification.

| Id | Match | Tower | Damage | Elixir | Cause | What happened |
|---|---|---|---|---|---|---|
| E1 | m1 | R | 2,372 | 7.0 | held | Bandit and Electro Wizard hero in the observation; Ice Golem and Log played early, then nothing until a Hog at 53.9 s. Tower to 72. |
| E2 | m1 | K | 1,401 | 2.2 | not in hand | Goblin Barrel on the King; Log was played 6 s earlier at a bridge. |
| E3 | m1 | K | 3,157 | 2.4 | offence | Prince and Electro Wizard hero; Skeletons and Ice Golem placed off the line, then a Fireball on the enemy tower at 119.2 s. |
| E6 | m2 | L | 3,020 | 6.1 | held | Hero Mini PEKKA (seen as a miner-like 1210 HP melee cell); one Skeletons play, then nothing; tower to 32. |
| E7 | m2 | R | 1,035 | 2.3 | misplaced | Ice Wizard hero, Spear Goblins, Skeleton Barrel; Cannon at (9,9), then single cards. |
| E9 | m3 | R | 2,377 | 3.1 | no elixir | Electro Giant evo push after spending on the previous one. |
| E11 | m4 | R | 3,052 | 7.8 | misplaced | Baby Dragon evo: Cannon placed against an air unit, Musketeer at the bridge; then 3-4.9 elixir. Tower 2,844 to 0 (sheet E11). |
| E17 | m5 | L | 1,052 | 9.7 | forced | Royal Ghost evo on the LEFT; the leak floor forced an Ice Golem to (14,15), the RIGHT bridge (sheet E17). |
| E18 | m5 | R | 1,358 | 5.3 | misplaced | P.E.K.K.A and Bandit; defence placed in the wrong lane. |
| E19 | m5 | R | 1,329 | 5.3 | offence | Royal Ghost evo and Magic Archer; a Hog sent instead of an answer. |
| E21 | m6 | L | 3,052 | 6.3 | misplaced | Electro Giant evo: elixir 7 to 1 on units into the reflect; tower 3,052 to 0 (sheet E21). |
| E22 | m6 | K | 3,395 | 1.6 | no elixir | Follow-up of E21; nothing left. |
| E27 | m7 | R | 1,903 | 7.5 | misplaced | Mortar evo; answers placed out of reach. |
| E29 | m8 | R | 1,540 | 7.4 | forced | The leak floor forced a Log to (3,15), the far lane. |
| E30 | m8 | R | 1,056 | 5.1 | **not seen** | Cannon Cart lost by the tracker for 40.0-44.5 s (section 5). |
| E34 | m9 | L | 1,074 | 6.0 | held | Executioner evo in the observation; the HUD OT flag was misread at this time. |
| E40 | m10 | L | 1,392 | 3.3 | no elixir | Lumberjack evo; elixir already spent. |
| E41 | m10 | L | 1,102 | 5.5 | held | Lumberjack evo in the left lane; hand Ice Spirit / Hog / Musketeer / Cannon; elixir 6 to 8; no play; tower 1,102 to destroyed (sheet E41). |
| E43 | m10 | K | 3,068 | 7.4 | misplaced | Giant Skeleton and Battle Ram evo on the King; 4-7 elixir with Fireball and Log in hand; King 4,446 to 1,666 (sheet E43). |

Corrections made during the audit [measured]:
- E11 was first classed as held. The contact sheet shows Cannon and Musketeer were played at onset, so it is misplaced.
- E34's onset observation column reads "none", but the timeline shows the Executioner evo cell (range 1279, distance 7) for most rows. It is still held.
- The m5 forced play at 64.0 s is an Ice Golem, not an Ice Spirit.

## 4. Card checks

| Card | In this session | What was measured |
|---|---|---|
| Royal Ghost (evo), gen2_l cloakboard | m5 | In the observation as a melee 1152 HP cell at conf 0.93-0.96, at 64 s and 119-127 s [measured]. Cloaked-Ghost-on-board works live. The bot still did not answer it: E17 was forced into the wrong lane, and E19 sent a Hog [measured]. |
| Suspicious Bush / Bush Goblins (rows 85 and 1011) | not in any deck | No bush detection in `perception.jsonl` [measured]. **Untested live this session.** |
| Hero Mini PEKKA | m2 (played 31.7, 132.9, 155.1 s) | Read as knight-hero (n 19) and miner (n 13); the track was lost f2019-f2049 [measured]. Injection test in section 5: not decisive. |
| Three Musketeers | absent | |
| Evo PEKKA | absent | The plain P.E.K.K.A (m5) read correctly, as a 3,760 HP melee cell [measured]. |
| Goblin Barrel (plain) | m1 | E2 on the King: the Log was not in hand (played 6 s earlier) [measured]. |
| Goblin Barrel (evo) | absent | Decoy mirroring is untested live. |
| Electro Giant evo | m2, m3, m6 | Read as royal-giant-ev1 [measured]. E21 is 3,052. The reflect is not represented under either class [inferred]. |
| Baby Dragon evo | m3, m4 | E11 is 3,052; the bot placed a Cannon against it [measured]. |
| Lumberjack evo | m8, m10 | Read as fisherman and mega-knight; E41 [measured]. |
| Mortar evo | m7 | E27 [measured]. |
| Executioner evo | m9 | E34 [measured]. |
| Giant Skeleton | m10 | Read as mega-knight (melee 3993); E43 [measured]. |
| Cannon Cart | m7, m8 | Lost by the tracker in E30; read as cannon [measured]. |
| Graveyard | m4 | Part of the m4 events [measured]. |
| Rocket | m9 | E35-E38 are spell chip [measured]. |
| Sparky | m6 | Present; the m6 losses are Electro Giant-led [measured]. |

Full misread list: `DETECTOR_MISREADS.md`.

## 5. Would naming the unit right have changed the policy? (`audit/inject.py`, `audit/inject.txt`)

The network was recomputed on the dumped observations twice: as recorded, and with the missing or misnamed unit written into the board.

| Case | Decisions | Mean p(play) before / after | P(at least one play) before / after |
|---|---|---|---|
| E6: Hero Mini PEKKA re-injected as Mini PEKKA (1361 HP, dps 46), replacing the miner cell | 17 | 0.109 / 0.122 | 0.88 / 0.90 |
| E6, with the tracker gaps filled | 24 | 0.089 / 0.119 | 0.91 / 0.96 |
| E30: Cannon Cart's own 39.5 s cell pasted back for 40.0-44.5 s | 9 | 0.086 / **0.213** | 0.56 / **0.88** |

All [measured].

The leak-floor forced decisions were recomputed without the force [measured]:
- m5 64.0 s, f8450, 9.7 elixir: the free p(play) was 0.26. The forced card was an Ice Golem at (14,15), the right bridge, while the Ghost was on the LEFT tower.
- m8 12.8 s, f13694, 9.6 elixir: the free p(play) was 0.39. The forced card was a Log at (3,15).

Reading [inferred]:
- **E6 is policy**: the unit was in the observation as a 1210 HP melee body, and the correct name changes little.
- **E30 is partly perception.**
- **The leak floor's forced card uses the where-head as it is, so it can place in the wrong lane.**

## 6. Threat play rate (`threat_rate.py`, `audit/sim/`)

"Under threat" means an enemy troop cell (channels 4-6) within radius 4 of a standing own Princess Tower. The numbers are weighted p(play) at 4-10 elixir, under threat against not under threat. All [measured].

| Checkpoint | Sim rows, gen2_k (11.9 % threat) | Sim rows, gen2_l (11.3 %) | Live 1127pm rows (6.0 %) | Live 1127am rows (8.0 %) | Live 1125pm rows (9.9 %) |
|---|---|---|---|---|---|
| ep0204071 | 0.260 vs 0.160 | 0.209 vs 0.141 | 0.130 vs 0.152 | 0.172 vs 0.097 | 0.226 vs 0.176 |
| ep0101074 | 0.168 vs 0.118 | 0.149 vs 0.112 | 0.115 vs 0.145 | 0.151 vs 0.086 | 0.193 vs 0.162 |
| ep0496977 | 0.359 vs 0.180 | 0.245 vs 0.163 | 0.116 vs 0.105 | 0.234 vs 0.101 | 0.159 vs 0.116 |

Pooled over the three live sessions (450 threat rows), p(play) under threat [measured, weighted arithmetic]:

| Checkpoint | Pooled p(play) |
|---|---|
| ep0204071 | about 0.189 |
| ep0496977 | about 0.173 |
| ep0101074 | about 0.164 |

Sim rollouts of ep0204071 on gen2_l (60 matches, 17,974 decisions) give p(play) under 7 elixir of 0.090 / 0.127 / 0.113 with 0 / 1 / 2 own towers down, and 0.429 at full bar [measured].

Reading [inferred]:
1. **The sim overstates the threat response for every checkpoint.** For ep0204071 it is 0.26 in the sim against 0.13 on its own live rows.
2. On its own session's live rows, ep0204071 plays slightly LESS under threat than not (ratio 0.85). Pooled, it answers threats as often as old main, or a little more. So the threat RATE is not what separates the checkpoints. WHERE the answer goes is (section 3).
3. The deploy gate's sim numbers do not predict this. Panel 65.6 % against ep0496977's 67.7 %, and defence probe 33.6 % against 41.2 %, both said "slightly worse". Live it is 0 of 10 against 3 of 12, though with a lower tower share taken.

## 7. Recommendations

These follow the standing rules:
- win rate against people is the target;
- Fireball and pocket stats are observe-only;
- no convoluted rewards and no extra bait weighting;
- the sim must not reward what fails against people.

1. **The defence policy is the lever, not the detector.** About 66-75 % of the damage that matters comes from seen threats that were held, misplaced, or answered with a Hog [measured share, inferred lever].
   - Train defence against opponents that react the way people do: reacting teacher, live ghosts, and this session's decks into the pool.
   - Judge by live win rate.
   - Do not add a defence reward term.
2. **Make the sim stop rewarding misplaced defence.** The sim's threat response is twice the live one, for every checkpoint [measured]. A misplaced answer must lose in the sim as it does live. Candidates [inferred]:
   - the Electro Giant evo reflect (modelled as a periodic area stun, `CLAUDE.md` patch 0048 "NOT modelled");
   - air units against a Cannon;
   - the wrong-lane answer.
   Check each on this session's ghost replays.
3. **The leak floor (`--leak-floor 9.5`) forces a card into the wrong lane** (E17, E29: 2,592 damage) [measured]. Make the forced placement lane-aware: aim at the nearest threatened tower, or at our King when nothing threatens. Otherwise drop the floor for the next session and compare.
4. **HUD reader faults to fix before the next session:**
   - m9's OT flag misread for t 59-183 s; the live policy saw overtime [measured];
   - the elixir view reading 0.0 against a HUD digit of 2.0 (f16896-f16906) [measured].
5. **Detector items, report only (no training here):**
   - Cannon Cart track loss (the only perception-caused push);
   - Hero Mini PEKKA read as knight-hero or miner;
   - Electro Giant evo read as royal-giant-ev1;
   - phantom royal-ghost and royal-hog boxes;
   - Lumberjack read as fisherman or mega-knight.
   See `DETECTOR_MISREADS.md`.
6. **Offence never finished a tower** (0 destroyed; 2 left under 15 %) [measured]. Watch it, but this is not a reason for a reward term. The lowest-HP towers (173, 321) were left at the end of regulation [measured].
7. **Use this session as ghost and eval data.** Correct m9's time first (the OT misread). Add the decks to the live-opponent pool share.
8. The Bush (rows 85 and 1011) and the evo Goblin Barrel are still untested live. Note it, and do not schedule a special session for them.
9. Fireball and pocket stats are observe-only. No bait weighting.

## 8. Files (all in `runs/live_1127pm/`)

**Top-level reports:**
- `AUDIT.md`: this report.
- `DETECTOR_MISREADS.md`: misreads with frames and true cards.
- `decision_rows.csv`: one row per tower-damage event. Columns: event, match, tower, time, damage, attackers, observation at onset, elixir, hand, p(play), cause.

**Classification and policy recompute (`audit/`):**
- `classify.py`, `cause_table.txt`
- `rows_auto.csv`, `events.*`, `event_facts.*`, `timelines.txt`
- `inject.py`, `inject.txt`: the injection and forced-play recomputes.

**Scoreboard and other inputs (`audit/`):**
- `scoreboard.txt` / `.json`: per-match results, all sessions.
- `spells.txt`, `lanes_*`, `real_final_humans.txt`, `enemy_classes.txt`

**Contact sheets and sim runs (`audit/`):**
- `sheets/E11.jpg`, `E17.jpg`, `E21.jpg`, `E41.jpg`, `E43.jpg`: frames checked by eye.
- `sim/`: the threat-rate outputs and the ep0204071 gen2_l sim rollouts.

Not mine, and not touched: `data_prep/` and `DATA_PLAN.md` (another agent).
