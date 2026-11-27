# Decision audit: live run 20261127-010105

**Run setup**
- Checkpoint: runs/pocket_1126/g2_pocket_fresh/snapshots/ep0101074.pth. This is PPO from imitation model C5. In the sim common panel it wins 58.4 %; the old main ep0496977 wins 68.0 %.
- Engine: build/gen2_k.
- Live flags: `--pocket-mask`, `--afford-project`, and the TOWER_DEATH_CONFIRM tower reader.
- 11 matches were re-read offline, m1 to m11, each matched to the official battle log.

**Labels**
- [measured] means counted from the recordings, the battle log, or a recomputation of the network on the recorded observations.
- [inferred] means a reading of those numbers.
- "By eye" means checked on frames. The frame numbers are cited (fNNNNNN, in runs/live/20261127-010105/raw/frames).

**Privacy:** opponent names and the player tag are not written anywhere in this audit or its files.

**Deliverables**
- This file.
- runs/live_1127/decision_rows.csv: one row per event, with the columns described in section 10.
- Working files in runs/live_1127/audit/.

**How to reproduce** (offline and read only; run from the repo root, with nice, while the trainer runs):

```
nice .venv/bin/python -I runs/live_1127/audit/events.py        # events.json / events.txt: every tower-damage push
.venv/bin/python -I runs/live_1127/audit/summarize.py           # summary.txt, rows_auto.csv
.venv/bin/python -I runs/live_1127/audit/classify.py            # decision_rows.csv, cause_table.txt (causes set by eye in the script)
.venv/bin/python -I runs/live_1127/audit/spells.py              # spells.txt: Log / Fireball moments
CR_ENGINE=$PWD/build/gen2_k/engine nice .venv/bin/python -I runs/live_1127/audit/inject.py   # inject.txt
CR_ENGINE=$PWD/build/gen2_k/engine nice -n 15 .venv/bin/python evals/tools/threat_rate.py \
  runs/pocket_1126/g2_pocket_fresh/snapshots/ep0101074.pth,runs/g2_ppo_s2r/snapshots/ep0496977.pth \
  --sim runs/live_1127/audit/sim/obs_sim.npy --live runs/live/20261127-010105 runs/live/20261125-2*   # sim/threat_rate.txt
```

The standard-step tools are live_ppo/reread.py (log in audit/reread_run.log), runs/live_1126/towers/lanes_live.py and runs/live_1126/d2/real_final.py. The scoreboard script is audit/scoreboard.py.

---

## 1. Results

### Per match [measured, battle log + re-read]

"Opp plays" counts the opponent's plays in the re-read. A match counts as active at 5 or more plays.

| Match | Result | Crowns (us-them) | Min | Opponent deck | Opp plays | Active |
|---|---|---|---|---|---|---|
| m1 | L | 0-2 | 3.00 | Tesla evo, Archer Queen, Knight evo, Skeletons, X-Bow, Electro Spirit, Rocket, Log | 27 | yes |
| m2 | L | 0-3 | 2.12 | Goblin Giant evo, Hero Mini PEKKA, Zap evo, Sparky, Dark Prince, Electro Wizard, Heal Spirit, Rage | 12 | yes |
| m3 | W | 3-0 | 0.83 | Royal Hogs evo, Witch, Golden Knight, Mirror, Baby Dragon, Princess, Skeleton Army, Sparky | **0** | idle |
| m4 | L | 0-1 | 3.00 | Musketeer evo, Tombstone hero, Monk, Barbarian Barrel, Three Musketeers, Tesla, Ice Spirit, Poison | 12 | yes |
| m5 | L | 1-3 | 2.89 | Miner, Bats, Ram Rider, X-Bow, Rage, Witch, Princess, Mega Knight | 21 | yes |
| m6 | L | 0-3 | 1.73 | PEKKA evo, Spear Goblins, Royal Ghost evo, Heal Spirit, Hog Rider, Earthquake, Arrows, Rage | 11 | yes |
| m7 | L | 0-3 | 2.60 | Skeleton Army evo, Valkyrie hero, Goblin Barrel evo, Cannon, Princess, Wall Breakers, Ice Spirit, Dart Goblin | 24 | yes |
| m8 | L | 0-3 | 2.46 | Battle Ram evo, Electro Wizard hero, Royal Ghost evo, Magic Archer, PEKKA, Bandit, Zap, Fireball | 13 | yes |
| m9 | W | 3-0 | 2.04 | Electro Giant evo, Bowler hero, Baby Dragon evo, Tornado, Knight, Lightning, Goblin Hut, Barbarian Barrel | **2** | idle |
| m10 | L | 0-1 | 3.00 | Baby Dragon evo, Balloon hero, Inferno Dragon evo, Tornado, Freeze, Barbarian Barrel, Bowler, Lumberjack | 19 | yes |
| m11 | L | 0-3 | 2.02 | Executioner evo, Knight hero, Bomber evo, Cannon Cart, Poison, Zap, Golem, Suspicious Bush | 10 | yes |

**Overall:** 2 wins of 11. Both wins came against idle opponents: m3 made 0 plays in 50 s and m9 made 2. By eye, both were real three-crown wins over opponents who did not play.

**Against active opponents: 0 wins of 9** [measured].
- Crowns per match: 0.58 dealt, 2.65 taken.
- 6 of the 9 matches were three-crown losses.
- Per match: 4.1 Hog hits and 4.0 elixir leaked.
- Pace: 10.2 plays per minute.

TRAINING_SIDE.md counts "12 matches, 3 wins". That count is by UTC day: its extra match is a win from before this run's first match at 09:04 UTC [inferred from battle-log timestamps].

### Compared with earlier sessions (same definition of active) [measured]

| Session | Checkpoint | Active matches | Wins | Crowns dealt / taken per match | Three-crown losses |
|---|---|---|---|---|---|
| **11-27, this run** | ep0101074 | 9 | **0** | 0.58 / 2.65 | 6 |
| 11-25 night | ep0496977 (old main) | 12 | 3 | 1.07 / 1.93 | 3 |
| 11-25 cloud | ep0395878 | 10 | 0 | 0.72 / 2.68 | 7 |
| 11-24 am | (that day's main) | 11 | 2 | 1.14 / 2.07 | not counted |

Tonight looks like the cloud-checkpoint session, not the old main. On its own, 0 of 9 against 3 of 12 is a small sample: Fisher p ≈ 0.2. The sim panel points the same way, 58.4 % against 68.0 % [inferred].

---

## 2. Verdict in one paragraph

**[measured]**
- I found 59 tower-damage events against us.
- In 36 of them the bot **could see the threat, could afford an answer, and the answer was in its hand, but it held or misplaced it**. Those 36 events cost 61,150 HP, which is 78 % of the HP lost to pushes.
- The policy plays under threat at about the same rate live (p(play) 0.10–0.15) as it does in the sim (0.11–0.20). Its behaviour is in distribution, so the cause is not out-of-distribution input.
- On the same sim rows, the old main ep0496977 plays under threat about twice as often (0.20–0.40).
- Perception gaps do exist: the walking Suspicious Bush, the cloaked Royal Ghost and the Hero Mini PEKKA. They cost about 9,000 HP directly.
- Writing the missing unit into the observation moved p(play) by 0.05 at most.

**Conclusion [inferred]:** the main cause tonight is the checkpoint's passive defence, and the sim had already shown it.

---

## 3. Cause table [measured; the cause per event was set by hand from the observation, the policy output and frames]

Definitions:
- **Event:** a run of drops on one of our towers, at least 100 HP, with gaps of 6 s or less.
- **KING FELL:** a Princess counted down at the end because the King fell. It is not a push.
- **Merged events:** E9 into E7, E25 into E24, E33 into E31, E62 into E61. These are the same push spilling over, and their damage is added to the parent.

| Cause | Events | HP lost | Share | Share without KING FELL |
|---|---|---|---|---|
| CHOSE OTHER: held (answer in hand and affordable, nothing played) | 17 | 27,768 | 33.1 % | 35.6 % |
| CHOSE OTHER: misplaced (played, but wrong tile, lane or card) | 15 | 27,335 | 32.6 % | 35.0 % |
| CHOSE OTHER: offence (spent on Hog or the bridge instead) | 4 | 6,047 | 7.2 % | 7.7 % |
| KING FELL | 2 | 5,780 | 6.9 % | n/a |
| NOT SEEN (not in the observation) | 4 | 5,472 | 6.5 % | 7.0 % |
| NO ELIXIR | 4 | 4,989 | 6.0 % | 6.4 % |
| SEEN AS WEAKER (wrong class) | 2 | 3,576 | 4.3 % | 4.6 % |
| GOOD (answered, chip only) | 5 | 1,054 | 1.3 % | 1.4 % |
| ANSWER NOT IN HAND | 2 | 970 | 1.2 % | 1.2 % |
| SPELL (spell chip; no answer exists) | 4 | 841 | 1.0 % | 1.1 % |
| MASKED | 0 | 0 | 0 | 0 |
| LATE (act delay) | 0 | 0 | 0 | 0 |
| **Total** | **59** | **83,832** | | 78,052 |

**MASKED is 0 [measured].** In the event windows, 71 of 6,824 slot-decisions were masked even though the HUD digit afforded the card. Two cases account for them:
- the slot that had just been played, which shows cost 0;
- short lags in the elixir view after bursts.

None of them is the main cause of any event.

**LATE is 0 [measured].** No event was lost to the act delay. The two events where a correct answer came 4 s late, E35 and E51, are policy timing and are counted under CHOSE OTHER.

---

## 4. The owner's reports, one by one

**1. "One win doesn't count: the opponent played no card."**
- CONFIRMED [measured]. The m3 opponent made 0 plays in 50 s.
- m9 also counts as idle, with 2 plays.
- Result: 0 wins of 9 against active opponents.

**2. "Hero Mini PEKKA not countered while we were winning" (m2, E5–E6, 3,576 HP).**
- CONFIRMED, cause SEEN AS WEAKER, then CHOSE OTHER [measured].
- The detector read it as a miner-class melee unit: 1,210 HP, dps 14.9. The detector has no Hero Mini PEKKA class; see FACTS_minipekka_3m.md.
- At 4–6 elixir, with Cannon and Skeletons in hand, the bot made no play for about 10 s.
- By eye: spawned at f002990, past the bridge at f003033, tower dead by f003108, no defender placed.
- With the unit written in as a Mini PEKKA, p(play) went from 0.089 to 0.101. Cannon never reached the top two cards; the top two were Hog at 0.07 and Skeletons at 0.07. **A correct read would not have produced a defence** [measured, inject.txt].

**3. "Three Musketeers (new skin) not defended" (m4, E12–E15, 6,757 HP).**
- CONFIRMED, cause CHOSE OTHER [measured].
- Each Musketeer was in the observation as a ranged unit with 721 HP, which is the right unit.
- The Cannon was held at 3–7 elixir from f005395 to f005640.
- Hog went to the other bridge (1,15).
- p(Fireball) was 0.000–0.027 while Fireball was affordable, until 98.6 s.
- By eye at f005640: the pair was clustered.

**4. "Evo P.E.K.K.A. destroyed us for no reason" (m6, E28/E30/E31, 7,242 HP, then the King fell).**
- CONFIRMED, cause CHOSE OTHER [measured]. The PEKKA was **seen** at 3,760 HP, melee, dps 46.8.
- E28: Cannon at (9,6) beside the King and Skeletons in the centre, neither in the PEKKA's path.
- E30: 4.2 elixir with Cannon affordable, p(play) ≤ 0.27, no play. By eye at f010273 and f010707: no defender.
- E31: PEKKA, Royal Ghost and Rage on the King, which went from 4824 to 775 in about 4 s.
- Same pattern in m8 with PEKKA: E47 (3,270), E48 (3,052) and E49 (1,463). The bot held at 5–8 elixir from 93 to 117 s, and the one Ice Golem went to the other lane.

**5. "Royal Ghost walked to the tower" (m6 E29 317 HP, m8 E44 1,315 HP).**
- CONFIRMED, cause NOT SEEN [measured].
- During this run the encoder dropped the cloaked ghost: no enemy cell in any decision. The sim does the same.
- By eye: faint at f010237 and f013855.
- E44: at 8 elixir with Cannon in hand, the bot made a forced Musketeer play to the back row.
- With a unit written in at the tracked tile [measured]:
  - E44: p(play) went from 0.064 to 0.110; P(at least one play over 26 decisions) from 0.83 to 0.96. But the top cards were Musketeer, then Hog, with Cannon ≤ 0.04.
  - E29: p(play) went from 0.147 to 0.155.
- live_ppo/cloaked_rows.py, added after this run per TRAINING_SIDE.md, now shows the walking ghost live. The sim still never shows it, so that input is new to the policy [inferred].

**6. "No Logs on Goblin Barrels" (m7, about 8,761 HP across E35–E42).**
- CONFIRMED, cause CHOSE OTHER (held or misplaced) [measured].
- Code 109 was in the observation's opponent-spell slot for 3 of the 4 barrels, with t_imp 2.3–2.7 s at first sight. That contradicts FACTS_minipekka_3m.md ("evolved barrel never in OPP_SPELLS"). The barrel agent (barrel/REPORT.md) also finds the tile known in time and blames the late Log.

| Barrel | Landing | Our hand and choice | HP lost |
|---|---|---|---|
| 1 | 33.0 s at (3,6) | Log in hand, p(Log) 0.063. Log and Ice Spirit went to the bridge (3,15) while it flew; Skeletons (3,8) 4 s late. By eye f011597. | 3,052 (tower to 0) |
| 2 | 78.45 s at (14,6) | Log in hand and affordable, p(Log) 0.11–0.15, held. By eye f012072. | 1,661 |
| 3 | 129.3 s at the King | Not in the spell slot. Forced Log (p(Log) 0.53) to (14,4), the wrong side; the goblins were at x 8–11. By eye f012598. | 816 |
| 4 | 146.7 s at the King | Log not in hand. Fireball at 149.4 to (14,24). Log in hand from 150.6 s with p(Log) 0.000. By eye f012790. | King to 61 |

In the sim, ep0101074 Logs 18 % of barrels within 3 s; people Log 59 % (TRAINING_SIDE.md).

**7. "Fireball not used when it should have been."**
- CONFIRMED [measured].
- 36 Fireball moments, 4 taken, 1 on target; details in section 6.

**8. "Generally not really defending."**
- CONFIRMED [measured]. CHOSE OTHER accounts for 78 % of HP lost to pushes.
- p(play) under threat: live 0.10–0.15, against 0.08–0.10 without a threat, so a threat lifts it very little.

**9. "Moments where it played really well."**
- Also true [measured]. 5 events were GOOD, chip damage only (1,054 HP in all):
  - E10 (m4): Monk chip, answered by Musketeer and Ice Golem.
  - E11 (m4): chip, answered by Ice Golem.
  - E16 (m4): chip, answered by a forced Cannon.
  - E18 (m5): Miner plus Princess, answered by Log, Ice Golem, Ice Spirit and Skeletons.
  - E34 (m7): Dart Goblin, answered by Ice Golem.
- m4 and m10 were close 0-1 losses.

**10. "Never really defends the Suspicious Bush."**
- CONFIRMED; see section 5.

---

## 5. Suspicious Bush (m11): every push

**Encoder [measured]**
- runs/live_1127/ENCODER_COVERAGE.md had not been written when this was finished.
- From TRAINING_SIDE.md, measured from the code: the live encoder's `card_table()` has no row for the Bush (85) or its goblins (−30). The walking Bush is dropped live, and the sim's observation does not show it either.
- I checked the recorded observations directly. While each Bush walked, **no decision had an enemy cell in the Bush's lane** (channels 4–7), although the tracker held it at confidence 0.75–0.96.

| Push | Bush played | Walk | Goblins in the observation | What we did | HP lost |
|---|---|---|---|---|---|
| Left (E56) | 11.7 s at (3,17) | 11.7–19.4 s, tracked all the way, not in the observation (f020177) | from 19.9 s: 2 melee cells, 404 HP | During the walk: Log to (14,15), the **other** lane's bridge, at 15.95 s, and Ice Spirit to (1,15). After the goblins appeared: held at 8–9.5 elixir until a forced Cannon at (8,9) at 23.9 s. | 1,536 |
| Right (E59) | 59.2 s at (14,17) | 59.2–69.4 s, tracked, not in the observation (f020640) | on arrival | Cannon at (8,9) at 61.4 s. By eye, the Bush moved toward it and then went back to the tower. Elixir view read 0.9–1.5 against 2–3 on the HUD. Log in hand from 70.6 s at 3.7–6.7 elixir with p(play) ≤ 0.04. | 2,304 |

**Injection [measured, inject.txt].** A building-targeter cell, with Hog Rider's signature, was written at the tracked tile on every walk decision:
- Left: p(play) went from 0.137 to 0.144 (15 decisions).
- Right: p(play) went from 0.090 to 0.094 (19 decisions).

**So [inferred]:**
- Encoding the Bush alone would not have produced a defence.
- After the goblins became visible, the bot still held at 8–9.5 elixir (E56) or ignored the Log (E59).

**Sim fidelity (TRAINING_SIDE.md).** In the sim, the Bush wrongly hits the tower for 256 itself before the goblins spawn.

**Cheapest answers in the sim** (TRAINING_SIDE.md):
- a Log mid-lane does 0 damage to the tower;
- a Cannon at (8.5,9) does 0;
- no answer costs 1,280.

**Exposure:** Bush decks are under-exposed in training, ×0.53.

---

## 6. Fireball and Log moments [measured; spells.txt and fb_live_20261127-010105_tracked.json, using the 11-26 definition]

**Fireball**
- By the main definition there were **36 moments, 4 taken, 1 on target**.
- The median of the highest p(Fireball) in each moment was **0.003**. Only 5 moments reached 0.1 or more.
- By the strict definition: 29 moments, 3 taken, 0 on target.

Low enemy Princess (under 15 %) with Fireball:
- m5: their right Princess sat at 1.5–2.6 % (about 45–80 HP) from 119 to 156 s, with Fireball in hand and affordable. p(Fireball) was 0.001–0.064. The bot cast it only when the leak floor forced a play at 156.1 s, to (12,25), and the tower fell at 161.6 s.
- m2: their left Princess sat at 186 HP (6 %) from 100 s until the end at 127.3 s. Fireball was not in hand until 126.3 s; Hog was chosen then.
- These are observed facts only: Sam's rule is that win rate is the target, not Fireball counts.

**Log on swarms and barrels.** The 16 log-able landings on our side, and our plays in the next 6 s:

| Unit | Landings | Result |
|---|---|---|
| Goblin Barrel | 4 | Log on target 0 times; details in section 4, item 6 |
| Skeleton Army (evo, m7) | 3 | no Log within 6 s (E39: Log in hand, p(play) ≤ 0.07) |
| Spear Goblins (m6) | 2 | no Log (E27: Log in hand at 4.2 elixir, p(play) ≤ 0.07) |
| Suspicious Bush | 2 | Log once, to the other lane's bridge |
| Barbarian Barrel | 5 | mostly on their side; no answer needed |

**Three Musketeers (m4)**
- The re-read logs the play at 91.6 s.
- p(Fireball) was 0.000–0.027 while Fireball was affordable, until 98.6 s.

---

## 7. Perception or policy?

**p(play) under threat, recomputed on the same rows** [measured, audit/sim/threat_rate.txt]
- Threat means an enemy troop cell near a standing own Princess.
- Columns are elixir 2–4 / 4–6 / 6–8 / 8–10.
- The network has no memory, so p(play) depends only on the observation.

| Checkpoint | Rows | Under threat | No threat |
|---|---|---|---|
| ep0101074 | sim (2,403 rows) | 0.110 / 0.158 / 0.170 / 0.204 | 0.077 / 0.107 / 0.097 / 0.182 |
| ep0101074 | tonight live (2,672) | 0.104 / 0.153 / 0.147 / 0.142 | 0.076 / 0.093 / 0.072 / 0.095 |
| ep0496977 | sim (2,403) | **0.198 / 0.349 / 0.398 / 0.279** | 0.118 / 0.181 / 0.155 / 0.226 |
| ep0496977 | tonight live (2,672) | 0.159 / 0.246 / 0.251 / 0.080 | 0.064 / 0.093 / 0.092 / 0.120 |

**What this shows [measured]**
- ep0101074's under-threat play rate is about the same live as in the sim. It does not collapse on live input.
- On the sim rows, ep0496977 plays under threat 1.4–2.3× as often.
- On tonight's live rows, ep0496977 plays under threat 1.5–1.7× as often at 2–8 elixir.

**Caveat [measured, inferred].** On the 11-25 night rows the order reverses: ep0101074 gives 0.14–0.21 and ep0496977 0.13–0.17. Those rows were recorded without `--afford-project`, which ep0101074's training assumes, so that comparison is not like for like [inferred]. The sim rows are the clean comparison.

**General play rate in the sim** [measured, sim_play_rate.log: ep0101074, 60 matches, 16,821 decisions]
- p(play) is 0.08–0.13 below 9.5 elixir.
- At a full bar (9.5+), p(play) is 0.41 with 0 towers down, 0.54 with 1 down, and 0.02 with 2 down.
- So the bot waits for the full bar, and the live leak floor then forces 17 % of its plays (section 8).

**Injection [measured, inject.txt].** Writing in the missing unit (Bush, Royal Ghost, Hero Mini PEKKA as Mini PEKKA) moved p(play) by at most 0.05. The cards it raised were Musketeer and Hog, not Cannon.

---

## 8. Standard steps

**Lanes (lanes_live.py) [measured]**

| Situation at the Hog play | Hogs | To the lower tower's lane | To the other lane | Towers level |
|---|---|---|---|---|
| Both Princesses up, lower one ≥ 30 % | 34 | 10 | 6 | 18 |
| Lower one 15–30 % | 1 | 1 | 0 | 0 |
| Lower one < 15 % | 2 | 2 | 0 | 0 |
| One tower down | 1 | to the dead lane | | |

- Reopen thresholds were not met.

**Real finals (real_final.py) [measured]**
- Matches ending with an enemy Princess alive under 15 %:
  - tonight: 1 of 11 (9 %; m2, tower at 6 %), under the 21 % reopen threshold;
  - people: 6.8 % (val, 424 matches) and 9.8 % (all, 8,547 matches).

**Clone/Mirror re-read [measured]**
- Only m3's deck holds Mirror, and that opponent made 0 plays, so there was nothing to check.

**TOWER_DEATH_CONFIRM [measured]**
- No false "destroyed" read, for opponent or own towers.
- Every dead flag matches the battle log. The lowest live readings match the battle-log final HP:

| Match | Tower | Lowest reading | Battle-log final HP |
|---|---|---|---|
| m2 | opponent's left Princess | 0.061 | 186 |
| m4 | our right Princess | 0.02 | 61 |
| m10 | our left Princess | 0.487 | 1486 |

- Deaths were confirmed at 379.4 s (m3), 792.1 s (m5), and 1341.5 and 1401.7 s (m9).
- One pocket line, "numeral 3052 read while latched: pocket closed again", appeared at about 409 s. That is the start of m4, a new match, so the reset is correct.

**Pocket mask [measured]**
- No pocket tile was refused.
- 8 pocket plays, all with `tile_in_pocket` True:
  - m3: Musketeer at (14,22).
  - m9: Skeletons, two Ice Golems, two Hogs at (9,22), a Musketeer and an Ice Spirit at (8,22).
- All placements (not only pocket ones): either oracle confirmed 214 of 218; the other 4 had no landing evidence.

**Afford-project [measured]**
- 2,680 decisions; projection changed the affordable set on 341 of them.
- Of 10 plays offered only under the projection, 9 landed and 1 failed (Ice Golem at 1.57 elixir in the view).
- Of 10 plays costing more than the elixir view, 9 landed and 1 failed.
- 5 of 259 plays failed in all.

**Forced plays (leak floor 9.5) [measured]**
- 45 of 259 plays (17 %) were forced; on 11-25 the share was 10 %.
- 35 plays went to the back row (y ≤ 1), 14 of them forced: Ice Spirit 9, Musketeer 9, Skeletons 8, Ice Golem 5, Hog 4.
- Plays per decision: 9.7 %, against 11.4 % on 11-25.

**Elixir view against the HUD [measured]**
- The view read more than 1.5 below the HUD on 1.6 % of decisions, against 0.5 % on 11-25.
- It mattered in E57, where the view said 7.5 while the HUD showed 9 (f020339).

---

## 9. Ranked changes

**1. Gate deployment on defence; use ep0496977 (or a later snapshot that passes) for the next session.**
- Evidence:
  - sim panel 58.4 % against 68.0 % [measured];
  - sim p(play) under threat about half the old main's [measured];
  - live 0 of 9 against 3 of 12 [measured];
  - 78 % of HP lost to held or misplaced answers [measured].
- Every number above was available before the session, so the sim had predicted the result [inferred].
- Needs, no new code:
  - a pre-deploy gate: common-panel win rate at least ep0496977's;
  - `evals/tools/threat_rate.py` on audit/sim/obs_sim.npy at least ep0496977's;
  - the defence-probe answer rate from runs/live_1127/defence, once that run finishes, at least the old main's.
- This also means the pocket run needs more training before it is deployed [inferred].

**2. Train against the decks that beat us (placement and answer choice): 15 misplaced events, 27.3k HP.**
- The main cases:
  - X-Bow sieges never answered by Hog or Fireball (E0, E1, E19, E22, E23: about 11.4k HP);
  - the barrel Log (m7: about 8.8k HP);
  - a centre Cannon (8,9) or (9,6) that does not pull edge-lane troops (E43 Ram; (7,9) does);
  - Ice Golem to the wrong lane (E55, 1.3k; E48, 3.1k, is counted as held).
- Needs, owned by the training agent:
  - deck re-weighting to the battle-log shares (barrel ×0.56, Bush ×0.53, PEKKA ×0.5; X-Bow share not measured here);
  - the sim fixes listed in TRAINING_SIDE.md.
- Expected effect: helps, but does not fix the passivity in item 1 [inferred].

**3. Encode the Bush and the cloaked Royal Ghost in the sim and live together; add a Hero Mini PEKKA detector class.**
- Cost: 6 events, about 9.0k HP (11.6 % of push damage).
- The injection test says these fixes alone barely move p(play) (≤ 0.05) [measured]. They pay off only with a policy that defends.
- Encoding them live only (cloaked_rows.py) gives the policy an input it never trained on [inferred].
- Needs:
  - an encoder change in the sim and the live code, owned by training and perception;
  - a detector class, owned by the perception agent.

**4. Elixir view after bursts (low priority).**
- The view read more than 1.5 below the HUD on 1.6 % of decisions, 3× the 11-25 rate.
- It cost at most part of E57 and E59.
- Needs: a perception check of the view after multi-card bursts, owned by the perception agent.

**5. Leak-floor forced plays (watch, no change).**
- 17 % of plays were forced, often to the back row.
- This is a symptom of the policy waiting for a full bar (sim p(play) 0.41 at 9.5+ against ≤ 0.13 below it).
- Changing the floor would be a live heuristic, against the "the policy must improve generally" ruling [inferred]. Item 1 should reduce it.

Fireball and Log rates are observe-only (win rate is the target) and are not ranked as targets.

---

## 10. All events

**What the table shows**
- Times are match seconds.
- Tower: L, R or K (ours).
- "In the observation at t0" sums the enemy cells on that lane at the first drop: count, HP, highest dps, type, lowest y.
- The hand is at threat onset; "+" means affordable.

**Columns in decision_rows.csv:** `id, m, tower, t0, t1, dmg, dmg_with_merged, hp_after, cause, note, frames, threat, obs_onset, obs_t0, e_view, e_digit, hand, pmax5, plays5, plays_later, yolo, onset_t, onset_frame, final`. Merged events (E9, E25, E33, E62) are in the CSV only, inside their parent's dmg_with_merged.

| id | m | t (s) | tower | HP lost | in the observation at t0 | elixir view/HUD | hand at onset (+ affordable) | max p(play), 5 s | our plays, first 5 s | cause | what happened |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 | 1 | 64-77 | L | 2680 | 2c HP 3366 dps 16.8 building+melee ymin20 | 8.4/8.0 | IG+ Can+ Log+ Mus+ | 0.23 | 49.1:Mus@8,10 50.4:IG@0,9 | CHOSE OTHER: misplaced | X-Bow at their bridge + Evo Knight; Cannon (9,11) and (9,9) out of the X-Bow's reach; Hog/FB not in hand at onset |
| E1 | 1 | 101-117 | R | 2623 | 2c HP 1815 dps 15.2 building+ranged ymin20 | 2.1/2.0 | IS+ FB- IG+ Can- | 0.25 | 98.8:IS@8,9 | CHOSE OTHER: misplaced | same X-Bow siege, right; 2 elixir at onset, FB affordable from ~102 s, never cast on the X-Bow; Cannon (9,9) centre |
| E2 | 1 | 147-147 | L | 337 | nothing | 6.1/6.0 | Mus+ FB+ Sk+ Can+ | 0.15 | 147.9:Sk@9,15 | SPELL | 337-HP tower finished by a Rocket |
| E3 | 1 | 166-168 | R | 429 | 1c HP 1000 dps 19.3 ranged ymin10 | 6.3/6.0 | Sk+ Hog+ IG+ Can+ | 0.14 | 165.1:Hog@8,0 168.4:IG@13,15 | CHOSE OTHER: offence | Archer Queen on a 429-HP tower; Hog at the back row (8,0), IG at the bridge |
| E4 | 2 | 32-36 | L | 1298 | 2c HP 1914 dps 19.0 melee ymin11 | 6.5/6.0 | IS+ FB+ Can+ Mus+ | 0.34 | 29.0:Mus@8,7 31.3:IS@8,15 | CHOSE OTHER: misplaced | Dark Prince + Rage; Musketeer far back (8,7), Ice Spirit late, Cannon in hand unplayed |
| E5 | 2 | 81-86 | L | 1754 | 1c HP 1210 dps 14.9 melee ymin8 | 4.1/4.0 | Sk+ FB+ Can+ Hog+ | 0.17 | - | SEEN AS WEAKER | Hero Mini PEKKA read as miner (1210 HP, dps 14.9); 4-6 elixir, Cannon/Skeletons in hand, no play for ~10 s; injected as Mini PEKKA p(play) 0.089->0.101 |
| E6 | 2 | 90-91 | K | 1822 | nothing | 2.7/2.0 | Sk+ FB- Can- Hog- | 0.24 | - | SEEN AS WEAKER | same unit on the King; 7 elixir at f003140, no defender; Hog to the left bridge |
| E7 | 2 | 121-127 | K | 3002 | 2c HP 4561 dps 33.3 bldg-tgt+ranged ymin7 | 6.4/6.0 | Sk+ Mus+ Can+ Log+ | 0.26 | - | CHOSE OTHER: offence | Goblin Giant evo + Sparky + Heal + Rage at the King; Cannon/Sk/Mus spent at 113 s, last plays IG (2,15) + Hog (1,15) at the bridge (+E9) |
| E8 | 2 | 127-127 | R | 3052 | nothing | 7.9/7.0 | Sk+ Mus+ Can+ Log+ | 0.43 | 113.3:Can@8,9 113.8:Sk@3,15 114.4:Mus@8,5 | KING FELL | right Princess counted down because the King fell |
| E10 | 4 | 16-20 | R | 135 | 1c HP 2214 dps 17.5 melee ymin13 | 8.8/8.0 | IG+ Can+ Sk+ Mus+ | 0.29 | 11.1:Mus@17,11 14.0:IG@10,13 | GOOD | Monk chip; Musketeer + IG answered |
| E11 | 4 | 64-67 | L | 135 | 3c HP 2131 dps 21.7 ranged ymin15 | 8.1/8.0 | IG+ Can+ IS+ FB+ | 0.39 | 56.6:IG@0,9 | GOOD | chip; IG answered |
| E12 | 4 | 65-70 | R | 1020 | 1c HP 721 dps 21.7 ranged ymin15 | 6.8/6.0 | Hog+ Can+ IS+ FB+ | 0.39 | - | CHOSE OTHER: offence | Three Musketeers (one right): nothing on the right; Hog (1,15), IS, Sk left/centre |
| E13 | 4 | 77-90 | L | 2917 | 2c HP 1442 dps 21.7 ranged ymin13 | 7.9/7.0 | Hog+ Can+ IS+ FB+ | 0.37 | 65.8:Hog@1,15 66.3:IS@4,15 | CHOSE OTHER: held | two Musketeers (3M) at the left bridge, read as Musketeers (721 HP); Cannon held at 3-7 elixir; FB not cast on the pair |
| E14 | 4 | 135-143 | R | 1596 | 2c HP 1442 dps 21.7 ranged ymin13 | 7.0/6.0 | Hog+ Can+ IG+ FB+ | 1.00 | 132.2:IG@15,15F | CHOSE OTHER: offence | 2 Musketeers + Poison; forced IG then Hog (13,13); 8-10 elixir; FB late onto their side |
| E15 | 4 | 140-147 | K | 1224 | 1c HP 721 dps 21.7 ranged ymin9 | 7.0/6.0 | Hog+ Can+ IG+ FB+ | 1.00 | 132.2:IG@15,15F | CHOSE OTHER: held | Musketeer on the King from the ruins; 6-10 elixir, Cannon in hand, nothing placed |
| E16 | 4 | 159-162 | R | 135 | 1c HP 721 dps 21.7 ranged ymin16 | 8.0/8.0 | Log+ Can+ Sk+ IS+ | 1.00 | 153.9:Can@11,10F | GOOD | chip; forced Cannon |
| E17 | 4 | 175-179 | R | 105 | nothing | 6.0/6.0 | Log+ Hog+ Sk+ IS+ | 0.07 | 162.1:Hog@16,15 | SPELL | Poison + Barbarian Barrel chip |
| E18 | 5 | 11-17 | L | 363 | 1c HP 261 dps 5.6 ranged ymin18 | 8.8/8.0 | Hog+ FB+ IG+ Log+ | 0.76 | 9.6:Log@3,15 10.6:IG@3,15 11.2:Hog@16,15 | GOOD | Miner + Princess chip; Log/IG/IS/Sk answered |
| E19 | 5 | 74-84 | R | 2627 | 3c HP 5854 dps 15.8 building+melee+ranged ymin11 | 8.2/8.0 | IS+ FB+ Mus+ Log+ | 0.58 | 61.1:Mus@9,8 62.8:IS@9,15 63.3:Log@10,13 | CHOSE OTHER: misplaced | X-Bow (11,18) + Mega Knight + Ram Rider; many plays, FB in hand never on the X-Bow; ran dry |
| E20 | 5 | 91-91 | R | 425 | 1c HP 1766 dps 15.9 bldg-tgt ymin7 | 2.4/2.0 | Can- FB- Mus- IG+ | 0.10 | - | NO ELIXIR | Ram Rider finishes 425; 2.4 elixir, only IG affordable |
| E21 | 5 | 93-95 | K | 813 | 3c HP 1928 dps 15.9 bldg-tgt+melee/air ymin4 | 2.4/2.0 | Can- FB- Mus- IG+ | 0.19 | 94.5:Can@8,9 | NO ELIXIR | Ram Rider + Bats on the King; 2.4 elixir, Cannon as soon as affordable |
| E22 | 5 | 128-145 | L | 2689 | 2c HP 1861 dps 15.2 building+ranged ymin18 | 6.3/6.0 | IS+ FB+ Mus+ Hog+ | 0.31 | 120.0:Mus@9,13 120.5:IS@9,9 122.1:Log@12,15 | CHOSE OTHER: misplaced | X-Bow (7,18) + Ram Rider + Miner; FB and Hog in hand at onset, never at the X-Bow; 9 defensive plays |
| E23 | 5 | 142-149 | K | 810 | 6c HP 3109 dps 15.9 bldg-tgt+melee+ranged ymin8 | 4.8/4.0 | Can+ FB+ IG+ Hog+ | 0.32 | 130.6:IG@8,8 | CHOSE OTHER: misplaced | same push on the King |
| E24 | 5 | 160-174 | K | 3201 | 1c HP 261 dps 5.6 ranged ymin11 | 7.0/6.0 | Can+ Mus+ Sk+ Hog+ | 1.00 | 162.8:Can@9,6F | CHOSE OTHER: held | both Princesses down; pack at the King while at 9 elixir until a forced Cannon (+E25) |
| E26 | 6 | 16-16 | L | 317 | 1c HP 1697 dps 19.8 bldg-tgt ymin9 | 2.2/2.0 | Can- FB- IG+ Log+ | 0.15 | - | NO ELIXIR | Hog Rider; 2.2 elixir, Cannon not affordable |
| E27 | 6 | 25-27 | R | 324 | 2c HP 266 dps 5.1 ranged ymin12 | 4.2/4.0 | Can+ FB+ IG+ Log+ | 0.07 | - | CHOSE OTHER: held | Spear Goblins; Log in hand at 4.2, p(play) <= 0.07 |
| E28 | 6 | 46-46 | L | 842 | 1c HP 3760 dps 46.8 melee ymin9 | 6.3/6.0 | Can+ FB+ Sk+ Log+ | 0.27 | 32.2:Sk@9,10 32.7:Can@9,6 | CHOSE OTHER: misplaced | Evo PEKKA seen (3760 HP); Cannon (9,6) beside the King, Skeletons centre |
| E29 | 6 | 56-56 | L | 317 | nothing | 1.8/1.0 | Mus- FB- IS+ Log- | 0.13 | - | NOT SEEN | Royal Ghost dropped by the encoder (also 1.8 elixir); injected p(play) 0.147->0.155 |
| E30 | 6 | 90-94 | L | 1576 | 1c HP 716 dps 13.7 melee ymin9 | 4.2/4.0 | Can+ FB+ Mus+ IS+ | 0.27 | - | CHOSE OTHER: held | Evo PEKKA seen (3760 HP, y 10) at the tower; 4.2 elixir, Cannon affordable, p(play) <= 0.27, no play |
| E31 | 6 | 99-103 | K | 4824 | 2c HP 5457 dps 46.8 bldg-tgt+melee ymin4 | 6.4/6.0 | Can+ FB+ Mus+ IS+ | 0.59 | 95.6:Can@6,6 | CHOSE OTHER: misplaced | PEKKA + Royal Ghost + Rage at the King; Cannon (6,6), IG/IS/Sk to the right bridge (+E33) |
| E32 | 6 | 104-104 | R | 2728 | nothing | 4.9/4.0 | Can+ FB+ Mus+ IS+ | 0.51 | - | KING FELL | right Princess counted down because the King fell |
| E34 | 7 | 14-14 | R | 286 | 1c HP 261 dps 17.9 ranged ymin15 | 8.1/8.0 | IG+ Hog+ Can+ Mus+ | 0.25 | 11.3:Hog@1,15 11.8:IG@12,9 | GOOD | Dart Goblin chip; IG answered |
| E35 | 7 | 37-45 | L | 3052 | 3c HP 665 dps 11.4 melee+ranged ymin8 | 4.5/4.0 | Hog+ Sk+ Can+ FB+ | 0.42 | 39.2:Sk@3,8 | CHOSE OTHER: misplaced | Evo Goblin Barrel at 33.0 on the left tower (code 109 in the observation); Log + IS went to the bridge (3,15) as it flew; Skeletons 4 s late |
| E36 | 7 | 46-48 | K | 625 | 4c HP 687 dps 11.4 melee ymin4 | 4.5/4.0 | Hog+ Sk+ Can+ FB+ | 0.42 | 39.2:Sk@3,8 | CHOSE OTHER: misplaced | same goblins move to the King |
| E37 | 7 | 75-84 | R | 1661 | 2c HP 2168 dps 17.7 melee+ranged ymin16 | 5.8/5.0 | Mus+ IS+ Log+ FB+ | 0.06 | - | CHOSE OTHER: held | Barrel 2 at 78.5 (14,6), code 109 in the observation; Log in hand and affordable, p(Log) 0.11-0.15, not thrown |
| E38 | 7 | 78-80 | K | 429 | 3c HP 2429 dps 17.9 melee+ranged ymin9 | 5.8/5.0 | Mus+ IS+ Log+ FB+ | 0.06 | - | CHOSE OTHER: held | Dart Goblin + Valkyrie hero on the King, same window |
| E39 | 7 | 106-108 | R | 1009 | 10c HP 1059 dps 25.2 bldg-tgt+melee ymin9 | 4.0/3.0 | Sk+ Can+ Log+ FB- | 0.07 | - | CHOSE OTHER: held | Evo Skeleton Army + Wall Breakers; Log in hand, 7 elixir at f012313, p(play) <= 0.07 |
| E40 | 7 | 121-123 | K | 286 | 1c HP 261 dps 17.9 ranged ymin9 | 7.9/7.0 | Sk+ IG+ Log+ FB+ | 1.00 | 119.8:IG@9,0F | CHOSE OTHER: held | Dart Goblin chip on the King; forced IG to the back row |
| E41 | 7 | 132-134 | K | 816 | 3c HP 606 dps 11.4 melee ymin2 | 8.0/8.0 | Sk+ IS+ Hog+ FB+ | 1.00 | 134.2:IS@11,8F | CHOSE OTHER: misplaced | Barrel 3 on the King (not in the spell slot); forced Log thrown at (14,4), the goblins at x 8-11 |
| E42 | 7 | 151-156 | K | 2607 | 15c HP 3368 dps 25.2 bldg-tgt+melee+ranged ymin1 | 6.0/6.0 | Sk+ Can+ IG+ FB+ | 0.00 | - | CHOSE OTHER: held | Barrel 4 + Skeleton Army + Wall Breakers at the King; Log not in hand at landing, then in hand at 7-9 elixir with p(Log)=0.000; Sk/Cannon held |
| E43 | 8 | 19-22 | L | 957 | nothing | 7.2/7.0 | ?- ?- ?- ?- | 0.24 | 8.6:Sk@9,0 | CHOSE OTHER: misplaced | Evo Battle Ram seen (967 HP); Hog in the same lane 1 s before, Cannon (8,9) did not pull the edge-lane Ram |
| E44 | 8 | 62-69 | L | 1315 | nothing | 3.7/3.0 | Can+ FB- Hog- Mus- | 0.01 | - | NOT SEEN | Royal Ghost dropped by the encoder; 8 elixir with Cannon in hand; forced Musketeer to the back row; injected p(play) 0.064->0.110 |
| E45 | 8 | 76-76 | L | 159 | nothing | 8.6/8.0 | Can+ FB+ Hog+ Mus+ | 1.00 | 63.3:Mus@8,0F | SPELL | opponent Fireball |
| E46 | 8 | 99-101 | L | 621 | 2c HP 1496 dps 20.4 bldg-tgt+ranged ymin9 | 4.2/4.0 | Mus+ FB+ Log+ IS+ | 0.18 | 92.5:Mus@9,7 | CHOSE OTHER: misplaced | Evo Battle Ram + Magic Archer; Log in hand, Musketeer (9,7) and IS centre |
| E47 | 8 | 105-117 | K | 3270 | 4c HP 5721 dps 46.8 melee+ranged ymin5 | 5.1/5.0 | Mus+ FB+ Log+ IS+ | 0.42 | 92.5:Mus@9,7 93.0:IS@8,15 | CHOSE OTHER: held | Magic Archer + Ram riders on the King; nothing placed 93-117 s at 5-8 elixir |
| E48 | 8 | 105-121 | R | 3052 | 1c HP 3760 dps 46.8 melee ymin9 | 5.3/5.0 | Mus+ FB+ Log+ IS+ | 0.42 | 92.5:Mus@9,7 93.0:IS@8,15 | CHOSE OTHER: held | PEKKA + Electro Wizard hero seen; nothing on the right; IG (2,15) to the other lane |
| E49 | 8 | 146-147 | K | 1463 | 3c HP 5256 dps 46.8 bldg-tgt+melee+ranged ymin5 | 7.7/7.0 | Hog+ FB+ IS+ Can+ | 1.00 | 133.8:Hog@8,0F | CHOSE OTHER: held | PEKKA + Battle Ram at the King; 7-9 elixir, forced Hog to the back row (8,0), Log to the bridge |
| E50 | 10 | 46-53 | R | 1545 | 2c HP 2971 dps 32.0 bldg-tgt/air+melee/air ymin11 | 8.4/8.0 | Log+ IS+ FB+ Mus+ | 1.00 | 40.8:Mus@17,8F 41.4:IS@10,15 | CHOSE OTHER: misplaced | Balloon hero + dragons (air); forced Musketeer at the edge (17,8), Cannon cannot hit air, FB unused at 8 |
| E51 | 10 | 87-97 | R | 1095 | 3c HP 4515 dps 32.0 melee+ranged+ranged/air ymin9 | 8.2/8.0 | Log+ Mus+ FB+ IG+ | 0.46 | 85.2:Mus@9,7 87.0:IG@10,15 | CHOSE OTHER: held | Lumberjack + Bowler + Baby Dragon; held 8 elixir 75-85 s, defenders late |
| E52 | 10 | 138-140 | R | 412 | 2c HP 2958 dps 32.0 bldg-tgt/air+melee ymin8 | 3.2/3.0 | IG+ Can+ Log+ Sk+ | 0.07 | 130.1:Log@14,15 131.4:Sk@14,15 | ANSWER NOT IN HAND | Lumberjack then Balloon hero; 3.2 elixir, no Musketeer in hand, Cannon cannot hit air |
| E53 | 10 | 142-144 | K | 558 | 4c HP 5863 dps 32.0 bldg-tgt/air+building+melee+ranged ymin3 | 3.2/3.0 | IG+ Can+ Log+ Sk+ | 0.07 | 130.1:Log@14,15 131.4:Sk@14,15 | ANSWER NOT IN HAND | same Balloon on the King; Musketeer later at the edge (17,8) |
| E54 | 10 | 154-154 | L | 240 | nothing | 6.5/6.0 | IG+ Can+ Hog+ Mus+ | 0.51 | 139.9:Can@8,13 141.7:Mus@17,8 | SPELL | Bowler boulder / bomb chip, nothing in the observation |
| E55 | 10 | 170-174 | L | 1326 | 1c HP 1282 dps 32.0 melee ymin8 | 6.2/6.0 | Sk+ Mus+ IS+ Can+ | 0.41 | 165.5:Mus@9,8 | CHOSE OTHER: misplaced | Lumberjack + Bowler + Inferno Dragon on the left; IG to the right, Cannon held |
| E56 | 11 | 22-26 | L | 1536 | 2c HP 404 dps 11.4 melee ymin9 | 8.1/8.0 | FB+ Can+ Sk+ Mus+ | 1.00 | 23.9:Can@8,9F | NOT SEEN | Suspicious Bush walked 11.7-19.4 s tracked at conf 0.75-0.96 but absent from the observation; goblins seen from 19.9 s, still held at 8-9.5 until a forced Cannon (8,9); injected p(play) 0.137->0.144 |
| E57 | 11 | 43-65 | L | 1516 | 2c HP 6065 dps 22.4 bldg-tgt+ranged ymin15 | 7.5/9.0 | FB+ Hog+ IG- Mus+ | 0.21 | - | CHOSE OTHER: held | Golem + Cannon Cart + Poison + Knight hero; 9 elixir (view 7.5) with FB/Hog/Mus, no play 35-40.8 s, then six plays to 0 elixir |
| E58 | 11 | 72-85 | K | 3434 | 4c HP 3979 dps 22.4 melee+ranged ymin7 | 2.9/2.0 | FB- IG+ Mus- Can- | 0.25 | 60.9:Can@8,9 | NO ELIXIR | Cannon Cart + Knight hero at the King after the flood; 2-3 elixir |
| E59 | 11 | 72-78 | R | 2304 | 2c HP 404 dps 11.4 melee ymin9 | 1.5/3.0 | FB- IG- Mus- Log- | 0.04 | - | NOT SEEN | Suspicious Bush walked 59.2-69.4 s tracked but absent from the observation; then 1.5 view / 3 HUD elixir; Log in hand from 70.6 s at 3.7-6.7, p(play) <= 0.04; injected p(play) 0.090->0.094 |
| E60 | 11 | 112-116 | R | 748 | 3c HP 7344 dps 22.4 bldg-tgt+ranged ymin9 | 6.7/6.0 | FB+ Hog+ Sk+ Log+ | 0.03 | - | CHOSE OTHER: held | Golem + Executioner evo + Cannon Cart; p(play) <= 0.03 at 6-8 elixir; forced Sk (17,10), Hog |
| E61 | 11 | 119-121 | K | 1390 | 3c HP 7344 dps 22.4 bldg-tgt+ranged ymin4 | 9.2/9.0 | FB+ Hog+ Sk+ Log+ | 1.00 | 105.5:Sk@17,10F 106.0:Hog@1,15 | CHOSE OTHER: held | same push at the King; 8-9 elixir, FB only at 119.7 s (+E62) |
