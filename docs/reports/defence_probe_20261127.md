# Defence probe: does the policy defend in the sim, and is PPO making it better? (2026-11-27)

Tags: [measured] = counted in these runs. [inferred] = my reading of the numbers.

## Short answer

- **Toward main.** The restart (C5 lineage) defends better than C5 on every measure, and it is moving toward main [measured].
  - Most of the gain came in the first 50k episodes.
  - From 50k to 133k it is nearly flat: held 15 → 12 %, answered 58 → 60 % [measured].
  - At 133k it sits about where main was at 100k on holds, answer rate and damage per threat [measured].
  - It is ahead of main-at-100k on wins (32 % vs 24 % pooled) [measured].
- **More training fixes the general hold, but slowly** [inferred from main's own trend].
  - Main took 400k more episodes to go from 11 % held to 8 %.
  - At the restart's pace, reaching main-497k's level needs a few hundred thousand more episodes.
- **More training will not fix the Goblin Barrel answer at a useful pace** [measured + inferred].
  - The barrel gap comes from the imitation start, not from PPO. C5 answers 22 % of barrels and Logs 2 %. Main's own start (ep 2,031) already answered 57 % and Logged 19 % before any real PPO.
  - The restart has gone 22 → 30 % in 133k episodes. Main's PPO added 18 points in 500k.
  - The restart has the Log in hand at the barrel far more often than main (255 vs 125 times) and uses it 1 % of the time (people 35 %).
- **"Waiting for a full bar" was C5's habit, and PPO has mostly removed it** [measured].
  - Under threat, C5 spends 18.7 % of its decisions at 9.5+ elixir. The restart at 115k–133k spends 5 %, main 2.7 %, people 3.3 %.
  - The restart's remaining gap to main is elsewhere:
    - it plays less often at 4–9.5 elixir under threat (about 0.8× main per decision);
    - it rarely adds a second card while one is in flight (0.07–0.10 per decision vs main 0.20–0.30).
  - The second gap is the policy, not the restart's mask: main run under the restart's mask and afford-lead setup still adds the second card at 0.21–0.30.
- **Recommendation** [inferred]:
  - Keep training for the general hold.
  - Fix the barrel on the imitation/anchor side; see TRAINING_SIDE.md fix C. The run is KL-anchored to C5, which does not Log barrels, until about 250k.
  - Defence-moment starts are a weak lever for the barrel. Barrel exposure is already above people's (0.40 vs 0.28 per match). Raising it 1.7× in evaluation did not change the answer rate (30 %).
  - No reward term: the evidence for one is not strong. The sim already rewards the right Log, and main learned and kept its barrel answer under the same reward.

## Per-checkpoint table

Setup for every row:
- Opponents: teacher, clone and bld, 400 matches each, paired seeds, 1,200 matches per arm.
- Engine: build/gen2_k, CR_P0088 with 14 switches, verified pocket table, act delay 13,15,1, temperature 1.0.
- Restart arms run with the pocket mask and CR_AFFORD_LEAD=1. Main arms run unmasked, as each lineage trains.
- People: 424 recorded Hog 2.6 matches, re-simulated against the recorded opponents.

| arm | episode | win | answered ≤ 5 s | held | held at 7+ elixir | tower dmg / threat (15 s) | offence instead | Goblin Barrel answered / Logged | Royal Ghost answered | Mini PEKKA dmg / threat | Log / match | Fireball / match | threatened decisions at 9.5+ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C5 | 0 | 8 % | 46 % | 24 % | 27 % | 539 | 11 % | 22 % / 2 % | 52 % | 1346 | 2.63 | 1.24 | 18.7 % |
| p050k | 50,501 | 22 % | 58 % | 15 % | 18 % | 428 | 10 % | 29 % / 3 % | 57 % | 1212 | 3.38 | 1.40 | 6.6 % |
| p101k | 101,074 | 29 % | 59 % | 14 % | 19 % | 403 | 10 % | 33 % / 5 % | 53 % | 907 | 3.58 | 1.20 | 5.4 % |
| p115k | 115,223 | 27 % | 59 % | 13 % | 16 % | 407 | 11 % | 28 % / 3 % | 57 % | 900 | 3.42 | 1.26 | 5.3 % |
| **p133k (newest, stage C)** | 133,387 | 32 % | 60 % | 12 % | 16 % | 389 | 11 % | 30 % / 3 % | 55 % | 908 | 3.39 | 1.19 | 4.9 % |
| main002k (main's start) | 2,031 | 12 % | 50 % | 22 % | 24 % | 504 | 9 % | 57 % / 19 % | 48 % | 1435 | 3.15 | 1.58 | 15.7 % |
| main100k | 100,983 | 24 % | 62 % | 11 % | 11 % | 404 | 9 % | 66 % / 22 % | 63 % | 1115 | 4.00 | 1.62 | 5.7 % |
| main199k | 199,998 | 30 % | 63 % | 10 % | 11 % | 373 | 9 % | 70 % / 22 % | 67 % | 878 | 4.30 | 1.43 | 4.0 % |
| main301k | 301,012 | 32 % | 64 % | 9 % | 8 % | 368 | 9 % | 70 % / 24 % | 66 % | 950 | 4.14 | 1.38 | 3.4 % |
| main400k | 400,010 | 38 % | 66 % | 9 % | 11 % | 348 | 9 % | 71 % / 30 % | 65 % | 830 | 4.33 | 1.19 | 3.4 % |
| **main497k (live)** | 496,977 | 41 % | 67 % | 8 % | 7 % | 339 | 8 % | 75 % / 27 % | 68 % | 687 | 4.19 | 1.13 | 2.7 % |
| people | – | 29 %* | 71 % | 7 % | 6 % | 303 | 7 % | 87 % / 42 % | 85 % | 764 | 4.09 | 1.92 | 3.3 % |

\*The people win rate is low because an open-loop re-simulation drifts from the real match. Their per-threat numbers are still the best reference available.

Standard errors (pooled):
- answered: about 0.4–0.6 points;
- held: about 0.3–0.4 points;
- Goblin Barrel answered: about 2 points (n of about 480 per arm);
- Royal Ghost: about 2.5 points.

Wins use paired seeds (same seeds, same opponents), so arm-to-arm differences are tighter than the binomial SE of about 1.3 points.

### Secondary set: recorded-human replays ("ghost" kind, 400 matches)

| arm | win | answered | held | held at 7+ | Goblin Barrel answered |
|---|---|---|---|---|---|
| C5 | 30 % | 51 % | 15 % | 15 % | 25 % |
| p050k | 46 % | 60 % | 12 % | 10 % | 42 % |
| p101k | 54 % | 61 % | 11 % | 10 % | 35 % |
| p115k | 52 % | 61 % | 10 % | 8 % | 47 % (n 58) |
| main497k | 65 % | 65 % | 8 % | 7 % | 73 % |

The ordering is the same as the main set. Barrel n is only about 60 here.

## Definitions

Threat:
- An opponent troop or spell play whose unit reaches our side of the river (y < 17 tiles), classed by card:
  - PEKKA (both forms);
  - tank;
  - Mini PEKKA (incl. Hero);
  - Prince type;
  - Three Musketeers;
  - Royal Ghost (both);
  - Goblin Barrel (both);
  - swarm;
  - Hog/Ram;
  - other.
- T0 is the moment it crosses.

Answered ≤ 5 s:
- One of our plays inside [T0 − 2 s, T0 + 5 s] whose bodies engage that threat's bodies within 6 s of landing, using the engine's target ids.
- A Log or Fireball counts if it hits.
- "Standing": a defender already on the field engages it.

Misplaced and offence:
- "Near" (misplaced): played on our side within 8 tiles but never engaged it. This is about 11–13 % for every policy arm and 10 % for people (wave-1 numbers).
- "Offence instead": our first play in the window lands across the river.

Held:
- The threat arrived with an affordable non-Hog answer in hand (cost ≤ elixir at T0) and we made no play in the window.
- About 95–99 % of threats qualify, because a 1-cost card is almost always affordable, so the table also splits holds by elixir at T0.

P(play):
- Realized share of the policy's decisions (env steps) at which a new play enters the act-delay queue, by elixir bin.
- "Under threat" means an enemy troop is on our side.
- This is a count of what happened, not the policy's own probability (DECISION_AUDIT.md uses logits). The two measures need not agree.

Spell moments:
- A Log or Fireball "moment" is a window where a cast on the spot would hit enough value.
- For policy arms it is scored with a 20-tick lag for the act delay; for people, lag 0.

## Coordinator's three questions, per checkpoint

### 1. Hold rate (affordable answer in hand, no play in the window) [measured]

| arm | held | at T0 elixir < 4 | 4–7 | 7+ |
|---|---|---|---|---|
| C5 | 24 % | 21 % | 21 % | 27 % |
| p050k | 15 % | 16 % | 13 % | 18 % |
| p101k | 14 % | 15 % | 11 % | 19 % |
| p115k | 13 % | 14 % | 11 % | 16 % |
| p133k | 12 % | 14 % | 10 % | 16 % |
| main002k | 22 % | 18 % | 19 % | 24 % |
| main100k | 11 % | 13 % | 11 % | 11 % |
| main497k | 8 % | 9 % | 7 % | 7 % |
| people | 7 % | 8 % | 8 % | 6 % |

- The restart's excess over main is mostly at 7+ elixir: 16 % vs 7 %.
- Against decks rich in the audited cards (see below), its 7+ hold rises to 22 % (main 14 %).

### 2. Chance of a play against elixir [measured]

Realized P(play per decision) with an enemy troop on our side. Pool kinds, about 180k decisions per arm.

| arm | 0–2 | 2–4 | 4–6 | 6–8 | 8–9.5 | 9.5+ |
|---|---|---|---|---|---|---|
| C5 | .058 | .074 | .085 | .090 | .117 | .094 |
| p050k | .068 | .090 | .117 | .134 | .173 | .092 |
| p101k | .072 | .091 | .125 | .145 | .178 | .094 |
| p115k | .065 | .089 | .124 | .142 | .173 | .093 |
| p133k | .063 | .088 | .130 | .143 | .178 | .103 |
| main002k | .034 | .071 | .092 | .116 | .144 | .121 |
| main100k | .033 | .076 | .120 | .161 | .212 | .194 |
| main497k | .039 | .093 | .152 | .195 | .239 | .223 |
| people | .040 | .102 | .129 | .141 | .188 | .304 |

Without a threat, every arm plays most at a full bar, people included: people .050 at 6–8 vs .243 at 9.5+. Waiting for a full bar when nothing is coming is human-like.

Split by whether one of our plays is already in flight (decided, not landed). Teacher opponent, under threat; each cell reads "nothing in flight / one in flight":

| arm (setup) | 4–6 | 6–8 | 8–9.5 | 9.5+ |
|---|---|---|---|---|
| p133k (mask + afford-lead) | .154 / .067 | .179 / .081 | .223 / .098 | .075 / .105 |
| main497k (as trained, unmasked) | .183 / .113 | .220 / .198 | .228 / .274 | .176 / .273 |
| main497k under the restart's mask + afford-lead | .180 / .134 | .196 / .209 | .219 / .262 | .141 / .300 |

The audited-deck opponents give the same picture (pend_aud.md):
- p115k: .08–.09 at 9.5+ with nothing in flight and .07 with one in flight.
- main497k: .09–.17 with nothing in flight and .20–.23 with one in flight.

What this shows:
- **C5 sits at a full bar under threat.** 18.7 % of its threatened decisions are at 9.5+, and its rate there is no higher than at 8–9.5 [measured].
  - PPO removed most of this by 50k: 6.6 %, then 4.9 % at 133k [measured].
  - Main's own start had the same habit (15.7 %). Main's PPO removed it by 100k (5.7 %) [measured].
- **With nothing in flight, at a full bar under threat, every policy is slow.** The restart is at .08–.14, main at .09–.18, people at .30. This is a small share of the time [measured].
- **The restart's gap to main**:
  - **(a)** It plays about 0.8× as often at 4–9.5 elixir under threat [measured].
  - **(b)** It rarely commits a second card while one is in flight. Main does this 2–3× as often [measured].
  - (b) is not caused by the restart's mask: main under that mask still does it [measured].
  - The mask does cost main some performance: wins against the teacher 30 → 18 %, held 7 → 10 %, offence instead 9 → 15 %. Main was not trained with it [measured].
  - So part of the restart-vs-main gap is setup, but the restart still holds more than main under the same setup (13 % vs 10 %) [measured].
- DECISION_AUDIT's 0.41 at 9.5+ is the policy's own probability over its states. My realized under-threat numbers show no full-bar spike for the restart. Both measures agree that the restart plays less than main at the same elixir under threat.

### 3. Toward main or away from C5? [measured]

- Toward main on every defence measure: held 24 → 12 %, answered 46 → 60 %, damage per threat 539 → 389, threatened time at 9.5+ 18.7 → 4.9 %.
- The pace dropped sharply after 50k. From 50k to 133k, held moved −3 points and answered +2.
- The first 9k episodes of stage C (KL loosened at 124k) moved wins 27 → 32 % and held 13 → 12 %. That is too short to call a trend.
- The barrel answer barely moved: 22 → 30 %.

## (a) C5 → 50k → 101k → newest: better or worse, and will it iron out?

Better at every step [measured].

| step | held | answered | dmg / threat | win |
|---|---|---|---|---|
| C5 → 50k | 24 → 15 % | 46 → 58 % | 539 → 428 | 8 → 22 % |
| 50k → 101k | 15 → 14 % | 58 → 59 % | 428 → 403 | 22 → 29 % |
| 101k → 115k | 14 → 13 % | flat | 403 → 407 | 29 → 27 % |
| 115k → 133k | 13 → 12 % | 59 → 60 % | 407 → 389 | 27 → 32 % |

- Main's lineage is the best guide to what comes next. After 100k it kept improving slowly: held 11 → 10 → 9 → 9 → 8 % at 100k / 199k / 301k / 400k / 497k, answered 62 → 67 %, damage per threat 404 → 339, wins 24 → 41 % [measured].
- The restart at 133k is roughly at main's 100k level on these measures. On the same slope it would reach main-497k's defence only after a few hundred thousand more episodes, about 1.5–2 days at 7–9k episodes an hour [inferred].
- Per class:
  - Hog/Ram (74 %), Prince type (68 %), tank (65 %) and Mini PEKKA (74 %) are near main.
  - The Goblin Barrel (30 % vs main 75 %) and the Royal Ghost (55 % vs 68 %) are not closing [measured].

## (b) Is main better or worse at defending?

Better on every defence measure [measured]:
- answered 67 vs 60 %;
- held 8 vs 12 %;
- damage per threat 339 vs 389;
- barrel 75 vs 30 % (Logged 27 vs 3 %);
- Royal Ghost 68 vs 55 %;
- Mini PEKKA damage per threat 687 vs 908;
- offence instead 8 vs 11 %.

Not better on the Fireball: 1.13 vs 1.19 a match, with 3 % vs 2 % of Fireball moments used (people 1.92 and 6 %). Fireball is observe-only.

Against the audited-card decks, main is better still: answered 66 vs 58 %, held 10 vs 16 %, wins 20 vs 14 %.

## (c) Threats the policy under-answers even in the sim

Pool kinds, newest restart vs main vs people; audited-deck runs where n is small [measured]:

| threat | restart (p133k) answered / no play / dmg | main497k | people | likely cause |
|---|---|---|---|---|
| Goblin Barrel | 30 % / 14 % / 945, offence instead 24 % | 75 % / 3 % / 586, 5 % | 87 % / 2 % / 292, 2 % | Imitation start. C5 22 % / Logged 2 %; main's start 57 % / 19 %. With the Log in hand the restart Logs 1 % (people 35 %). Exposure is not short: 0.40 a match vs people 0.28, and 0.67 a match on the audited decks with no change in the answer rate. The sim rewards the Log (TRAINING_SIDE.md §2). The 24 % "offence instead" looks like a racing reply to barrels that C5 already had (21 %) [inferred]. |
| Royal Ghost | 55 % / 22 % / 734 | 68 % / 11 % / 639 | 85 % / 4 % / 365 | Sim observation. A cloaked unit is not drawn on the sim board, so no policy sees the ghost until it hits (TRAINING_SIDE.md, "observation" row). All policies are low here. Main's lead suggests part of it is still learnable [inferred]. |
| Mini PEKKA (incl. Hero) | 74 % / 6 % / 908 | 75 % / 4 % / 687 | 76 % / 4 % / 764 | The answer rate is human-level but the damage is higher: the answers are late or weak. Hero's sim ability is too weak (TRAINING_SIDE), so live is harder still [inferred]. |
| Three Musketeers (audited decks, n 101–125) | p115k 74 % / 18 % / 1483 | 83 % / 5 % / 1251 | 79 % / 12 % / 918 (n 24) | People answer with the Fireball 32 % of the time; the policies hardly ever do. The sim's formation is wrong (in a line; real is a triangle). Rare in real decks [inferred]. |
| PEKKA (audited decks, n 201–250) | p115k 79 % / 8 % / 699 | 80 % / 6 % / 656 | 86 % / 0 % / 94 (n 36) | Answered about as often as main. Damage is high for every policy. People's n is too small to rank [measured]. |

Value or reward racing bias:
- The general "offence instead" rate is only a little above people's (11 % vs 7 %).
- The barrel is the exception (24 %).
- I see no sign of a general racing bias beyond what C5 brought in [inferred].

## (d) Defence-moment starts vs the existing starts

Existing exposure [measured]:
- The existing mid starts are uniform over ticks 200–2,400 (p_midmatch 0.215). In the people panel, such a start lands within 5 s of a threat crossing 30 % of the time.
- For the audited classes (PEKKA, Mini/Hero PEKKA, Three Musketeers, Royal Ghost, Goblin Barrel) that share is only 3.4 %. Whole matches hold 1.13 audited threats each, and about 91 s remain after a mid start.
- The behind starts and live-ghost mid starts (p_mid 0.67) draw from the same kind of uniform window. Only live3 holds real moments, and it has about 44 audited moments in all.
- The v1 human store has hundreds of thousands of such moments, so any moment table should come from v1, with live3 as a top-up.

What moment starts could add [inferred arithmetic]:
- Moving all 21.5 % of mid starts to "0.5 s after an audited play" raises audited-threat decisions by only about 24 %.
- The barrel is already over-exposed relative to people (0.40 vs 0.28 a match).
- Main learned and kept its barrel answer with no moment starts.
- So moment starts are unlikely to move the barrel or the general hold. They could matter for the rare classes (PEKKA, Three Musketeers, about 0.02–0.05 a match), whose sim fidelity is also the weakest.

Cheapest experiment [inferred cost]:
- **Code:** a site hook (new file under runs/live_1127/defence/, no training/ edits) that overrides PoolEnv._draw0's start_tick for mid-start episodes. It places the start 0.5 s after an opponent's audited play, drawn from a table built from the v1 store.
- **Run:** fork the current state and train about 20k episodes.
- **Compare:** against the parent run at the same episode count, with this probe: teacher, clone, bld and the audited decks, about 15 min per arm.
- **Cost:**
  - 1–2 h of engineering.
  - About 2.5–3 h of trainer time. On the Mac that means pausing or sharing with the running trainer; a RunPod 4090 is about $2 of the remaining cap.
- **Expected:** a small effect on the barrel. I would run it only after the barrel's anchor problem is addressed.

## What to change, ranked [inferred]

1. **Barrel answer: imitation/anchor side.** TRAINING_SIDE.md fix C: re-weight human barrel-answer states into a new anchor, or add a targeted barrel-state anchor or auxiliary loss in new files and resume.
   - Evidence: the gap is present at step 0 and survives 133k episodes of PPO.
   - The run's KL anchor is C5, which does not Log barrels, until kl-final reaches 0.05 at 250k.
2. **General hold and second-card commitment: keep training.**
   - Main's lineage shows these improve with PPO (held 22 → 11 % in its first 100k, then slowly to 8 %).
   - Re-check at 175k and 250k with this probe, about 15 min per arm, as stage C's looser KL takes effect.
3. **Royal Ghost: sim observation** (TRAINING_SIDE.md fix B `cloakboard`, or the perception-side alternative). Not a training-method change.
4. **Defence-moment starts and pool re-weighting:** low expected value for these failures. Pool calibration (TRAINING_SIDE.md fix A) is cheap and harmless, but my audited-deck runs show more barrels in front of the policy does not change how it answers them.
5. **No new reward term.** The evidence does not point at the reward.

## Caveats

- **Open-loop people panel.** People are re-simulated against the recorded opponents' plays (open loop), so their wins are not comparable. Their per-threat numbers use the same threat definition.
- **Two P(play) measures.** Mine is realized per decision; DECISION_AUDIT's is the policy probability.
- **In-flight plays.** Decisions with a play in flight are counted in my P(play) tables. The split tables separate them; the effect is visible mainly at 6+ elixir.
- **Different setups.** The restart runs masked with afford-lead; main runs unmasked, as each was trained. Main under the restart's setup is worse (wins against the teacher 30 → 18 %). Part of the gap is setup, but holds and the second card stay worse for the restart under the same setup.
- **Small classes.** PEKKA and Three Musketeers are rare in the pool decks (n 22–55 per arm). The audited-deck runs give n of about 100–250.
- **Main's lineage is not the same recipe.** Main differs in its init (g2_big_L0_H10), engine changes during its run, delays and pool. Its points show what PPO did there, not a controlled ablation.
- **Re-runs are deterministic.** Re-running the same seeds gave identical matches (400 of 400), so the hist re-runs into out_hist/ replace the original files cleanly.

## Files

All under ~/cr-robot/runs/live_1127/defence/:

- **Probe and runners:**
  - defence_probe.py — the probe (pool and human paths, hist and hist2 histograms);
  - run_probe.sh — env per lineage, MASK=0/1;
  - run_all.sh, run_more.sh, run_tail.sh, run_tail2.sh — the run chains, with *.log.
- **Reports:**
  - report.py — the report builder;
  - report_pool.md / report_pool.json — the main set;
  - report_ghost.md — the ghost set;
  - report_aud.md / report_aud.json — the audited decks;
  - report_teacher.md — teacher only, includes main497kM.
- **In-flight split:** pend_table.py, pend_aud.md, pend_teacher.md.
- **Audited decks and exposure:** decks_audited.json (1,939 meta decks holding the audited cards, equal weight per class) and live_moments.py (exposure counts).
- **Raw outputs:**
  - out/*.json — one file per arm and kind;
  - out_hist/ — re-runs with the histogram;
  - out_pend/main497k_teacher.json.
- **Stale:** report_wave1.md / .json — superseded by report_pool.md.
