# Goblin Barrel landing predictor: verdict (2026-11-27)

Written by the coordinator from the barrel agent's final message (the agent was not allowed to write this file).
All numbers come from `summary.txt` in this folder. Scripts: measure_live.py, decisions.py, log_window_sweep.py, summarize.py.

## Verdict: do not build the predictor

On a plain barrel a mid-air predictor would gain at most about 0.1 s. In live play no barrel's tile reached the bot too late. Tower HP is lost for two other reasons:
- the policy plays the Log late;
- the sim gives the evolved barrel a landing tile that live play can never have.

## Data

The live reader was replayed on all 11 recorded sessions:
- 104 barrel tracks in all;
- 60 followed from the launch: 51 plain, 9 evo;
- 40 plain barrels landed cleanly on our side.

Truth is where the goblins appear. Times are seconds after the first sighting, given as p10 / median / p90.

## Current live estimate, plain barrel

**How early it arrives [measured]**
- The estimate is known at 0.20 / 0.28 / 0.46 s.
- Impact comes at 2.81 s, so about 2.55 s of flight remain.

**First decision that shows the tile [measured, n 32]**
- 0.24 / 0.43 / 0.73 s, including 0.11 s of waiting for the next decision.

**Lower detector threshold [measured]**
- No gain: the first sighting at any confidence is 0.00 s earlier at the median.

**Accuracy [measured]**

| Measure | Result |
|---|---|
| Side right | 40 of 40 |
| Tile error | 0.64 / 1.00 / 1.83 tiles |
| Impact-time error (median) | 0.05 s |
| y bias | −0.52 tiles, partly from using the goblins as truth [inferred] |

## Log deadline (engine sweep on build/gen2_k)

- With no Log the tower loses 875 HP [measured].
- For a full save the Log must be cast [measured]:
  - 16–24 ticks after the throw from (3,2);
  - 20–28 ticks after the throw from (3,4).
- With the 14-tick action delay, the decision must come by +1.0 s from (3,2), or by +1.4 s from (3,4) [inferred].
- The live tile arrives 0.57 s before the first deadline (p10 0.27 s), and 0.97 s before the second. None of the 32 was late [measured].

## What the policy did [measured]

- The Log was in hand by +1.0 s for 27 of 40 barrels.
- For only 3 of 40 was a Log decided in the window behind the right tower.
- Most Logs were decided at +1.5 to +2.9 s, after the landing, which costs 250–500 HP.
- This matches the sim: the policy Logs within 3 s on 18 % of barrels, against 59 % for people (TRAINING_SIDE.md).

## The new predictor [measured]

| Sightings used | Time | Side right | Tile error | Time error |
|---|---|---|---|---|
| First 3 | 0.20 s | 40 of 40 | 1.02 | 0.07 s |
| First 2 | 0.10 s | wrong side once | 2.47 | – |

- Three sightings are about 0.08 s earlier than now and no more accurate.
- The barrel moves sideways about 0.19 tiles per frame, which is about the box noise. So the current 3-sighting rule is already at the physics limit [inferred].

## Sim against live

**Plain barrel.** The sim shows the tile at about 0.33 s (median, read from the `training/obs_noise.py` code). Live already matches it at 0.28 s [measured live]. There is no gap to close.

**Evo barrel: the real gap.**
- Live gets a tile for 0 of 9 throws [measured].
- The sim names card 132 "Goblin Barrel", so it gives the evo a tile 93 % of the time [measured].
- By eye (frame strip f010151–f010179 of session 20261124-094533, plus two other throws) [measured]:
  - one fused sprite splits into two barrels at mirrored targets;
  - goblins land at both of our Princess Towers;
  - the detector holds them as one box until about +1.1–1.45 s, which is already the deadline.
- No predictor can tell the side in time [inferred].

## How often [measured]

- 14 of 76 live matches had an enemy barrel.
- Tonight it was 1 of 11: 3 plain throws, 1 evo.
- About 15 of the 60 tracked throws were evo (an approximate count by eye).

## Other spells [inferred]

An earlier landing estimate would not help any of them:

| Spell | Why not |
|---|---|
| Fireball, Giant Snowball | Already fitted; the deck has no answer to them |
| Rocket | Can't be countered or dodged |
| Arrows | Flies about 1.0 s, under the 1.4 s action delay |
| Poison, Earthquake | Not thrown; they appear where they land |

## Recommendations

1. **Training side.** Make the sim's evo barrel match live, by one of:
   - setting the evo (card 132) estimate to always fail in READER2;
   - building the planned evo-barrel decoy scenario.
2. **Policy and imitation data.** Teach the Log in the window behind the target tower:
   - decided +0.2 to +1.0 s after the throw from (3,2);
   - decided +0.6 to +1.4 s from (3,4).
3. **Don't build:**
   - an early-frame predictor;
   - a post-split evo estimate. It would also mean editing `opp_spells.py`, which the perception agent owns.
