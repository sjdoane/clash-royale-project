# Overnight report, 2026-11-18

_Draft, written 04:40 balerion time. The PPO and detector sections are refreshed at the end._

## TL;DR

- **Data collection is done.**
  - Imitation dataset: 8,552 real Hog 2.6 matches (1.99M decision rows, 442K plays), replayed through the simulator, plus a 5,753-match secondary set.
  - Opponent pool: 4,221 real decks.
  - Video perception (hand, elixir, clock, towers, units with tile positions) has finished on all 19 supported videos. That gives 11,937 usable plays of ours, 8,407 of them from good-quality videos.
- **Training is running.** BC → PPO on balerion's 5090, PID 3434021.
  - Eval win rate: 8.9% (BC alone) → 47% at episode 41k → **about 59–61%** at episodes 124k–128k.
  - Against real decks: 5% → 49% → **about 62–64%**.
  - The early "plateau" call was wrong. It is still climbing slowly, at roughly +1 point per 10k episodes; see §3.
  - All metrics stream live to W&B: https://wandb.ai/clash-royale/clash-royale-ai.
  - Elixir leaked per match: 5.3 → 0.2.
- **Not done:**
  - Phone testing (nothing was played on the phone overnight).
  - Engine patches for the gaps found. They are reported only, per the engine rules.
- **Nothing is committed.**

## 1. Imitation dataset (`data/bc/hog26/REPORT.md` has full detail)

- **Source.** IL_Replay (RoyaleAPI replays). Each match carries both decks with their Evolution/Hero forms and levels, and every play with card, tile and tick.
  - Each match is re-simulated in the engine with forced hands and elixir top-ups, so every row has a full 13,977-wide engine observation.
- **Size.**
  - Primary: 8,552 matches, 1,986,624 rows, 441,845 plays, 8,487 Hero Musketeer ability steps.
  - Secondary: 5,753 matches.
  - Validation: 424 held-out matches, split by replay tag.
- **Fidelity.**
  - The simulated winner agrees with the real result in 92% of matches.
  - 98% of plays are legal in the simulation; the other 2% were moved to the nearest legal cell and flagged.
  - The simulator destroys about 3x fewer towers than the real games did.
- **Design: how it lets PPO learn beyond the data without forgetting it.**
  - Lost matches are kept at weight 0.5, chosen on validation.
  - The no-op is reweighted, so the policy samples timing at the human rate.
  - PPO starts from BC with critic warmup, then a KL-to-BC term plus a BC aux loss, both decaying 0.1 → 0.01 over 30k episodes. Early on it stays human-like; later it is free to improve.
- **Engine gaps found, not patched:**
  - Triple elixir starts at 3:00; the real game starts it at 4:00.
  - No pocket placement after a Princess Tower falls.
  - Seven Hero and Evolution forms are missing.
  - A 20-tick hand cooldown the real game doesn't have.
  - Spell crown-tower damage is 100%.

## 2. Video perception (full game state from YouTube 2.6 gameplay)

Pipeline: `video_dataset/tray.py` (our hand/plays) + `video_dataset/detector/` (HUD, units, towers) + `postproc.py`.

Held-out video uVf1EUszTIM, 30 hand-labelled frames:

| field | accuracy |
|---|---|
| hand slots (card identity incl. evo/hero) | **105/108 = 97.2%**, from 82.4% before the fix; Hog 20/20 |
| next card | 22/22 |
| elixir | 30/30 |
| hero button, overtime | 30/30 each |
| elixir multiplier | 29/30 |
| clock | 27/30 (misses are the red final seconds) |
| tower HP | 170/180 = 94.4%; destroyed towers 53/53 |
| our play tile vs truth | median 0.46 tiles, p90 1.95 |

**Unit detector** on the same frames, measured as KataCR (the off-the-shelf detector) → ours v0 → ours v2:

| metric | KataCR | v0 | v2 |
|---|---|---|---|
| detection precision (side and box) | 0.86 | 0.96 | 0.91 |
| detection recall | 0.68 | 0.68 | **0.82** |
| class accuracy of detected units | 0.82 | 0.93 | 0.92 |
| strict recall (class and side) | 0.61 | 0.69 | **0.78** |

**Known limits:**
- A pending-placement "ghost" gets read as a Hog; a dedicated pending class is needed.
- Our Hero Musketeer is recognised 7/15.
- A few persistent digit confusions on tower HP.
- 57 of the 191 unit classes have no training boxes yet, including most Heroes and Evolutions. Units of these classes are detected by side only; the card isn't identified.

**Full run, done over 19 videos** (about 19 hours of gameplay, 352 matches):
- **Our plays:** 11,937 usable in total. Usable means confident and consistent with the card rotation. The 8,407 from the 13 good-quality videos form the video BC label set.
- **Low quality:** 6 videos fall under the 80% usable bar and are kept only for perception validation. They are 62x30, 7p8l, IaJ4, MoMO, SAag and fiNg (79.6%).
- **Per video:**
  - 2.8–3.6 units tracked per frame.
  - Clock read on 90–92% of frames.
  - Princess Tower state known on 94–100% of frames (IaJ4 67%).
  - The King tower HP read is low by design, because it shows no bar until it takes damage.
  - Only 0.2–0.6% of our-side boxes fall outside our deck.
- **Skipped:** 14 other-channel videos, because their screen layout isn't supported.
- **Files:**
  - Per-video table: `data/det/state/SUMMARY.md`.
  - Detector write-up, including the 191-class coverage: `video_dataset/detector/REPORT.md`.
  - Per-video data: `data/det/state/<vid>/*_pp.parquet`.

## 3. Training (balerion, `~/cr-robot-remote/train/`)

**Setup.**
- BC: `checkpoints/bc_hog26.pth`.
  - Held-out card match given the human played: 0.519.
  - Exact cell 0.224, against 0.112 for the per-card modal cell.
- PPO run `ppo_bc26`:
  - 16 environments; opponents are the teacher curriculum plus the real-deck pool, sampled by PFSP.
  - Reward: ±1 for the result, zero-sum tower PBRS and a leak penalty. Nothing else, so the reward can't be gamed through shaping.

**Eval panel.** 192 sampled matches per eval, 12 per opponent:

| episodes | overall | rungs | real decks | leak/match |
|---|---|---|---|---|
| BC only | 8.9% | 15% | 5% | 5.3 |
| 0 | 9.9% | 11% | 9% | 5.0 |
| 4,340 | 25.5% | 25% | 26% | 0.4 |
| 8,530 | 30.2% | 32% | 29% | 0.1 |
| 14,837 | 34.9% | 28% | 39% | 0.1 |
| 16,911 | 38.0% | 35% | 40% | 0.2 |
| 19,208 | 37.0% | 36% | 38% | 0.2 |
| 21,532 | 42.7% | 32% | 49% | 0.1 |
| 25,761 | 44.3% | 33% | 51% | 0.05 |
| 27,859 | **46.4%** | 40% | **50%** | 0.05 |

| 30,117 | 36.5% | 33% | 38% | 0.0 |
| 32,300 | 42.7% | 42% | 43% | 0.03 |
| 36,623 | 45.3% | 43% | 47% | 0.06 |
| 38,942 | 40.1% | 32% | 45% | 0.07 |
| 41,088 | 46.9% | 43% | 49% | 0.04 |
| 49,949 | 49.5% | 43% | 53% | 0.09 |
| 63,807 | 53.6% | 47% | 57% | 0.05 |
| 75,064 | 59.9% | 53% | 64% | 0.01 |
| 88,469 | 54.7% | 43% | 62% | 0.07 |
| 106,409 | 57.3% | 53% | 60% | 0.02 |
| 117,483 | 58.9% | 47% | 66% | 0.01 |
| 124,094 | 59.4% | 54% | 62% | 0.01 |
| 126,242 | **61.5%** | 57% | 64% | 0.06 |
| 128,393 | 58.9% | 50% | 64% | 0.04 |

_Correction:_ an earlier draft said it had plateaued at about episode 25k.
- It was a slow climb, not a plateau: the trailing 5-eval mean went from about 46% at 41k to about 59% at 128k.
- Rungs 6–10 (the hard teachers) are still the weak spot, at 0–42% on 12 games each.
- At episode 130,773 the curriculum is at stage 5, with KL to BC 0.85, explained variance 0.82 and entropies 0.43 (card) and 0.55 (place).

Earlier notes from the 41k draft:
- Eval noise is about ±3.5 points, because each eval is only 192 matches.
- Rungs 0 and 2 are at 90%+.
- Rung 4 is at about 50%.
- Rungs 6–10 are still at 0–25%.
- The next lever is frozen-snapshot self-play and/or longer rung-6 exposure; more of the same is unlikely to move it.

At episode 42,215:
- Explained variance is 0.86.
- KL to BC is 0.75, with the coefficient held at its 0.01 floor.
- Entropies are 0.47 for card and 0.59 for place.

At episode 29,601 the KL/BC coefficients reached their 0.01 floor.
- KL to BC is 0.71.
- Explained variance is 0.75.
- Entropies are 0.52 for card and 0.60 for place, so the policy has not collapsed.
- The curriculum is at stage 2.

**Health at episode 19,520.**
- Explained variance 0.83.
- KL to BC 0.43, with the coefficient down to 0.022. It is drifting from the human policy as designed.
- Card entropy 0.54, place entropy 0.62; no collapse.
- Curriculum stage 1.

**Weak spot.** Teacher rungs 6–10 are still at about 0–8%; low rungs are at 60–90%.

**Commands.**
- Monitor, stop, resume and fetch weights: see `training/STATUS.md`.
- A copy as of 05:38 is in `remote_results/`: `bc_hog26.pth`, `ppo_bc26.pth` at episode about 42k, and the run logs. It is not gitignored, so don't `git add` it. To refresh it:
  `rsync -a -e "ssh -o ControlPath=$HOME/.ssh/cm-balerion" <user>@<gpu-host>:cr-robot-remote/train/{checkpoints,runs} remote_results/`

**Weights & Biases.** https://wandb.ai/clash-royale/clash-royale-ai. The Forge UI shows the same project.
- The sidecar `training/wandb_sidecar.py` runs on balerion and has backfilled every run.
- What it logs:
  - `train/*`: PPO losses, KL and BC terms, entropies, explained variance.
  - `opp/*`: win rate and game count per PFSP opponent.
  - `eval/*` and `eval_opp/<opponent>/*`: the eval panel per opponent (win, draw, loss, plays, leak, damage).
  - `tb/*`: all 105 TensorBoard scalars, including placement entropy per card, aux next-card accuracy and the curriculum.
  - `bc/*`: BC validation per epoch.
  - `balerion-system`: GPU utilisation, memory, power and temperature, plus load.
- New runs (the league arm, net_v2 BC) appear automatically once they write to `runs/<run>/`.

## 4. Suggested next steps

1. Let PPO finish the KL decay (~30k episodes), then add frozen-snapshot self-play opponents.
2. Train on the video-derived plays as a second BC source. YersonCz videos only (the SoTired ones are low quality).
3. Run a phone test in Training Camp through iPhone Mirroring (`play_live.py`).
4. Decide on the engine patches above. Fixing the elixir schedule matters most for the late game.
