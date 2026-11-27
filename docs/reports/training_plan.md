# Training plan: Hog 2.6 (evo/hero meta) from top-player video, then RL

Status 2026-11-18. Owner: Sam. Nothing here is built yet unless marked done.

## 0. Why change course

The ep-76728 checkpoint played oddly live: Musketeer in the back corner at 0:04,
The Log rolled from our own back row. Two causes are open and both are
addressed below:

- **The policy:** trained on a reward of about a dozen shaped terms
  (`python_ai/rewards/weights.py`), several of them deliberately biasing.
- **The observation gap:** the simulator gives the policy a perfect state; live,
  the state is rebuilt from screenshots, and one coordinate step is still
  "derived, not measured" (`perception/live/adapter.py`).

## 1. The deck (current meta, all cards at level 11)

| Slot | Card | Engine id |
|---|---|---|
| 0 Evolution | **Evo Skeletons** | 125 |
| 1 Hero | **Hero Musketeer** | 168 |
| 2 Wild | **Evo Cannon** | 129 |
| 3-7 | Hog Rider, Ice Golem, Ice Spirit, The Log, Fireball | 15, 40, 72, 33, 7 |

- Average 2.625 elixir. The cheapest 4-card cycle costs 6.
- This is the most-played Hog/Hero Musketeer list: 58-59% over about 13-29K ranked battles (RoyaleTracker,
  Timesaver 2026-10-22).
- The runner-up variant is Evo Musketeer + Evo Skeletons + Hero Ice Golem, at 57.5% over
  7.9K battles. The engine has every card for it, so it stays an option.
- The engine already accepts the deck:
  `CLASH_DECK="evo:skeletons,hero musketeer,evo:cannon,hog rider,ice golem,ice spirit,the log,fireball"`.
- Every card in the engine is level 11, the tournament standard, the same for
  everyone, as in a Classic Challenge.

**Engine fixes needed before training.** Each is a patch in `patches/`, and each number needs a
source before it goes in:

| Card | Engine now | Current game (to verify) |
|---|---|---|
| Hero Musketeer turret | 200 HP, 90 dmg, at her feet, ability on a 22 s cooldown | ~1536 HP, 204 splash on landing then 140 per 0.4 s, range ~4, 10 s; lands ~3 tiles ahead; **once per deployment** (since 2026-08-04) |
| Evo Cannon barrage | reuses the Cannon hit (202), no knockback | ~261 (Oct 2026, unofficial), 9 projectiles, 2.5 tiles, knockback |
| Evo Skeletons | cap 8, 2 cycles | cap 6 or 8 and cycles 1-3 disputed |
| Ice Spirit | as before | HP 230 → 215; no longer connects with towers alone (Aug 2026, official) |
| Skeletons | as before | spawn radius reportedly shrunk (Oct 2026) |

- **Verifying the numbers:** read the in-game card info screens on Sam's phone
  through iPhone Mirroring, then scale them back to level 11 by the ratio of the
  base card's HP. For example, the shown Musketeer HP over 721 gives the scale
  factor for the turret.
- **Opponents:** update the meta deck pool
  (`python_ai/opponents/decks/meta_decks.json`) to current decks with their
  evos and heroes. Otherwise the policy trains against last year's game.

## 2. One observation, everywhere

The policy's input is defined as **what perception can read from a screen**:

- units (card, side, tile, HP bucket), tower HP, our hand and next card,
  our elixir, our evo counters, hero-ability availability, clock and overtime.

The same spec is filled from three sources:

- **the simulator:** from the true state, with noise matched to the measured
  perception errors (missed units, tile jitter, misreads, elixir off by one);
- **videos:** from the detector on recorded frames;
- **the live game:** from the detector on the phone.

The **critic** may see privileged simulator state (opponent elixir, hand and
cycle); the **actor** never does (asymmetric actor-critic). This closes the
observation gap by construction, and it is what lets video data train the same
network.

Before anything trains, two checks:

- audit the existing observation against this spec;
- measure the detector-to-engine tile mapping on frames with known placements.

## 3. Data: top-player video → (state, action) trajectories

**Sources.** Raw portrait phone recordings of full ladder sessions, uploaded
Sep-Nov 2026, the player's own POV, no edits:

| Channel | What | Volume |
|---|---|---|
| YersonCz | top-10 world, 2.6 only, near-daily | 40-60 min, ~13-17 matches each; ~11 videos in Oct-Nov, ~30 more back from mid-Oct |
| SoTired | top-50 | 6 videos, 16-73 min |
| OYASSUU | 3000+ | 3 videos, ~50 min |
| Extra Ian77 | uncut livestream recordings | 2 recordings, 2-3 h each (check stream overlays) |

- Total: about 300-400 matches, or roughly 12-16K of our player's own placements.
- **Mohamed Light and Ian77's main channels are a poor fit:** edited 16:9
  commentary videos. Mohamed Light's long sessions are on Twitch. He appears as
  the *opponent* in four YersonCz and SoTired videos.
- Full candidate list: section 7.

**Pipeline (`tools/video_dataset/`, new):**

1. **Download** with yt-dlp, at native resolution, into `data/video/raw/`. These files stay
   local and are never committed or redistributed. YouTube's terms restrict
   downloading; keep the use to private research. Asking the creators is cheap
   and worth doing.
2. **Calibrate** each video, reusing `tools/calibrate_ios.py`'s two-band fit:
   it already maps any portrait screen onto the 549x976 frame.
3. **Segment** the video into matches with the detector's screen classifier,
   keeping in-battle frames only, sampled at 10 fps.
4. **Perceive** each frame with the same detector stack as live play, giving per-frame state.
5. **Label our player's actions.** A card leaves the hand and the queue shifts,
   which gives the card and the time. The tile is where the new friendly unit or spell
   appears within its deploy window. A hero ability is labelled when the turret appears.
   Low-confidence events are dropped.
6. **Label waiting too.** Every decision tick without a play becomes an explicit
   "wait" example. *When* to play is learned from the humans, not hand-coded.
7. **Label opponent actions** from new enemy units, for an opponent model and
   as data on which counters top players face.
8. **Quality check.** Hand-label 3 matches, measure label precision and recall
   and tile error, then render overlays of a random sample to check by eye.

## 4. Training

**Stage A, behaviour cloning.** Train on the video trajectories plus the
existing mirror-teacher data, weighted toward video. Hold out whole videos for validation.

**Stage B, PPO from the BC weights.**

- **KL penalty:** keep the policy close to the BC policy at first, then decay the
  penalty so it can leave the human prior where self-play finds better play.
- **Opponent league:** self-play snapshots, the BC policy, current meta decks,
  and exploiters (`trainers/league.py`, `trainers/exploiter.py` exist).
- **Scripted precision opponents,** for the "bot-only tech" idea:
  - a frame-perfect 4-3 Cannon placed as the Hog hits the river;
  - perfect card counting, with the Hog sent exactly when the counter is out of cycle;
  - Skeletons and Ice Spirit kiting.
- The research found **no documented case** of bots doing superhuman techs: public
  bots are weak at placement and elixir management. These opponents are a
  hypothesis that is cheap to test, not something to copy from footage.

**No hand-written play heuristics in the policy.** Only legality masks. Drop the
live "tactical advisor": its decisions should be learned. It may stay as an
optional early-training teacher.

**Evaluation.**

- **Fixed panel:** the built-in teacher, the meta-deck pool, the BC policy,
  the previous best, and the scripted precision opponents.
- **Report:** Elo plus per-matchup win rate.
- **Live check:** Training Camp on the phone at milestones.

## 5. Reward: minimal, three terms, chosen by experiment

Start from the smallest reward and add a term only if an ablation shows it
helps at a fixed compute budget.

1. **Outcome.** +1 win, −1 loss, 0 draw.
2. **Tower potential, potential-based** (Ng, Harada & Russell 1999):
   `F = γΦ(s') − Φ(s)`, where Φ is our tower HP fraction minus theirs.
   - Potential-based shaping provably leaves the optimal policy unchanged, so it
     cannot be farmed; it only makes the outcome easier to learn.
   - **The exponential variant goes here:** Φ uses a convex transform of each
     tower's damage, `(e^{k·d} − 1)/(e^k − 1)`. Finishing a hurt tower is then
     worth more than chipping a full one. It stays potential-based, so it is
     still unexploitable. Sweep k ∈ {0, 1, 3}, where 0 is linear.
3. **Elixir leak.** −c per elixir lost while sitting at the 10 cap.
   - It is the only elixir term, and it measures a real loss, not a proxy.
   - It cannot be gamed: the only way to avoid it is to spend, and bad spending
     costs towers, which the outcome and tower terms already price.
   - Size c so a match's worst-case leak is at most about 0.2 of a win.

**Ablation order:** outcome only → +tower → +leak → exponential tower.

**Deliberately absent:** elixir-trade bonuses, troop HP, win-condition damage,
flawless-defence and spell-value terms. Each of these can be exploited, which
is why weights.py needed fixes on top of them.

**Optional elixir-advantage potential,** if leak alone under-teaches trading:
`Φ_e = α·(our elixir − their elixir)` using the critic's privileged state.
Being potential-based, it also cannot be exploited.

## 6. Order of work

1. Engine fixes for the deck and opponent decks (§1), verified on the phone.
2. Observation spec, sim noise model, and tile-mapping measurement (§2).
3. Video pipeline on 2 videos end to end, with the QC numbers (§3).
4. If the labels are good: all videos, then BC (§4A).
5. Reward ablations, short runs (§5).
6. Long PPO + league run (§4B), then live Training Camp tests.
7. Live side: evo and hero card templates for the hand classifier; tap the hero
   ability button; capture that survives the Mirroring window losing focus.

## 7. Candidate videos (from the 2026-11-18 search; overlays not yet checked)

YersonCz:
4s1o1sk-hPM, QD_1suWFNKo, EiUz9NwunVo, h37LHPnfZX4, uVf1EUszTIM, iVgmnDX-GPY,
kn4StFVuJeg, OJ_a2wtSFEA, LcbpMXMyABw, iZvLzDxRUCg, hSdq46S1Kqw, IknQJSoWDuA
(vs Mohamed Light), N72Rk5OG-2Q (vs Mohamed Light).

SoTired:
7p8lHdf_siI, Y5COPSGStTA, IaJ4covgmoU, 62x30JG1b44, fiNgZ2MgKpk, icixBi0JTmE.

OYASSUU:
E0VuP3MUVew, 78qs5-_Mnug, IZrTA60YB7I.

Extra Ian77 livestreams:
2M7uZCmiiiE, 3ujIGnViodY.

Ian77 and Mohamed Light edited videos (landscape, lower value):
3o0uSXkbwwI, ofdf32-T218, zGSBVtDDgzo, G7cTO29IaQA, f6hPzgsYbw8, hS21wINxWxc,
HXRrQFkmWy0.

Upload feeds for pulling more:
`https://www.youtube.com/feeds/videos.xml?channel_id=UCPKaY_SMlcT2cTyscSgpofA`
(YersonCz) and `...UCee7RQgrnwbVVLF5EFXcGSg` (SoTired).
