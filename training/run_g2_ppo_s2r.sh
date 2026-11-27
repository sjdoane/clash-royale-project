#!/bin/zsh
# Start or resume the gen-2 PPO run g2_ppo_s2r (resumes from runs/g2_ppo_s2r/state.pth when it exists).
# Stop it with SIGTERM: it finishes the update and saves. Log is appended.
#
# 2026-11-23 01:4x PST, from episode ~40,400 on: --anchor-free-elixir 9.5 (no WHEN reference at a
# full elixir bar). State before the change: runs/g2_ppo_s2r/state_ep*_pre_freeelixir.pth.
# To go back: stop, copy that file over state.pth, drop the flag below, start again.
#
# 2026-11-23 02:50 PST, from episode 51,732 on: the policy is shown what live shows it in two more ways.
#   --board-hp-live: every troop at full health (live never reads troop health). Measured cost of hiding it
#     on ep0040380: -0.8 +- 1.4 points under training noise.
#   --sim2real-elixir-est --sim2real-elixir-est-share 0.8: the fractional elixir estimate the live driver can
#     now feed (--elixir-est digit) on 80 % of integer-digit episodes, the plain integer on the rest.
# State before this change: runs/g2_ppo_s2r/state_ep*_pre_hplive_elest.pth. To go back: stop, copy it over
# state.pth, drop the three flags, start again.
#
# 2026-11-23 03:52 PST, from episode 62,365 on: --tail-preset reader2. The opponent tail is now noised as the
# live reader of 2026-11-22 fills it (landing estimates for a thrown Goblin Barrel / Fireball / Giant Snowball
# in the opponent-spell block, which training had always left empty; fewer missed follow-up plays).
# Measured on ep0050513 before the change: 54.7 with this tail vs 53.7 without (+1.0 +- 1.3), so no harm.
# State before this change: runs/g2_ppo_s2r/state_ep0062365_pre_reader2.pth. To go back: stop, copy it over
# state.pth, drop the flag, start again.
#
# 2026-11-23 06:31 PST, from episode 92,936 on: --kl-end 0.3 -> 0.15 and --kl-max 0.06 -> 0.08 (cloning loss
# unchanged at 0.2). Held-out win rate was flat from 60k to 90k (true observations 59.8, 60.4, 60.5, 59.9;
# live 56.9, 56.1, 56.3, 54.1) and the total KL to the clone had sat at 0.040-0.041 since 50k with 1 % of
# samples clipped: the policy was resting where the anchor balances the reward. Watch the guard lines
# (plays per minute, card shares; Fireball share ratio was 0.73 before) and the 100k / 110k evaluations.
# State before this change: runs/g2_ppo_s2r/state_ep0092936_pre_kl015.pth. To go back: stop, copy it over
# state.pth, restore 0.3 and 0.06, start again.
#
# 2026-11-23 09:38 PST, from episode 123,998 on (after Sam's second live session with ep0121202, six matches,
# his note: Cannon placed slightly late / slightly off, spells weak, "make sure training allows further
# improvement and doesn't lock it in this exact state"). Three changes:
#   act delay 3,6,1 -> 4,7,1. Measured in runs/live/20261123-091124: frame captured -> decision 0.10 s,
#     decision -> card visible on the board 0.45-0.54 s by card, so observed state -> card on board is about
#     0.56 s. Training modelled 0.3-0.6 s (mean 0.45); it now models 0.4-0.7 s (mean 0.55).
#   KL anchor, second stage (new option --kl-final in training/ppo_pool.py): the coefficient goes on down
#     from 0.15 to 0.05 (log scale) between 124k and 250k instead of staying at 0.15, and the ceiling
#     --kl-max goes 0.08 -> 0.12 so it does not cancel that. After 06:31 the total KL had settled
#     again (0.050-0.055) with held-out results flat on the live conditions (100k/110k/120k: 59.3/58.5/58.2).
#     Cloning loss stays at 0.2 (it is what holds card and spell usage near the human data).
#   Episode cap 200,000 -> 400,000.
# Watch: guard lines, Fireball share ratio (0.66-0.83 before), plays per minute (13.5-14.2 before).
# State before this change: runs/g2_ppo_s2r/state_ep0123834_pre_klfinal_delay47.pth (about 100 episodes
# earlier). To go back: stop, copy it over state.pth, drop --act-delay and the three --kl-final flags,
# restore --kl-max 0.08, start again.
#
# 2026-11-23 11:05 PST, from episode 134,423 on (after Sam's third live session with ep0121202, five matches;
# his note on the Minion Giant match: the Cannons kept not pulling it, about one tile off).
#   Reactive opponents now take decks in use today: "meta_share" 0.75 for clone and teacher episodes from
#   tick 0 (new option in training/pool_env.py; decks of data/decks/meta_api_pool.json drawn by count, level 11).
#   Until now every ghost, clone and teacher deck came from the recordings, which stop before Minion Giant
#   existed; it is in 8.2 % of the decks of the official-API pool of 2026-11-18 (and Knight 22.9 % there vs
#   9.2 % in training, Ice Wizard 15.7 vs 3.9, Furnace 15.1 vs 6.8). Measured mix after the change: 22.6 % of
#   all episodes use a meta deck (clone 15.0, teacher 7.5), 1.9 % of all episodes face a Minion Giant deck.
#   No card or archetype is weighted: decks are drawn at their ladder frequency.
#   The trainer's by_ckpt table gets rows "clone:<name>+meta" and "teacher:meta" for these episodes.
# Simulator check behind it: a Cannon at tile (8,9) does not pull a Minion Giant coming down the edge lane,
# (7,9) does (evals/cfg/minion_giant_cannon_grid_left.json); the policy puts 0.6-0.7 of its mass on column 8.
# State before this change: runs/g2_ppo_s2r/state_ep0134423_pre_metadecks.pth. To go back: stop, copy it over
# state.pth, remove the two "meta_share" entries from --pool-override, start again.
#
# 2026-11-23 12:12 PST, from episode 141,477 on (Sam: use the recorded live matches as
# replay opponents too, and weight them deliberately).
#   Our own live matches are a second, small ghost store: data/ghost/live1 (training/live_ghost_build.py from
#   training/live_sessions.json; 19 matches of four sessions: 14 for training, the newest session's 5 held out).
#   "live": share 0.03 of ALL episodes, taken out of the ghost kind (dataset ghosts 0.35 -> 0.32); the
#   opponent is always the human side; every match has the same weight.
#   Why 3 %: 0.2 % of all episodes per match (cap 10 % once there are 50+ matches). One match is then replayed
#   about 500 times a day (a dataset match less than once), which is the limit set by memorising 14 scripts,
#   not by their worth; the check is the held-out session (CR_GHOST_STORE=data/ghost/live1 eval_ghost --split
#   val): training matches rising while it stays flat = memorising, and the share goes down.
#   Why two thirds mid-match ("p_mid" 0.67; the pool's own chance is 0.43): from the first tick the policy
#   already beats these replays (ep0131291, deployed setting: 92.9 % on the 14, 92.5 % on the held-out ones
#   inside the spend range, against 0 wins in 21 real matches). The reader misses about a fifth of the
#   opponent's elixir, and a script stops fitting the match once our play differs. A mid-match start replays
#   both real sides to the hand-over, so what follows is still the opponent's answer to that position, and the
#   position is one our policy really got into. "towers": the six towers are then set to the measured health
#   (error at the hand-over 0.107 -> 0.011 of full health).
#   Left out: matches whose kept opponent elixir / available is outside 0.6-1.1 ("spend"; the held-out
#   Golem match at 1.27 has invented plays) and the two recordings that begin mid-match (app left).
#   No weighting by card, deck or result. by_ckpt rows: "ghost:live", "ghost:live+mid".
# State before this change: runs/g2_ppo_s2r/state_ep0141477_pre_liveghosts.pth. To go back: stop, copy it over
# state.pth, remove the "live" entry from --pool-override, start again.
#
# 2026-11-23, from episode 152,400: the live store is rebuilt from the offline re-read of every recorded
#   match (live_ppo/reread.py -> training/live_ghost_build.py), same path, same tags, same share.
#   What changed in the replays: (1) the opponent's deck is the true one with its forms (official battle
#   log) instead of the eight most-read cards plus fillers; (2) plays are the re-read's (37 of 37 and 19 of
#   20 named plays right on two matches counted by hand, no extra play; the live ledger kept 32 of 39 on
#   the first); (3) opponent rows were 2-3 rows too near the river behind their bridge row (reader rows
#   are the 32 visual rows and went through an engine-row conversion they did not need); (4) the cast tile
#   of their Log / Barbarian Barrel is one roll length behind where the roll ended, not the first sight of
#   the thrown sprite (0.5-5 rows further back). Plays the replay had to move to a legal tile: 161 -> 50.
#   Both sides replayed against reality (evals/tools/live_ghost_check.py), share of tower health lost:
#   ours 0.47 -> 0.51 (real 0.67), theirs 0.50 -> 0.45 (real 0.17). Still easier than the real opponents:
#   Lightning is never read, hero ability taps are not replayed, and our own attack does more in the
#   simulator than it did (a match with every play right: 0.65 dealt against 0.22 real).
#   Not drawn: the held-out 52 s recording (spend 0.57). Old store: data/ghost/live1_ledger_20261123.
# State before this change: runs/g2_ppo_s2r/state_ep0152400_pre_reread.pth. To go back: stop, copy it over
# state.pth, move data/ghost/live1_ledger_20261123 back to data/ghost/live1, start again.
#
# 2026-11-23 14:14 PST, from episode 156,780: three changes in one stop, all from the recorded live matches.
#   (1) --act-delay 16,18,1 (was 4,7,1). From the frame the policy decided on to the card taking effect is
#       1.7 s, not 0.5 s. The 0.5 s (capture + the two clicks) only gets the card out of the hand; the game
#       then takes about 1.2 s more before the unit lands (its deploy stopwatch appears) or the spell leaves
#       the King Tower. That comes before the 1 s deploy the simulator already has. Measured four ways:
#       stopwatch first seen 1.65-1.72 s after the decision frame (median per session, about 330 plays,
#       10th-90th percentile 1.58-1.84); our Fireball first visible at our King Tower in the raw frames
#       1.74 s after it (26 casts, 1.64-1.88; pixels only, no detector) and exploding 1.65 s later, which is
#       the simulator's flight; the Hog's first tower hit and the Fireball's 159 on a tower 2.1-2.2 s later
#       than the simulator's from the same tile (tower numbers read about 0.4 s late); and replaying the
#       recorded matches with our plays 1.2 s later removes tower hits that never happened.
#       Cost on the policy as it was (ep ~154.1k, 600 matches, live2, T 0.5): 65.5 % -> 57.8 % wins, tower
#       health lost 53.1 % -> 60.0 %, elixir leaked 0.82 -> 2.32. Expect that drop at first.
#   (2) data/ghost/live1 rebuilt with our own plays dated at the landing (OWN_LAND_S in
#       training/live_ghost_build.py). Old store: data/ghost/live1_reread_20261123.
#   (3) engine build/gen2_e (gen2_c + patch 0083): the first Goblins of a Goblin Drill and the first
#       Skeletons / Bats of a Witch / Night Witch come at the published start time, not one whole interval
#       late (first Drill stab 2.5-3.0 s after the play, was 5.4-5.9; recorded 2.1-2.4).
#   Both sides of the 19 recorded matches replayed, share of tower health lost: ours 0.513 -> 0.600 (real
#   0.668), theirs 0.447 -> 0.388 (real 0.171).
# State before this change: runs/g2_ppo_s2r/state_ep0156780_pre_delay1618.pth. To go back: stop, copy it over
# state.pth, move data/ghost/live1_reread_20261123 back to data/ghost/live1, set CR_ENGINE to build/gen2_c/engine
# and --act-delay 4,7,1, start again.
#
# 2026-11-23 15:55 PST, from episode 171,116: two changes in one stop, from the question "what differs
# between the simulator and the live matches".
#   (1) Opponent mix: teacher 0.10 -> 0.30, clone 0.35 -> 0.20, ghost 0.35 -> 0.30 (self 0.20 and the live
#       replays' 0.03 of all episodes unchanged); "p_midmatch" 0.30 -> 0.215 so that a ghost / clone episode
#       still starts mid-match 43 % of the time.
#       Why: by opponent kind the simulator already gives the live result, but only against the opponent that
#       answers what we do. Episodes 157k-170k (delay 16,18,1), sum over three towers of the health share lost:
#         teacher  win  7.6 %  dealt 1.00  taken 2.40     live, 19 matches: win 0 %  dealt 0.61  taken 2.08
#         ghost    win 47.0 %  dealt 1.44  taken 1.48     clone: 45.2 % 1.39 1.55     self: 57.0 % 1.64 1.49
#       A replay cannot answer our plays and the clone answers weakly, and together they were 70 % of the
#       episodes. The win rate against the teacher has not moved since 100k (cumulative 0.127 -> 0.14).
#       No card, deck or archetype is weighted; the teacher keeps its rungs (8, 9, 10 at 1:1:2) and its decks
#       (0.75 from the ladder pool).
#       Watch: the teacher row of by_kind_recent in runs/g2_ppo_s2r/progress.jsonl
#       (evals/tools/kind_trend.py), the guard lines, plays per minute.
#   (2) Engine build/gen2_f (gen2_e + patches 0084, 0085): the pocket opens when a Princess Tower falls
#       (replayed pocket plays were moved back to the river bank), and the Hunter fires ten pellets in a fan.
#       The 19 recorded matches replayed with both sides scripted, share of tower health lost: ours
#       0.600 -> 0.661 (real 0.668), theirs 0.388 -> 0.375 (real 0.171; by card, per match: Hog Rider 2102
#       against a real 1218, Hero Musketeer 1302 against 183).
# State before this change: runs/g2_ppo_s2r/state_ep0171116_pre_gen2f_teacher30.pth. To go back: stop, copy it
# over state.pth, set CR_ENGINE to build/gen2_e/engine, remove "weights" and "p_midmatch" from --pool-override,
# start again.
#
# 2026-11-23 16:45 PST, from episode 179,612: engine build/gen2_g (gen2_f + patch 0086), nothing else changed.
#   Which building or tower a troop turns to, fitted to 84 decided pull events read frame by frame from the 19
#   recorded live matches (before: 60 right, 6 wrong, 18 within measurement error; now 70, 0, 14):
#   (1) a Crown Tower is ranked against other targets by its published body (1.0 / 1.4), not the engine's
#       1.5 / 2.0: a Cannon pulls a Hog when its centre is more than 0.4 nearer than the tower's (was 0.9);
#   (2) the published SightClip no longer cuts sight along the arena (a Hog leaving the bridge sees the tower
#       it runs at); the sideways cut stays;
#   (3) a Tesla is underground and no target unless an enemy is inside its own sight (0.8 s hide time).
#   The 19 recorded matches replayed with both sides scripted barely move: tower health lost, ours
#   0.661 -> 0.650 (real 0.668), theirs 0.375 -> 0.365 (real 0.171); Hog Rider 2102 -> 2069 a match (real
#   1218), Hero Musketeer 1302 -> 1211 (real 183). So the pull rule was wrong in 6 of 84 cases and is not what
#   makes our attack about twice as strong in these replays as it was in the real matches.
# State before this change: runs/g2_ppo_s2r/state_ep0179612_pre_gen2g.pth. To go back: stop, copy it over
# state.pth, set CR_ENGINE to build/gen2_f/engine, start again.
#
# 2026-11-23 20:27 PST, from episode 197,844: the fifth live session (ep0187861, 0 wins of 9; 0 of 30 so far).
#   The policy stops playing when our towers are down and theirs are untouched: p(play) at 8+ elixir with a
#   tower down 0.004 and 0.047 in the two recorded runs (evals/tools/live_obs_probe.py), 0.22 / 0.42 when the
#   same observations show the enemy towers at half / a quarter health, 0.44 with the enemy towers blanked.
#   The observation is right; the state is missing from training: live we are behind on towers 47 % of the
#   seconds from a third of the match on, in the simulator 8 % of the decisions, and with our tower down the
#   enemy towers are both untouched in 5 % of the simulator's rows (p(play) there 0.036 at 7+ elixir).
#   Four changes, engine unchanged (build/gen2_g):
#   (1) --anchor-free-behind: decisions with more of our Princess Towers down than theirs carry no WHEN
#       reference (no WHEN KL to the clone, no cloning WHEN target), like --anchor-free-elixir: the clone's
#       humans give up there, and that habit is what the anchor was holding in place.
#   (2) pool "behind": 20 % of the teacher, clone and replay episodes start with the six towers as a live
#       training match had them (data/ghost/live2/towers.npy, a second between 33 % and 85 % of that match; a
#       mid-match start takes the second of its hand-over). Rows "<kind>:...+behind" in the per-checkpoint
#       table; evals/tools/kind_trend.py prints the teacher row without them.
#   (3) live store data/ghost/live2: the seven matches of the fifth session added (21 training matches, 20
#       inside the spend range; the fourth session stays held out), share 0.03 -> 0.04 (0.2 % a match).
#   (4) deck pool data/decks/meta_api_pool_live.json for the clone and teacher: the 33 decks our live opponents
#       played (official battle log; not the held-out session's) as 20 % of the pool's draws
#       (training/live_decks_pool.py). The ladder top-1000 pool had none of Electro Giant + Hero cards, Mirror +
#       Clone, Golem + Lava Hound + Three Musketeers.
#   To watch: live_obs_probe.py "as recorded" on runs/live/20261123-192303 and -192829 per snapshot (0.004 /
#   0.047 at ep0187861), and the teacher plain-start row.
# State before this change: runs/g2_ppo_s2r/state_ep0197844_pre_live5.pth. To go back: stop, copy it over
# state.pth, take "behind", "meta" and the live "path" out of --pool-override (live share 0.03), remove
# --anchor-free-behind, start again.
#
# 2026-11-24 02:25 PST, from episode 210,021: the scripted teacher gets our placement lag in 70 % of its episodes.
#   Sam asked whether to loosen the tie to the human clone. Measured instead:
#   (a) the tie has loosened four times already (KL coefficient 0.30 -> 0.07, KL to the clone up 1.5-3x) and the
#       win rate against the teacher did not rise (15.5-17.8 % for 100k-150k, 9-11 % since act delay 16,18,1);
#   (b) evals/tools/pilot_probe.py, 100 same-seed episodes: the engine's search teacher piloting OUR deck through
#       our act delay wins 31 % (8 % against the teacher), our policy ep0206073 38 % (18 %), the same pilot with
#       no delay 68 % (50 %). The delay decides, not the play;
#   (c) evals/tools/answer_time.py: in the 26 recorded live matches the opponent's plays after our Hog lands are
#       at chance for 2 s (0.26 per Hog against 0.31) and peak at 2.5-3.5 s; the teacher lands 0.99 per Hog in
#       those 2 s (chance 0.47), most of it inside 0.5 s. No person answers that fast.
#   Change: pool teacher "lag_share" 0.7 (training/pool_env.py _LaggedTeacher): in those episodes the teacher's
#   plays go through the same act delay as ours (it decides on what it sees, the card lands 1.6-1.8 s later; a
#   card in flight is not offered again). With it the teacher's answers start at 2.0-2.5 s like the live ones;
#   ep0206073 wins 38 % of 40 against it. 30 % stay instant: prediction plays exist, and the old teacher row keeps
#   its history. Rows "teacher:...+lag" in the per-checkpoint table; evals/tools/kind_trend.py prints both.
#   Risk: the simulator's attack is already about 1.8x the real one, so a fairer teacher can make the simulator
#   read rosier. To watch: the instant-teacher row must not fall, held-out evals (instant teacher), leak.
# State before this change: runs/g2_ppo_s2r/state_ep0210021_pre_teacherlag.pth. To go back: stop, copy it over
# state.pth, take "lag_share" out of --pool-override, start again.
#
# 2026-11-24 10:28 PST, from episode 284,938: the sixth live session (ep0284847, runs/live/20261124-094533;
#   first on the round-5 detector, --leak-floor 9.5 and the lagged teacher): 2 wins of 11 (one 1-0, one by the
#   opponent leaving at 0-2), the first live wins (0 of 30 before). Battle log, forfeit match left out: tower
#   health dealt 0.95 a match (0.61-0.78 in the sessions before), taken 2.08 (2.29-2.77), leaked 3.4 (13.3 the
#   evening before). Health lost while sitting on 6+ elixir without playing: 4 % of what we lost (22 % the
#   evening before); p(play) at 8+ elixir when behind on towers 0.28 (0.05) (evals/tools/live_idle_damage.py).
#   Three quarters of the health we lose is now lost while we are playing: defence, not passivity.
#   The machine was restarted at 09:43, which ended the trainer at this episode; nothing else was changed by it.
#   Two data changes, engine and flags unchanged:
#   (1) live store data/ghost/live3: the ten usable matches of this session added (31 training matches, 29
#       inside the spend range; the fourth session stays held out), share 0.04 -> 0.06 (0.2 % a match).
#   (2) deck pool data/decks/meta_api_pool_live3.json: this session's opponent decks added to the live 20 %.
#   Opponents' answers to our Hog over all 36 matches still start 2 s after it lands (0.23 per Hog in the first
#   2 s against 0.30 by chance; evals/tools/answer_time.py live --store data/ghost/live3): lag_share 0.7 stays.
# State before this change: runs/g2_ppo_s2r/state_ep0284938_pre_live7.pth. To go back: stop, copy it over
# state.pth, put data/ghost/live2 (share 0.04) and data/decks/meta_api_pool_live.json back in --pool-override,
# start again.
#
# 2026-11-24 12:57 PST, from episode 309,341: engine build/gen2_h (gen2_g + patch 0087, commit e49c3a9;
#   patches/pending/0087-river-crossing-on-tower-column-hog-diagonal-jump.patch), nothing else changed.
#   The river crossing runs on the tower column (3 / 14) and the Hog makes its diagonal jump there.
#   Why: replaying the 36 recorded live3 matches, our replayed Hog ran 0.4-1.1 tiles wide of the recorded
#   path (toward the wall) at rows 18-23 on gen2_g, within 0.2 on gen2_h (0.4 at row 23 right). Hits a Hog
#   play when the opponent answers with a building: real 0.58, gen2_g 1.33, gen2_h 0.93 (n 58); all plays
#   real 1.14, gen2_g 2.30, gen2_h 2.15. Clone 60 matches: win 58 % -> 53 %.
#   Evaluation files from here on are _eh_ (evals/tools/eval_pair.sh, eval_liveghost.sh point at gen2_h;
#   watcher log runs/logs/eval_watch_gen2h.log); ep0284847 and ep0296981 are evaluated on both as the bridge.
# State before this change: runs/g2_ppo_s2r/state_ep0309341_pre_0087.pth. To go back: stop, copy it over
# state.pth, set CR_ENGINE to build/gen2_g/engine (and the two eval scripts back to gen2_g / _eg_), start again.
#
# 2026-11-24 14:25 PST, from episode 321,483: pool change (weights, teacher lag share, building answers to our Hog).
#   weights ghost/clone/teacher/self 0.30/0.20/0.30/0.20 -> 0.20/0.15/0.45/0.20; teacher lag_share 0.7 -> 0.3;
#   teacher build_answer_p 1.0 + build_reserve; clone build_answer_p 1.0 (training/pool_env.py _BuildCommit).
#   Why (people = 24 clean live3 matches; sim = 60 matches each, ep0301012, gen2_h, seeds 8700+): people allow
#   1.16 Hog hits a play and deal 7,674 tower damage to us a match. Instant teacher 1.93 / 7,832 (policy wins
#   13 %), lagged teacher 2.42 / 6,562 (37 %), clone 2.55 / 6,294, ghost 3.51 / 5,989. The instant teacher is
#   the only opponent at people's strength; the lag added at ep 210,021 made it softer than people (it
#   executes stale decisions late; people answer late but well). The building answers move clone and teacher
#   toward people's building rate (building came in building-deck Hog plays: clone 47 -> 63 %, teacher lag 0
#   31 -> 44 %, lag 1 42 -> 55 %) and cut hits a play (2.55 -> 2.17, 1.93 -> 1.69, 2.42 -> 2.26) at no measured
#   cost (policy win +5 / 0 / +3 points, damage to us within 1 %, spend +3-6 %). Pool-wide Hog hits a play
#   was about 2.45 before. Held-out evaluations are unchanged (instant teacher, no options).
# State before this change: runs/g2_ppo_s2r/state_ep0321483_pre_pool1124.pth. To go back: stop, copy it over
# state.pth, set weights back to 0.30/0.20/0.30/0.20, teacher to {"meta_share":0.75,"lag_share":0.7} and clone
# to {"when_bias":0.2,"meta_share":0.75} in --pool-override, start again.
#
# 2026-11-24 15:35 PST, from episode 330,896: act delay 16,18,1 -> 14,16,1, nothing else changed.
#   ActuationDelay(lo, hi, jitter): an episode delay uniform in [lo, hi] (inclusive) plus a per-play jitter
#   uniform in [0, jitter], so 16,18,1 = 16-19 ticks a play and 14,16,1 = 14-17 ticks (1.4-1.7 s); 14,17,1
#   would give 14-18. Why: the live-latency measurements of 2026-11-24, from the board the decision used to our
#   deploy stopwatch: click median 1.70-1.77 s, --tap-mode drag 1.51 s (p10 1.39, p90 1.68), 0 failed
#   placements in 69 plays. Live play moves to drag. Evaluations from here are _d1417_ ... _ed_ (eval_pair.sh,
#   eval_liveghost.sh); ep0329297 is evaluated under both delays as the bridge.
# State before this change: runs/g2_ppo_s2r/state_ep0330896_pre_actdelay1417.pth. To go back: stop, copy it
# over state.pth, set --act-delay 16,18,1, start again.
#
# 2026-11-24 16:51 PST, from episode 338,464: CLOUD HANDOFF (the Mac is closed). The local trainer was stopped
#   with SIGTERM after its state write (update 8044); this script's launch line below is NOT what runs in the cloud.
#   Pod 5j6n0195hvivs4 (RunPod cpu3c 32 vCPU, $0.96/h, US-CA-2), network volume 4mpqfxjc7c (20 GB) at /workspace;
#   repo at /workspace/cr-robot, $HOME/cr-robot is a symlink to it on the pod (absolute paths in the
#   state). Engine gen2_h source at e49c3a9 built on Linux (build_linux, Release). Notes: /workspace/HANDOFF.md.
#   Pod launch line (/workspace/bin/run_pod.sh) = the line below with these changes:
#     --device cpu --act-device cpu --num-envs 24 --rollout-seconds 250 --torch-threads 16 (24 x 250 = 12 x 500:
#     same experience per update); --max-episodes 480000 (only the stop test reads it);
#     --pool-override weights ghost/clone/teacher/self 0.10/0.15/0.55/0.20 (was 0.20/0.15/0.45/0.20),
#     live {"path":"data/ghost/live3","share":0.0,"p_mid":0.67} (was share 0.06), teacher "lag_share":0.0 (was 0.3).
#   Why the pool change (opponent scorecard 2026-11-24): live-ghost replays are the softest opponent (83 % wins for
#   us, 3.9 Hog hits a play against people's 1.16) and miss opponent spells; dataset ghosts are next softest on
#   defence (3.3 hits a play); the unlagged teacher is the only kind at people's strength (1.66, 13 % wins).
#   Owner's instruction today: opponents react with no added delay (lag_share 0.0).
#   Cloning term on the pod: data/bc2/hog_g2 holds only every 7th training shard (k = 7, 16 of 110 shards, 2.72 GB:
#   train-hog-0000 0003-m 0007 0010-m ... 0049 0052-m); noise cache rebuilt on the pod. All 167 snapshots copied.
#   Watchdog: trainer SIGTERM 6 h 45 min after the start (budget $8), then the pod terminates itself.
# State before this change: runs/g2_ppo_s2r/state_ep0338464_pre_cloud1124.pth (covers all three pool changes:
#   lag_share 0.0, live share 0.0, weights 0.10/0.15/0.55/0.20; and the k=7 shard subset).
# To come back: rsync /workspace/cr-robot/runs/g2_ppo_s2r/ (state.pth, latest.pth, snapshots/, progress.jsonl) and
#   runs/logs/g2_ppo_s2r.log (append) from the volume (start a pod on volume 4mpqfxjc7c if none is running), then
#   resume locally with this script (decide whether to carry the pool changes into the line below first).
# 2026-11-25 02:02 UTC, from episode 340,875 (update 8104): SWITCHED TO GPU POD qrx76gkxsxdzux (RTX 4090, 16 vCPU,
#   $0.89/h, EU-RO-1), network volume dmcb7hvk2x (20 GB). The CPU pod 5j6n0195hvivs4 is terminated; its volume
#   4mpqfxjc7c keeps the pre-switch copy. Same training as the CPU pod: --num-envs 24 --rollout-seconds 250
#   --torch-threads 8, only --device cuda --act-device cuda changed (rollout length unchanged). About 2.7 ep/s
#   (CPU pod 0.57). State at the switch: runs/g2_ppo_s2r/state_ep0340875_pre_gpu1125.pth (on the volume).
#   On the volume: run dir /workspace/cr-robot-runs/g2_ppo_s2r, log /workspace/cr-robot-runs/logs/g2_ppo_s2r.log,
#   launch /workspace/bin/run_pod.sh, notes /workspace/HANDOFF.md. Trainer stops 06:45 UTC, pod terminates 07:00 UTC.
# 2026-11-25 08:40 UTC, from episode 396,849 (update 9470): BACK ON THE MAC. Cloud run copied back (state sha256
#   fdef4059340e..., cloud logs in runs/cloud_logs_1125/). The line below now carries the cloud pool (weights
#   0.10/0.15/0.55/0.20, lag_share 0.0, live share 0.0) and --max-episodes 480000; envs stay 12 x 500 s on mps.
#   State at the return: runs/g2_ppo_s2r/state_ep0396849_post_cloud1125.pth. The Mac state from before the cloud
#   (episode 338,464) is also kept as state_local_before_copyback1125.pth. To go back to the old pool: restore
#   weights 0.20/0.15/0.45/0.20, lag_share 0.3, live share 0.06 and start from state_ep0338464_pre_cloud1124.pth.
# 2026-11-25 09:20 UTC: ROLLED BACK to episode 338,464 (state_ep0338464_pre_cloud1124.pth) and the old pool above.
#   The cloud run (340,875 -> 396,849 on weights 0.10/0.15/0.55/0.20, lag 0, live 0) got worse on every measure:
#   held-out replays, as deployed, 65.3 % (ep0329297) -> 58.6 % (ep0395878), paired -6.7 +- 1.5; in training, wins
#   against the lag-0 teacher 16.8 -> 12.7 %, clone 41.4 -> 38.0 %, own snapshots 61.5 -> 57.5 %, damage taken up
#   against every kind, elixir leaked 0.5 -> 2.7 a minute. Cause not isolated (candidates: the harder pool, the
#   play-ratio guard's KL boost stuck at 3.375, 24 x 250 s rollouts, the 7-shard BC subset on the pod).
#   Kept, not used: state_ep0396849_post_cloud1125.pth, state_ep0398665_cloudpool_stopped1125.pth (1,800 more
#   episodes on the Mac), cloud snapshots in runs/g2_ppo_s2r/snapshots_cloud1125/ (ep0339xxx-ep0397xxx).
# 2026-11-25 01:43 PST, at episode 342,412 (update 8143): THIS RUN IS PAUSED for the anchor test (does the imitation
#   pull help?). Stopped with SIGTERM; state copied to runs/g2_ppo_s2r/state_ep0342412_pre_anchor_arms1125.pth
#   (revert point) and from it to three arms in runs/anchor_1125/ (launch scripts training/run_g2_anc_{ctrl,loose,free}.sh,
#   logs runs/anchor_1125/logs/). The arms are this line with 4 envs x 1500 s (= 12 x 500 s), --torch-threads 2,
#   --out runs/anchor_1125/g2_anc_<arm>, and only the anchor flags differing:
#     ctrl  = this line's anchor unchanged (the control arm continues this run);
#     loose = --kl-max 0.24;
#     free  = --kl-start 0 --kl-end 0 --kl-final 0 --kl-max 0 --bc-start 0 --bc-end 0 (no KL, no cap, no cloning).
#   The arms read this run's snapshots for the self pool but write nothing here. Do not restart this script
#   while the arms run; to continue the run later, copy the chosen arm's state or restart from the revert point.
# 2026-11-25 ~04:55 PST, from episode 351,411 (update 8368): ANCHOR TEST RESULT + ENGINE gen2_i (patch 0088).
#   Anchor arms at +8,000 (as-deployed held-out, 1,000 paired matches, gen2_h, _ei_ files): ep0341378 65.2 %;
#   ctrl ep0350446 65.5, loose ep0350415 65.0 (loose - ctrl -0.5 +- 1.5), free ep0350428 44.0 (free - ctrl
#   -21.5 +- 1.7; it stopped answering with Fireball and Hog, KL to the clone 1.26). Coordinator's rule chose loose
#   (held-out not below -1 SE; 4k-8k window: no-lag teacher 16.3 vs 16.2 %, damage taken 1.48 vs 1.51, Fireball share
#   3.4 %). This run continues from the loose arm's state (runs/anchor_1125/g2_anc_loose, its progress rows appended
#   here, its snapshots copied into snapshots/; the state's snapshot list still names runs/anchor_1125/g2_anc_loose/
#   snapshots, keep that folder). Changes on the line: --kl-max 0.12 -> 0.24; CR_ENGINE gen2_h -> gen2_i with
#   CR_P0088=tornado,jump,log,turret,king,speed,bankrows,stomp,jumpcol (physdist, jumpspeed, evocannon off).
#   Evaluations from here are the _ei_ series on gen2_i with that switch set (evals/tools/eval_pair_ei.sh); they do
#   not compare with _ed_ files. Revert points: engine change -> runs/g2_ppo_s2r/state_ep0351411_pre_0088.pth (copy
#   over state.pth, CR_ENGINE back to gen2_h, drop the CR_P0088 line; kl-max stays 0.24 = loose arm); anchor arms ->
#   state_ep0342412_pre_anchor_arms1125.pth with latest_ep0342412_pre_anchor_arms1125.pth (copy over state.pth and
#   latest.pth, --kl-max 0.12, gen2_h, restore progress_pre_anchor_merge1125.jsonl). Script before: run_g2_ppo_s2r.sh.bak_pre_0088_1125.
# 2026-11-25 10:44 PST, from episode 394,480 (update 9430), owner's decision: THE CLONE OPPONENT PLAYS WITHOUT THE ACT
#   DELAY. Pool override: "no_delay":true in the clone entry (new option in training/pool_env.py, off by default;
#   backup pool_env.py.bak_pre_clone_nodelay1125). Nothing else on the line changed.
#   Why: the clone imitates people's COMMAND times and people have no pipeline delay, so putting our act delay on it
#   too made it late. Measured on gen2_i, 600 matches a row (runs/anchor_lead_1125/clone_sc_summary.txt): our win
#   rate against it went from 52.8 % (delay on) to 37.8 % (delay off), and our Hog hits a play from 2.32 to 2.06.
#   This restart also picks up the new option --bc-head-weights in training/ppo_pool.py (default 1,1,1 = as before;
#   backup ppo_pool.py.bak_pre_bchead1125), which this line does not pass.
# State before this change: runs/g2_ppo_s2r/state_ep0394480_pre_clone_nodelay1125.pth. To go back: stop, copy it
#   over state.pth, remove "no_delay":true from the clone entry, start again. Script before:
#   run_g2_ppo_s2r.sh.bak_20261125_pre_clone_nodelay.
# 2026-11-26 04:2x UTC, from episode 480,009 (stopped at its limit): --max-episodes 480000 -> 600000, nothing else changed (backup run_g2_ppo_s2r.sh.bak_20261126_pre_max600k).
# 2026-11-26 09:1x UTC, from episode 509,159: --act-delay 14,16,1 -> 13,15,1 (live delay median 14.0, 2.5-97.5 % 12.9-16.4 ticks, n 431; Sam: training range must cover 95 %). Revert: state_ep0509159_pre_delay1316_1126.pth + run_g2_ppo_s2r.sh.bak_20261126_pre_delay1315.
cd $HOME/cr-robot || exit 1
export CR_ENGINE=$HOME/cr-robot/build/gen2_i/engine PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CR_P0088=tornado,jump,log,turret,king,speed,bankrows,stomp,jumpcol
nohup .venv/bin/python -m training.ppo_pool --run g2_ppo_s2r \
  --init runs/bc3/g2_big_L0_H10/ep4.pth \
  --clone-ckpt runs/bc3/all_big_L0_H10/ep5.pth \
  --pool-override '{"weights":{"ghost":0.20,"clone":0.15,"teacher":0.45,"self":0.20},"p_midmatch":0.215,"clone":{"when_bias":0.2,"meta_share":0.75,"build_answer_p":1.0,"no_delay":true},"teacher":{"meta_share":0.75,"lag_share":0.3,"build_answer_p":1.0,"build_reserve":true},"live":{"path":"data/ghost/live3","share":0.06,"p_mid":0.67},"meta":{"path":"data/decks/meta_api_pool_live3.json"},"behind":{"share":0.2}}' \
  --sim2real v1 --sim2real-scale 0:1.5 --sim2real-widen 10000 --sim2real-anchor-clean --priv-critic \
  --sim2real-elixir-est --sim2real-elixir-est-share 0.8 --board-hp-live --tail-preset reader2 \
  --num-envs 12 --device mps --act-device cpu --torch-threads 4 \
  --lr 5e-5 --warmup-lr 5e-4 --critic-warmup 6 --rollout-seconds 500 --chunk-seconds 50 --epochs 4 --minibatches 8 \
  --kl-start 1.0 --kl-end 0.15 --kl-mix 0.5 --kl-max 0.24 --bc-start 0.5 --bc-end 0.2 --anchor-decay 40000 \
  --kl-final 0.05 --kl-final-from 124000 --kl-final-at 250000 --act-delay 13,15,1 \
  --anchor-free-elixir 9.5 --anchor-free-behind \
  --ent-when 0 --ent-what 0 --ent-where 0 --snapshot-every 2000 --max-episodes 600000 \
  >> runs/logs/g2_ppo_s2r.log 2>&1 < /dev/null &
echo "started pid $!"
