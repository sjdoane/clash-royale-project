#!/bin/zsh
# Start or resume the gen-2 PPO run (resumes from runs/$RUN/state.pth when it exists).
# Stop it with SIGTERM: it finishes the update and saves. Log: runs/logs/$RUN.log
#
# Needs: checkpoints/bc_init.pth + bc_clone_any.pth (scripts/fetch_assets.sh),
#        data/ghost/v1 + data/decks (scripts/build_data.sh ghosts),
#        data/bc2/hog_g2 for the cloning loss (scripts/build_data.sh bc), or NO_BC=1 to train without it.
#   RUN=my_run DEVICE=cuda NUM_ENVS=16 training/run_g2_ppo_s2r.sh
#
# The original run also replayed our own recorded live matches (data/ghost/live1, pool entry
# "live":{"share":0.03,"p_mid":0.67}); those recordings hold other players' matches and are not shipped.
#
# What follows is the run's change log, kept as it was written during training.
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
cd "$(dirname "$0")/.." || exit 1
export CR_ENGINE="${CR_ENGINE:-$PWD/engine}" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN=${RUN:-g2_ppo_s2r}
BC=(--bc-start 0.5 --bc-end 0.2)
[ -n "${NO_BC:-}" ] && BC=(--bc-start 0 --bc-end 0)
mkdir -p runs/logs
nohup .venv/bin/python -m training.ppo_pool --run $RUN \
  --init checkpoints/bc_init.pth \
  --clone-ckpt checkpoints/bc_clone_any.pth \
  --pool-override '{"clone":{"when_bias":0.2,"meta_share":0.75},"teacher":{"meta_share":0.75}}' \
  --sim2real v1 --sim2real-scale 0:1.5 --sim2real-widen 10000 --sim2real-anchor-clean --priv-critic \
  --sim2real-elixir-est --sim2real-elixir-est-share 0.8 --board-hp-live --tail-preset reader2 \
  --num-envs ${NUM_ENVS:-12} --device ${DEVICE:-mps} --act-device cpu --torch-threads 4 \
  --lr 5e-5 --warmup-lr 5e-4 --critic-warmup 6 --rollout-seconds 500 --chunk-seconds 50 --epochs 4 --minibatches 8 \
  --kl-start 1.0 --kl-end 0.15 --kl-mix 0.5 --kl-max 0.12 "${BC[@]}" --anchor-decay 40000 \
  --kl-final 0.05 --kl-final-from 124000 --kl-final-at 250000 --act-delay 4,7,1 \
  --anchor-free-elixir 9.5 \
  --ent-when 0 --ent-what 0 --ent-where 0 --snapshot-every 2000 --max-episodes 400000 \
  >> runs/logs/$RUN.log 2>&1 < /dev/null &
echo "started pid $! (log: runs/logs/$RUN.log)"
