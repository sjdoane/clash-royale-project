#!/bin/zsh
# 2026-11-25 anchor A/B/C arm g2_anc_loose -- loose: cap doubled, --kl-max 0.24, everything else as control.
# All three arms (training/run_g2_anc_{ctrl,loose,free}.sh) start from runs/g2_ppo_s2r/state_ep0342412_pre_anchor_arms1125.pth
# (copied to runs/anchor_1125/g2_anc_<arm>/state.pth with latest.pth) and differ ONLY in the anchor line marked below.
# Shared machine split: 4 envs x 1500 s rollout = the main run's 12 x 500 s (same samples per update), 2 torch threads.
# Run dir runs/anchor_1125/g2_anc_loose (own snapshots, progress.jsonl, state); log runs/anchor_1125/logs/g2_anc_loose.log.
# The self pool reads the main run's snapshots listed in the copied state (read only). Stop with SIGTERM.
cd $HOME/cr-robot || exit 1
export CR_ENGINE=$HOME/cr-robot/build/gen2_h/engine PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
DEV=${DEV:-mps}
nohup .venv/bin/python -m training.ppo_pool --run g2_anc_loose --out runs/anchor_1125/g2_anc_loose \
  --init runs/bc3/g2_big_L0_H10/ep4.pth \
  --clone-ckpt runs/bc3/all_big_L0_H10/ep5.pth \
  --pool-override '{"weights":{"ghost":0.20,"clone":0.15,"teacher":0.45,"self":0.20},"p_midmatch":0.215,"clone":{"when_bias":0.2,"meta_share":0.75,"build_answer_p":1.0},"teacher":{"meta_share":0.75,"lag_share":0.3,"build_answer_p":1.0,"build_reserve":true},"live":{"path":"data/ghost/live3","share":0.06,"p_mid":0.67},"meta":{"path":"data/decks/meta_api_pool_live3.json"},"behind":{"share":0.2}}' \
  --sim2real v1 --sim2real-scale 0:1.5 --sim2real-widen 10000 --sim2real-anchor-clean --priv-critic \
  --sim2real-elixir-est --sim2real-elixir-est-share 0.8 --board-hp-live --tail-preset reader2 \
  --num-envs 4 --device $DEV --act-device cpu --torch-threads 2 \
  --lr 5e-5 --warmup-lr 5e-4 --critic-warmup 6 --rollout-seconds 1500 --chunk-seconds 50 --epochs 4 --minibatches 8 \
  --kl-start 1.0 --kl-end 0.15 --kl-mix 0.5 --kl-max 0.24 --bc-start 0.5 --bc-end 0.2 --anchor-decay 40000 \
  --kl-final 0.05 --kl-final-from 124000 --kl-final-at 250000 \
  --act-delay 14,16,1 \
  --anchor-free-elixir 9.5 --anchor-free-behind \
  --ent-when 0 --ent-what 0 --ent-where 0 --snapshot-every 2000 --max-episodes 480000 \
  >> runs/anchor_1125/logs/g2_anc_loose.log 2>&1 < /dev/null &
echo "started pid $!"
