#!/bin/zsh
# usage: eval_watch.sh <date> <target_episode> [<target_episode> ...]
# For each target, wait for the first g2_ppo_s2r snapshot at or past it, then run eval_pair.sh on it.
# Stops early if the trainer is no longer running and no such snapshot exists.
cd "$(dirname "$0")/../.." || exit 1
day=$1; shift
for target in "$@"; do
  while true; do
    snap=$(ls runs/g2_ppo_s2r/snapshots | sed -n 's/^ep\([0-9]*\)\.pth$/\1/p' | awk -v t=$target '$1+0 >= t {print; exit}')
    [ -n "$snap" ] && break
    pgrep -f "training.[p]po_pool --run g2_ppo_s2r" > /dev/null || { echo "trainer gone before $target"; exit 2; }
    sleep 60
  done
  echo "$(date +%H:%M) evaluating ep$snap"
  evals/tools/eval_pair.sh runs/g2_ppo_s2r/snapshots/ep$snap.pth ep$snap $day
  evals/tools/eval_liveghost.sh runs/g2_ppo_s2r/snapshots/ep$snap.pth ep$snap $day
  echo "$(date +%H:%M) done ep$snap"
done
