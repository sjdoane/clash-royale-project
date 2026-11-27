#!/bin/zsh
# Per new PPO snapshot: p(play) at 8+ elixir with one of our towers down, on the recorded observations of the
# two live runs of 2026-11-23 evening (evals/tools/live_obs_probe.py ... quick). The number the "it just won't
# play" fix (training/run_g2_ppo_s2r.sh, block of 2026-11-23 20:27) is meant to move; 0.005 / 0.074 at ep0197968.
#   nohup zsh evals/tools/giveup_watch.sh >> runs/logs/giveup_probe.log 2>&1 &
# Stops when the trainer has been gone for 10 minutes.
cd $HOME/cr-robot || exit 1
export CR_ENGINE=$PWD/build/gen2_c/engine PYTHONDONTWRITEBYTECODE=1
done_list=runs/logs/giveup_probe.done
touch $done_list
gone=0
while true; do
  for s in runs/g2_ppo_s2r/snapshots/ep*.pth(N); do
    n=${${s:t:r}#ep}
    (( 10#$n < 197968 )) && continue
    grep -qx "$n" $done_list && continue
    for r in 20261123-192303 20261123-192829; do
      line=$(.venv/bin/python evals/tools/live_obs_probe.py runs/live/$r $s quick 2>/dev/null | grep "as recorded")
      echo "$(date '+%m-%d %H:%M') ep$n $r $line"
    done
    echo "$n" >> $done_list
  done
  if pgrep -f "training.[p]po_pool --run g2_ppo_s2r" > /dev/null; then gone=0; else gone=$((gone + 1)); fi
  (( gone >= 10 )) && { echo "$(date '+%m-%d %H:%M') trainer gone, stopping"; exit 0; }
  perl -e 'sleep 60'
done
