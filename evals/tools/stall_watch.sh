#!/bin/zsh
# usage: stall_watch.sh   (left running next to the trainer)
# Two updates of g2_ppo_s2r took 859 s and 585 s on 2026-11-23 (normal: 5 s) with nothing to show why.
# When the trainer's log has not moved for 120 s, take one 5 s stack sample of the trainer process
# (macOS `sample`) into runs/logs/stall_<time>.txt, then wait for the log to move again.
cd "$(dirname "$0")/../.." || exit 1
log=runs/logs/g2_ppo_s2r.log
while true; do
  pid=$(pgrep -f "training.[p]po_pool --run g2_ppo_s2r" | head -1)
  [ -z "$pid" ] && { sleep 60; continue; }
  age=$(( $(date +%s) - $(stat -f %m $log) ))
  if [ $age -ge 120 ]; then
    out=runs/logs/stall_$(date +%m%d_%H%M%S).txt
    { echo "log idle for $age s"; tail -1 $log | cut -c1-80; vm_stat | head -12; sysctl vm.swapusage; pmset -g assertions | head -12; } > $out 2>&1
    sample $pid 5 -file $out.sample > /dev/null 2>&1
    while [ $(( $(date +%s) - $(stat -f %m $log) )) -ge 120 ]; do sleep 20; done
    echo "resumed after $(( $(date +%s) - $(stat -f %m $out) )) s more" >> $out
  fi
  sleep 30
done
