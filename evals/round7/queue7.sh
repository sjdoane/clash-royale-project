# serial queue: round-7 probes (killed and retried if a live session starts), then the act-delay bridge evals
cd $HOME/cr-robot
S=/private/tmp/claude-501/-Users-samueldoane-cr-robot/40e94860-6611-4922-a6d8-a1b149e53030/scratchpad
live() { pgrep -f "python[0-9.]* [^ ]*play_live[.]py" >/dev/null; }
quiet() { q=0; while [ $q -lt 30 ]; do if live; then q=0; else q=$((q+1)); fi; sleep 10; done; }
probe() {   # name pooljson [seed]
  while true; do
    quiet; rm -f $S/r7_$1.log
    SEED=${3:-8700} zsh $S/run7.sh $1 teacher "$2" &
    pid=$!
    while kill -0 $pid 2>/dev/null; do
      if live; then pkill -f run7.sh; pkill -f r7_probe.py; echo "$(date +%H:%M) paused $1"; fi
      sleep 10
    done
    grep -q "exit 0" $S/r7_$1.log 2>/dev/null && { echo "$(date +%H:%M) done $1"; return; }
  done
}
D='"build_answer_p":1.0,"build_reserve":true'
for spec in "$@"; do
  case $spec in
    probe:*) a=(${(s/:/)spec}); probe $a[2] "{\"teacher\":{\"lag_share\":$a[3],$D,\"attack_bank\":$a[4]}}" $a[5] ;;
    bridge) quiet; echo "$(date +%H:%M) bridge old"; zsh $S/eval_pair.sh.pre_ed runs/g2_ppo_s2r/snapshots/ep0329297.pth ep0329297 20261124
            quiet; zsh $S/eval_liveghost.sh.pre_ed runs/g2_ppo_s2r/snapshots/ep0329297.pth ep0329297 20261124
            quiet; echo "$(date +%H:%M) bridge new"; zsh evals/tools/eval_pair.sh runs/g2_ppo_s2r/snapshots/ep0329297.pth ep0329297 20261124
            quiet; zsh evals/tools/eval_liveghost.sh runs/g2_ppo_s2r/snapshots/ep0329297.pth ep0329297 20261124
            echo "$(date +%H:%M) bridge done" ;;
  esac
done
