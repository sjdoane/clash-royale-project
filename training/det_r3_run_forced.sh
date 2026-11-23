#!/usr/bin/env bash
# Mac-side driver for the round-3 detector pod (video_dataset/detector/pod_train_r3.sh).
# Runs next to training/pod_watch.sh, which pulls /root/out and terminates the pod on DONE.
#
#   1 waits for the A/B stage (AB_DONE), pulls each arm's last.pt and scores it against the
#     best detector so far on our by-eye frames (training/det_check.py, rules fixed there)
#   2 picks the data for the main run: dense and/or real only if that arm is clearly better
#     than ctl; no main run at all unless the picked arm clearly beats the reference
#   3 starts the main run, scores epochs 2, 5 and 9 as they appear and stops the run if a
#     check says STOP (round 2 would have been stopped at the first check)
#   4 scores the final weights and writes <dest>/VERDICT.txt
# "Stop" = kill the training and touch /root/out/DONE; pod_watch.sh then terminates the pod.
#
# usage: training/det_r3_run.sh <ssh_host> <ssh_port> <local_out_dir> [main_hours]
set -uo pipefail
HOST=$1; PORT=$2; DEST=$3; MAIN_HOURS=${4:-2.5}
REPO=$(cd "$(dirname "$0")/.." && pwd)
KEY=${KEY:-$HOME/.ssh/runpod_skillmimic_20260916}
PY=${PY:-$REPO/.venv-det/bin/python}
POLL=${POLL:-60}
SSH=(ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=20 "root@$HOST")
POD_SH=/root/cr/video_dataset/detector/pod_train_r3.sh
mkdir -p "$DEST/checks"
LOG=$DEST/det_r3_run.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
remote() { "${SSH[@]}" "$@"; }
fetch() {   # fetch <remote path under /root/out> : copy one file into $DEST, same relative path
  mkdir -p "$DEST/$(dirname "$1")"
  rsync -az --partial -e "ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new" "root@$HOST:/root/out/$1" "$DEST/$1"
}
check() {   # check <name> <weights> <rule> [vs] ; prints the table, returns det_check's exit code
  local name=$1 w=$2 rule=$3 vs=${4:-}
  local args=("$w" --rule "$rule" --json "$DEST/checks/$name.json")
  [[ -n $vs ]] && args+=(--vs "$vs")
  (cd "$REPO" && "$PY" training/det_check.py "${args[@]}") 2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}
stop_run() {   # end the pod session: pod_watch.sh terminates the pod when it sees DONE
  log "ending the pod session: $1"
  # [3] / [n]: the patterns must not match this ssh command line itself
  remote "pkill -f 'pod_train_r[3].sh'; pkill -f 'pytho[n] -'; sleep 5; touch /root/out/DONE" || log "could not reach the pod to stop it"
}
rc_of() { cat "$DEST/checks/$1.rc" 2>/dev/null || echo none; }
wait_remote() {   # wait_remote <file under /root/out> : 0 it exists, 1 the session ended first,
                  # 2 the pod script is not running any more, 3 the pod has been unreachable for 15 min
  local dead=0 gone=0
  while true; do
    if remote true 2>/dev/null; then
      gone=0
      remote "test -f /root/out/$1" && return 0
      remote "test -f /root/out/DONE" && return 1
      if remote "pgrep -f 'pod_train_r[3].sh' >/dev/null"; then dead=0; else dead=$((dead + 1)); fi
      (( dead >= 2 )) && return 2
    else
      gone=$((gone + 1)); log "pod not reachable (waiting for $1)"
      (( gone * POLL >= 900 )) && return 3
    fi
    /bin/sleep "$POLL"
  done
}

# ---- 1 A/B arms -----------------------------------------------------------------------------
log "waiting for the A/B stage"
wait_remote AB_DONE; w=$?
if [[ $w != 0 ]]; then
  log "A/B stage did not finish (wait code $w): see $DEST/pod_train.log"
  [[ $w == 2 ]] && stop_run "the pod script died in the A/B stage"
  exit 1
fi
ARMS=$(remote "ls /root/out/ab")
rm -f "$DEST"/checks/*.rc
for arm in $ARMS; do
  fetch "ab/$arm/weights/last.pt" || { log "could not pull arm $arm"; continue; }
  check "ab_$arm" "$DEST/ab/$arm/weights/last.pt" arm; echo $? > "$DEST/checks/$arm.rc"
done
[[ $(rc_of ctl) != none ]] || { stop_run "no ctl arm to compare with"; exit 1; }

# ---- 2 pick the data for the main run -------------------------------------------------------
DATA="${FORCE_DATA:-}"       # FORCE_DATA=dense: the owner asked to keep the crowded frames
for arm in dense real; do
  [[ $DATA == *$arm* ]] && continue
  [[ $(rc_of $arm) != none && $(rc_of $arm) != 3 ]] || continue
  if check "ab_${arm}_vs_ctl" "$DEST/ab/$arm/weights/last.pt" better "$DEST/ab/ctl/weights/last.pt"; then
    DATA="$DATA$arm"
  fi
done
[[ -n $DATA ]] || DATA=ctl
PICK=${DATA/densereal/dense}       # the single arm whose score gates the main run
[[ $DATA == densereal ]] && log "both dense and real beat ctl: main run uses both (gate: the dense arm)"
log "data for the main run: $DATA (arm verdicts, 0 pass / 3 stop / 4 no clear gain: $(for a in $ARMS; do printf '%s=%s ' "$a" "$(rc_of $a)"; done))"
GATE=$(rc_of $PICK)
if [[ -n ${FORCE_DATA:-} && $GATE == 4 && $(rc_of ctl) == 0 ]]; then
  log "forced data $FORCE_DATA: arm $PICK has no clear strict-F1 gain but no regression (exit 4) and ctl passed: the main run goes ahead"
  GATE=0
fi
if [[ $GATE != 0 ]]; then
  best=""; for arm in $ARMS; do [[ $(rc_of $arm) == 0 ]] && best=$arm; done
  echo "no main run: arm $PICK did not clearly beat the reference (exit $(rc_of $PICK)). Best detector: ${best:+arm $best ($DEST/ab/$best/weights/last.pt)}${best:-the reference (unchanged)}" | tee "$DEST/VERDICT.txt" | tee -a "$LOG"
  stop_run "no main run"
  exit 0
fi

# ---- 3 main run + early checks ---------------------------------------------------------------
log "starting the main run: --data $DATA --max-hours $MAIN_HOURS"
remote "nohup bash $POD_SH --stage main --data $DATA --max-hours $MAIN_HOURS --no-install >> /root/pod_train.out 2>&1 < /dev/null &"
for ep in 1 4 8; do
  wait_remote "train/weights/epoch$ep.pt" || break
  fetch "train/weights/epoch$ep.pt" || continue
  check "main_epoch$ep" "$DEST/train/weights/epoch$ep.pt" early
  if [[ $? == 3 ]]; then
    echo "main run stopped at epoch $((ep + 1)) by the early check. Best detector: arm $PICK ($DEST/ab/$PICK/weights/last.pt)" | tee "$DEST/VERDICT.txt" | tee -a "$LOG"
    stop_run "early check failed at epoch $((ep + 1))"
    exit 0
  fi
done

# ---- 4 final ------------------------------------------------------------------------------------
log "waiting for the main run to finish"
wait_remote DONE; w=$?
[[ $w == 2 ]] && stop_run "the pod script died in the main run"
fetch "train/weights/last.pt" || { /bin/sleep 90; }      # pod_watch.sh may have terminated the pod: it pulled the file
[[ -f $DEST/train/weights/last.pt ]] || { log "no final weights in $DEST"; exit 1; }
check main_final "$DEST/train/weights/last.pt" arm; rc=$?
check main_final_vs_arm "$DEST/train/weights/last.pt" better "$DEST/ab/$PICK/weights/last.pt"; rc2=$?
if [[ $rc == 0 && $rc2 == 0 ]]; then v="main run ($DEST/train/weights/last.pt)"
else v="arm $PICK ($DEST/ab/$PICK/weights/last.pt); the main run did not clearly add to it (vs reference exit $rc, vs arm exit $rc2)"; fi
echo "Best detector: $v" | tee "$DEST/VERDICT.txt" | tee -a "$LOG"
