#!/usr/bin/env bash
# Mac-side watcher for a RunPod training pod.
#  - pulls /root/out back to the Mac every INTERVAL seconds
#  - when /root/out/DONE exists and the final pull succeeded, terminates the pod
#  - if the GPU sits idle for IDLE_LIMIT seconds after /root/out/TRAINING_STARTED
#    appears (training crashed or hung), pulls once more and stops the pod
#  - at DEADLINE (epoch seconds, optional) pulls and stops the pod regardless (cost fuse)
# Stop/terminate go through training/rp_api.sh from the Mac: the pod itself cannot reach
# the RunPod API, so on-pod runpodctl calls are not relied on.
# usage: training/pod_watch.sh <ssh_host> <ssh_port> <local_out_dir> <pod_id>
set -uo pipefail
HOST=$1; PORT=$2; DEST=$3; POD=$4
API=$(cd "$(dirname "$0")" && pwd)/rp_api.sh
DEADLINE=${DEADLINE:-}
KEY=${KEY:-$HOME/.ssh/runpod_skillmimic_20260916}
INTERVAL=${INTERVAL:-600}
IDLE_LIMIT=${IDLE_LIMIT:-1800}
SSH=(ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=20 "root@$HOST")
mkdir -p "$DEST"
idle_since=""

pull() {
  rsync -az --partial -e "ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new" \
    "root@$HOST:/root/out/" "$DEST/"
}
remote() { "${SSH[@]}" "$@"; }

while true; do
  ts=$(date '+%F %T')
  if pull; then echo "[$ts] pulled"; else echo "[$ts] pull FAILED"; fi

  if remote test -f /root/out/DONE; then
    pull && echo "[$ts] DONE + final pull ok; terminating pod" \
      && "$API" terminate "$POD" && exit 0
    echo "[$ts] final pull or terminate FAILED; stopping instead"; "$API" stop "$POD"; exit 1
  fi

  if remote test -f /root/out/TRAINING_STARTED; then
    util=$(remote "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1" 2>/dev/null | tr -dc 0-9)
    if [[ -n "$util" && "$util" -lt 5 ]]; then
      [[ -z "$idle_since" ]] && idle_since=$(date +%s)
      idle=$(( $(date +%s) - idle_since ))
      echo "[$ts] GPU idle ${idle}s (util=${util}%)"
      if (( idle >= IDLE_LIMIT )); then
        pull; echo "[$ts] idle limit hit; stopping pod"
        "$API" stop "$POD"
        exit 2
      fi
    else
      idle_since=""; echo "[$ts] GPU util=${util:-?}%"
    fi
  fi
  if [[ -n $DEADLINE ]] && (( $(date +%s) >= DEADLINE )); then
    pull; echo "[$ts] deadline reached; stopping pod"; "$API" stop "$POD"; exit 3
  fi
  sleep "$INTERVAL"
done
