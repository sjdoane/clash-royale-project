#!/bin/zsh
# usage: replay_live.sh <checkpoint> <tag> [extra driver flags]
# Replays Sam's five recorded matches (runs/live/20261123-004549) through the live driver with a checkpoint.
# Reads frames only: no phone, no taps. Log: runs/logs/replay_<tag>.log (its "run dir" line names the output).
cd $HOME/cr-robot || exit 1
export CR_ENGINE=$HOME/cr-robot/build/gen2_e/engine PYTHONUNBUFFERED=1
ck=$1; tag=$2; extra=$3
.venv/bin/python play_live.py --checkpoint $ck --replay runs/live/20261123-004549/raw --no-record --no-video --seconds 700 \
  --driver-flags="--loop-timestamps --ability tap $extra" > runs/logs/replay_$tag.log 2>&1
