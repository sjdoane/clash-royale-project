#!/bin/bash
# Re-run the whole state-estimation eval (runs/perception/STATE_ACCURACY_REPORT.md).
#
#   live_ppo/eval/run_state_accuracy.sh [RUN] [--pipeline] [--labels]
#
#   RUN         output dir under runs/perception (default v4o)
#   --pipeline  re-run the state pipeline (production flags, state_pipeline.DEFAULT_FLAGS)
#               over every eval source first (~30 min per video, 4 at a time)
#   --labels    rebuild the candidate labels from the BASE runs (runs/perception/base,
#               the pre-fix pipeline; labels never come from the run being scored)
#
# Then: per-field accuracy with block-bootstrap CIs (dev and held-out test),
# the 30 by-eye frames of uVf1EUszTIM, the encoded-obs agreement against the
# truth (live_vs_sim.py labels), the engine-vs-encoder check on constructed
# sim states (live_vs_sim.py sim), troop state and opponent plays.
# A run made before driver.py had state_filter.KingFloor (v3 and older) needs
#   live_ppo/eval/replay_towers.py SRC --run RUN --write && live_ppo/eval/remap_obs_rows.py SRC --run RUN
# first (that is how v3k was made from v3).
set -e
cd "$(dirname "$0")/../.."
RUN=${1:-v4o}; shift || true
PIPE=0; LAB=0
for a in "$@"; do
  case $a in --pipeline) PIPE=1 ;; --labels) LAB=1 ;; esac
done
SRCS="rec_dry1 vid_QD_1suWFNKo vid_uVf1EUszTIM vid_iVgmnDX-GPY"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
PY=.venv/bin/python
if [ $PIPE = 1 ]; then
  for s in $SRCS; do
    if [ "$s" = rec_dry1 ]; then fps=4; else fps=60; fi
    nice -n 5 $PY live_ppo/state_pipeline.py runs/live/$s --out runs/perception/$RUN/$s.state.jsonl \
      --yolo-cache runs/live/$s.yolo.jsonl --fps $fps --obs runs/perception/$RUN/$s.obs.npz \
      > runs/perception/$RUN/$s.log 2>&1 &
  done
  wait
fi
if [ $LAB = 1 ]; then
  for s in $SRCS; do $PY live_ppo/eval/build_labels.py $s --sheets | cut -c1-200; done
fi
$PY live_ppo/eval/state_accuracy.py --run $RUN --base base --out runs/perception/state_accuracy_$RUN.json
$PY live_ppo/eval/verify30.py --run $RUN | head -20
$PY live_ppo/eval/live_vs_sim.py labels --run $RUN
$PY live_ppo/eval/live_vs_sim.py sim | tail -25
# troop state (99 by-eye frames / 966 units, held-out) and opponent plays (gold labels)
$PY live_ppo/eval/eval_units.py | tail -30
$PY live_ppo/eval/eval_opp_plays.py --watch --tier gold | tail -20
