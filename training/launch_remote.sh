#!/usr/bin/env bash
# Run ON balerion from ~/cr-robot-remote/train (or on the Mac from the repo root with
# NUM_ENVS=8). Refuses the GPU if a compute process not started from ~/cr-robot-remote/
# is running (Sam's rule); the trainer then falls back to CPU via CUDA_VISIBLE_DEVICES="".
#   RUN=ppo_bc26 INIT=checkpoints/bc_hog26.pth NUM_ENVS=16 bash training/launch_remote.sh
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:-ppo_bc26}
INIT=${INIT:-checkpoints/bc_hog26.pth}
NUM_ENVS=${NUM_ENVS:-16}
PY=${PY:-.venv/bin/python}
mkdir -p runs checkpoints
if command -v nvidia-smi >/dev/null; then
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    cwd=$(readlink /proc/$p/cwd || echo unknown)
    case "$cwd" in
      "$HOME"/cr-robot-remote/*) ;;
      *) echo "foreign GPU process $p ($cwd): not using the GPU"; export CUDA_VISIBLE_DEVICES="";;
    esac
  done
fi
nohup setsid $PY -m training.ppo_bc --run "$RUN" --init "$INIT" --num-envs "$NUM_ENVS" \
  --lr 5e-5 --seed 26 --save-every 200 ${EXTRA:-} >> "runs/$RUN.log" 2>&1 < /dev/null &
echo "trainer PID $!"
CUDA_VISIBLE_DEVICES="" nohup setsid $PY -m training.eval_panel --watch "$RUN" --bc "$INIT" \
  --every 2000 --procs 3 >> "runs/${RUN}_eval.log" 2>&1 < /dev/null &
echo "eval PID $!"
