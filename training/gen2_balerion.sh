#!/usr/bin/env bash
# Gen-2 on balerion: its own tree (~/cr-robot-gen2), engine build and venv.
# Run ON balerion, from ~/cr-robot-gen2, after training/gen2_push.sh (Mac) has
# copied the tree over:
#
#   bash training/gen2_balerion.sh setup     # venv (same pins as the gen-1 venv) + engine build + tests
#   bash training/gen2_balerion.sh bc        # BC net_v2 at 1 Hz on the gen-2 dataset (GPU)
#   bash training/gen2_balerion.sh ppo       # league PPO, opponent level 11 -> 15, act delay
#   bash training/gen2_balerion.sh eval      # eval watcher: T=0.5, delay, levels 11/13/15 (CPU)
#   bash training/gen2_balerion.sh link      # gen2_* run dirs visible to the W&B sidecar
#   bash training/gen2_balerion.sh status
#
# ~/cr-robot-remote is only READ (its venv's package list and its CMake deps).
# The single write there is `link`: symlinks ~/cr-robot-remote/train/runs/gen2_*
# -> ~/cr-robot-gen2/runs/gen2_*, which is how the sidecar finds a run.
set -euo pipefail
cd "$(dirname "$0")/.."
G=$(pwd)
OLD=$HOME/cr-robot-remote/train
PY=$G/.venv/bin/python
BC_RUN=${BC_RUN:-gen2_bc_v2_1hz}
RUN=${RUN:-gen2_league}
NUM_ENVS=${NUM_ENVS:-}
mkdir -p runs checkpoints

gpu_ok() {
  # Sam's rule: the GPU only if EVERY process on it is ours (user sjdoane).
  # One process of any other user (jonat's included) and we do not launch;
  # their processes are never touched.
  command -v nvidia-smi >/dev/null || return 1
  me=$(id -un)
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    u=$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')
    [ -z "$u" ] && continue          # exited since the query
    if [ "$u" != "$me" ]; then
      echo "GPU process $p belongs to $u, not $me: not launching on the GPU"
      return 1
    fi
  done
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  echo "GPU free MiB: $free"
  [ "${free:-0}" -ge "${GPU_MIN_FREE_MIB:-6000}" ] || { echo "not enough GPU memory free"; return 1; }
}

pick_envs() {
  # Keep the 1-min load under ~20: each env worker is about one core.
  load=$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)
  n=$(( 19 - load - ${1:-0} ))
  [ $n -gt 16 ] && n=16
  [ $n -lt 4 ] && { echo "load $load too high for another job" >&2; exit 3; }
  echo $n
}

case "${1:-}" in
setup)
  if [ ! -x "$PY" ]; then
    "$OLD/.venv/bin/pip" freeze --exclude-editable > /tmp/gen2_req.txt
    python3.12 -m venv .venv
    .venv/bin/pip install -q --upgrade pip
    .venv/bin/pip install -q -r /tmp/gen2_req.txt --extra-index-url https://download.pytorch.org/whl/cu128
  fi
  deps=""
  for d in pybind11 catch2; do
    src=$(ls -d "$OLD"/engine/build_linux/_deps/${d}-src 2>/dev/null || true)
    [ -n "$src" ] && deps="$deps -DFETCHCONTENT_SOURCE_DIR_${d^^}=$src"
  done
  cmake -S engine -B engine/build_linux -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PY" $deps >/dev/null
  nice -n 10 cmake --build engine/build_linux -j 6 --target clash_royale_env ClashRoyaleTests
  cp engine/build_linux/clash_royale_env*.so engine/python_ai/
  "$PY" -c "import sys; sys.path.insert(0,'engine/python_ai'); import clash_royale_env as E; \
d=[125,168,129,15,40,72,33,7]; n=E.ClashRoyaleEnv(d, d).observation_size(); \
assert E.ClashRoyaleEnv.OPP_STATE_START + E.ClashRoyaleEnv.OPP_STATE_SIZE == n; \
assert n == 14459, n; print('gen-2 engine OK, obs', n)"
  (cd engine && "$PY" -m pytest -q python_ai/tests -p no:cacheprovider -x -k "seed_reproducible or opponent_cycle or scalar_encoder" | tail -2)
  "$PY" -m pytest -q training/tests -p no:cacheprovider | tail -2
  ;;
bc)
  gpu_ok || { echo "refusing the GPU: BC waits"; exit 2; }
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid -f nice -n 5 "$PY" -m training.bc_v2 train \
    --run "$BC_RUN" --data hog26 --minutes "${BC_MINUTES:-90}" --seed 26 \
    >> "runs/$BC_RUN.log" 2>&1 < /dev/null
  echo "BC launched -> runs/$BC_RUN.log"
  ;;
ppo)
  [ -f "checkpoints/$BC_RUN.pth" ] || { echo "no checkpoints/$BC_RUN.pth yet"; exit 2; }
  gpu_ok || { echo "refusing the GPU: PPO waits"; exit 2; }
  n=${NUM_ENVS:-$(pick_envs 4)}
  echo "num envs $n"
  # --fork-from "" and --snap-dirs "": gen-1 checkpoints and snapshots read a
  # 13977-wide observation and cannot play this engine.
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid -f "$PY" -m training.league \
    --run "$RUN" --arch net_v2 --fork-from "" --snap-dirs "" --init "checkpoints/$BC_RUN.pth" \
    --num-envs "$n" --lr 1e-4 --seed 26 --save-every 200 \
    --p-mirror 0.4 --pfsp-window 300 --act-delay 2,5,2 \
    --opp-level-lo 11 --opp-level-hi 15 --opp-level-p-cap 0.5 \
    --opp-level-every "${LEVEL_EVERY:-12000}" --opp-level-gate "${LEVEL_GATE:-0.3}" \
    --opp-level-window 400 \
    >> "runs/$RUN.log" 2>&1 < /dev/null
  echo "PPO launched -> runs/$RUN.log"
  ;;
eval)
  CUDA_VISIBLE_DEVICES="" setsid -f nice -n 5 "$PY" -m training.eval_panel --watch "$RUN" \
    --bc "checkpoints/$BC_RUN.pth" --every 4000 --procs 3 \
    --temp 0.5 --act-delay 2,5,2 --opp-levels 11,13,15 \
    >> "runs/${RUN}_eval.log" 2>&1 < /dev/null
  echo "eval watcher launched -> runs/${RUN}_eval.log"
  ;;
link)
  for r in "$BC_RUN" "$RUN"; do
    mkdir -p "runs/$r"
    [ -e "$OLD/runs/$r" ] || ln -s "$G/runs/$r" "$OLD/runs/$r"
    ls -ld "$OLD/runs/$r"
  done
  ;;
status)
  uptime
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader || true
  pgrep -af "cr-robot-gen2|training.league --run $RUN|bc_v2 train --run $BC_RUN|eval_panel --watch $RUN" | cut -c1-160 || true
  tail -n 2 "runs/$BC_RUN.log" "runs/$RUN.log" "runs/${RUN}_eval.log" 2>/dev/null || true
  ;;
*)
  sed -n 2,14p "$0"; exit 1;;
esac
