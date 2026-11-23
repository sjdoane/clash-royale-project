#!/bin/bash
# IL pipeline worker that claims videos (mkdir lock), so any number of workers
# can share one list: live_ppo/eval/il_queue2.sh VID [VID ...]
# Waits for each video's extraction (il_queue.sh extract), then runs the full
# state pipeline (production flags, encoded obs) into runs/perception/v2.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
O=runs/perception/v2
for v in "$@"; do
  while [ ! -f "runs/live/vid_$v.extract.done" ]; do sleep 60; done
  [ -f "$O/vid_$v.state.jsonl.done" ] && continue
  mkdir "$O/vid_$v.claim" 2>/dev/null || continue
  echo "$(date) pipeline $v"
  nice -n 10 .venv/bin/python live_ppo/state_pipeline.py "runs/live/vid_$v" \
    --out "$O/vid_$v.state.jsonl" --yolo-cache "runs/live/vid_$v.yolo.jsonl" \
    --fps 60 --obs "$O/vid_$v.obs.npz" > "$O/vid_$v.log" 2>&1 \
    && touch "$O/vid_$v.state.jsonl.done"
done
echo "$(date) worker done"
