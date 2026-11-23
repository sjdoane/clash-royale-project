#!/bin/bash
# IL video queue: extract live-format frames (+ YOLO side-car), then run the
# full state pipeline (production flags, encoded obs) over them. Low priority.
#   live_ppo/eval/il_queue.sh extract  VID [VID ...]   sequential extraction
#   live_ppo/eval/il_queue.sh pipeline VID [VID ...]   waits for each extraction
# Run one extract worker and one or two pipeline workers on disjoint lists.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
mode=$1; shift
for v in "$@"; do
  if [ "$mode" = extract ]; then
    [ -f "runs/live/vid_$v.extract.done" ] && continue
    echo "$(date) extract $v"
    rm -f "runs/live/vid_$v.yolo.jsonl"
    nice -n 10 .venv-det/bin/python live_ppo/eval/video_frames.py "$v" --t0 0 --t1 100000 \
      > "runs/perception/il/extract_$v.log" 2>&1 \
      && touch "runs/live/vid_$v.extract.done" || echo "extract $v failed"
  else
    while [ ! -f "runs/live/vid_$v.extract.done" ]; do sleep 60; done
    [ -f "runs/perception/v2/vid_$v.state.jsonl.done" ] && continue
    echo "$(date) pipeline $v"
    nice -n 10 .venv/bin/python live_ppo/state_pipeline.py "runs/live/vid_$v" \
      --out "runs/perception/v2/vid_$v.state.jsonl" --yolo-cache "runs/live/vid_$v.yolo.jsonl" \
      --fps 60 --obs "runs/perception/v2/vid_$v.obs.npz" > "runs/perception/v2/vid_$v.log" 2>&1 \
      && touch "runs/perception/v2/vid_$v.state.jsonl.done"
  fi
done
echo "$(date) $mode queue done"
