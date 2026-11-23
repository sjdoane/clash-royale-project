#!/usr/bin/env bash
# Unit-detector retrain on one RunPod GPU pod. Idempotent and resumable: re-run the
# same command after any interruption and finished steps are skipped.
#
#   (pod)  mkdir -p /root/cr && tar -xf /root/pod_bundle.tar -C /root/cr
#          nohup bash /root/cr/video_dataset/detector/pod_train.sh --max-hours 10 > /root/pod_train.out 2>&1 &
#
# Steps (markers in $WORK):
#   1 pip install (pinned ultralytics) + libGL      4 build train/val lists + data yaml
#   2 HF download: SudhishNeel synthetic_arena,     5 touch $OUT/TRAINING_STARTED; train (time-boxed)
#     README, their detector weights                6 held-out eval -> $OUT/metrics.json; touch $OUT/DONE
#   3 ext_convert.py (their ids -> ours)
# Contract with training/pod_watch.sh: everything the Mac needs is in $OUT (/root/out),
# kept small (train/weights/{last,best}.pt, results.csv, logs, metrics). Cost fuse: the
# pod stops itself MAX_HOURS + 1.5 h after the FIRST start of this script.
#
# Data: their synthetic_arena (105k real-pixel synthetic frames, converted by
# ext_convert.py) + our real frames (CRDD part2 boxes, data/det/yolo/crdd) repeated
# REAL_REPEAT times. Val (used only for best.pt selection): our real_dev frames + their
# synth val backdrops. Final numbers: pod_eval.py on the 30 held-out uVf1EUszTIM frames.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)                      # bundle root (repo layout)
OUT=${OUT:-/root/out}
WORK=${WORK:-/root/work}
MAX_HOURS=10            # wall-clock budget for the whole script (setup + train + eval)
ARCH=yolov8m.pt         # COCO init; yolov8s.pt -> init from our v2 instead (see below)
IMGSZ=1024
BATCH=16                # YOLOv8m @1024 with AMP: ~22 GB on a 32 GB card
EPOCHS=100              # upper bound only; ultralytics `time=` sets the real length
EXTEND=1                # 1: append their 7 unmapped unit classes (198 total); 0: strict 191, paint out
REAL_REPEAT=3
DEVICE=0
WORKERS=$(nproc 2>/dev/null || sysctl -n hw.ncpu)
CONVERT_LIMIT=""        # frames per shard (dry run only)
HF_SYNTH="synthetic_arena/*"
FRACTION=1.0
INSTALL=1
FUSE=1
EVAL_MARGIN_H=0.3       # reserved at the end for eval + final pull

while [[ $# -gt 0 ]]; do
  case $1 in
    --max-hours) MAX_HOURS=$2; shift 2;;
    --arch) ARCH=$2; shift 2;;
    --imgsz) IMGSZ=$2; shift 2;;
    --batch) BATCH=$2; shift 2;;
    --epochs) EPOCHS=$2; shift 2;;
    --extend) EXTEND=$2; shift 2;;
    --real-repeat) REAL_REPEAT=$2; shift 2;;
    --device) DEVICE=$2; shift 2;;
    --workers) WORKERS=$2; shift 2;;
    --convert-limit) CONVERT_LIMIT=$2; shift 2;;
    --hf-synth) HF_SYNTH=$2; shift 2;;
    --fraction) FRACTION=$2; shift 2;;
    --no-install) INSTALL=0; shift;;
    --no-fuse) FUSE=0; shift;;
    *) echo "unknown arg $1"; exit 1;;
  esac
done

mkdir -p "$OUT" "$WORK"
exec > >(tee -a "$OUT/pod_train.log") 2>&1
log() { echo "[$(date '+%F %T')] $*"; }
PY=${PY:-python}
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
cd "$HERE"

# ---- cost fuse: started once, timed from the first start ------------------------------
[[ -f "$WORK/start_ts" ]] || date +%s > "$WORK/start_ts"
START=$(cat "$WORK/start_ts")
elapsed() { echo $(( $(date +%s) - START )); }
if [[ $FUSE == 1 ]]; then
  FUSE_S=$(awk -v h="$MAX_HOURS" -v e="$(elapsed)" 'BEGIN{s=int(h*3600+5400-e); print (s>60?s:60)}')
  if [[ -f "$WORK/fuse.pid" ]] && kill -0 "$(cat "$WORK/fuse.pid")" 2>/dev/null; then
    log "cost fuse already armed (pid $(cat "$WORK/fuse.pid"))"
  else
    nohup bash -c "sleep $FUSE_S; runpodctl pod stop \"\$RUNPOD_POD_ID\" || runpodctl stop pod \"\$RUNPOD_POD_ID\"" \
      > "$WORK/fuse.log" 2>&1 &
    echo $! > "$WORK/fuse.pid"
    log "cost fuse armed: pod stops in ${FUSE_S}s"
  fi
fi
log "ROOT=$ROOT OUT=$OUT max_hours=$MAX_HOURS arch=$ARCH imgsz=$IMGSZ batch=$BATCH extend=$EXTEND workers=$WORKERS"

# ---- 1 install --------------------------------------------------------------------------
if [[ $INSTALL == 1 && ! -f "$WORK/.installed" ]]; then
  log "install"
  if ! ldconfig -p 2>/dev/null | grep -q libGL.so.1; then
    (apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0) || log "apt libGL failed (opencv may still work)"
  fi
  pip install -q "ultralytics==8.4.215" "huggingface_hub>=0.34" hf_xet pyarrow pyyaml
  $PY -c "import ultralytics, cv2, torch; print('ultralytics', ultralytics.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  touch "$WORK/.installed"
fi

# ---- 2 download -------------------------------------------------------------------------
SN=$ROOT/data/ext/sudhishneel
if [[ ! -f "$WORK/.downloaded" ]]; then
  log "HF download ($HF_SYNTH)"
  $PY - "$SN" "$HF_SYNTH" <<'EOF'
import sys
from huggingface_hub import snapshot_download
sn, synth = sys.argv[1], sys.argv[2]
for i in range(5):
    try:
        snapshot_download("SudhishNeel/clash-royale-detection", repo_type="dataset", local_dir=f"{sn}/dataset",
                          allow_patterns=[synth, "README.md", "LABELLING.md", "SPEC.md", "classes.txt"], max_workers=16)
        snapshot_download("SudhishNeel/clash-royale-detector", local_dir=f"{sn}/model",
                          allow_patterns=["v10/*", "v8/*", "README.md"], max_workers=8)
        break
    except Exception as e:  # transient HF errors: retry, already-complete files are skipped
        print("download retry", i, repr(e)[:200])
else:
    raise SystemExit("HF download failed")
EOF
  touch "$WORK/.downloaded"
fi

# ---- 3 convert --------------------------------------------------------------------------
CONV_TAG="extend=$EXTEND limit=${CONVERT_LIMIT:-all} synth=$HF_SYNTH"
if [[ "$(cat "$WORK/.converted" 2>/dev/null)" != "$CONV_TAG" ]]; then
  log "ext_convert ($CONV_TAG)"
  args=(--workers "$WORKERS")
  [[ $EXTEND == 1 ]] && args+=(--extend)
  [[ -n $CONVERT_LIMIT ]] && args+=(--limit "$CONVERT_LIMIT")
  $PY ext_convert.py "${args[@]}"
  cp "$ROOT/data/det/ext_synth/convert_stats.json" "$OUT/convert_stats.json"
  echo "$CONV_TAG" > "$WORK/.converted"
  rm -f "$WORK/data.yaml"
fi

# ---- 4 dataset lists + yaml, cache mode -------------------------------------------------
if [[ ! -f "$WORK/data.yaml" || ! -f "$WORK/cache_mode" ]]; then
  log "dataset yaml"
  $PY - "$ROOT" "$WORK" "$EXTEND" "$REAL_REPEAT" "$IMGSZ" <<'EOF'
import json, os, shutil, sys
from pathlib import Path
import yaml
root, work, extend, rep, imgsz = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] == "1", int(sys.argv[4]), int(sys.argv[5])
sys.path.insert(0, str(root / "video_dataset" / "detector"))
from ext_classmap import target_classes
names = target_classes(extend)
det = root / "data" / "det"
imgs = lambda d: sorted(str(p) for p in (det / d / "images").glob("*.jpg"))
synth, crdd, dev, sval = imgs("ext_synth"), imgs("yolo/crdd"), imgs("yolo/real_dev"), imgs("ext_synth_val")
train = synth + crdd * rep
val = dev + sval[:1500]
(work / "train.txt").write_text("\n".join(train) + "\n")
(work / "val.txt").write_text("\n".join(val) + "\n")
(work / "data.yaml").write_text(yaml.safe_dump({"path": str(root), "train": str(work / "train.txt"),
                                                "val": str(work / "val.txt"),
                                                "names": dict(enumerate(names))}, sort_keys=False))
# ultralytics caches the image resized to imgsz on the long side (568x896 -> ~650x1024x3)
per = imgsz * round(imgsz * 568 / 896) * 3
need = len(train) * per
mem = next(int(l.split()[1]) * 1024 for l in open("/proc/meminfo") if l.startswith("MemAvailable")) if os.path.exists("/proc/meminfo") else 0
disk = shutil.disk_usage(det).free
# a container sees the host's /proc/meminfo; the real limit is the cgroup's
for cg in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
    try:
        lim = open(cg).read().strip()
        if lim.isdigit():
            mem = min(mem, int(lim)) if mem else int(lim)
    except OSError:
        pass
cache = "ram" if need < 0.6 * mem else "disk" if need < disk - 30e9 else "none"
info = {"n_classes": len(names), "train_list": len(train), "synth": len(synth), "crdd": len(crdd), "crdd_repeat": rep,
        "val_list": len(val), "real_dev": len(dev), "synth_val_used": min(len(sval), 1500),
        "cache_need_gb": round(need / 1e9, 1), "mem_avail_gb": round(mem / 1e9, 1), "disk_free_gb": round(disk / 1e9, 1),
        "cache": cache}
(work / "cache_mode").write_text(cache)
(work / "dataset_info.json").write_text(json.dumps(info, indent=1))
print(info)
EOF
  cp "$WORK/data.yaml" "$WORK/dataset_info.json" "$OUT/"
fi
CACHE=$(cat "$WORK/cache_mode")

# ---- 5 train ----------------------------------------------------------------------------
W=$OUT/train/weights
if [[ ! -f "$WORK/.trained" ]]; then
  TRAIN_H=$(awk -v h="$MAX_HOURS" -v e="$(elapsed)" -v m="$EVAL_MARGIN_H" 'BEGIN{t=h-e/3600-m; printf "%.3f", (t>0.05?t:0.05)}')
  INIT=$ARCH
  [[ $ARCH == yolov8s.pt && -f "$ROOT/data/det/runs/v2/weights/best.pt" ]] && INIT=$ROOT/data/det/runs/v2/weights/best.pt
  log "train: ${TRAIN_H} h left for training, init=$INIT cache=$CACHE"
  touch "$OUT/TRAINING_STARTED"
  # Hard stop slightly after the budget even on resume (ultralytics restarts its own
  # `time` clock on resume): SIGTERM (SIGINT is ignored under `nohup ... &`), last.pt of the
  # previous epoch stays with its optimizer state, so a rerun resumes.
  TIMEOUT_S=$(awk -v t="$TRAIN_H" 'BEGIN{print int(t*3600+600)}')
  TO=(); command -v timeout >/dev/null && TO=(timeout --signal=TERM --kill-after=120 "$TIMEOUT_S")
  set +e
  ${TO[@]+"${TO[@]}"} $PY - "$WORK/data.yaml" "$OUT" "$INIT" "$IMGSZ" "$BATCH" \
      "$EPOCHS" "$TRAIN_H" "$DEVICE" "$WORKERS" "$CACHE" "$FRACTION" <<'EOF'
import sys
from pathlib import Path
from ultralytics import YOLO
data, out, init, imgsz, batch, epochs, hours, device, workers, cache, fraction = sys.argv[1:]
last = Path(out) / "train" / "weights" / "last.pt"
if last.exists():
    import torch
    ck = torch.load(last, map_location="cpu", weights_only=False)
    if ck.get("optimizer") is None:
        # ultralytics strips the optimizer when a run completes; resume=True on such a
        # file silently starts a NEW run with default args, so treat it as finished.
        print("last.pt is a finished run (no optimizer state): not training again")
    else:
        print("resuming from", last, "epoch", ck.get("epoch"))
        YOLO(str(last)).train(resume=True)
else:
    dev = device if device in ("cpu", "mps") else int(device)
    YOLO(init).train(
        data=data, epochs=int(epochs), time=float(hours), imgsz=int(imgsz), batch=int(batch), device=dev,
        workers=int(workers), cache={"none": False}.get(cache, cache), amp=True, fraction=float(fraction),
        project=out, name="train", exist_ok=True, save_period=-1, plots=False, val=True, patience=100,
        # augmentation as train_det.py (arena geometry: no rotation, mild scale)
        mosaic=0.5, close_mosaic=3, fliplr=0.5, degrees=0, scale=0.25, translate=0.05,
        hsv_h=0.01, hsv_s=0.4, hsv_v=0.3, cos_lr=True)
EOF
  rc=$?
  set -e
  log "train exit code $rc"
  [[ -f "$W/last.pt" ]] || { log "no last.pt: training failed"; exit 1; }
  touch "$WORK/.trained"
fi

# ---- 6 eval + DONE ----------------------------------------------------------------------
if [[ ! -f "$OUT/DONE" ]]; then
  BEST=$W/best.pt; [[ -f "$BEST" ]] || BEST=$W/last.pt
  EDEV=$DEVICE; [[ $EDEV =~ ^[0-9]+$ ]] && EDEV=cuda:$EDEV
  log "held-out eval of $BEST"
  $PY pod_eval.py score --device "$EDEV" --out "$OUT/metrics.json" \
    --model "new_best=$BEST:ours:$IMGSZ" \
    --model "new_last=$W/last.pt:ours:$IMGSZ" \
    --model "new_best_theirframe=$BEST:theirs:$IMGSZ" \
    --model "baseline_v2=$ROOT/data/det/runs/v2/weights/best.pt:ours:896" \
    --model "sn_v10=$SN/model/v10/best.pt:theirs:800"
  log "elapsed $(( $(elapsed) / 60 )) min"
  touch "$OUT/DONE"
fi
log "all done"
