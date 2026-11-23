#!/usr/bin/env bash
# Round-2 unit-detector training on one RunPod GPU pod (199 classes: the 198 of round 1
# + deploy-pending). Idempotent and resumable like pod_train.sh: re-run the same command
# after any interruption and finished steps are skipped.
#
#   (pod)  mkdir -p /root/cr && tar -xf /root/pod_bundle_r2.tar -C /root/cr
#          nohup bash /root/cr/video_dataset/detector/pod_train_r2.sh --max-hours 9 > /root/pod_train.out 2>&1 &
#
# Steps (markers in $WORK):
#   1 pip install                          5 dataset lists + data yaml (names: classes_round2.json)
#   2 HF download SudhishNeel synth        6 touch $OUT/TRAINING_STARTED; train from round-1 best (time-boxed)
#   3 ext_convert.py (their ids -> ours)   7 held-out eval -> $OUT/metrics.json; touch $OUT/DONE
#   4 generate synth2 round-2 frames on the pod (CPU)
# Data: synth2 (our generator, exact labels, all classes) + SudhishNeel synth + CRDD real
# boxes x CRDD_REPEAT + real_r2/train (consensus-filtered real frames) x REAL_REPEAT.
# Val (best.pt selection only): real_r2/val (held-out videos) + a small synth2 val.
# Stopping is done from the Mac (training/pod_watch.sh + rp_api.sh): the pod can't reach the
# RunPod API, so there's no on-pod cost fuse here.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
OUT=${OUT:-/root/out}
WORK=${WORK:-/root/work}
MAX_HOURS=9
INIT=""                 # default: data/det/init/r1_best.pt (round-1 v3 best)
IMGSZ=1024
BATCH=16
EPOCHS=100              # upper bound; `time=` sets the real length
CRDD_REPEAT=2
REAL_REPEAT=4
SYNTH2_N=200000
SYNTH2_VAL_N=1000
DEVICE=0
WORKERS=12              # the pod's cgroup CPU limit is ~12.75 (nproc shows the host's 120)
INSTALL=1
EVAL_MARGIN_H=0.3

while [[ $# -gt 0 ]]; do
  case $1 in
    --max-hours) MAX_HOURS=$2; shift 2;;
    --init) INIT=$2; shift 2;;
    --imgsz) IMGSZ=$2; shift 2;;
    --batch) BATCH=$2; shift 2;;
    --epochs) EPOCHS=$2; shift 2;;
    --crdd-repeat) CRDD_REPEAT=$2; shift 2;;
    --real-repeat) REAL_REPEAT=$2; shift 2;;
    --synth2-n) SYNTH2_N=$2; shift 2;;
    --device) DEVICE=$2; shift 2;;
    --workers) WORKERS=$2; shift 2;;
    --no-install) INSTALL=0; shift;;
    *) echo "unknown arg $1"; exit 1;;
  esac
done

mkdir -p "$OUT" "$WORK"
exec > >(tee -a "$OUT/pod_train.log") 2>&1
log() { echo "[$(date '+%F %T')] $*"; }
PY=${PY:-python}
cd "$HERE"
[[ -f "$WORK/start_ts" ]] || date +%s > "$WORK/start_ts"
START=$(cat "$WORK/start_ts")
elapsed() { echo $(( $(date +%s) - START )); }
[[ -n $INIT ]] || INIT=$ROOT/data/det/init/r1_best.pt
[[ -f $INIT ]] || { log "init weights $INIT missing"; exit 1; }
log "ROOT=$ROOT max_hours=$MAX_HOURS init=$INIT imgsz=$IMGSZ batch=$BATCH synth2_n=$SYNTH2_N workers=$WORKERS"

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
  log "HF download (synthetic_arena/*)"
  $PY - "$SN" <<'EOF'
import sys
from huggingface_hub import snapshot_download
sn = sys.argv[1]
for i in range(5):
    try:
        snapshot_download("SudhishNeel/clash-royale-detection", repo_type="dataset", local_dir=f"{sn}/dataset",
                          allow_patterns=["synthetic_arena/*", "README.md", "LABELLING.md", "SPEC.md", "classes.txt"],
                          max_workers=16)
        break
    except Exception as e:
        print("download retry", i, repr(e)[:200])
else:
    raise SystemExit("HF download failed")
EOF
  touch "$WORK/.downloaded"
fi

# ---- 3 convert (their ids -> our 198; deploy-pending = 198 is never in their data) -------
if [[ ! -f "$WORK/.converted" ]]; then
  log "ext_convert --extend"
  $PY ext_convert.py --workers "$WORKERS" --extend
  touch "$WORK/.converted"
fi

# ---- 4 synth2 generation on the pod -----------------------------------------------------
# Interface (written by the synth2 agent): data/det/synth2_bundle/gen_r2.sh <n> <seed> <workers> <out_dir>
# writes <out_dir>/images/*.jpg + <out_dir>/labels/*.txt in classes_round2.json ids.
S2=$ROOT/data/det/synth2_r2
S2B=$ROOT/data/det/synth2_bundle
if [[ ! -f "$WORK/.synth2" ]]; then
  if [[ ! -f "$S2B/gen_r2.sh" ]]; then
    log "unpack synth2 bundle"
    tar -xf "$S2B/synth2_r2_bundle.tar" -C "$S2B"
  fi
  log "synth2: $SYNTH2_N train + $SYNTH2_VAL_N val frames"
  rm -rf "$S2/train" "$S2/val"
  PYTHON=$(command -v "$PY") bash "$S2B/gen_r2.sh" "$SYNTH2_N" 1 "$WORKERS" "$S2/train"
  PYTHON=$(command -v "$PY") bash "$S2B/gen_r2.sh" "$SYNTH2_VAL_N" 777 "$WORKERS" "$S2/val"
  n=$(find "$S2/train/images" -name '*.jpg' | wc -l)
  (( n >= SYNTH2_N * 99 / 100 )) || { log "synth2 made only $n of $SYNTH2_N frames"; exit 1; }
  touch "$WORK/.synth2"
fi

# ---- 5 dataset lists + yaml --------------------------------------------------------------
if [[ ! -f "$WORK/data.yaml" ]]; then
  log "dataset yaml"
  $PY - "$ROOT" "$WORK" "$CRDD_REPEAT" "$REAL_REPEAT" <<'EOF'
import json, sys
from pathlib import Path
import yaml
root, work, crep, rrep = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
det = root / "data" / "det"
names = json.load(open(det / "classes_round2.json"))
assert len(names) == 199 and names[198] == "deploy-pending", (len(names), names[-1])
imgs = lambda d: sorted(str(p) for p in (det / d / "images").glob("*.jpg"))
rimgs = lambda s: sorted(str(p) for p in (det / "real_r2" / "images" / s).glob("*.jpg"))
s2, s2v = imgs("synth2_r2/train"), imgs("synth2_r2/val")
sn, crdd = imgs("ext_synth"), imgs("yolo/crdd")
rtr, rva = rimgs("train"), rimgs("val")
for k, v in dict(synth2=s2, sn=sn, crdd=crdd, real_train=rtr, real_val=rva).items():
    assert v, f"empty: {k}"
train = s2 + sn + crdd * crep + rtr * rrep
val = rva + s2v
(work / "train.txt").write_text("\n".join(train) + "\n")
(work / "val.txt").write_text("\n".join(val) + "\n")
(work / "data.yaml").write_text(yaml.safe_dump({"path": str(root), "train": str(work / "train.txt"),
                                                "val": str(work / "val.txt"), "names": dict(enumerate(names))},
                                               sort_keys=False))
info = {"n_classes": len(names), "train_list": len(train), "synth2": len(s2), "sn": len(sn), "crdd": len(crdd),
        "crdd_repeat": crep, "real_train": len(rtr), "real_repeat": rrep, "val_real": len(rva), "val_synth2": len(s2v)}
(work / "dataset_info.json").write_text(json.dumps(info, indent=1))
print(info)
EOF
  cp "$WORK/data.yaml" "$WORK/dataset_info.json" "$OUT/"
fi

# ---- 6 train ----------------------------------------------------------------------------
W=$OUT/train/weights
if [[ ! -f "$WORK/.trained" ]]; then
  TRAIN_H=$(awk -v h="$MAX_HOURS" -v e="$(elapsed)" -v m="$EVAL_MARGIN_H" 'BEGIN{t=h-e/3600-m; printf "%.3f", (t>0.05?t:0.05)}')
  log "train: ${TRAIN_H} h, init=$INIT (cache off: the list is far bigger than RAM)"
  touch "$OUT/TRAINING_STARTED"
  TIMEOUT_S=$(awk -v t="$TRAIN_H" 'BEGIN{print int(t*3600+600)}')
  set +e
  timeout --signal=TERM --kill-after=120 "$TIMEOUT_S" $PY - "$WORK/data.yaml" "$OUT" "$INIT" "$IMGSZ" "$BATCH" \
      "$EPOCHS" "$TRAIN_H" "$DEVICE" "$WORKERS" <<'EOF'
import sys
from pathlib import Path
from ultralytics import YOLO
data, out, init, imgsz, batch, epochs, hours, device, workers = sys.argv[1:]
last = Path(out) / "train" / "weights" / "last.pt"
if last.exists():
    import torch
    ck = torch.load(last, map_location="cpu", weights_only=False)
    if ck.get("optimizer") is None:
        print("last.pt is a finished run (no optimizer state): not training again")
    else:
        print("resuming from", last, "epoch", ck.get("epoch"))
        YOLO(str(last)).train(resume=True)
else:
    # 198 -> 199 classes: ultralytics transfers every matching tensor and re-inits the class
    # logits of the detect head, so round-1 features and boxes carry over.
    YOLO(init).train(
        data=data, epochs=int(epochs), time=float(hours), imgsz=int(imgsz), batch=int(batch), device=int(device),
        workers=int(workers), cache=False, amp=True, project=out, name="train", exist_ok=True,
        save_period=-1, plots=False, val=True, patience=100, warmup_epochs=1.0,
        mosaic=0.5, close_mosaic=2, fliplr=0.5, degrees=0, scale=0.25, translate=0.05,
        hsv_h=0.01, hsv_s=0.4, hsv_v=0.3, cos_lr=True)
EOF
  rc=$?
  set -e
  log "train exit code $rc"
  [[ -f "$W/last.pt" ]] || { log "no last.pt: training failed"; exit 1; }
  touch "$WORK/.trained"
fi

# ---- 7 eval + DONE ----------------------------------------------------------------------
if [[ ! -f "$OUT/DONE" ]]; then
  BEST=$W/best.pt; [[ -f "$BEST" ]] || BEST=$W/last.pt
  log "held-out eval of $BEST"
  $PY pod_eval.py score --device "cuda:$DEVICE" --out "$OUT/metrics.json" \
    --model "r2_best=$BEST:ours:$IMGSZ" \
    --model "r2_last=$W/last.pt:ours:$IMGSZ" \
    --model "r1_best=$INIT:ours:$IMGSZ" \
    --model "baseline_v2=$ROOT/data/det/runs/v2/weights/best.pt:ours:896" || log "eval failed (weights are safe)"
  log "elapsed $(( $(elapsed) / 60 )) min"
  touch "$OUT/DONE"
fi
log "all done"
