#!/usr/bin/env bash
# Round-3 unit-detector training on one RunPod GPU pod: 198 classes (the round-1 set; no
# deploy-pending), started from the best detector so far (A/B arm B), with that run's recipe.
#
#   (pod)  mkdir -p /root/cr && tar -xf /root/pod_bundle_r3.tar -C /root/cr
#          nohup bash /root/cr/video_dataset/detector/pod_train_r3.sh --stage ab > /root/pod_train.out 2>&1 &
#   (Mac)  training/pod_watch.sh <host> <port> runs/det_r3_pod <pod_id>     # pulls, cost fuses, terminates on DONE
#          training/det_r3_run.sh <host> <port> runs/det_r3_pod             # scores the arms (det_check.py),
#                                    # starts `--stage main --data <pick>`, stops a run that fails its early check
#
# What round 2 got wrong, and what this script does instead:
#   * a unit under a deploy stopwatch was unlabelled (synth2 and real_r2 alike)
#       -> synth2_r2.py --deploy unit: the unit keeps its label, the watch has no box
#   * 199 classes re-initialised the class logits of the detect head
#       -> 198 classes, same ids as the init weights: every tensor transfers (asserted below)
#   * real_r2's labels miss ~23 % of the visible units (taught as background)
#       -> real_r2 is not used. data/det/real_r3 (recall-first relabel) is an optional A/B arm
#   * optimizer=auto ran SGD at lr0 0.01 for 9 h
#       -> optimizer=SGD lr0=0.002 written out (the A/B recipe)
#   * human-labelled real frames (CRDD) were 4 % of the images; in arm B they were 32 %
#       -> CRDD is repeated to stay near a third of every list
#   * best.pt was picked on real_r2/val (same label flaws), nothing was checked for 9 h
#       -> val=False, a checkpoint every epoch; the Mac scores them on our by-eye frames
#
# Stages (markers in $WORK; re-running a stage skips what is finished):
#   common  1 pip install   2 synth2 frames (CPU, on the pod)   3 lists + yaml
#   ab      ARM_HOURS per arm from the same init, same list size:
#             ctl    synth2 + CRDD
#             dense  ctl with BUSY_SHARE of the synth2 frames swapped for busy ones (8-18 plays;
#                    the default frames average 4 units, our busy real frames 10+)
#             real   ctl + real_r3 (only if data/det/real_r3 is in the bundle)
#           Writes $OUT/AB_DONE. training/det_check.py on the Mac decides which arms helped.
#   main    --data ctl|dense|real|densereal, time-boxed by --max-hours. Writes $OUT/DONE.
# Stopping is done from the Mac (training/pod_watch.sh + rp_api.sh): the pod can't reach the
# RunPod API. pod_watch.sh terminates the pod on $OUT/DONE and stops it after 30 idle minutes.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
OUT=${OUT:-/root/out}
WORK=${WORK:-/root/work}
STAGE=""
DATA=ctl
MAX_HOURS=2.5           # main stage: training time (not counting setup)
ARM_HOURS=0.55          # ab stage: per arm (arm B of round 2 ran 0.55 h)
INIT=""                 # default: data/det/init/r3_init.pt (arm B, optimizer stripped)
IMGSZ=1024
BATCH=16
SYNTH2_N=60000          # unique synth2 frames; the ab arms use the first AB_SYNTH2_N of them
AB_SYNTH2_N=30000
BUSY_SHARE=0.3          # dense lists: this share of the synth2 frames are busy ones
CRDD_REPEAT_AB=2        # 13.9k CRDD of a 43.9k list = 32 %, as arm B
CRDD_REPEAT_MAIN=4      # 27.9k CRDD of an 87.9k list = 32 %
REAL_REPEAT_AB=1
REAL_REPEAT_MAIN=2
LR0=0.002
DEVICE=0
WORKERS=12              # the pod's cgroup CPU limit is ~12.75 (nproc shows the host's 120)
INSTALL=1
ARMS_ARG=""             # ab stage: arms to run (default: ctl dense, + real if it is in the bundle)

while [[ $# -gt 0 ]]; do
  case $1 in
    --stage) STAGE=$2; shift 2;;
    --data) DATA=$2; shift 2;;
    --max-hours) MAX_HOURS=$2; shift 2;;
    --arm-hours) ARM_HOURS=$2; shift 2;;
    --init) INIT=$2; shift 2;;
    --imgsz) IMGSZ=$2; shift 2;;
    --batch) BATCH=$2; shift 2;;
    --synth2-n) SYNTH2_N=$2; shift 2;;
    --ab-synth2-n) AB_SYNTH2_N=$2; shift 2;;
    --busy-share) BUSY_SHARE=$2; shift 2;;
    --arms) ARMS_ARG=$2; shift 2;;
    --lr0) LR0=$2; shift 2;;
    --device) DEVICE=$2; shift 2;;
    --workers) WORKERS=$2; shift 2;;
    --no-install) INSTALL=0; shift;;
    *) echo "unknown arg $1"; exit 1;;
  esac
done
[[ $STAGE == ab || $STAGE == main || $STAGE == prep ]] || { echo "--stage prep|ab|main"; exit 1; }
[[ $DATA == ctl || $DATA == dense || $DATA == real || $DATA == densereal ]] || { echo "--data ctl|dense|real|densereal"; exit 1; }
BUSY_N=$(awk -v n="$SYNTH2_N" -v s="$BUSY_SHARE" 'BEGIN{print int(n*s+0.5)}')
AB_BUSY_N=$(awk -v n="$AB_SYNTH2_N" -v s="$BUSY_SHARE" 'BEGIN{print int(n*s+0.5)}')

mkdir -p "$OUT" "$WORK"
exec > >(tee -a "$OUT/pod_train.log") 2>&1
log() { echo "[$(date '+%F %T')] $*"; }
PY=${PY:-python}
cd "$HERE"
[[ -n $INIT ]] || INIT=$ROOT/data/det/init/r3_init.pt
[[ -f $INIT ]] || { log "init weights $INIT missing"; exit 1; }
HAVE_REAL=0
[[ -d $ROOT/data/det/real_r3/images/train ]] && HAVE_REAL=1
log "stage=$STAGE data=$DATA init=$INIT imgsz=$IMGSZ batch=$BATCH synth2_n=$SYNTH2_N lr0=$LR0 real_r3=$HAVE_REAL"
[[ $DATA == *real && $HAVE_REAL == 0 ]] && { log "--data $DATA but data/det/real_r3 is not in the bundle"; exit 1; }

# ---- 1 install --------------------------------------------------------------------------
if [[ $INSTALL == 1 && ! -f "$WORK/.installed" ]]; then
  log "install"
  need=""
  ldconfig -p 2>/dev/null | grep -q libGL.so.1 || need="libgl1 libglib2.0-0"
  command -v rsync >/dev/null || need="$need rsync"      # pod_watch.sh pulls with rsync
  [[ -z $need ]] || (apt-get update -qq && apt-get install -y -qq $need) || log "apt install failed: $need"
  pip install -q "ultralytics==8.4.215" pyyaml
  $PY -c "import ultralytics, cv2, torch; print('ultralytics', ultralytics.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  touch "$WORK/.installed"
fi
free_gb=$(df -BG --output=avail /root | tail -1 | tr -dc 0-9)
log "disk free under /root: ${free_gb} GB"
(( free_gb >= 15 )) || { log "less than 15 GB free: synth2 frames + checkpoints will not fit"; exit 1; }

# ---- 2 synth2 generation on the pod -----------------------------------------------------
# The generator in the synth2 tar is the round-2 one; the patched copy next to this script
# (--deploy unit) replaces it. Images are byte-identical to round 2's for the same seed and
# index; only the labels differ (units under a watch labelled, no class 198).
S2=$ROOT/data/det/synth2_r3
S2B=$ROOT/data/det/synth2_bundle
if [[ ! -f "$WORK/.synth2" ]]; then
  if [[ ! -d "$S2B/synth2" ]]; then
    log "unpack synth2 bundle"
    tar -xf "$S2B/synth2_r2_bundle.tar" -C "$S2B"
  fi
  cp "$HERE/synth2_r2.py" "$S2B/synth2_r2.py"
  grep -q -- '--deploy' "$S2B/synth2_r2.py" && grep -q -- '--busy' "$S2B/synth2_r2.py" \
      || { log "generator has no --deploy / --busy option"; exit 1; }
  log "synth2: $SYNTH2_N train frames"
  rm -rf "$S2/train"
  SYNTH2_ROOT="$S2B/synth2" $PY -I "$S2B/synth2_r2.py" --n "$SYNTH2_N" --seed 1 --workers "$WORKERS" \
      --out "$S2/train" --split train --deploy unit
  n=$(find "$S2/train/images" -name '*.jpg' | wc -l)
  (( n >= SYNTH2_N * 99 / 100 )) || { log "synth2 made only $n of $SYNTH2_N frames"; exit 1; }
  log "synth2: $BUSY_N busy frames"
  rm -rf "$S2/busy"
  SYNTH2_ROOT="$S2B/synth2" $PY -I "$S2B/synth2_r2.py" --n "$BUSY_N" --seed 2 --workers "$WORKERS" \
      --out "$S2/busy" --split train --deploy unit --busy 1.0
  n=$(find "$S2/busy/images" -name '*.jpg' | wc -l)
  (( n >= BUSY_N * 99 / 100 )) || { log "synth2 made only $n of $BUSY_N busy frames"; exit 1; }
  touch "$WORK/.synth2"
fi

# ---- 3 dataset lists + yaml --------------------------------------------------------------
if [[ ! -f "$WORK/.lists" ]]; then
  log "dataset lists"
  $PY - "$ROOT" "$WORK" "$AB_SYNTH2_N" "$CRDD_REPEAT_AB" "$CRDD_REPEAT_MAIN" "$REAL_REPEAT_AB" "$REAL_REPEAT_MAIN" \
       "$AB_BUSY_N" "$BUSY_N" <<'EOF'
import json, sys
from collections import Counter
from pathlib import Path
import yaml
root, work = Path(sys.argv[1]), Path(sys.argv[2])
ab_n, c_ab, c_main, r_ab, r_main, ab_busy, busy_n = map(int, sys.argv[3:10])
det = root / "data" / "det"
names = json.load(open(det / "classes_round2.json"))
assert len(names) == 199 and names[198] == "deploy-pending", (len(names), names[-1])
names = names[:198]
HELD_OUT = ("uVf1EUszTIM", "iVgmnDX-GPY", "QD_1suWFNKo", "rec_dry1", "LekrcVeQjiw", "kaLMeKpsuGY")


def imgs(d):
    out = sorted(str(p) for p in (det / d).glob("*.jpg"))
    bad = [p for p in out if any(h in p for h in HELD_OUT)]
    assert not bad, f"held-out frames in {d}: {bad[:3]}"
    return out


def check_labels(name, images, sample=4000):
    """every sampled label row: class 0..197, box inside the frame; returns boxes/image."""
    step = max(len(images) // sample, 1)
    n = rows = 0
    cls = Counter()
    for p in images[::step]:
        lp = Path(p.replace("/images/", "/labels/")).with_suffix(".txt")
        assert lp.exists(), f"{name}: no label file for {p}"
        n += 1
        for line in lp.read_text().splitlines():
            c, x, y, w, h = line.split()[:5]
            assert 0 <= int(c) < 198, f"{name}: class {c} in {lp}"
            assert all(0.0 <= float(v) <= 1.0 for v in (x, y, w, h)), f"{name}: box out of frame in {lp}"
            rows += 1
            cls[int(c)] += 1
    assert rows, f"{name}: no boxes in {n} sampled label files"
    return {"images": len(images), "sampled": n, "boxes_per_image": round(rows / n, 2), "classes_seen": len(cls)}


s2 = imgs("synth2_r3/train/images")
busy = imgs("synth2_r3/busy/images")
crdd = imgs("yolo/crdd/images")
real = imgs("real_r3/images/train") if (det / "real_r3" / "images" / "train").is_dir() else []
info = {"n_classes": len(names), "synth2": check_labels("synth2", s2), "synth2_busy": check_labels("busy", busy),
        "crdd": check_labels("crdd", crdd)}
if real:
    info["real_r3"] = check_labels("real_r3", real)
assert len(s2) >= ab_n and len(busy) >= busy_n >= ab_busy, (len(s2), ab_n, len(busy), busy_n, ab_busy)
# dense lists are the same size as ctl: busy frames take the place of plain ones
lists = {"ab_ctl": s2[:ab_n] + crdd * c_ab,
         "ab_dense": s2[:ab_n - ab_busy] + busy[:ab_busy] + crdd * c_ab,
         "main_ctl": s2 + crdd * c_main,
         "main_dense": s2[:len(s2) - busy_n] + busy[:busy_n] + crdd * c_main}
if real:
    lists["ab_real"] = lists["ab_ctl"] + real * r_ab
    lists["main_real"] = lists["main_ctl"] + real * r_main
    lists["main_densereal"] = lists["main_dense"] + real * r_main
for k, v in lists.items():
    (work / f"{k}.txt").write_text("\n".join(v) + "\n")
    # ultralytics validates once at the end even with val=False: keep that pass to 600 frames
    # (training frames, so the number means nothing; real scoring is training/det_check.py)
    (work / f"{k}_val.txt").write_text("\n".join(s2[:300] + crdd[:300]) + "\n")
    (work / f"{k}.yaml").write_text(yaml.safe_dump({"path": str(root), "train": str(work / f"{k}.txt"),
                                                     "val": str(work / f"{k}_val.txt"),
                                                     "names": dict(enumerate(names))}, sort_keys=False))
    info[k] = {"list": len(v), "crdd_share": round(sum("/yolo/crdd/" in p for p in v) / len(v), 3),
               "busy_share": round(sum("/synth2_r3/busy/" in p for p in v) / len(v), 3),
               "real_share": round(sum("/real_r3/" in p for p in v) / len(v), 3)}
(work / "dataset_info.json").write_text(json.dumps(info, indent=1))
print(json.dumps(info, indent=1))
EOF
  cp "$WORK/dataset_info.json" "$OUT/"
  touch "$WORK/.lists"
fi
[[ $STAGE == prep ]] && { log "prep done"; exit 0; }

# ---- train one run ------------------------------------------------------------------------
# $1 list name, $2 run dir under $OUT, $3 hours
train_run() {
  local list=$1 run=$2 hours=$3
  local timeout_s
  timeout_s=$(awk -v t="$hours" 'BEGIN{print int(t*3600+900)}')
  log "train $run: $list, ${hours} h"
  touch "$OUT/TRAINING_STARTED"
  set +e
  timeout --signal=TERM --kill-after=120 "$timeout_s" $PY - "$WORK/$list.yaml" "$OUT" "$run" "$INIT" "$IMGSZ" \
      "$BATCH" "$hours" "$DEVICE" "$WORKERS" "$LR0" <<'EOF'
import sys
from pathlib import Path
import torch
from ultralytics import YOLO
data, out, run, init, imgsz, batch, hours, device, workers, lr0 = sys.argv[1:]
last = Path(out) / run / "weights" / "last.pt"
if last.exists():
    ck = torch.load(last, map_location="cpu", weights_only=False)
    if ck.get("optimizer") is None:
        print("last.pt is a finished run (no optimizer state): not training again")
    else:
        print("resuming from", last, "epoch", ck.get("epoch"))
        YOLO(str(last)).train(resume=True)
    sys.exit(0)
m = YOLO(init)
nc = m.model.model[-1].nc
assert nc == 198, f"init has {nc} classes: the head would be re-initialised"
ref = {k: v.clone() for k, v in m.model.state_dict().items()}


def all_weights_transferred(trainer):
    """ultralytics rebuilds the model from yaml and copies matching tensors; a class-count or
    architecture mismatch silently leaves tensors at their random init."""
    sd = trainer.model.state_dict()
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    miss = [k for k, v in ref.items() if k not in sd or sd[k].shape != v.shape
            or not torch.equal(sd[k].float().cpu(), v.float().cpu())]
    miss = [k for k in miss if "num_batches_tracked" not in k and not k.endswith("dfl.conv.weight")]
    print(f"init transfer: {len(ref) - len(miss)}/{len(ref)} tensors identical")
    assert not miss, f"{len(miss)} tensors not transferred, e.g. {miss[:5]}"
    ref.clear()


m.add_callback("on_train_start", all_weights_transferred)
m.train(
    data=data, epochs=100, time=float(hours), imgsz=int(imgsz), batch=int(batch), device=int(device) if device.isdigit() else device,
    workers=int(workers), cache=False, amp=True, project=out, name=run, exist_ok=True,
    optimizer="SGD", lr0=float(lr0), lrf=0.2, momentum=0.937, cos_lr=False,
    warmup_epochs=0.3, warmup_bias_lr=0.0,
    mosaic=0.5, close_mosaic=0, fliplr=0.5, degrees=0, scale=0.25, translate=0.05,
    hsv_h=0.01, hsv_s=0.4, hsv_v=0.3,
    val=False, save_period=1, plots=False, patience=100, seed=0)
EOF
  local rc=$?
  set -e
  log "train $run exit code $rc"
  [[ -f "$OUT/$run/weights/last.pt" ]] || { log "no last.pt in $run: training failed"; exit 1; }
}

# ---- ab -----------------------------------------------------------------------------------
if [[ $STAGE == ab ]]; then
  ARMS="ctl dense"; [[ $HAVE_REAL == 1 ]] && ARMS="ctl dense real"
  [[ -n $ARMS_ARG ]] && ARMS=${ARMS_ARG//,/ }
  for arm in $ARMS; do
    if [[ ! -f "$WORK/.ab_$arm" ]]; then
      train_run "ab_$arm" "ab/$arm" "$ARM_HOURS"
      touch "$WORK/.ab_$arm"
    fi
  done
  touch "$OUT/AB_DONE"
  log "ab done: score $OUT/ab/*/weights/last.pt on the Mac, then run --stage main --data <arm>"
  exit 0
fi

# ---- main ---------------------------------------------------------------------------------
if [[ ! -f "$WORK/.trained" ]]; then
  train_run "main_$DATA" "train" "$MAX_HOURS"
  touch "$WORK/.trained"
fi
touch "$OUT/DONE"
log "all done"
