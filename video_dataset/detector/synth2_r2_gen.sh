#!/bin/sh
# synth2 round 2: gen_r2.sh <n> <seed> <workers> <out_dir>
# Writes <out_dir>/images/*.jpg and <out_dir>/labels/*.txt (YOLO, ids of classes_round2.json,
# 199 classes, 198 = deploy-pending). Seed 777 = val (backgrounds held out from train), any
# other seed = train. Needs python3 with numpy + opencv. Runs from any cwd.
set -eu
if [ $# -ne 4 ]; then
    echo "usage: $0 <n> <seed> <workers> <out_dir>" >&2
    exit 2
fi
DIR=$(cd "$(dirname "$0")" && pwd)
SYNTH2_ROOT="$DIR/synth2"
export SYNTH2_ROOT
split=train
[ "$2" = "777" ] && split=val
PY=${PYTHON:-python3}
exec "$PY" -I "$DIR/synth2_r2.py" --n "$1" --seed "$2" --workers "$3" --out "$4" --split "$split"
