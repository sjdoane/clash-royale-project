#!/usr/bin/env bash
# Mac -> balerion: stage the gen-2 tree and copy it to ~/cr-robot-gen2.
#
#   [GEN2_DATA=/path/to/gen2/data] bash training/gen2_push.sh
#
# Engine: a copy of engine/ (which carries 0001-0003 in its working tree) with
# patches/pending/0004..0019 (and any later 00*.patch) applied in order. engine/ itself is never touched;
# this is the gen-2 apply command.
# Data: the gen-2 BC dataset (data/bc/hog26, rebuilt on the gen-2 engine) and
# the deck pools. That is the same derived data gen-1 trained from on balerion;
# the raw IL_Replay replays are NOT copied anywhere.
# Uses only the existing SSH master socket (training/rsh); never opens a new connection.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
GEN2_DATA=${GEN2_DATA:-}   # empty: engine + training only (the dataset comes later)
STAGE=${STAGE:-/tmp/gen2_stage}
SSH="ssh -o ControlPath=$HOME/.ssh/cm-balerion -o BatchMode=yes"
HOST=user@balerion
DEST=cr-robot-gen2

ssh -o ControlPath="$HOME/.ssh/cm-balerion" -O check $HOST 2>/dev/null \
  || { echo "no SSH master socket to balerion; ask Sam to reopen it"; exit 2; }

rm -rf "$STAGE" && mkdir -p "$STAGE"
rsync -a --exclude .git --exclude .venv --exclude '*.so' --exclude '*.pyd' \
  --exclude '/build/' --exclude '/build_*/' --exclude __pycache__ --exclude .pytest_cache \
  "$REPO/engine/" "$STAGE/engine/"
for p in "$REPO"/patches/pending/00*.patch; do
  (cd "$STAGE/engine" && git apply --check "$p" && git apply "$p") || { echo "FAILED $p"; exit 1; }
  echo "applied $(basename "$p")"
done
grep -q "NUM_LEVEL_SCALARS" "$STAGE/engine/include/core/ClashEnv.h"

$SSH $HOST "mkdir -p $DEST/data/bc $DEST/checkpoints $DEST/runs"
RS="rsync -a -e \"$SSH\""
eval $RS --delete "$STAGE/engine/" $HOST:$DEST/engine/
eval $RS --exclude __pycache__ "$REPO/training/" $HOST:$DEST/training/
if [ -n "$GEN2_DATA" ]; then
  eval $RS "$GEN2_DATA/bc/hog26" $HOST:$DEST/data/bc/
  eval $RS "$GEN2_DATA/decks" $HOST:$DEST/data/
else
  eval $RS "$REPO/data/decks" $HOST:$DEST/data/   # deck pools, for now
fi
echo "pushed. On balerion: cd ~/$DEST && bash training/gen2_balerion.sh setup"
