#!/bin/bash
# Rebuild the data that is NOT shipped (it all comes from the IL_Replay dataset on
# Hugging Face, which has no licence: download it yourself, keep it private, never
# commit it). Each step skips what already exists.
#
#   scripts/build_data.sh ghosts     # ~2 GB download + ~5 min: replay opponents + deck pool
#                                    #   (needed for ghost evaluations and for PPO training)
#   scripts/build_data.sh bc         # + the behaviour-cloning datasets (hours, ~46 GB on disk)
#                                    #   (needed to retrain the clones, or for PPO's cloning loss)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CR_ENGINE="${CR_ENGINE:-$PWD/engine}" PYTHONDONTWRITEBYTECODE=1
what=${1:-ghosts}
PROCS=${PROCS:-8}

bash video_dataset/fetch_il_replay.sh                                   # data/il_replay (raw parquet parts)
[ -s data/il_replay/hog26/matches.parquet ] || $PY video_dataset/il_replay_26.py
[ -s data/decks/il_replay_pool.json ]       || $PY video_dataset/il_replay_bc.py pool
[ -s data/decks/train_pool.json ]           || $PY -c "from training import opponents as O; O.build_pool()"
[ -d data/ghost/v1 ]                        || $PY -I training/ghost_data.py build --procs "$PROCS"
[ "$what" = bc ] || { echo "ghost store and deck pools ready"; exit 0; }

# The released clones were built with the engine at tag gen2-data (patch 0046). Building with
# the newer engine works too; rows then differ slightly from the released checkpoints' data.
for set in hog_g2 all_g2; do
  root=data/bc2/$set
  [ -f "$root/READY" ] && continue
  if [ $set = hog_g2 ]; then others=(--others 0); mirror=both; else others=(--others 50000 --floor 10); mirror=no; fi
  $PY -m training.replay_bc plan  --root "$root" "${others[@]}"
  $PY -m training.replay_bc build --root "$root" --convention conv2g --grid 5 --mirror $mirror --procs "$PROCS"
  $PY -m training.replay_bc finish --root "$root"
done
echo "BC datasets ready: data/bc2/hog_g2, data/bc2/all_g2"
