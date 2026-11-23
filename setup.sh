#!/usr/bin/env bash
# One-shot setup. Safe to re-run.
#   ./setup.sh            engine + main venv (.venv)            -> simulator, training, evaluation
#   ./setup.sh --live     also the detector venv (.venv-det)    -> live play / detector work
set -euo pipefail
cd "$(dirname "$0")"

LIVE=0; [ "${1:-}" = "--live" ] && LIVE=1
BUILD_DIR=engine/build
PY=.venv/bin/python

need() { command -v "$1" >/dev/null 2>&1 || { echo "setup.sh: '$1' not found. $2" >&2; exit 1; }; }
need git   "Install git."
need uv    "Install uv: https://docs.astral.sh/uv/  (macOS: brew install uv)"
need cmake "Install cmake 3.15+: brew install cmake  (or: uv tool install cmake)"

echo "== engine submodule (sjdoane/ClashRoyaleAi, branch gen2)"
git submodule update --init --recursive

echo "== Python 3.11 venv at .venv"
[ -x "$PY" ] || uv venv --python 3.11 .venv
uv pip install --python "$PY" -r requirements.txt

echo "== build the simulator's Python module"
cmake -S engine -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
      -DPYBIND11_FINDPYTHON=ON -DPython_EXECUTABLE="$PWD/$PY" >/dev/null
cmake --build "$BUILD_DIR" --target clash_royale_env -j
cp "$BUILD_DIR"/clash_royale_env*.so engine/python_ai/

if [ "$LIVE" = 1 ]; then
  echo "== detector venv at .venv-det"
  [ -x .venv-det/bin/python ] || uv venv --python 3.11 .venv-det
  uv pip install --python .venv-det/bin/python -r requirements-det.txt
fi

echo "== check: the engine imports and a match runs"
"$PY" -c "from training import common as C; print('engine OK:', C.ENGINE, '| obs', C.CE.NUM_CHANNELS, 'channels')"

echo
echo "Done. Next: scripts/fetch_assets.sh, then $PY play_sim.py"
