#!/bin/sh
# Opponent-play reading table for any detector weights (see opp_plays_ab.py):
#   live_ppo/eval/opp_plays_ab.sh <weights.pt> [more weights...]
# Builds the box caches it needs (one detector process at a time, .venv-det), then scores
# under .venv. Reads the weights only.
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
exec "$REPO/.venv/bin/python" "$REPO/live_ppo/eval/opp_plays_ab.py" "$@"
