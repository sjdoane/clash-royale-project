#!/bin/bash
# Download the trained models (and the small deck file) from the private release
# sjdoane/clash-royale-project-assets. Needs the GitHub CLI signed in (gh auth login)
# with access to that repo: ask Sam for an invite.
#
#   scripts/fetch_assets.sh            # everything (~120 MB)
#   scripts/fetch_assets.sh --no-det   # skip the 50 MB detector (only needed for live play)
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=sjdoane/clash-royale-project-assets
TAG=${ASSETS_TAG:-v1}
command -v gh >/dev/null || { echo "install the GitHub CLI first: brew install gh && gh auth login"; exit 1; }
mkdir -p checkpoints data/decks data/det/runs/r3/weights
get() { [ -s "$2" ] && { echo "have $2"; return; }; gh release download "$TAG" -R "$REPO" -p "$1" -O "$2" && echo "got  $2"; }
get ppo_latest.pth      checkpoints/ppo_latest.pth       # PPO policy, newest snapshot (sim win rate 65 %)
get ppo_ep0121202.pth   checkpoints/ppo_ep0121202.pth    # the snapshot used in the live sessions
get bc_init.pth         checkpoints/bc_init.pth          # behaviour clone of Hog 2.6 players: PPO starts here
get bc_clone_any.pth    checkpoints/bc_clone_any.pth     # behaviour clone of all players: a PPO opponent
get meta_api_pool.json  data/decks/meta_api_pool.json    # current ladder decks (official API, 2026-11-18)
[ "${1:-}" = "--no-det" ] || get detector_r3.pt data/det/runs/r3/weights/best.pt   # YOLOv8 unit detector
echo "done. Try: .venv/bin/python play_sim.py"
