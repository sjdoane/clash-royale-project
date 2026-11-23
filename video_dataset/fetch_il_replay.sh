#!/bin/bash
# Fetch the IL_Replay dataset (FirstLight_CR, Hugging Face) into data/il_replay/.
# RoyaleAPI replays, anonymised: decks with forms and levels, both players'
# plays as card + tile + 20 Hz tick. No licence is stated on the dataset page:
# private research use only, never committed or redistributed (data/ is gitignored).
set -euo pipefail
cd "$(dirname "$0")/.."
B=https://huggingface.co/datasets/VanguardX101/IL_Replay/resolve/main
D=data/il_replay
mkdir -p "$D/actions" "$D/replays"
curl -sfL -A "Mozilla/5.0" -o "$D/manifest.json" "$B/manifest.json"
curl -sfL -A "Mozilla/5.0" -o "$D/SHA256SUMS" "$B/SHA256SUMS"
python3 -c "import json;[print(f['path']) for f in json.load(open('$D/manifest.json'))['files']]" |
while read -r f; do
  [ -s "$D/$f" ] && continue
  curl -sfL --retry 5 -A "Mozilla/5.0" -o "$D/$f.part" "$B/$f" && mv "$D/$f.part" "$D/$f"
  echo "got $f"
done
(cd "$D" && tr -d '\r' < SHA256SUMS | shasum -a 256 -c - 2>/dev/null | grep -vc ': OK$' | sed 's/^/checksum failures: /') || true
