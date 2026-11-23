#!/bin/sh
# Download every video in videos.txt into data/video/raw/ (private research use
# only: never commit or redistribute these files). Resumable: finished ids are
# recorded in data/video/raw/archive.txt and skipped on the next run.
#
# Video only, H.264, at most 1920 tall (the phone recordings are 884-1080 x
# 1920-2400 portrait); metadata JSON beside each file.
set -eu
cd "$(dirname "$0")/.."
OUT=data/video/raw
mkdir -p "$OUT"
grep -v '^#' video_dataset/videos.txt | awk 'NF {print "https://youtu.be/" $1}' > "$OUT/urls.txt"
exec .venv-tools/bin/yt-dlp \
  --js-runtimes node \
  -f "bv*[vcodec^=avc1][height<=1920]/bv*[vcodec^=avc1]/bv*" \
  --download-archive "$OUT/archive.txt" \
  --write-info-json --no-write-comments \
  --concurrent-fragments 4 --retries 10 --fragment-retries 10 \
  --ignore-errors --no-overwrites \
  -o "$OUT/%(id)s.%(ext)s" \
  -a "$OUT/urls.txt"
