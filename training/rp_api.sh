#!/usr/bin/env bash
# rp_api.sh get|stop|terminate <pod_id>
# RunPod GraphQL (api.runpod.io) with the key from the local runpod-key MCP config. The key is
# read at call time and handed to curl on stdin (never in argv, child env, or logs).
set -euo pipefail
case $1 in
  get)       q="query { pod(input:{podId:\\\"$2\\\"}) { id name desiredStatus costPerHr } }" ;;
  stop)      q="mutation { podStop(input:{podId:\\\"$2\\\"}) { id desiredStatus } }" ;;
  terminate) q="mutation { podTerminate(input:{podId:\\\"$2\\\"}) }" ;;
  *) echo "usage: $0 get|stop|terminate <pod_id>" >&2; exit 64 ;;
esac
python3 -I -c '
import json, os
k = json.load(open(os.path.expanduser("~/.claude.json")))["mcpServers"]["runpod-key"]["env"]["RUNPOD_API_KEY"]
print("header = \"Authorization: Bearer %s\"" % k)' \
  | curl -sS -K - -H 'Content-Type: application/json' -w '\nHTTP %{http_code}\n' \
      -d "{\"query\": \"$q\"}" https://api.runpod.io/graphql
