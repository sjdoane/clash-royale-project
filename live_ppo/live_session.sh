#!/bin/bash
# One live session, the standard way (2026-11-24): the PPO trainer is paused for the
# whole session and resumed however it ends, and play_live.py runs with the standard
# flags including --tap-mode drag. Training's act delay (14-17 ticks) assumes both.
#
#   live_ppo/live_session.sh            # dry run (no taps)
#   live_ppo/live_session.sh --act      # real taps, after play_live's typed confirmation
#
# Extra arguments go to play_live.py after the defaults, so they win
# (e.g. --checkpoint runs/g2_ppo_s2r/snapshots/ep0330000.pth, --seconds 600).
# Passing --driver-flags REPLACES the standard driver flags below; copy them in if you do.
#
# Never kills anything: only kill -STOP / kill -CONT on the trainer ("training.ppo_pool")
# and its child processes, found by pattern at launch. eval_watch.sh, caffeinate and
# other probes are left alone. The trainer is resumed by an EXIT/INT/TERM/HUP trap, and
# by a watchdog if this script itself is killed with -9.

cd "$(dirname "$0")/.." || exit 1
ROOT=$PWD
PATTERN="^[^ ]*python[^ ]* -m training[.]ppo_pool"
LOCK="$ROOT/runs/live/.live_session.lock"

# Engine per checkpoint (2026-11-27, gen2_l live prep; live_ppo/engine_pick.py): the checkpoint that
# play_live.py will load (the last --checkpoint in the arguments, else the newest main snapshot) names
# the engine and CR_P0088 it was trained with, from its run's launch.txt. A checkpoint without a
# launch.txt (the main run runs/g2_ppo_s2r) keeps the engine exactly as before (CR_LIVE_ENGINE, default
# build/gen2_c/engine) and CR_P0088 as the caller set it. With a launch.txt, CR_LIVE_ENGINE / CR_P0088
# may be left unset; set to something else, the session refuses. Checked before anything is paused.
# The driver repeats the check on the engine it loads and sets the Ghost mode / Bush Goblins itself.
CKPT=$(ls -t "$ROOT"/runs/g2_ppo_s2r/snapshots/ep*.pth 2>/dev/null | head -1)
EFF=$CKPT; prev=""
for a in "$@"; do
    [ "$prev" = "--checkpoint" ] && EFF=$a
    case $a in --checkpoint=*) EFF=${a#--checkpoint=} ;; esac
    prev=$a
done
PICK=$("$ROOT/.venv/bin/python" "$ROOT/live_ppo/engine_pick.py" --shell "$EFF") || {
    echo "live_session: refusing to start: no engine for checkpoint $EFF (reason above). Nothing was paused."
    exit 1
}
eval "$PICK"
if [ "$PICK_KIND" = run ]; then
    LIVE_ENGINE=$PICK_ENGINE
    export CR_P0088=$PICK_P0088
else
    LIVE_ENGINE="$ROOT/${CR_LIVE_ENGINE:-build/gen2_c/engine}"
fi

mkdir -p "$ROOT/runs/live"
if ! mkdir "$LOCK" 2>/dev/null; then
    other=$(cat "$LOCK/pid" 2>/dev/null)
    if [ -n "$other" ] && kill -0 "$other" 2>/dev/null; then
        echo "live_session: another live session is running (pid $other); not starting a second one."
        exit 1
    fi
    rm -rf "$LOCK"; mkdir "$LOCK" || exit 1          # stale lock from a dead session
fi
echo $$ > "$LOCK/pid"

# trainer + all its descendants
descendants() { local c; for c in $(pgrep -P "$1"); do echo "$c"; descendants "$c"; done; }
PIDS=""
for p in $(pgrep -f "$PATTERN"); do
    PIDS="$PIDS $p $(descendants "$p" | tr '\n' ' ')"
done
PIDS=$(echo $PIDS)

resume() {
    [ -n "$PIDS" ] && kill -CONT $PIDS 2>/dev/null
}
report() {
    if [ -n "$PIDS" ]; then
        echo "live_session: trainer resumed:"
        ps -o pid=,stat=,etime=,command= -p "$(echo $PIDS | cut -d' ' -f1)" | cut -c1-110
        local stopped
        stopped=$(ps -o stat= -p "$(echo $PIDS | tr ' ' ',')" 2>/dev/null | grep -c T)
        echo "live_session: processes still stopped: $stopped (should be 0)"
    fi
}
cleanup() {
    trap - EXIT INT TERM HUP
    declare -F record_session >/dev/null && record_session
    rm -rf "$LOCK"
    resume
    report
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

echo "live_session: $(date '+%Y-%m-%d %H:%M:%S %Z'), load average $(sysctl -n vm.loadavg)"
if [ -z "$PIDS" ]; then
    echo "live_session: no trainer ($PATTERN) found; nothing to pause, carrying on."
else
    n=$(echo $PIDS | wc -w | tr -d ' ')
    kill -STOP $PIDS
    echo "live_session: trainer paused ($n processes: $PIDS)"
    # watchdog: resumes the trainer if this script dies without running its trap (kill -9).
    # It acts only when the lock still names this session (the trap removes the lock first),
    # so it can never resume a trainer that a LATER session has paused.
    ME=$$
    ( trap '' INT HUP TERM
      while kill -0 $ME 2>/dev/null; do sleep 2; done
      if [ "$(cat "$LOCK/pid" 2>/dev/null)" = "$ME" ]; then
          kill -CONT $PIDS 2>/dev/null; rm -rf "$LOCK"
      fi ) >/dev/null 2>&1 &
fi

READERS=$(grep -E '^(ROBUST_TOWERS|OPP_KING_READER) *=' "$ROOT/live_ppo/hud_live.py" | sed 's/ *#.*//' | tr '\n' ' ')
echo "live_session: reader switches (live_ppo/hud_live.py): $READERS"
ARGS="$*"; MARK="$LOCK/start"; touch "$MARK"
record_session() {   # session.json in each run dir this session created: settings for later analysis
    local d
    for d in $(find "$ROOT/runs/live" -maxdepth 1 -type d -name '20*' -newer "$MARK" 2>/dev/null); do
        printf '{"wrapper": "live_session.sh", "tap_mode": "drag", "trainer_paused": %s, "readers": "%s", "args": "%s", "engine": "%s", "p0088": "%s", "engine_pick": "%s"}\n' \
            "$([ -n "$PIDS" ] && echo true || echo false)" "$READERS" "$ARGS" "$LIVE_ENGINE" "${CR_P0088-}" "$PICK_KIND" > "$d/session.json"
    done
}

echo "live_session: checkpoint $CKPT (newest snapshot; pass --checkpoint to choose another)"
echo "live_session: engine $LIVE_ENGINE, CR_P0088=${CR_P0088-(unset)} for $EFF"
echo "live_session:   ($PICK_DESC)"

# CR_LIVE_ENGINE (default off -> build/gen2_c as before): another engine root, e.g. build/gen2_j/engine for a
# pocket-trained checkpoint with --pocket-mask (export CR_P0088 too). 2026-11-26, pocket/data owner.
# Since 2026-11-27 only for a checkpoint without a launch.txt; others take theirs (engine pick above).
CR_ENGINE="$LIVE_ENGINE" "$ROOT/.venv/bin/python" "$ROOT/play_live.py" \
    --checkpoint "$CKPT" --seconds 3600 \
    --driver-flags="--loop-timestamps --ability tap --dump-obs --leak-floor 9.5 --elixir-est digit --tap-mode drag" \
    "$@"
rc=$?
echo "live_session: play_live.py exited with $rc"
exit $rc
