#!/bin/zsh
# usage: eval_pair_ei.sh <engine_dir> <p0088|off> <checkpoint> <label> <date> [condition ...]
#   _ei_ series (2026-11-25): eval_pair.sh's conditions with the engine and CR_P0088 as arguments, run through
#   evals/tools/eval_ghost_ei.py (training/eval_ghost.py plus read-only forensics: own Hog tower hits a play,
#   Cannon answers to enemy bridge Hogs). Same 421 held-out replays, 1,000 matches, same seeds as _ed_.
#   Output: evals/ghost_g2_ppo_s2r_<label>_<cond>_ei_<eng>_<sw>_<date>.json, eng = basename of the build folder
#   (gen2_h, gen2_i), sw = off or the switch list with commas -> '+'. Default condition: live2_d1417_t050.
#   Conditions: live2_d1417_t050 (as deployed, T 0.5), live2_d1417 (T 1.0), cleanT_d1417.
#   PROCS (default 4) and NICE (default 10) from the environment.
cd $HOME/cr-robot || exit 1
eng=$1; sw=$2; ck=$3; lab=$4; day=$5; shift 5
[ -d "$eng" ] || { echo "no engine dir $eng"; exit 1; }
[ "$sw" = off ] && sw=""
export CR_ENGINE=$eng CR_P0088=$sw PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
tag=$(basename $(dirname $eng))_${${sw:-off}//,/+}
conds=("$@"); [ ${#conds} -eq 0 ] && conds=(live2_d1417_t050)
ev() { out=$1; shift; [ -f evals/$out.json ] && return; nice -n ${NICE:-10} .venv/bin/python evals/tools/eval_ghost_ei.py --temp ${T:-1.0} --n 1000 --procs ${PROCS:-4} --act-delay 14,16,1 --no-human --out evals/$out.json "$@" > runs/logs/eval_$out.log 2>&1; }
LIVE=(--sim2real v1 --sim2real-elixir-est --obs-noise evals/cfg/board_default_hp_live.json --tail-noise evals/cfg/tail_reader2_spells.json)
for c in $conds; do
  case $c in
    live2_d1417_t050) T=0.5 ev ghost_g2_ppo_s2r_${lab}_${c}_ei_${tag}_${day} --weights $ck --label s2r_${lab}_${c} $LIVE ;;
    live2_d1417)      ev ghost_g2_ppo_s2r_${lab}_${c}_ei_${tag}_${day} --weights $ck --label s2r_${lab}_${c} $LIVE ;;
    cleanT_d1417)     ev ghost_g2_ppo_s2r_${lab}_${c}_ei_${tag}_${day} --weights $ck --label s2r_${lab}_${c} --tail-noise spells ;;
    *) echo "unknown condition $c"; exit 1 ;;
  esac
done
