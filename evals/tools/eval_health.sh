#!/bin/zsh
# usage: eval_health.sh <checkpoint> <label> <date>
# Sizes the unit-health gap: live never reads troop health, so the driver writes every unit at full health.
#   hponly  : clean, except every unit shown at full health            (compare with *_clean_ec_*)
#   s2rB    : --sim2real v1 + board noise + tail noise (training conditions, true health)
#   s2rBhp  : the same with every unit shown at full health (live conditions)   (compare with s2rB)
cd "$(dirname "$0")/../.." || exit 1
export CR_ENGINE="${CR_ENGINE:-$PWD/engine}" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ck=$1; lab=$2; day=$3
ev() { out=$1; shift; .venv/bin/python -m training.eval_ghost --temp 1.0 --n 1000 --procs 4 --act-delay 3,6,1 --no-human --out evals/$out.json "$@" > runs/logs/eval_$out.log 2>&1; }
ev ghost_g2_ppo_s2r_${lab}_hponly_ec_${day} --weights $ck --label s2r_${lab}_hponly --obs-noise evals/cfg/board_hp_live_only.json --tail-noise off
ev ghost_g2_ppo_s2r_${lab}_s2rB_ec_${day} --weights $ck --label s2r_${lab}_s2rB --sim2real v1 --obs-noise default
ev ghost_g2_ppo_s2r_${lab}_s2rBhp_ec_${day} --weights $ck --label s2r_${lab}_s2rBhp --sim2real v1 --obs-noise evals/cfg/board_default_hp_live.json
echo done
