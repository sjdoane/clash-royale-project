#!/bin/zsh
# usage: eval_live_one.sh <checkpoint> <label> <date>   -> only the "live" evaluation of eval_pair.sh
cd "$(dirname "$0")/../.." || exit 1
export CR_ENGINE="${CR_ENGINE:-$PWD/engine}" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ck=$1; lab=$2; day=$3
out=ghost_g2_ppo_s2r_${lab}_live_ec_${day}
.venv/bin/python -m training.eval_ghost --temp 1.0 --n 1000 --procs 4 --act-delay 3,6,1 --no-human --out evals/$out.json \
  --weights $ck --label s2r_${lab}_live --sim2real v1 --sim2real-elixir-est --obs-noise evals/cfg/board_default_hp_live.json > runs/logs/eval_$out.log 2>&1
