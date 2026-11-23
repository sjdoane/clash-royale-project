#!/bin/zsh
# usage: eval_decomp.sh <checkpoint> <label> <date>
# The checkpoints were trained with the opponent tail as the live reader gives it (names policy "spells").
# eval_ghost only turns that on when --obs-noise is a non-empty spec, so the plain "clean" and "s2r1"
# evaluations show the policy a tail it never trained on. This adds the tail one step at a time:
#   cleanT : clean + tail noise
#   s2rT   : --sim2real v1 + tail noise
#   s2rBT  : --sim2real v1 + board noise + tail noise, true troop health   (= training conditions before 2026-11-23)
# (s2rBhp from eval_health.sh = s2rBT with troops at full health; "live" = that plus the elixir estimate)
cd "$(dirname "$0")/../.." || exit 1
export CR_ENGINE="${CR_ENGINE:-$PWD/engine}" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ck=$1; lab=$2; day=$3
ev() { out=$1; shift; .venv/bin/python -m training.eval_ghost --temp 1.0 --n 1000 --procs 4 --act-delay 3,6,1 --no-human --out evals/$out.json "$@" > runs/logs/eval_$out.log 2>&1; }
ev ghost_g2_ppo_s2r_${lab}_cleanT_ec_${day} --weights $ck --label s2r_${lab}_cleanT --tail-noise spells
ev ghost_g2_ppo_s2r_${lab}_s2rT_ec_${day} --weights $ck --label s2r_${lab}_s2rT --sim2real v1 --tail-noise spells
ev ghost_g2_ppo_s2r_${lab}_s2rBT_ec_${day} --weights $ck --label s2r_${lab}_s2rBT --sim2real v1 --obs-noise training/obs_noise_default.json
echo done
