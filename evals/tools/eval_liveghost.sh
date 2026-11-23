#!/bin/zsh
# usage: eval_liveghost.sh <weights.pth> <label> <date>
# The deployed setting (noise, delay 4-7 ticks, T 0.5) against the replays of our own live matches, 30 per
# match from the first tick: training matches and the held-out session (evals/tools/liveghost_summary.py).
cd "$(dirname "$0")/../.." || exit 1
export CR_ENGINE=$PWD/build/gen2_c/engine PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CR_GHOST_STORE=$PWD/data/ghost/live1
n=$(.venv/bin/python -c "import json; print(30 * len(json.load(open('data/ghost/live1/meta.json'))['live']))")
out=evals/liveghost_g2_ppo_s2r_$2_live2_d47_t050_$3.json
nice -n 5 .venv/bin/python -m training.eval_ghost --temp 0.5 --n $n --procs 5 --act-delay 4,7,1 --no-human \
  --split all --faced exact --out $out --weights $1 --label s2r_$2_liveghost --sim2real v1 --sim2real-elixir-est \
  --obs-noise evals/cfg/board_default_hp_live.json --tail-noise evals/cfg/tail_reader2_spells.json \
  > runs/logs/eval_liveghost_$2.log 2>&1
.venv/bin/python evals/tools/liveghost_summary.py $out
