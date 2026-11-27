#!/bin/zsh
# usage: eval_liveghost.sh <weights.pth> <label> <date>
# The deployed setting (noise, delay 16-18 ticks since 2026-11-23 14:14, T 0.5; engine gen2_e; files before
# that are _d47_ on gen2_c against the store without the landing shift and do not compare) against the replays of our own live matches, 30 per
# match from the first tick: training matches and the held-out session (evals/tools/liveghost_summary.py).
# 2026-11-23 15:55: engine build/gen2_f (pocket, Hunter); files are named _ef_ and do not compare with the
# earlier ones (the recorded pocket plays now land where they were made, so these replays are harder).
# 2026-11-23 16:45: engine build/gen2_g (patch 0086, which building pulls a troop); files are named _eg_.
# 2026-11-24 12:57: engine build/gen2_h (patch 0087, river crossing on the tower column); files are named _eh_.
# 2026-11-24 15:35: act delay 16,18,1 -> 14,16,1 (14-17 ticks a play, the drag tap); files are named _d1417_ _ed_.
cd $HOME/cr-robot || exit 1
export CR_ENGINE=$PWD/build/gen2_h/engine PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CR_GHOST_STORE=$PWD/data/ghost/live1
n=$(.venv/bin/python -c "import json; print(30 * len(json.load(open('data/ghost/live1/meta.json'))['live']))")
out=evals/liveghost_g2_ppo_s2r_$2_live2_d1417_t050_ed_$3.json
nice -n 5 .venv/bin/python -m training.eval_ghost --temp 0.5 --n $n --procs 5 --act-delay 14,16,1 --no-human \
  --split all --faced exact --out $out --weights $1 --label s2r_$2_liveghost --sim2real v1 --sim2real-elixir-est \
  --obs-noise evals/cfg/board_default_hp_live.json --tail-noise evals/cfg/tail_reader2_spells.json \
  > runs/logs/eval_liveghost_$2.log 2>&1
.venv/bin/python evals/tools/liveghost_summary.py $out
