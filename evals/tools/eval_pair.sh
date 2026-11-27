#!/bin/zsh
# usage: eval_pair.sh <checkpoint> <label> <date>   -> two evaluations, 1,000 matches each against the 421 held-out replays
#   cleanT : true observations, with the opponent tail as the policy was trained on it (spell names only)
#   live   : what the live driver gives the policy as far as it is modelled: --sim2real v1 with the fractional
#            elixir estimate, board noise, tail noise, every troop shown at full health
#   live2  : live, with the opponent tail as the live reader fills it since 2026-11-22 (preset reader2:
#            landing estimates for thrown spells, fewer missed follow-up plays). Added 2026-11-23 03:40.
#            ep0050513: live 53.7, live2 54.7 (paired +1.0 +- 1.3).
#   live2_d47_t050 : live2 as the policy is deployed from 2026-11-23 09:55 on: WHAT / WHERE temperature 0.5
#            (play_live.py default for gen-2) and act delay 4,7,1 (measured live: 0.56 s from observed frame
#            to card on board). ep0121202: 67.7 against live2 59.7 (paired +8.0 +- 1.5).
# Until 2026-11-23 02:15 this ran "clean" and "s2r1" instead. Both showed the policy a tail with every card
# named, which it never saw in training (eval_ghost turns tail noise on only with a non-empty --obs-noise), so
# they read about 5 points (clean) and 10 points (noisy) low. evals/tools/eval_decomp.sh has the steps between.
# 2026-11-23 14:14 (trainer episode 156,780 on): SERIES BREAK. Engine build/gen2_e (patch 0083) and act delay
# 16,18,1 in every condition: from the decision frame to the card taking effect is 1.7 s live, not 0.5 s
# (training/run_g2_ppo_s2r.sh has the measurements). Files are named _ee_ instead of _ec_; they do not compare
# with the _ec_ ones (ep ~154.1k, live2 T 0.5: 65.5 % at delay 4,7,1, 57.8 % at 16,18,1, both on gen2_c).
#   cleanT_d1618       : true observations, T 1.0
#   live2_d1618        : live reader noise, T 1.0
#   live2_d1618_t050   : as deployed (T 0.5). This is the headline number.
# The old "live" (first reader's tail) and the delay-4,7,1 conditions are no longer run.
# 2026-11-23 15:55 (trainer episode 171,116 on): engine build/gen2_f (patches 0084 pocket, 0085 Hunter);
# files are named _ef_. Against the held-out dataset replays the change is small (a tower has to fall first).
# 2026-11-23 16:45 (trainer episode 179,612 on): engine build/gen2_g (patch 0086: a Crown Tower is ranked as a
# target by its published body, no along-arena sight cut, an underground Tesla is no target); files are named
# _eg_. ep0173730 is evaluated on both engines (_ef_ and _eg_) as the bridge between the two series.
# 2026-11-24 12:57 (trainer episode 309,341 on): engine build/gen2_h (patch 0087: river crossing on the tower
# column, the Hog's diagonal jump); files are named _eh_. ep0284847 and ep0296981 are on both (_eg_, _eh_).
# 2026-11-24 15:35 (trainer episode ~330.8k on): act delay 16,18,1 -> 14,16,1 (14-17 ticks a play, 1.4-1.7 s:
# live play moves to --tap-mode drag, measured 1.51 s median, p10 1.39, p90 1.68). Files are named _d1417_ ... _ed_;
# ep0329297 is evaluated under both delays (_d1618_ _eh_ and _d1417_ _ed_) as the bridge.
cd $HOME/cr-robot || exit 1
export CR_ENGINE=$HOME/cr-robot/build/gen2_h/engine PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ck=$1; lab=$2; day=$3
ev() { out=$1; shift; [ -f evals/$out.json ] && return; .venv/bin/python -m training.eval_ghost --temp ${T:-1.0} --n 1000 --procs 4 --act-delay 14,16,1 --no-human --out evals/$out.json "$@" > runs/logs/eval_$out.log 2>&1; }
T=0.5 ev ghost_g2_ppo_s2r_${lab}_live2_d1417_t050_ed_${day} --weights $ck --label s2r_${lab}_live2_d1417_t050 --sim2real v1 --sim2real-elixir-est --obs-noise evals/cfg/board_default_hp_live.json --tail-noise evals/cfg/tail_reader2_spells.json
ev ghost_g2_ppo_s2r_${lab}_live2_d1417_ed_${day} --weights $ck --label s2r_${lab}_live2_d1417 --sim2real v1 --sim2real-elixir-est --obs-noise evals/cfg/board_default_hp_live.json --tail-noise evals/cfg/tail_reader2_spells.json
ev ghost_g2_ppo_s2r_${lab}_cleanT_d1417_ed_${day} --weights $ck --label s2r_${lab}_cleanT_d1417 --tail-noise spells
