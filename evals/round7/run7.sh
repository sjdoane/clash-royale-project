# usage: zsh run_p.sh name kind pooljson [n]
cd $HOME/cr-robot
S=/private/tmp/claude-501/-Users-samueldoane-cr-robot/40e94860-6611-4922-a6d8-a1b149e53030/scratchpad
export CR_ENGINE=$PWD/build/gen2_h/engine PYTHONPATH=$S:$PWD/build/gen2_h/engine PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
CK=${CK:-runs/g2_ppo_s2r/snapshots/ep0301012.pth}
.venv/bin/python $S/r7_probe.py $CK --kind $2 --n ${4:-60} --seed ${SEED:-8700} --procs 3 --pool-json "$3" --out $S/r7_$1.json > $S/r7_$1.log 2>&1
echo exit $? >> $S/r7_$1.log
