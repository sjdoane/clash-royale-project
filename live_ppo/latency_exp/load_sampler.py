#!/usr/bin/env python3
"""Once a second: wall time, load averages, trainer state (R/S = running, T = paused).
Covariate log for the 2026-11-24 latency experiment; touches nothing else.

    nohup .venv/bin/python live_ppo/latency_exp/load_sampler.py runs/latency_exp/load.jsonl &
"""
import json, os, subprocess, sys, time

out = open(sys.argv[1], "a", buffering=1)
end = time.time() + float(sys.argv[2] if len(sys.argv) > 2 else 4 * 3600)
while time.time() < end:
    try:
        ps = subprocess.run(["pgrep", "-f", "training.ppo_pool --run g2_ppo_s2r"], capture_output=True, text=True).stdout.split()
        pid = ps[0] if ps else None
        st = cpu = None
        if pid:
            r = subprocess.run(["ps", "-o", "stat=,%cpu=", "-p", pid], capture_output=True, text=True).stdout.split()
            st, cpu = (r[0], float(r[1])) if len(r) == 2 else (None, None)
        evals = subprocess.run(["pgrep", "-f", "eval_ghost|pilot_probe|_probe.py"], capture_output=True, text=True).stdout.split()
        out.write(json.dumps({"t_unix": round(time.time(), 3), "load": os.getloadavg(),
                              "trainer_pid": pid, "trainer_stat": st, "trainer_cpu": cpu,
                              "n_eval_procs": len(evals)}) + "\n")
    except Exception as e:  # never die mid-session
        out.write(json.dumps({"t_unix": time.time(), "err": str(e)}) + "\n")
    time.sleep(1.0)
