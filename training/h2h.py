"""Head-to-head: a checkpoint against fixed reference snapshots of our own lineage.

    # watch a run: every --every episodes, checkpoints/<run>.pth vs each --refs
    .venv/bin/python -m training.h2h --watch ppo_league26 --every 2000
    # one-off (e.g. the A arm), written OUTSIDE that run's own files
    .venv/bin/python -m training.h2h --weights checkpoints/ppo_bc26.pth --label ppo_bc26 \
        --out runs/ppo_league26/ref_ppo_bc26_h2h.jsonl

Both seats are networks (training/league_env.LeagueEnv in mirror mode, King
term off): team 1 gets its own mirrored observation, LSTM state and Hero
abilities, exactly as the league's neural opponents do. Both play CLASH_DECK.
Each pairing plays --n seeded matches per seat with the seats swapped, so the
seat advantage cancels and the rows are comparable across checkpoints (same
seeds every time). Both policies SAMPLE, as eval_panel's does. CPU only.

Output: one json line per (checkpoint, reference): candidate win / draw rate,
per-seat win rate, plays per match for both sides.
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import time

from training import common as C

import numpy as np
import torch

SNAP_DIR = os.path.join(C.REPO, "checkpoints", "ppo_bc26_snapshots")
DEFAULT_REFS = ("1795011811032_pipeline1_ep00060675.pth", "*_ep00119319.pth")


def _resolve(ref):
    import glob
    p = ref if os.path.isabs(ref) else os.path.join(SNAP_DIR, ref)
    hits = sorted(glob.glob(p))
    if not hits and os.path.exists(os.path.join(C.REPO, ref)):
        hits = [os.path.join(C.REPO, ref)]
    if not hits:
        raise FileNotFoundError(ref)
    return hits[-1]


def _load(path):
    from training.net_v2 import load_any
    return load_any(path)


def parse_delay(s):
    """'lo,hi,jitter' -> tuple of ints, '' / None -> None (no actuation delay)."""
    return tuple(int(v) for v in s.split(",")) if s else None


def play(cand, ref, job, temp=1.0, act_delay=None):
    """One match. job: seed, cand_seat (0 or 1). Returns the candidate's result.

    Both seats sample at `temp` (0 = argmax); act_delay = (lo, hi, jitter)
    puts both seats behind training/reward.ActuationDelay."""
    from training.league_env import LeagueEnv, MIRROR, policy_step, opp_rows
    from python_ai.rl import abilities as A
    torch.manual_seed(job["seed"])
    spec = {"gamma": 0.999, "w_tower": 1.0, "k": 0.0, "leak_worst": 0.2, "draw": 0.0}
    env = LeagueEnv({"opponent": "teacher", "teacher_stage": 0, "deck_pool": False,
                     "defensive_scenario_prob": 0.0, "scenario_seed": job["seed"],
                     "reward_spec": spec, "league_mix": (0.0, 1.0, 0.0), "act_delay": act_delay})
    slots = A.ability_engine_slots(C.deck_ids())
    obs, info = env.reset(seed=job["seed"])
    assert info["opp_kind"] == MIRROR
    nets = (cand, ref) if job["cand_seat"] == 0 else (ref, cand)
    H = cand.LSTM_HIDDEN
    hc = [(torch.zeros(1, H), torch.zeros(1, H)) for _ in range(2)]

    def ready(seat, inf):
        key = "champion_ability_slot{}_ready" if seat == 0 else "opp_ability_slot{}_ready"
        return torch.as_tensor([[bool(inf.get(key.format(s), False)) for s in slots]],
                               dtype=torch.bool).reshape(1, len(slots))

    plays = [0, 0]
    r, steps = 0.0, 0
    with torch.no_grad():
        while True:
            o0 = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            o1 = torch.as_tensor(info["obs1"], dtype=torch.float32).unsqueeze(0)
            out0 = policy_step(nets[0], o0, hc[0], ready(0, info), temp)
            out1 = policy_step(nets[1], o1, hc[1], ready(1, info), temp)
            hc[0], hc[1] = (out0["h"], out0["c"]), (out1["h"], out1["c"])
            act = {"card_index": int(out0["card"]), "target_x": float(out0["x"]),
                   "target_y": float(out0["y"]), "opp": opp_rows(out1, slots)[0]}
            act.update({k: int(v[0]) for k, v in A.action_dict(out0["ab"], slots, 1).items()})
            obs, r, term, trunc, info = env.step(act)
            plays[0] += int(int(out0["card"]) != C.NOOP)
            plays[1] += int(int(out1["card"]) != C.NOOP)
            steps += 1
            if term or trunc:
                break
    s = job["cand_seat"]
    r_c = float(r) if s == 0 else -float(r)
    return {"seat": s, "win": float(r_c > 0.5), "draw": float(abs(r_c) < 0.5),
            "plays_cand": plays[s], "plays_ref": plays[1 - s], "steps": steps}


def _worker(args):
    # A watcher started before --temp/--act-delay existed passes 3 fields.
    cand_path, ref_path, jobs, *rest = args
    temp, act_delay = (list(rest) + [1.0, None])[:2]
    torch.set_num_threads(1)
    cand, ref = _load(cand_path), _load(ref_path)
    return [play(cand, ref, j, temp, act_delay) for j in jobs]


def jobs_for(n):
    return [{"seed": 777000 + i, "cand_seat": s} for i in range(n) for s in (0, 1)]


def evaluate(cand_path, ref_path, n, procs, temp=1.0, act_delay=None):
    jobs = jobs_for(n)
    chunks = [jobs[i::procs] for i in range(procs)]
    with mp.get_context("spawn").Pool(procs) as pool:
        res = sum(pool.map(_worker, [(cand_path, ref_path, c, temp, act_delay) for c in chunks]), [])
    m = lambda key, rows: float(np.mean([x[key] for x in rows])) if rows else float("nan")  # noqa: E731
    return {"n": len(res), "win_rate": m("win", res), "draw_rate": m("draw", res),
            "win_rate_seat0": m("win", [x for x in res if x["seat"] == 0]),
            "win_rate_seat1": m("win", [x for x in res if x["seat"] == 1]),
            "plays_cand": m("plays_cand", res), "plays_ref": m("plays_ref", res),
            "steps": m("steps", res)}


def _episodes(path):
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        return int(ck.get("episodes_completed", 0)) if isinstance(ck, dict) and "model" in ck else 0
    except Exception:  # noqa: BLE001
        return -1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights")
    p.add_argument("--label", default=None)
    p.add_argument("--watch", default=None, help="run name: poll checkpoints/<run>.pth")
    p.add_argument("--every", type=int, default=2000, help="episodes between evals (watch)")
    p.add_argument("--refs", default=",".join(DEFAULT_REFS),
                   help="comma list; names/globs in checkpoints/ppo_bc26_snapshots or repo paths")
    p.add_argument("--n", type=int, default=24, help="seeds per seat (matches = 2n per ref)")
    p.add_argument("--procs", type=int, default=2)
    p.add_argument("--out", default=None)
    p.add_argument("--temp", type=float, default=1.0, help="sampling temperature, 0 = argmax")
    p.add_argument("--act-delay", default="", help="lo,hi,jitter ticks (empty = none)")
    a = p.parse_args()
    act_delay = parse_delay(a.act_delay)
    refs = [_resolve(r) for r in a.refs.split(",") if r]
    run = a.watch or a.label or "h2h"
    out = a.out or os.path.join(C.REPO, "runs", run, "h2h.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"[h2h] refs: {[os.path.basename(r) for r in refs]} -> {out}", flush=True)

    def record(path, label, ep):
        for ref in refs:
            t0 = time.time()
            s = evaluate(path, ref, a.n, a.procs, a.temp, act_delay)
            rec = {"t": time.time(), "label": label, "episodes": ep, "weights": path,
                   "ref": os.path.basename(ref), "eval_s": round(time.time() - t0, 1), **s}
            if a.temp != 1.0 or act_delay:
                rec.update(temp=a.temp, act_delay=act_delay)
            with open(out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[h2h] {label} ep {ep} vs {os.path.basename(ref)}: win {s['win_rate']:.3f} "
                  f"(seat0 {s['win_rate_seat0']:.3f} seat1 {s['win_rate_seat1']:.3f}) draw "
                  f"{s['draw_rate']:.3f} plays {s['plays_cand']:.1f}/{s['plays_ref']:.1f} "
                  f"n {s['n']} in {rec['eval_s']}s", flush=True)

    if a.weights:
        path = a.weights
        if a.label:   # never read a live checkpoint mid-write: evaluate a copy
            snap = os.path.join(os.path.dirname(out), f"{a.label}.h2h_copy.pth")
            shutil.copyfile(a.weights, snap)
            path = snap
        record(path, a.label or os.path.basename(a.weights), _episodes(path))
    if not a.watch:
        return
    src = os.path.join(C.REPO, "checkpoints", f"{a.watch}.pth")
    last = -10 ** 9
    while True:
        ep = _episodes(src) if os.path.exists(src) else -1
        if ep >= 0 and ep - last >= a.every:
            snap = src + ".h2h_copy"
            shutil.copyfile(src, snap)
            record(snap, a.watch, ep)
            last = ep
        time.sleep(60)


if __name__ == "__main__":
    main()
