"""Smoke run and throughput measurements for the opponent pool (training/pool_env.py).

  smoke   N episodes of a fixed learner checkpoint (no learning) against the
          pool; win rate per opponent kind, per start (tick 0 / mid-match), per
          teacher rung. Episode i uses seed base + i: the result does not depend
          on --procs.
  core    env steps per second on ONE core for each opponent kind and decision
          interval (the learner's own inference is timed apart from the env).
  vec     an AsyncVectorEnv of n workers with the learner batched in this
          process: wall time per vector step for opp_inference worker | main.
  infer   one batched forward of a checkpoint on CPU for several batch sizes
          and thread counts.
  profile where one core's time goes for one opponent kind (cProfile).

    CR_ENGINE=build/gen2/engine .venv/bin/python -m training.pool_smoke smoke \
        --pool training/pool_gen2_default.json --episodes 2000 --procs 4 --out evals/pool_smoke.json
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import time

from training import common as C  # noqa: I001  (first: engine path, CLASH_DECK)

import numpy as np

from training import pool_env as P

LEARNER = os.path.join(C.REPO, "remote_results", "ppo_league26_ep138708.pth")
DEFAULT_POOL = os.path.join(C.REPO, "training", "pool_gen2_default.json")
KIND_CASES = (("ghost", 0), ("ghost", 1), ("clone", 0), ("clone", 1), ("teacher", 0), ("self", 0))


def only(pool, kind, mid):
    """`pool` with all weight on one kind, every (mid=1) or no episode mid-match."""
    w = {k: float(k == kind) for k in P.KIND_NAMES}
    return P.merge_pool(pool, {"weights": w, "p_midmatch": 1.0 if mid else 0.0,
                               "midmatch": {"kinds": [kind] if mid else ["ghost", "clone"]}})


def play(env, learner, seed, temp=1.0, max_steps=None):
    """One episode of `learner` (an OppPolicy on our seat) -> row."""
    import torch
    t0, c0 = time.perf_counter(), time.process_time()
    obs, info = env.reset(seed=seed)
    t_reset, c_reset = time.perf_counter() - t0, time.process_time() - c0
    learner.configure(env.ai_deck, [int(c) for c in info["opp_deck"] if c >= 0], temp)
    gen = torch.Generator()
    gen.manual_seed(int(seed) + 15)
    state = learner.state(1)
    ready = torch.zeros(1, 2, dtype=torch.bool)
    net0 = (env._net.calls, env._net.secs)
    t_env = t_pol = c_env = 0.0
    steps, r, slow = 0, 0.0, 0.0
    first = info
    while True:
        t0 = time.perf_counter()
        card, x, y, ab, state = learner.act(torch.as_tensor(obs).unsqueeze(0), state, ready, gen)
        act = {"card_index": int(card[0]), "target_x": float(x[0]), "target_y": float(y[0]),
               "activate_ability_slot1": int(ab[0, 0]), "activate_ability_slot2": int(ab[0, 1])}
        t1, c1 = time.perf_counter(), time.process_time()
        obs, r, term, trunc, info = env.step(act)
        t2 = time.perf_counter()
        c_env += time.process_time() - c1
        ready[0, 0] = bool(info.get("champion_ability_slot1_ready", False))
        ready[0, 1] = bool(info.get("champion_ability_slot2_ready", False))
        t_pol += t1 - t0
        t_env += t2 - t1
        slow = max(slow, t2 - t1)
        steps += 1
        if term or trunc or (max_steps and steps >= max_steps):
            break
    return {"seed": int(seed), "kind": P.KIND_NAMES[first["opp_kind"]], "mid": int(first["midmatch"]),
            "start_tick": int(first["start_tick"]), "fallback": int(first.get("start_fallback", 0)),
            "deck_id": int(first["opp_deck_id"]),
            "ckpt": first["opp_ckpt"], "rung": int(first["opp_rung"]), "reward": float(r),
            "done": bool(term or trunc), "steps": steps, "end_tick": int(env.game.get_current_tick()),
            "t_reset": t_reset, "t_env": t_env, "c_reset": c_reset, "c_env": c_env, "t_learner": t_pol, "t_slowest_step": slow,
            "opp_calls": env._net.calls - net0[0], "t_opp_net": env._net.secs - net0[1]}


def _worker(args):
    import torch
    torch.set_num_threads(1)
    cfg, learner_path, seeds, temp, max_steps = args
    env = P.PoolEnv(cfg)
    learner = P.OppPolicy(learner_path)
    return [play(env, learner, s, temp, max_steps) for s in seeds]


def run(jobs, procs):
    """jobs: [(env cfg, learner path, seeds, temp, max_steps)] -> rows per job."""
    procs = max(1, min(int(procs), len(jobs)))
    if procs == 1:
        return [_worker(j) for j in jobs]
    with mp.get_context("spawn").Pool(procs) as pool:
        return pool.map(_worker, jobs)


def _env_cfg(args, pool, seed=0, dt=None):
    cfg = {"scenario_seed": seed, "pool": pool, "decision_ticks": int(dt or args.decision_ticks)}
    if args.act_delay:
        cfg["act_delay"] = tuple(int(v) for v in args.act_delay.split(","))
    return cfg


def _rate(rows):
    n = len(rows)
    if not n:
        return None
    w = sum(r["reward"] > 0.5 for r in rows)
    d = sum(abs(r["reward"]) <= 0.5 for r in rows)
    p = w / n
    return {"n": n, "win": p, "draw": d / n, "loss": (n - w - d) / n,
            "se": float(np.sqrt(max(p * (1 - p), 1e-9) / n)),
            "steps": float(np.mean([r["steps"] for r in rows]))}


def summarize(rows):
    out = {"all": _rate(rows), "by_kind": {}, "by_kind_start": {}, "teacher_by_rung": {}, "by_ckpt": {}}
    for k in P.KIND_NAMES:
        out["by_kind"][k] = _rate([r for r in rows if r["kind"] == k])
        for mid in (0, 1):
            sub = [r for r in rows if r["kind"] == k and r["mid"] == mid]
            if sub:
                out["by_kind_start"][f"{k}/{'mid' if mid else 't0'}"] = _rate(sub)
    for rung in sorted({r["rung"] for r in rows if r["kind"] == "teacher"}):
        out["teacher_by_rung"][str(rung)] = _rate([r for r in rows if r["kind"] == "teacher" and r["rung"] == rung])
    for ck in sorted({r["ckpt"] for r in rows if r["ckpt"]}):
        out["by_ckpt"][ck] = _rate([r for r in rows if r["ckpt"] == ck])
    out["midmatch_share"] = float(np.mean([r["mid"] for r in rows]))
    out["midmatch_fallbacks"] = int(sum(r.get("fallback", 0) for r in rows))
    out["distinct_decks"] = len({r["deck_id"] for r in rows})
    return out


def cmd_smoke(args):
    pool = P.load_pool_config(args.pool)
    seeds = [args.seed + i for i in range(args.episodes)]
    procs = max(1, min(args.procs, len(seeds)))
    jobs = [(_env_cfg(args, pool, seed=args.seed + 1000003 * k), args.learner, seeds[k::procs], args.temp, None)
            for k in range(procs)]
    t0 = time.perf_counter()
    parts = run(jobs, procs)
    rows = [None] * len(seeds)
    for k, part in enumerate(parts):
        rows[k::procs] = part
    s = summarize(rows)
    s["wall_s"] = time.perf_counter() - t0
    print(f"{len(rows)} episodes, learner {os.path.basename(args.learner)} at temperature {args.temp}, "
          f"decision_ticks {args.decision_ticks}, {s['wall_s']:.0f} s on {procs} processes")
    print(f"mid-match share {s['midmatch_share']:.3f} (+ {s['midmatch_fallbacks']} that fell back to tick 0), "
          f"distinct opponent decks {s['distinct_decks']}")
    for name, tab in (("kind", s["by_kind"]), ("kind/start", s["by_kind_start"]),
                      ("teacher rung", s["teacher_by_rung"]), ("checkpoint", s["by_ckpt"])):
        for k, v in tab.items():
            if v:
                print(f"  {name:12s} {k:24s} n {v['n']:5d}  win {v['win']:.3f} +- {v['se']:.3f}  "
                      f"draw {v['draw']:.3f}  loss {v['loss']:.3f}  decisions/episode {v['steps']:.0f}")
    v = s["all"]
    print(f"  {'all':12s} {'':24s} n {v['n']:5d}  win {v['win']:.3f} +- {v['se']:.3f}  draw {v['draw']:.3f}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"meta": {"pool": pool, "learner": args.learner, "temp": args.temp, "seed": args.seed,
                                "decision_ticks": args.decision_ticks, "act_delay": args.act_delay,
                                "engine": os.environ.get("CR_ENGINE")},
                       "summary": s, "rows": rows}, fh, indent=1)
        print("wrote", args.out)


def cmd_core(args):
    pool = P.load_pool_config(args.pool)
    cases = [(k, m, dt) for dt in (5, 10) for k, m in KIND_CASES] + [("mix", 0, 5), ("mix", 0, 10)]
    jobs = []
    for i, (k, m, dt) in enumerate(cases):
        p = pool if k == "mix" else only(pool, k, m)
        jobs.append((_env_cfg(args, p, seed=args.seed + i, dt=dt), args.learner,
                     [args.seed + 100 * i + j for j in range(args.episodes)], args.temp, None))
    parts = run(jobs, args.procs)
    out = []
    print("steps/s (cpu) = decisions per second of this process's own CPU time (env.step + reset), the per-core "
          "number; (wall) includes waiting for a core.")
    print("kind      start  dt  episodes  decisions  steps/s cpu  steps/s wall  ms/step  reset ms  "
          "opp-net ms/step  learner ms/step  slowest step ms")
    for (k, m, dt), rows in zip(cases, parts):
        n = sum(r["steps"] for r in rows)
        te = sum(r["t_env"] for r in rows)
        tr = sum(r["t_reset"] for r in rows)
        row = {"kind": k, "mid": m, "dt": dt, "episodes": len(rows), "decisions": n,
               "steps_per_s": n / (te + tr),
               "steps_per_s_cpu": n / sum(r["c_env"] + r["c_reset"] for r in rows),
               "ms_step": 1e3 * sum(r["c_env"] for r in rows) / n,
               "ms_reset": 1e3 * sum(r["c_reset"] for r in rows) / len(rows),
               "ms_opp_net": 1e3 * sum(r["t_opp_net"] for r in rows) / n,
               "ms_learner": 1e3 * sum(r["t_learner"] for r in rows) / n,
               "ms_slowest": 1e3 * float(np.mean([r["t_slowest_step"] for r in rows])),
               "steps_per_episode": n / len(rows)}
        out.append(row)
        print(f"{k:9s} {'mid' if m else 't0 ':5s} {dt:3d} {len(rows):9d} {n:10d} {row['steps_per_s_cpu']:12.0f} "
              f"{row['steps_per_s']:13.0f} "
              f"{row['ms_step']:8.2f} {row['ms_reset']:9.1f} {row['ms_opp_net']:16.2f} "
              f"{row['ms_learner']:16.2f} {row['ms_slowest']:16.1f}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)


def cmd_infer(args):
    import torch
    env = P.PoolEnv(_env_cfg(args, P.load_pool_config(args.pool)))
    obs, _ = env.reset(seed=1)
    for path in (args.learner,) + tuple(args.ckpt or ()):
        pol = P.OppPolicy(path)
        pol.configure(env.ai_deck, env.ai_deck, 1.0)
        for threads in (1, 2, 4):
            torch.set_num_threads(threads)
            for b in (1, 4, 9, 16):
                o = torch.as_tensor(np.stack([obs] * b))
                st, ready = pol.state(b), torch.zeros(b, 2, dtype=torch.bool)
                for _ in range(5):
                    pol.act(o, st, ready)
                t0 = time.perf_counter()
                for _ in range(args.reps):
                    pol.act(o, st, ready)
                ms = 1e3 * (time.perf_counter() - t0) / args.reps
                print(f"{os.path.basename(path):28s} threads {threads}  batch {b:2d}  {ms:6.2f} ms  "
                      f"{ms / b:5.2f} ms/obs")


def cmd_vec(args):
    import gymnasium as gym
    import torch
    torch.set_num_threads(args.threads)
    pool = P.merge_pool(P.load_pool_config(args.pool), {"opp_inference": args.mode})
    if args.kind != "mix":
        pool = only(pool, args.kind, 0)
    n = args.envs
    envs = gym.vector.AsyncVectorEnv(
        [P.make_pool_env(args.seed + i, pool, decision_ticks=args.decision_ticks,
                         act_delay=tuple(int(v) for v in args.act_delay.split(",")) if args.act_delay else None)
         for i in range(n)], context="spawn")
    learner = P.OppPolicy(args.learner)
    deck = list(C.deck_ids())
    learner.configure(deck, deck, args.temp)
    drv = P.PoolOppDriver(n, deck, seed=args.seed) if args.mode == "main" else None
    try:
        obs, infos = envs.reset(seed=[args.seed + i for i in range(n)])
        if drv:
            drv.observe(infos)
        state = learner.state(n)
        ready = torch.zeros(n, 2, dtype=torch.bool)
        t = collections.Counter()
        steps_t = []
        for it in range(args.warmup + args.steps):
            t0 = time.perf_counter()
            card, x, y, ab, state = learner.act(torch.as_tensor(obs), state, ready)
            act = {"card_index": card.numpy(), "target_x": x.numpy().astype(np.float32),
                   "target_y": y.numpy().astype(np.float32),
                   "activate_ability_slot1": np.zeros(n, np.int64), "activate_ability_slot2": np.zeros(n, np.int64)}
            t1 = time.perf_counter()
            if drv:
                act["opp"] = drv.actions()
            t2 = time.perf_counter()
            obs, r, term, trunc, infos = envs.step(act)
            t3 = time.perf_counter()
            if drv:
                drv.observe(infos)
            t4 = time.perf_counter()
            if it >= args.warmup:
                t["learner"] += t1 - t0
                t["opp_main"] += (t2 - t1) + (t4 - t3)
                t["env_step"] += t3 - t2
                steps_t.append(t3 - t2)
        tot = sum(t.values())
        k = args.steps
        print(f"mode {args.mode:6s} kind {args.kind:7s} envs {n:2d} dt {args.decision_ticks:2d} threads {args.threads}  "
              f"vector step {1e3 * tot / k:6.2f} ms = envs.step {1e3 * t['env_step'] / k:6.2f} "
              f"(median {1e3 * float(np.median(steps_t)):5.2f}, p95 {1e3 * float(np.percentile(steps_t, 95)):6.2f}) "
              f"+ learner {1e3 * t['learner'] / k:5.2f} + opp in main {1e3 * t['opp_main'] / k:5.2f}  "
              f"-> {n * k / tot:7.0f} learner decisions/s")
    finally:
        envs.close()


def _bucket(fname, name):
    """Which part of a decision a profiled function belongs to (by its own time)."""
    if "numpy.array" in name or "numpy.asarray" in name or "get_observation_for_team" in name:
        return "observation: engine list -> numpy (both seats)"
    if "step_self_play" in name or "clash_royale_env.step" in name:
        return "engine simulation (step_self_play, incl. next-card probes)"
    if "clash_royale_env" in name:
        return "engine getters, snapshot/restore"
    if "gym_wrapper.py" in fname:
        return "engine wrapper python (step, info dict)"
    if "/torch/" in fname or "torch" in name or fname.endswith(("net.py", "net_v2.py", "net_v3.py")):
        return "opponent network forward (torch)"
    if fname.endswith("obs_compat.py"):
        return "observation adapter (obs_compat)"
    for key, label in (("obs_v2.py", "tail: tracker, render, next card (obs_v2)"),
                       ("reward.py", "reward shaping, action delay, in-flight cards (reward.py)"),
                       ("ghost.py", "ghost script, prefix replay, guard (ghost.py)"),
                       ("pool_env.py", "pool env: draw, publish, privileged vector"),
                       ("teacher.py", "scripted teacher"), ("/opponents/", "scripted teacher"),
                       ("/numpy/", "numpy (small ops)")):
        if key in fname:
            return label
    return "other python"


def cmd_profile(args):
    """Own-time shares of one core's work, learner inference excluded (cProfile:
    python-heavy parts are inflated a little relative to C calls)."""
    import cProfile
    import pstats
    import torch
    torch.set_num_threads(1)
    pool = P.load_pool_config(args.pool)
    if args.kind != "mix":
        pool = only(pool, args.kind, args.mid)
    env = P.PoolEnv(_env_cfg(args, pool))
    learner = P.OppPolicy(args.learner)
    play(env, learner, args.seed, args.temp)                # warm: checkpoint loads, caches
    pr = cProfile.Profile()
    orig = learner.act

    def unprofiled(*a, **k):
        pr.disable()
        try:
            return orig(*a, **k)
        finally:
            pr.enable()
    learner.act = unprofiled
    pr.enable()
    rows = [play(env, learner, args.seed + 1 + j, args.temp) for j in range(args.episodes)]
    pr.disable()
    steps = sum(r["steps"] for r in rows)
    parts = collections.Counter()
    for (fname, _line, name), v in pstats.Stats(pr).stats.items():
        parts[_bucket(fname, name)] += v[2]
    total = sum(parts.values())
    print(f"kind {args.kind} {'mid' if args.mid else 't0'} dt {args.decision_ticks}: {steps} decisions, "
          f"{1e3 * total / steps:.2f} ms/decision under the profiler")
    for name, v in parts.most_common():
        print(f"  {100 * v / total:5.1f} %  {1e3 * v / steps:7.3f} ms  {name}")
    return {"kind": args.kind, "mid": args.mid, "dt": args.decision_ticks, "steps": steps,
            "shares": {k: v / total for k, v in parts.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("smoke", "core", "infer", "vec", "profile"):
        p = sub.add_parser(name)
        p.add_argument("--pool", default=DEFAULT_POOL)
        p.add_argument("--learner", default=LEARNER)
        p.add_argument("--temp", type=float, default=1.0)
        p.add_argument("--seed", type=int, default=20261122)
        p.add_argument("--decision-ticks", type=int, default=10)
        p.add_argument("--act-delay", default=None, help="lo,hi,jitter ticks")
        p.add_argument("--episodes", type=int, default=2000 if name == "smoke" else 8)
        p.add_argument("--procs", type=int, default=4)
        p.add_argument("--out", default=None)
        if name == "infer":
            p.add_argument("--ckpt", action="append")
            p.add_argument("--reps", type=int, default=100)
        if name == "vec":
            p.add_argument("--envs", type=int, default=4)
            p.add_argument("--mode", default="worker", choices=("worker", "main"))
            p.add_argument("--threads", type=int, default=2)
            p.add_argument("--steps", type=int, default=1500)
            p.add_argument("--warmup", type=int, default=100)
        if name in ("vec", "profile"):
            p.add_argument("--kind", default="mix", choices=("mix",) + P.KIND_NAMES)
        if name == "profile":
            p.add_argument("--mid", type=int, default=0)
    args = ap.parse_args()
    {"smoke": cmd_smoke, "core": cmd_core, "infer": cmd_infer, "vec": cmd_vec, "profile": cmd_profile}[args.cmd](args)


if __name__ == "__main__":
    main()
