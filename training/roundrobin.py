"""Round-robin Bradley-Terry Elo over a set of checkpoints (the selection rule).

    .venv/bin/python -m training.roundrobin --n 24 --temp 0.5 --out runs/rr/final.json \
        a=checkpoints/ppo_v2.pth b=checkpoints/ppo_bc26_snapshots/x.pth ...

Every pair plays --n seeded matches per seat, seats swapped (training/h2h.play,
so the same LeagueEnv mirror setup the h2h watcher uses; v1 and net_v2
checkpoints mix freely). A Bradley-Terry model is fitted to all games (a draw
counts half a win to each side) by the MM algorithm, mapped to Elo
(400*log10), anchored so the first entry is 0. CIs come from --boot bootstrap
resamples of the games within each pair.

Opponent activity is logged per entry: its plays per game, and the plays per
game of the entries it beat, so a rating earned against passive play is
visible (FLZ's caution: a net that wins by out-waiting a passive one).
"""
import argparse
import itertools
import json
import math
import multiprocessing as mp
import os
import time

import numpy as np
import torch

from training import h2h as H


def _worker(args):
    i, j, pi, pj, jobs, temp, act_delay = args
    torch.set_num_threads(1)
    a, b = H._load(pi), H._load(pj)
    rows = []
    for job in jobs:
        r = H.play(a, b, job, temp, act_delay)
        rows.append({"i": i, "j": j, "score_i": r["win"] + 0.5 * r["draw"],
                     "plays_i": r["plays_cand"], "plays_j": r["plays_ref"], "seat_i": r["seat"]})
    return rows


def bt_fit(games, k, iters=500, prior=0.5):
    """games: list of (i, j, score_i). Returns log-strengths (k,), MM updates.

    `prior` adds that many virtual draws between every entry and a virtual
    average player, which keeps a 100% / 0% entry finite."""
    W = np.zeros((k, k))
    for i, j, s in games:
        W[i, j] += s
        W[j, i] += 1.0 - s
    N = W + W.T
    p = np.ones(k)
    for _ in range(iters):
        wins = W.sum(1) + prior
        denom = (N / (p[:, None] + p[None, :])).sum(1) + 2 * prior / (p + 1.0)
        new = wins / denom
        new /= np.exp(np.mean(np.log(new)))
        if np.max(np.abs(new - p)) < 1e-10:
            p = new
            break
        p = new
    return np.log(p)


def elo(logp):
    e = 400.0 / math.log(10) * logp
    return e - e[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entries", nargs="+", help="label=path")
    ap.add_argument("--n", type=int, default=24, help="seeds per seat per pair")
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--act-delay", default="")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    labels, paths = zip(*(e.split("=", 1) for e in a.entries))
    k = len(labels)
    act_delay = H.parse_delay(a.act_delay)
    jobs = H.jobs_for(a.n)
    tasks = []
    for i, j in itertools.combinations(range(k), 2):
        per = max(1, len(jobs) // max(1, a.procs // 2))
        for c in range(0, len(jobs), per):
            tasks.append((i, j, paths[i], paths[j], jobs[c:c + per], a.temp, act_delay))
    t0 = time.time()
    rows = []
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for r in pool.imap_unordered(_worker, tasks):
            rows.extend(r)
    games = [(r["i"], r["j"], r["score_i"]) for r in rows]
    e = elo(bt_fit(games, k))

    rng = np.random.default_rng(0)
    by_pair = {}
    for g in games:
        by_pair.setdefault(g[:2], []).append(g)
    boots = []
    for _ in range(a.boot):
        g = [pg[t] for pg in by_pair.values() for t in rng.integers(0, len(pg), len(pg))]
        boots.append(elo(bt_fit(g, k, iters=200)))
    boots = np.asarray(boots)
    lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)

    plays = {x: [] for x in range(k)}
    beaten_plays = {x: [] for x in range(k)}
    score = {x: [] for x in range(k)}
    pair = {}
    for r in rows:
        i, j, s = r["i"], r["j"], r["score_i"]
        plays[i].append(r["plays_i"])
        plays[j].append(r["plays_j"])
        score[i].append(s)
        score[j].append(1 - s)
        if s > 0.5:
            beaten_plays[i].append(r["plays_j"])
        elif s < 0.5:
            beaten_plays[j].append(r["plays_i"])
        pair.setdefault(f"{labels[i]}|{labels[j]}", []).append(s)
    table = []
    for x in np.argsort(-e):
        table.append({"label": labels[x], "path": paths[x], "elo": round(float(e[x]), 1),
                      "ci95": [round(float(lo[x]), 1), round(float(hi[x]), 1)],
                      "score": round(float(np.mean(score[x])), 4), "games": len(score[x]),
                      "plays_per_game": round(float(np.mean(plays[x])), 2),
                      "beaten_opp_plays_per_game": round(float(np.mean(beaten_plays[x])), 2)
                      if beaten_plays[x] else None,
                      "p_best": round(float(np.mean(np.argmax(boots, 1) == x)), 3)})
    rec = {"t": time.time(), "temp": a.temp, "act_delay": act_delay, "n_per_seat": a.n,
           "wall_s": round(time.time() - t0, 1), "table": table,
           "pair_scores": {kk: round(float(np.mean(v)), 3) for kk, v in pair.items()}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"[rr] {len(games)} games in {rec['wall_s']}s, T={a.temp}, delay={act_delay}")
    for t in table:
        print(f"  {t['label']:<24} elo {t['elo']:>7.1f} [{t['ci95'][0]:>7.1f},{t['ci95'][1]:>7.1f}] "
              f"score {t['score']:.3f} plays/g {t['plays_per_game']:5.1f} "
              f"beaten-opp plays/g {t['beaten_opp_plays_per_game']} P(best) {t['p_best']}")


if __name__ == "__main__":
    main()
