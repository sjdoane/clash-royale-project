"""eval_panel.py's fixed-seed panel at any decision step, for v1 or v2 weights.

    .venv/bin/python -m training.eval_panel_hz --weights checkpoints/bc_v2_5hz.pth --label bc_v2_5hz \
        --decision-ticks 2 --procs 14
    .venv/bin/python -m training.eval_panel_hz --weights checkpoints/bc_hog26.pth --label bc_v1 \
        --decision-ticks 10 --procs 14

Same 192 jobs as eval_panel (its `panel()`; teacher rungs 0..10 on the mirror
deck, the top-10 real decks at rung 2, the same seeds), the same sampling
policy, the same per-match statistics. What changes with --decision-ticks k:

  * our agent decides every k ticks (`env.step(action, k)`);
  * the UtilityTeacher still decides every 10 ticks. Its plan clock
    (COMBO_FOLLOWUP_DELAY_TICKS) and its per-decision epsilon assume one call
    per 10 ticks; calling it 5x as often would make a different (more random,
    faster-reacting) opponent. On off-beat steps it is swapped for a holding
    stub, so it sees exactly the decision schedule of the 1 Hz panel;
  * the leak statistic uses k ticks of regeneration per step.

At k = 10 this is eval_panel.play exactly (no off-beat steps).
"""
import argparse
import json
import multiprocessing as mp
import os
import time

from training import common as C

import numpy as np
import torch
from torch.distributions import Categorical

OPP_DECISION_TICKS = 10      # the teacher's decision clock (opponents/teacher.py COMBO_FOLLOWUP_DELAY_TICKS)


class _Hold:
    """Teacher stand-in for off-beat steps: hold, no ability."""

    def act(self, game, obs):
        return C.HAND_SIZE, 0.0, 0.0

    def ability_flags(self, game, obs):
        return False, False


def make_env_class():
    from python_ai.envs import gym_wrapper
    from training.reward import RewardEnv, elixir_leak

    class RewardEnvHz(RewardEnv):
        """RewardEnv stepping `decision_ticks` per decision, teacher on its own clock."""

        def __init__(self, env_config=None):
            cfg = dict(env_config or {})
            self.decision_ticks = int(cfg.pop("decision_ticks", 2))
            self.opp_ticks = int(cfg.pop("opp_decision_ticks", OPP_DECISION_TICKS))
            assert self.opp_ticks % self.decision_ticks == 0, "opponent clock must align with ours"
            super().__init__(cfg)
            self._tick = 0

        def reset(self, seed=None, options=None):
            self._tick = 0
            return super().reset(seed=seed, options=options)

        def step(self, action, skip_frames=None):
            k = self.decision_ticks
            obs0 = self._last_obs
            teacher = self.teacher
            if teacher is not None and self._tick % self.opp_ticks:
                self.teacher = _Hold()
            try:
                obs, reward, term, trunc, info = gym_wrapper.MicroRoyaleEnv.step(self, action, k)
            finally:
                self.teacher = teacher
            self._tick += k
            F, _ = self._spec.shaping(obs0, obs, [float(term)])
            leak = elixir_leak(obs0, obs, ticks=k)
            info["r_tower"] = float(F[0])
            info["leak_elixir"] = float(leak[0])
            info["r_shape"] = float(F[0] - self._spec.c_leak * leak[0])
            if term and self._sampler is not None:
                self._sampler.record(self._cur_bucket, reward > 0.5)
            self._last_obs = obs
            return obs, reward, term, trunc, info

    return RewardEnvHz


def play(net, job, spec, decision_ticks):
    from python_ai.rl import abilities as A
    from training.reward import tower_damage_fracs
    Env = make_env_class()
    torch.manual_seed(job["seed"])
    env = Env({"opponent": "teacher", "teacher_stage": job["rung"], "deck_pool": False,
               "defensive_scenario_prob": 0.0, "scenario_seed": job["seed"],
               "reward_spec": spec, "fixed_opp_deck": job["deck"], "decision_ticks": decision_ticks})
    env.set_teacher_stage(job["rung"])
    slots = A.ability_engine_slots(C.deck_ids())
    obs, info = env.reset(seed=job["seed"])
    h = (torch.zeros(1, net.LSTM_HIDDEN), torch.zeros(1, net.LSTM_HIDDEN))
    plays, leak, steps, r = 0, 0.0, 0, 0.0
    t_net = 0.0
    ready = A.ready_from_infos({k: np.asarray([v]) for k, v in info.items()}, 1, slots)
    with torch.no_grad():
        while True:
            t0 = time.perf_counter()
            o = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            mask = net.affordability_mask(o)
            f, emb, sp, hi = net.extract_features_hires(o)
            cl, a1, a2, _, h = net.step_lstm_and_card(f, h, mask)
            ci = Categorical(logits=cl).sample()
            pl = net.placement_given_card(h[0], emb, ci, o, sp, hires_map=hi)
            cell = Categorical(logits=pl).sample()
            ab, _, _ = A.sample([x for x in (a1, a2) if x is not None], ready)
            t_net += time.perf_counter() - t0
            x, y = net.cell_to_xy(cell)
            act = {"card_index": int(ci), "target_x": float(x), "target_y": float(y)}
            act.update({k: int(v[0]) for k, v in A.action_dict(ab, slots, 1).items()})
            obs, r, term, trunc, info = env.step(act)
            plays += int(int(ci) != C.NOOP)
            leak += info.get("leak_elixir", 0.0)
            steps += 1
            ready = A.ready_from_infos({k: np.asarray([v]) for k, v in info.items()}, 1, slots)
            if term or trunc:
                break
    d_own, d_en = tower_damage_fracs(obs)
    return {"opp": job["opp"], "win": float(r > 0.5), "draw": float(abs(r) < 0.5),
            "plays": plays, "leak": leak, "steps": steps, "net_ms": 1000 * t_net / max(1, steps),
            "dmg_dealt": float(d_en.sum()), "dmg_taken": float(d_own.sum())}


def _worker(args):
    path, jobs, spec, k = args
    torch.set_num_threads(1)
    from training.net_v2 import load_any
    net = load_any(path)
    return [play(net, j, spec, k) for j in jobs]


def evaluate(path, jobs, procs, decision_ticks, spec=None):
    spec = spec or {"gamma": 0.999, "w_tower": 1.0, "k": 0.0, "leak_worst": 0.2, "draw": 0.0}
    chunks = [jobs[i::procs] for i in range(procs)]
    with mp.get_context("spawn").Pool(procs) as pool:
        res = sum(pool.map(_worker, [(path, c, spec, decision_ticks) for c in chunks]), [])
    agg = {}
    for m in res:
        a = agg.setdefault(m["opp"], {"n": 0, **{k: 0 for k in ("win", "draw", "plays", "leak", "steps",
                                                                 "dmg_dealt", "dmg_taken")}})
        a["n"] += 1
        for k in ("win", "draw", "plays", "leak", "steps", "dmg_dealt", "dmg_taken"):
            a[k] += m[k]
    per = {}
    for k, a in agg.items():
        n = a.pop("n")
        per[k] = {kk: round(v / n, 4) for kk, v in a.items()}
        per[k]["n"] = n
    rung = [m for m in res if m["opp"].startswith("rung")]
    real = [m for m in res if not m["opp"].startswith("rung")]
    summ = {"win_rate": float(np.mean([m["win"] for m in res])),
            "draw_rate": float(np.mean([m["draw"] for m in res])),
            "plays_per_match": float(np.mean([m["plays"] for m in res])),
            "leak_per_match": float(np.mean([m["leak"] for m in res])),
            "dmg_dealt": float(np.mean([m["dmg_dealt"] for m in res])),
            "dmg_taken": float(np.mean([m["dmg_taken"] for m in res])),
            "net_ms_per_decision": float(np.mean([m["net_ms"] for m in res])),
            "n": len(res), "decision_ticks": decision_ticks,
            "win_rate_rungs": float(np.mean([m["win"] for m in rung])),
            "win_rate_real_decks": float(np.mean([m["win"] for m in real]))}
    return summ, per, res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--label", default=None)
    p.add_argument("--decision-ticks", type=int, default=2)
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--n-rung", type=int, default=12)
    p.add_argument("--n-deck", type=int, default=12)
    p.add_argument("--deck-rung", type=int, default=2)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    from training.eval_panel import panel
    jobs = panel(a.n_rung, a.n_deck, a.deck_rung)
    label = a.label or os.path.basename(a.weights)
    out = a.out or os.path.join(C.REPO, "runs", "eval_hz", "eval.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()
    summ, per, res = evaluate(a.weights, jobs, a.procs, a.decision_ticks)
    rec = {"t": time.time(), "label": label, "weights": a.weights, "eval_s": round(time.time() - t0, 1),
           **summ, "per_opponent": per}
    with open(out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    with open(out.replace(".jsonl", f".{label}.matches.jsonl"), "w") as fh:
        for m in res:
            fh.write(json.dumps(m) + "\n")
    print(f"[eval_hz] {label} @ {a.decision_ticks} ticks: win {summ['win_rate']:.3f} "
          f"(rungs {summ['win_rate_rungs']:.3f}, real decks {summ['win_rate_real_decks']:.3f}) "
          f"plays {summ['plays_per_match']:.1f} leak {summ['leak_per_match']:.1f} "
          f"net {summ['net_ms_per_decision']:.2f} ms/decision in {rec['eval_s']}s", flush=True)
    for k, v in per.items():
        print(f"  {k:>8}: win {v['win']:.2f} plays {v['plays']:.1f} leak {v['leak']:.1f}", flush=True)


if __name__ == "__main__":
    main()
