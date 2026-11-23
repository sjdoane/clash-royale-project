"""p(play) per decision in the simulator, by elixir and by own Princess towers down: the simulator-side
twin of live_play_rate.py. Conditions: the "live" evaluation (eval_pair.sh) or --clean (true observations,
trained tail).

usage: CR_ENGINE=$PWD/build/gen2_c/engine .venv/bin/python evals/tools/sim_play_rate.py CKPT [--n 200] [--clean] [--out x.json]
"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from training import common as C  # noqa: E402
from training import eval_ghost as EG  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("weights")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--clean", action="store_true")
    p.add_argument("--out", default=None, help="save the per-decision records (.npy: match, elixir, towers down, p_play)")
    p.add_argument("--part", default="0/1", help="i/k: take every k-th match starting at i (to run in parallel)")
    p.add_argument("--force-idle", type=float, default=0.0,
                   help="seconds: once per match, from a seeded time in 20-100 s, every action is replaced by "
                        "'play nothing' for this long (a stretch in which the bot did not act, whatever the cause)")
    p.add_argument("--dump-obs", default=None, help="save every 7th decision's full observation to this .npy")
    p.add_argument("--merge", nargs="*", default=None, help="print the table from saved .npy records instead of playing")
    a = p.parse_args()
    if a.merge:
        table(np.concatenate([np.load(f) for f in a.merge]), a.weights)
        return
    from training import ghost as G
    from training import obs_compat, sim2real
    from training.h2h import parse_delay
    from training.net_v2 import sample_logits
    from python_ai.rl import abilities as A
    torch.set_num_threads(2)
    cv = EG.convention_of({"convention": None, "timing": None, "levels": None})
    jobs = EG.panel(G.get_store(timing=cv.timing), a.n, EG.DEFAULT_SEED, "val", "exact")
    pi, pk = (int(v) for v in a.part.split("/"))
    jobs = jobs[pi::pk]
    net, info0 = obs_compat.load_policy(a.weights)
    S, X = C.SPATIAL_SIZE, int(C.EXTRA)
    NBC = C.BOARD_H * C.BOARD_W
    recs = []
    keep = []
    for job in jobs:
        torch.manual_seed(job["seed"])
        env_cfg = {"ghost": (job["m"], job["side"]), "ghost_convention": cv, "scenario_seed": job["seed"],
                   "reward_spec": dict(G.DEFAULT_SPEC), "act_delay": parse_delay("3,6,1"),
                   "obs_v2_noise": {"names": info0.get("names") or "spells"}}
        if cv.tower_troops and G.HAS_TOWER_TROOPS:
            st = G.get_store(timing=cv.timing)
            tk = int(st.tower_key[job["m"], job["side"]])
            env_cfg["opp_tower_troop"] = G.tower_troop(st.keys[tk] if tk >= 0 else None)
        if a.clean:
            env = G.GhostEnv(env_cfg)
        else:
            env_cfg["obs_noise"] = {"path": "evals/cfg/board_default_hp_live.json"}
            env_cfg["sim2real"] = sim2real.cli_spec("v1", elixir_est=True)
            env = sim2real.ghost_env_class()(env_cfg)
        obs, info = env.reset(seed=job["seed"])
        if hasattr(net, "set_opp_deck"):
            net.set_opp_deck(env.ghost.deck)
        net.set_sampling(what_temp=1.0, where_temp=1.0, when_temp=1.0, when_bias=0.0)
        dt = int(info0.get("decision_ticks") or 5)
        slots = A.ability_engine_slots(C.deck_ids())
        h = (torch.zeros(1, net.LSTM_HIDDEN), torch.zeros(1, net.LSTM_HIDDEN))
        ready = A.ready_from_infos({k: np.asarray([v]) for k, v in info.items() if k.startswith("champion")}, 1, slots)
        idle0 = float(np.random.default_rng([int(job["seed"]), 77]).uniform(20.0, 100.0)) if a.force_idle > 0 else -1.0
        k = 0
        with torch.no_grad():
            while True:
                tsec = k * dt / 10.0
                k += 1
                idle_s = tsec - idle0 if (idle0 >= 0 and idle0 <= tsec < idle0 + a.force_idle) else -1.0
                since = tsec - (idle0 + a.force_idle) if (idle0 >= 0 and tsec >= idle0 + a.force_idle) else -1.0
                o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                if a.dump_obs and len(recs) % 7 == 0:
                    keep.append(np.asarray(obs, np.float32).copy())
                mask = net.affordability_mask(o)
                f, emb, sp, hi = net.extract_features_hires(o)
                cl, a1, a2, _, h = net.step_lstm_and_card(f, h, mask)
                pr = torch.softmax(cl, -1)[0]
                recs.append((job["i"], float(obs[S]) * 10.0, int(obs[X + 4] <= 0.0) + int(obs[X + 5] <= 0.0),
                             1.0 - float(pr[C.NOOP]), idle_s, since,
                             float((obs[10 * NBC:11 * NBC].reshape(C.BOARD_H, C.BOARD_W)[:15] > 0).sum()),
                             float(max(obs[c * NBC:(c + 1) * NBC].reshape(C.BOARD_H, C.BOARD_W)[:15].max() for c in (4, 5, 6))),
                             float(obs[len(obs) - 148 + 145])))
                ci = sample_logits(cl, 1.0)
                if idle_s >= 0:
                    ci = torch.full_like(ci, C.NOOP)
                pl = net.placement_given_card(h[0], emb, ci, o, sp, hires_map=hi)
                cell = sample_logits(pl, 1.0)
                ab, _, _ = A.sample([x for x in (a1, a2) if x is not None], ready)
                x, y = net.cell_to_xy(cell)
                act = {"card_index": int(ci), "target_x": float(x), "target_y": float(y)}
                act.update({k: int(v[0]) for k, v in A.action_dict(ab, slots, 1).items()})
                obs, r, term, trunc, info = env.step(act, skip_frames=dt)
                ready = A.ready_from_infos({k: np.asarray([v]) for k, v in info.items() if k.startswith("champion")}, 1, slots)
                if term or trunc:
                    break
    R = np.asarray(recs, dtype=np.float64)
    if a.out:
        np.save(a.out, R)
    if a.dump_obs:
        np.save(a.dump_obs, np.stack(keep))
    table(R, f"{a.weights} ({'clean' if a.clean else 'live conditions'})")


def table(R, label):
    print(f"{label}: {len(np.unique(R[:, 0]))} matches, {len(R)} decisions")
    for name, lo, hi in (("under 7 elixir", -1, 7), ("7-9 elixir", 7, 9.5), ("full bar (9.5+)", 9.5, 11)):
        row = []
        for nd in (0, 1, 2):
            m = (R[:, 1] >= lo) & (R[:, 1] < hi) & (R[:, 2] == nd)
            k = int(m.sum())
            row.append(f"{nd} down {R[m, 3].mean():.3f} (n={k})" if k else f"{nd} down - (n=0)")
        print(f"  p(play) {name:16s}: " + ", ".join(row))
    if R.shape[1] > 4 and (R[:, 4] >= 0).any():
        print("  inside the forced idle stretch, p(play) per decision at a full bar by seconds idle:")
        for lo, hi in ((0, 3), (3, 6), (6, 10), (10, 15), (15, 20), (20, 30)):
            m = (R[:, 4] >= lo) & (R[:, 4] < hi) & (R[:, 1] >= 9.5)
            if m.any():
                print(f"    {lo:2d}-{hi:2d} s: {R[m, 3].mean():.3f} (n={int(m.sum())})")
        print("  after it ends (the policy is free again), p(play) per decision by seconds since:")
        for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 20)):
            m = (R[:, 5] >= lo) & (R[:, 5] < hi)
            if m.any():
                print(f"    {lo:2d}-{hi:2d} s: {R[m, 3].mean():.3f} (n={int(m.sum())}), at a full bar "
                      f"{R[m & (R[:, 1] >= 9.5), 3].mean() if (m & (R[:, 1] >= 9.5)).any() else float('nan'):.3f}")


if __name__ == "__main__":
    main()
