"""unit_fate_sim.work with extra logging (scratch copy, the tool itself is untouched):
per body a trajectory and its tower-hit ticks, per decision the opponent's hand and elixir, and for the
teacher every scored candidate (tick, card, x, y own frame, kind, score, margin) and the choice."""
import json, sys, re
sys.path.insert(0, "/Users/samueldoane/cr-robot")
sys.path.insert(0, "/Users/samueldoane/cr-robot/evals/tools")
import unit_fate_sim as U

src = open("/Users/samueldoane/cr-robot/evals/tools/unit_fate_sim.py").read()
w = src[src.index("def work(job):"):src.index("def main():")]
rep = [
    ("    k = \"ghost\" if kind == \"live\" else kind\n",
     "    k = \"ghost\" if kind == \"live\" else kind\n    _patch_teacher()\n"),
    ("        body = {}                                    # unit id -> dict\n",
     "        body = {}                                    # unit id -> dict\n        hlog = []\n        leak = [0.0]\n        TLOG.clear()\n"),
    ("                tick = int(game.get_current_tick())\n",
     "                tick = int(game.get_current_tick())\n                hlog.append((tick, list(game.get_hand_for_team(1))[:4], round(float(game.get_elixir_for_team(1)), 2)))\n"
     "                leak[0] += (5 * (1 + (tick >= 1200) + (tick >= 2400)) / 28.0) if float(game.get_elixir_for_team(1)) >= 9.99 else 0.0\n"),
    ("                    b.update(t1=tick, maxy=max(b[\"maxy\"], u[\"y\"]), mind=min(b[\"mind\"], d), hp=u[\"hp\"])\n",
     "                    b.update(t1=tick, maxy=max(b[\"maxy\"], u[\"y\"]), mind=min(b[\"mind\"], d), hp=u[\"hp\"])\n"
     "                    b.setdefault(\"traj\", []).append((tick, round(u[\"x\"], 2), round(u[\"y\"], 2)))\n"),
    ("                dmg[hrow[3]] += hrow[6]\n",
     "                dmg[hrow[3]] += hrow[6]\n                hitt.setdefault(hrow[3], []).append((int(hrow[0]), int(hrow[5]), float(hrow[6])))\n"),
    ("        dmg = collections.Counter()\n", "        dmg = collections.Counter()\n        hitt = {}\n"),
    ("            b[\"dmg\"] = float(dmg.get(uid, 0))\n", "            b[\"dmg\"] = float(dmg.get(uid, 0))\n            b[\"hit_ticks\"] = hitt.get(uid, [])\n"),
    ("\"opp_plays\": opp_plays, \"bodies\": list(body.values())})",
     "\"opp_plays\": opp_plays, \"bodies\": list(body.values()), \"hand\": hlog, \"tlog\": list(TLOG),\n                    \"deck1\": deck1, \"spent1\": float(game.get_elixir_spent(1)),"
     " \"dmg1\": float(game.get_tower_damage_dealt(1)), \"leak1\": leak[0], \"spec\": dict(env.spec),"
     " \"balog\": list((getattr(env._net, \"build_answer_log\", None) if env.spec[\"kind\"] == \"clone\" else getattr(env._util, \"build_answer_log\", None)) or [])})"),
]
for a, b in rep:
    assert w.count(a) == 1, a
    w = w.replace(a, b)

TLOG = []


def _patch_teacher():
    from python_ai.opponents import teacher as T
    if getattr(T.UtilityTeacher, "_probe", False):
        return
    T.UtilityTeacher._probe = True
    s0, a0 = T.UtilityTeacher.score, T.UtilityTeacher.act
    cur = {"t": -1, "el": 0.0}

    def score(self, env, cand, baseline, obs_own):
        sc = s0(self, env, cand, baseline, obs_own)
        try:
            m = self.margin_for(cand, cur["el"])
            pass
        except Exception as e:  # noqa
            TLOG.append(("err", cur["t"], str(e)))
        return sc

    def act(self, env, obs_own):
        cur["t"] = int(env.get_current_tick())
        cur["el"] = float(T.tactics.own_elixir(obs_own))
        hand = [int(c) for c in env.get_hand_for_team(self.team)][:4]
        cur["role"] = None
        out = a0(self, env, obs_own)
        if int(out[0]) < 4:
            ours = [(int(u["card_id"]), round(float(u["y"]), 1), int(u["hp"])) for u in env.get_units()
                    if u["team"] != self.team and not u["tower"] and u["hp"] > 0]
            TLOG.append(("a", cur["t"], self.team, hand, round(float(env.get_elixir_for_team(self.team)), 2),
                         int(out[0]), float(out[1]), float(out[2]), self.last_kind,
                         round(float(T.tactics.threat_level(obs_own)), 1), cur["role"], ours))
        else:
            TLOG.append(("a", cur["t"], self.team, hand, round(float(env.get_elixir_for_team(self.team)), 2),
                         int(out[0]), float(out[1]), float(out[2]), self.last_kind))
        return out

    c0 = T.UtilityTeacher.commit

    def commit(self, cand):
        cur["role"] = getattr(cand, "role", None)
        return c0(self, cand)

    T.UtilityTeacher.score, T.UtilityTeacher.act, T.UtilityTeacher.commit = score, act, commit


g = dict(U.__dict__)
g.update(TLOG=TLOG, _patch_teacher=_patch_teacher, __name__="r7_probe")
exec(compile(w, "work", "exec"), g)
U.work = g["work"]
work = g["work"]

if __name__ == "__main__":
    import multiprocessing as mp
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--kind"); ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--procs", type=int, default=4); ap.add_argument("--seed", type=int, default=8700)
    ap.add_argument("--pool-json", default=""); ap.add_argument("--out")
    a = ap.parse_args()
    seeds = [a.seed + i for i in range(a.n)]
    over = json.loads(a.pool_json) if a.pool_json else None
    jobs = [(a.ckpt, seeds[k::a.procs], a.kind, over, [15]) for k in range(a.procs)]
    import r7_probe as BP  # spawn pickles work as bld_probe.work
    with mp.get_context("spawn").Pool(a.procs) as p:
        res = [e for chunk in p.map(BP.work, jobs, chunksize=1) for e in chunk]
    json.dump(res, open(a.out, "w"))
    print("done", len(res))
