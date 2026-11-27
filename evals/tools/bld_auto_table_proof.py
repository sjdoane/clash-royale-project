"""Proof that training/bld_teacher.py with the engine-picked pull table ("table": "auto", TABLES) behaves exactly as
the backup training/bld_teacher.py.bak_pre_gen2k1126 on gen2_j with the bld opponent ON (2026-11-26). One arm a process:
  arm old   training.bld_teacher is loaded from the backup (the site hook, if any, still applied)
  arm new   today's file
Each arm writes evals/bld_opp/auto_proof_<arm>.json:
  A) 300 env.reset(seed) with the running gen2_j trainer's --pool-override (bld share 0.15): spec, observation hash,
     game state after reset;
  B) whole matches with the policy (opp_scorecard_bld.work, act delay 13,15,1), kind bld (share 1.0): spec, every
     opponent play, the build-answer log (cells fired), result, tower damage, units met.
usage: (gen2_j run env) .venv/bin/python evals/tools/bld_auto_table_proof.py ARM [N_MATCH]
"""
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "evals/tools")
ARM = sys.argv[1]
NM = int(sys.argv[2]) if len(sys.argv) > 2 else 40
CKPT = "runs/bc3/g3j_L13_Wnone_pk_fxafbw/ep4.pth"
BACKUP = "training/bld_teacher.py.bak_pre_gen2k1126"
#: PID 42937's --pool-override (2026-11-26, gen2_j g2_pocket_fresh)
POOL = json.loads('{"weights":{"ghost":0.20,"clone":0.15,"teacher":0.45,"self":0.20},"p_midmatch":0.215,"clone":'
                  '{"when_bias":0.2,"meta_share":0.75,"build_answer_p":1.0,"no_delay":true},"teacher":{"meta_share":'
                  '0.75,"lag_share":0.3,"build_answer_p":1.0,"build_reserve":true},"live":{"path":"data/ghost/live3",'
                  '"share":0.06,"p_mid":0.67},"meta":{"path":"data/decks/meta_api_pool_live4.json"},"behind":'
                  '{"share":0.2},"bld":{"share":0.15}}')

if ARM == "old":
    class _Old(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "training.bld_teacher":
                return None
            spec = importlib.util.spec_from_file_location(
                name, BACKUP, loader=importlib.machinery.SourceFileLoader(name, BACKUP))
            orig = spec.loader.exec_module

            def exec_module(module, _o=orig):
                _o(module)
                hook = getattr(sys.modules.get("sitecustomize"), "_HOOKS", {}).get(name)
                if hook is not None:
                    hook(module)
            spec.loader.exec_module = exec_module
            return spec
    sys.meta_path.insert(0, _Old())

from training import bld_teacher as BT  # noqa: E402
assert BT.__file__.endswith(".bak_pre_gen2k1126") == (ARM == "old"), BT.__file__
assert hasattr(BT, "TABLES") == (ARM == "new")
from training import pool_env as PE  # noqa: E402
from training import ppo_pool as PP  # noqa: E402
import opp_scorecard_bld as B  # noqa: E402


def h(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]


def game_state(g):
    us = sorted((u["id"], u["card_id"], u["team"], round(float(u["x"]), 3), round(float(u["y"]), 3), float(u["hp"]))
                for u in g.get_units())
    return dict(tick=int(g.get_current_tick()), elx=[round(float(g.get_elixir_for_team(t)), 4) for t in (0, 1)],
                hand=[[int(c) for c in g.get_hand_for_team(t)] for t in (0, 1)],
                tow=[[float(g.get_tower_hp(t, s)) for s in range(3)] for t in (0, 1)], units=us)


def part_a():
    from training import obs_noise, sim2real
    po = json.loads(json.dumps(POOL))
    po["self"] = dict(po.get("self") or {}, ckpts=[CKPT])
    pool, _ = PP.build_pool(None, po, [B.CLONE], "ghost")
    s2r = dict(sim2real.cli_spec("v1", "0:1.5", elixir_est=True), elixir_est_share=0.8)
    env = PE.make_pool_env(5000, pool=pool, act_delay=PP.parse_delay("13,15,1"),
                           obs_noise=obs_noise.cli_spec("default", "", hp_live=True), decision_ticks=5,
                           obs_v2_noise={"names": "spells", "preset": "reader2"}, sim2real=s2r)()
    out = []
    for s in range(5000, 5300):
        obs, info = env.reset(seed=s)
        out.append(dict(seed=s, spec=json.loads(json.dumps(env.spec, default=str)), obs=h(obs),
                        state=game_state(env.game)))
    return out


def part_b():
    out = []
    res = B.work((CKPT, list(range(9100, 9100 + NM)), "bld", {}, "13,15,1", B.CLONE))
    for e in res:
        out.append({f: e.get(f) for f in ("kind", "spec", "opp_plays", "ba_log", "reward", "end", "taken", "dealt",
                                          "td", "hogs", "bodies", "leak1", "spent1")})
    return out


A = part_a()
nb = sum(1 for r in A if r["spec"].get("bld"))
res = dict(arm=ARM, bld_teacher=BT.__file__, A=A, B=part_b())
json.dump(res, open(f"evals/bld_opp/auto_proof_{ARM}.json", "w"), sort_keys=True, default=str)
fires = sum(1 for e in res["B"] for x in e["ba_log"] if x[0] == "fire")
print(ARM, "A", len(A), "bld resets", nb, "B", len(res["B"]), "fires", fires,
      "sha A", hashlib.sha256(json.dumps(A, sort_keys=True, default=str).encode()).hexdigest()[:16],
      "sha B", hashlib.sha256(json.dumps(res["B"], sort_keys=True, default=str).encode()).hexdigest()[:16],
      "sha all", hashlib.sha256(json.dumps([A, res["B"]], sort_keys=True, default=str).encode()).hexdigest()[:16])
