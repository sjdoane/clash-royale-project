"""Proof that training/pool_env.py with the bld hooks behaves exactly as the backup when the "bld" section is off
(2026-11-26). One arm a process:
  arm old      training.pool_env is loaded from training/pool_env.py.bak_pre_bldopp1126 (the site hook still applied)
  arm new      today's file, pool without a "bld" key
  arm new_bld0 today's file, pool with "bld": {"share": 0.0}
Each arm writes evals/bld_opp/off_proof_<arm>.json:
  A) 300 env.reset(seed) with the fresh pocket run's whole POOL (pocket_stages.sh: mixed weights, p_midmatch 0.215,
     behind 0.2, live 0.06): the spec, a hash of the observation and the game state after reset;
  B) whole matches with the policy (opp_scorecard_bld.work, act delay 13,15,1) for kinds lag0 lag1 clone ghost live
     self: the spec, every opponent play, the result, tower damage, units met.
usage: (run env) .venv/bin/python evals/tools/bld_off_proof.py ARM [N_MATCH_PER_KIND]
"""
import hashlib
import importlib.machinery
import importlib.util
import json
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "evals/tools")
ARM = sys.argv[1]
NM = int(sys.argv[2]) if len(sys.argv) > 2 else 6
CKPT = "runs/bc3/g3j_L13_Wnone_pk_fxafbw/ep4.pth"
BACKUP = "training/pool_env.py.bak_pre_bldopp1126"

if ARM == "old":
    import importlib.abc

    class _Old(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "training.pool_env":
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

from training import pool_env as PE  # noqa: E402
assert PE.__file__.endswith(".bak_pre_bldopp1126") == (ARM == "old"), PE.__file__
from training import ppo_pool as PP  # noqa: E402
import opp_scorecard_bld as B  # noqa: E402

BLD = {"bld": {"share": 0.0}} if ARM == "new_bld0" else {}
_SH = open("runs/pocket_1126/fix/pocket_stages.sh").read()
_DECKS = "data/decks/meta_api_pool_live4.json"          # run_g2_pocket_fresh.sh DECKS=
POOL = json.loads(_SH.split("POOL='")[1].split("'\n")[0].replace("\"'$DECKS'\"", json.dumps(_DECKS)))


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
    po = dict(POOL, **BLD)
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
    for k in ("lag0", "lag1", "clone", "ghost", "live", "self"):
        res = B.work((CKPT, list(range(9100, 9100 + NM)), k, dict(BLD), "13,15,1", B.CLONE))
        for e in res:
            out.append({f: e.get(f) for f in ("kind", "spec", "opp_plays", "reward", "end", "taken", "dealt", "td",
                                              "hogs", "bodies", "leak1", "spent1")})
    return out


res = dict(arm=ARM, pool_env=PE.__file__, A=part_a(), B=part_b(),
           bld_imported="training.bld_teacher" in sys.modules)
blob = json.dumps(res, sort_keys=True, default=str)
json.dump(res, open(f"evals/bld_opp/off_proof_{ARM}.json", "w"), sort_keys=True, default=str)
print(ARM, "A", len(res["A"]), "B", len(res["B"]), "bld_imported", res["bld_imported"],
      "sha A", hashlib.sha256(json.dumps(res["A"], sort_keys=True, default=str).encode()).hexdigest()[:16],
      "sha B", hashlib.sha256(json.dumps(res["B"], sort_keys=True, default=str).encode()).hexdigest()[:16])
