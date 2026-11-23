"""Shared setup for everything in training/: paths, the deck, engine imports.

Import this module FIRST (before numpy/torch): it sets CLASH_DECK and the
engine's import path, and `python_ai` caps OpenBLAS threads at import.

Nothing here edits engine/; engine modules are imported read-only from
engine/python_ai via sys.path.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: CR_ENGINE picks another engine tree, e.g. build/gen2/engine (a copy of engine/
#: with patches/pending applied). live_ppo/driver.py reads the same variable.
ENGINE = os.path.abspath(os.environ.get("CR_ENGINE") or os.path.join(REPO, "engine"))
TRAINING = os.path.join(REPO, "training")

#: The Hog 2.6 evo/hero list (TRAINING_PLAN.md section 1). Ids 125,168,129,15,40,72,33,7.
HOG26_DECK = "125,168,129,15,40,72,33,7"   # = evo:skeletons,hero musketeer,evo:cannon,hog rider,ice golem,ice spirit,the log,fireball
os.environ.setdefault("CLASH_DECK", HOG26_DECK)

for p in (ENGINE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
# Spawned AsyncVectorEnv workers (macOS default) must be able to import
# `training.*` and `python_ai.*` when they unpickle the env factory.
_pp = os.environ.get("PYTHONPATH", "")
for p in (ENGINE, REPO):
    if p not in _pp.split(os.pathsep):
        _pp = p + (os.pathsep + _pp if _pp else "")
os.environ["PYTHONPATH"] = _pp

import python_ai  # noqa: E402,F401  (caps BLAS threads, makes the .so importable)
import clash_royale_env as E  # noqa: E402

CE = E.ClashRoyaleEnv
BOARD_W, BOARD_H = CE.BOARD_WIDTH, CE.BOARD_HEIGHT
HAND_SIZE = CE.HAND_SIZE
NOOP = HAND_SIZE
SPATIAL_SIZE = CE.NUM_CHANNELS * BOARD_H * BOARD_W
EXTRA = CE.EXTRA_SCALARS_START          # 0 time | 1-2 spend | 3-5 own K/L/R | 6-8 enemy | 9 mult/3
MAX_BUILDING_HP = float(CE.MAX_BUILDING_HP)
#: Engine UPSTREAM item 44 (gen 2): each tower scalar is hp over THAT tower's own
#: max, so 1.0 is full at any level. Older engines write hp / MAX_BUILDING_HP.
#: Detected by the level section the same patch adds.
TOWER_OBS_OWN_MAX = hasattr(CE, "LEVEL_START")
MAX_MATCH_ELIXIR = float(CE.MAX_MATCH_ELIXIR)
MAX_ELIXIR_MULT = float(E.MAX_ELIXIR_MULTIPLIER)


def _start_tower_hp():
    """(3,) starting HP of King, left, right Princess, read off a fresh board."""
    g = CE(list(range(8)), list(range(8)), 100)
    return [float(g.get_tower_max_hp(0, s)) for s in range(3)]


TOWER_HP0 = _start_tower_hp()


def data_dir(*parts):
    return os.path.join(REPO, "data", *parts)


def deck_ids():
    from python_ai.deck import DEFAULT_DECK
    return list(DEFAULT_DECK)


def new_net(device="cpu"):
    """The exact network the PPO trainer builds (same deck => same ability heads)."""
    from python_ai.models.net import MicroRoyaleNet
    from python_ai.rl.abilities import ability_engine_slots
    return MicroRoyaleNet(num_ability_slots=len(ability_engine_slots(deck_ids()))).to(device)


def load_weights(net, path):
    import torch
    from python_ai.models.policy_io import load_state_dict_flexible
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    load_state_dict_flexible(net, sd, f"training load ({path})")
    return net
