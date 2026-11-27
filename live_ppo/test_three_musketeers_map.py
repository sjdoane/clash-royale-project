"""Round-6 prep (2026-11-27): a `three-musketeers` detector class must reach card 80.

Today Three Musketeers bodies come out of r5 as `musketeer` (card 6). The round-6 class
list `data/det/classes_round6_3m.json` reuses retired slot 180 (`mirror`, no labels in any
source) for `three-musketeers`. This pins that the live code maps the new name to card 80
in both paths with NO code change, and that every other class keeps its id. Run:
CR_P0088=<the live switches> .venv/bin/python -m pytest -q live_ppo/test_three_musketeers_map.py
"""
import ast
import json
import os
import sys

import pytest

REPO = "/Users/samueldoane/cr-robot"
ENG = os.environ.get("CR_ENGINE") or REPO + "/build/gen2_k/engine"
sys.path[:0] = [ENG + "/perception", ENG, ENG + "/python_ai", REPO + "/live_ppo"]


@pytest.fixture(scope="module")
def yolo_card_id():
    """driver.yolo_card_id, compiled from driver.py's own source (importing driver.py
    would parse the live command line and start the side-cars)."""
    from live.unit_to_card import UnitMappingError, _norm, card_id_for  # noqa: PLC0415
    from python_ai.models import perception_encoder as PE  # noqa: PLC0415
    src = open(REPO + "/live_ppo/driver.py").read()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "yolo_card_id")
    ns = {"card_id_for": card_id_for, "UnitMappingError": UnitMappingError, "_norm": _norm,
          "E": PE.engine, "_YOLO_ID": {}}
    exec(compile(ast.Module([fn], []), "driver.py:yolo_card_id", "exec"), ns)
    return ns["yolo_card_id"]


def test_new_class_maps_to_card_80(yolo_card_id):
    import opp_plays  # noqa: PLC0415
    assert yolo_card_id("three-musketeers") == 80
    assert opp_plays.card_for_class("three-musketeers") == 80


@pytest.mark.parametrize("cls,cid", [("musketeer", 6), ("musketeer-ev1", 6), ("musketeer-hero", 6),
                                     ("mini-pekka", 5), ("mini-pekka-hero", 5),
                                     ("pekka", 13), ("pekka-ev1", 13),
                                     ("royal-ghost", 47), ("goblin", 4), ("mirror", None)])
def test_other_classes_unchanged(yolo_card_id, cls, cid):
    assert yolo_card_id(cls) == cid


def test_card_80_has_an_observation_row():
    from python_ai.models import perception_encoder as PE  # noqa: PLC0415
    assert 80 in PE.card_table()


def test_class_list_differs_only_at_slot_180():
    old = json.load(open(REPO + "/data/det/classes_round2.json"))[:198]
    new = json.load(open(REPO + "/data/det/classes_round6_3m.json"))
    assert len(new) == 198 and new[180] == "three-musketeers" and old[180] == "mirror"
    assert [a for i, a in enumerate(old) if i != 180] == [b for i, b in enumerate(new) if i != 180]
