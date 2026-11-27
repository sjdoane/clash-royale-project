"""Tests for body_rows.ProxyConfirm / --body-rows on-confirmed (2026-11-27).

    .venv/bin/python -m pytest -q live_ppo/eval/test_body_rows_confirm.py
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "live_ppo"))
import body_rows as B  # noqa: E402

DT = 1 / 11.7


def box(cls, x, y, side=1, conf=0.6):
    return {"cls": cls, "side": side, "conf": conf, "tile": [x, y]}


def kept(gate, dets, t):
    drop = gate.drop(dets, t)
    return [d["cls"] for d in dets if id(d) not in drop]


def test_variant_has_no_brawler_and_keeps_the_rest():
    assert "goblin-brawler" in B.PROXY and "goblin-brawler" not in B.PROXY_CONFIRMED
    assert {k: v for k, v in B.PROXY.items() if k != "goblin-brawler"} == B.PROXY_CONFIRMED


def test_lone_box_dropped_then_confirmed():
    g = B.ProxyConfirm()
    assert kept(g, [box("elixir-golem-big", 5, 10), box("hog-rider", 3, 12)], 0.0) == ["hog-rider"]
    assert kept(g, [box("elixir-golem-big", 5.4, 10.6)], DT) == ["elixir-golem-big"]


def test_golem_big_and_mid_confirm_each_other_but_not_other_cards():
    g = B.ProxyConfirm()
    g.drop([box("elixir-golem-big", 5, 10)], 0.0)
    assert kept(g, [box("elixir-golem-mid", 5, 11)], DT) == ["elixir-golem-mid"]   # both card 90
    g = B.ProxyConfirm()
    g.drop([box("phoenix-big", 5, 10)], 0.0)
    assert kept(g, [box("elixir-golem-mid", 5, 10)], DT) == []


def test_side_distance_and_age():
    g = B.ProxyConfirm()
    g.drop([box("phoenix-big", 5, 10, side=0)], 0.0)
    assert kept(g, [box("phoenix-big", 5, 10, side=1)], DT) == []                    # other side
    g = B.ProxyConfirm()
    g.drop([box("phoenix-big", 5, 10)], 0.0)
    assert kept(g, [box("phoenix-big", 5, 11.6)], DT) == []                          # 1.6 tiles
    g = B.ProxyConfirm()
    g.drop([box("phoenix-big", 5, 10)], 0.0)
    for k in range(1, 4):
        g.drop([], k * DT)                                                           # 3 empty frames
    assert kept(g, [box("phoenix-big", 5, 10)], 4 * DT) == []
    g = B.ProxyConfirm()
    g.drop([box("phoenix-big", 5, 10)], 0.0)
    assert kept(g, [box("phoenix-big", 5, 10)], 1.2) == []                           # 1.2 s later


def test_other_classes_never_dropped_and_reset():
    g = B.ProxyConfirm()
    dets = [box("goblin-brawler", 5, 10), box("royal-ghost", 1, 13), box("golemite", 4, 9)]
    assert kept(g, dets, 0.0) == ["goblin-brawler", "royal-ghost", "golemite"]
    g.drop([box("barbarian-barrel-hero", 5, 10)], DT)
    g.reset()
    assert kept(g, [box("barbarian-barrel-hero", 5, 10)], 2 * DT) == []


def test_the_165545_false_positive_is_dropped():
    """20261124-165545 m1 f1333: one elixir-golem-mid box (0.52) at (1, 8), none before."""
    g = B.ProxyConfirm()
    for k in range(3):
        g.drop([box("lava-pup", 1, 9), box("barbarian", 1, 9)], k * DT)
    assert kept(g, [box("elixir-golem-mid", 1, 8, conf=0.52), box("barbarian", 1, 9)], 3 * DT) == ["barbarian"]


def test_driver_defaults_on_confirmed_and_keeps_the_reverts():
    """Coordinator ruling 2026-11-27: on-confirmed and all five play rules are the defaults."""
    src = (REPO / "live_ppo" / "driver.py").read_text()
    m = re.search(r'"--body-rows", choices=\(([^)]*)\), default="([\w-]+)"', src)
    assert m and m.group(2) == "on-confirmed"
    assert all(f'"{c}"' in m.group(1) for c in ("on", "on-confirmed", "rows", "off"))
    m = re.search(r'"--opp-rules", default="([^"]*)"', src)
    assert m and m.group(1) == "own_units,badge_marks,three_musk,rage,gang_bodies"
    assert "PROXY_GATE.reset()" in src and "body_rows.PROXY_CONFIRMED" in src
