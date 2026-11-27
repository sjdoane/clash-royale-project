"""live_ppo/cloaked_gate.py (2026-11-27). Run:
CR_P0088=<the live switches> .venv/bin/python -m pytest -q live_ppo/test_cloaked_gate.py
The synthetic cases need no engine; the last two measure the reveal rule from the engine."""
import os
import sys

import pytest

REPO = "/Users/samueldoane/cr-robot"
ENG = os.environ.get("CR_ENGINE") or REPO + "/build/gen2_k/engine"
sys.path[:0] = [ENG + "/perception", ENG, ENG + "/python_ai", REPO + "/live_ppo"]

import cloaked_gate as G  # noqa: E402

DT = 0.1                       # 10 frames a second


def _box(x, y, side=1, tf=True):
    d = {"side": side, "tile": [int(x), int(y)]}
    if tf:
        d["tile_f"] = [x, y]
    return d


def _walk(t0, t1, x, y0, vy):
    """frames (t, x, y) from t0 to t1, moving vy tiles a second along y"""
    n = int(round((t1 - t0) / DT))
    return [(round(t0 + i * DT, 3), x, y0 + vy * i * DT) for i in range(n)]


def _run(gate, track, side=1):
    return [(t, gate.keep(_box(x, y, side), t)) for t, x, y in track]


def _shown_times(res):
    return [t for t, s in res if s]


def test_deploy_then_walk_is_never_shown():
    g = G.RevealGate(deploy=1.0)
    track = _walk(0.0, 1.0, 3.0, 23.0, 0.0) + _walk(1.0, 8.0, 3.0, 23.0, -2.0)
    assert _shown_times(_run(g, track)) == []


def test_still_beside_a_target_from_landing_waits_for_deploy_and_window():
    # a ghost dropped in range is cloaked while deploying (the simulator: deploy 1.0 s, first
    # hit 0.6 s later); the gate shows it once a whole still window lies after the deploy
    g = G.RevealGate(deploy=1.0)
    res = _run(g, _walk(0.0, 4.0, 3.0, 9.0, 0.0))
    first = _shown_times(res)[0]
    assert 2.0 - 1e-6 <= first <= 2.0 + DT + 1e-6
    assert all(s for t, s in res if t >= first)


def test_walk_stop_hit_walk_off():
    g = G.RevealGate(deploy=1.0)
    deploy = _walk(0.0, 1.0, 3.0, 23.0, 0.0)
    walk = _walk(1.0, 6.0, 3.0, 23.0, -2.0)           # to y 13
    y_stop = 23.0 - 2.0 * 5.0
    stop = _walk(6.0, 10.0, 3.0, y_stop, 0.0)         # hitting for 4 s
    off = _walk(10.0, 14.0, 3.0, y_stop, -2.0)        # target dead, walks on
    res = _run(g, deploy + walk + stop + off)
    shown = _shown_times(res)
    # shown from about one window after the stop, never before the stop
    assert min(shown) >= 6.0 + 0.6 - 1e-6 and min(shown) <= 6.0 + G.WINDOW_S + DT + 1e-6
    # kept while it walks off (the simulator: 1.8 s after its last hit)
    assert 10.0 + G.REVEAL_S in [round(t, 3) for t in shown]
    # gone at most the still-window slack (0.8 tiles at 2 tiles a second) later
    assert max(shown) <= 10.0 + G.STILL_TILES / 2.0 + G.REVEAL_S + DT + 1e-6
    assert all(not s for t, s in res if t > 10.0 + G.STILL_TILES / 2.0 + G.REVEAL_S + DT + 1e-6)


def test_two_ghosts_are_kept_apart_by_position():
    g = G.RevealGate(deploy=1.0)
    hitting = _walk(0.0, 5.0, 3.0, 9.0, 0.0)
    walking = _walk(0.0, 5.0, 14.0, 26.0, -2.0)
    got = {}
    for (t, xa, ya), (_, xb, yb) in zip(hitting, walking):
        got.setdefault("a", []).append(g.keep(_box(xa, ya), t))
        got.setdefault("b", []).append(g.keep(_box(xb, yb), t))
    assert any(got["a"]) and not any(got["b"])


def test_sides_are_separate():
    # a still box of the other side at the walking ghost's spot does not reveal it
    g = G.RevealGate(deploy=1.0)
    walking = _walk(0.0, 5.0, 3.0, 23.0, -2.0)
    res = []
    for t, x, y in walking:
        g.keep(_box(3.0, 18.0, side=0), t)
        res.append(g.keep(_box(x, y, side=1), t))
    assert not any(res)


def test_reset_forgets():
    g = G.RevealGate(deploy=1.0)
    _run(g, _walk(0.0, 4.0, 3.0, 9.0, 0.0))
    g.reset()
    res = _run(g, _walk(4.0, 5.0, 3.0, 9.0, 0.0))
    assert not any(s for _, s in res)


def test_integer_tile_without_tracker():
    g = G.RevealGate(deploy=1.0)
    res = [(t, g.keep(_box(x, y, tf=False), t)) for t, x, y in _walk(0.0, 4.0, 3.0, 9.0, 0.0)]
    assert _shown_times(res)


def test_log_is_the_last_frame():
    g = G.RevealGate(deploy=1.0)
    g.keep(_box(3.0, 9.0), 0.0)
    g.keep(_box(14.0, 9.0), 0.0)
    assert len(g.log) == 2
    g.keep(_box(3.0, 9.0), 0.1)
    assert len(g.log) == 1 and set(g.log[0]) == {"tile", "side", "age", "hit", "shown"}


# ---------------------------------------------------------------- engine
@pytest.fixture(scope="module")
def PE():
    from python_ai.models import perception_encoder as pe  # noqa: PLC0415
    return pe


def _ghost_trace(PE, cid):
    """Ghost (team 0) walks to an enemy Elixir Collector (never attacks) beside our left
    Princess Tower and hits it until it falls. Per tick: (tick, attacked, targetable)."""
    env = PE._E(PE._PROBE_DECK, PE._PROBE_DECK)
    env.reset()
    env.inject(99, 3.0, 11.0, 1)
    env.inject(cid, 3.0, 8.0, 0)
    out, prev = [], 0.0
    for tick in range(1, 120):
        env.step(PE.HAND_SIZE, 0.0, 0.0, 1)
        # get_units reports an injected Evolution (136) under its base id 47
        g = [u for u in env.get_units() if u["team"] == 0 and u["card_id"] in G.CLOAKED_CARDS]
        if not g:
            break
        cd = float(g[0]["cooldown"])
        out.append((tick, cd > prev + 0.5, bool(g[0]["targetable"])))
        prev = cd
    return out


@pytest.mark.parametrize("cid", G.CLOAKED_CARDS)
def test_engine_reveals_from_first_hit_until_reveal_s_after_last(PE, cid):
    tr = _ghost_trace(PE, cid)
    hits = [t for t, a, _ in tr if a]
    assert len(hits) >= 2, tr
    first_vis = next(t for t, _, v in tr if v)
    assert first_vis == hits[0]                        # cloaked until the first hit lands
    # the first engagement only: the probe deck's own AI sometimes walks a second target in
    cloak = next(t for t, _, v in tr if t > hits[0] and not v)
    last = max(t for t in hits if t < cloak)
    assert len([t for t in hits if t < cloak]) >= 2
    assert all(v for t, _, v in tr if hits[0] <= t < cloak)
    assert cloak - last == round(G.REVEAL_S * 10)      # 18 ticks: patch 0040


@pytest.mark.parametrize("cid", G.CLOAKED_CARDS)
def test_deploy_from_engine(PE, cid):
    ticks = PE.engine.get_card_stats(cid)["base"]["deploy_ticks"]
    assert G.deploy_s(PE.engine, cid) == pytest.approx(ticks / 10.0)
    assert G.deploy_s(PE.engine, cid) == pytest.approx(G.DEFAULT_DEPLOY_S)
