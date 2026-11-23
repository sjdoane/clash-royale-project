"""The gen-1 checkpoint adapter (training/obs_compat.py): layout, scaling, and
agreement with the gen-1 engine itself.

    CR_ENGINE=build/gen2/engine .venv/bin/python -m pytest training/tests -q

Needs the gen-2 engine (skipped otherwise). The comparison with the gen-1
engine runs engine/ in a second Python process (skipped when it cannot be
imported); the checkpoint tests are skipped when remote_results/ is absent.
"""
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from training import common as C  # noqa: F401  (sets CLASH_DECK first)
from training import obs_compat as OC

CE = C.CE
pytestmark = pytest.mark.skipif(not OC.IS_GEN2, reason="the adapter is for the gen-2 engine")
CKPTS = [os.path.join(C.REPO, "remote_results", n)
         for n in ("bc_hog26.pth", "ppo_league26_ep138708.pth")]
G1_EXTRA, G1_CYCLE = 13597, 13607          # gen-1 offsets (engine/include/core/ClashEnv.h)


def _dump(weights=()):
    """Observations of a seeded mirror match, the same calls on any engine:
    at reset, one tick after both sides deploy a card, and 40 ticks later.
    With `weights`, also each checkpoint's card and placement logits at reset."""
    deck = C.deck_ids()
    g = CE(list(deck), list(deck), 3600)
    g.seed(7)
    out = {"hand0": np.array(g.get_hand_for_team(0)), "hand1": np.array(g.get_hand_for_team(1)),
           "reset0": np.array(g.get_observation_for_team(0), np.float32),
           "reset1": np.array(g.get_observation_for_team(1), np.float32)}
    for team in (0, 1):
        g.set_elixir_for_team(team, 10.0)
    slots = [next(i for i, c in enumerate(g.get_hand_for_team(t)) if c not in (33, 7)) for t in (0, 1)]
    g.step_self_play(slots[0], 9.0, 10.0, slots[1], 9.0, 10.0, 1, False, False, False, False)
    out["t1_0"] = np.array(g.get_observation_for_team(0), np.float32)
    out["t1_1"] = np.array(g.get_observation_for_team(1), np.float32)
    g.step_self_play(C.NOOP, 0.0, 0.0, C.NOOP, 0.0, 0.0, 40, False, False, False, False)
    out["t41_0"] = np.array(g.get_observation_for_team(0), np.float32)
    for k, path in enumerate(weights):
        net, _ = OC.load_policy(path)
        net.eval()
        with torch.no_grad():
            o = torch.as_tensor(out["reset0"]).unsqueeze(0)
            mask = net.affordability_mask(o)
            f, emb, sp, hi = net.extract_features_hires(o)
            h = (torch.zeros(1, net.LSTM_HIDDEN), torch.zeros(1, net.LSTM_HIDDEN))
            cl, _, _, _, h = net.step_lstm_and_card(f, h, mask)
            ci = torch.argmax(cl, dim=-1)
            pl = net.placement_given_card(h[0], emb, ci, o, sp, hires_map=hi)
        out[f"card{k}"], out[f"place{k}"] = cl.numpy(), pl.numpy()
    return out


@pytest.fixture(scope="module")
def gen1(tmp_path_factory):
    """_dump() from the gen-1 engine (engine/), in its own process."""
    path = str(tmp_path_factory.mktemp("gen1") / "obs.npz")
    have = [p for p in CKPTS if os.path.exists(p)]
    env = {k: v for k, v in os.environ.items() if k not in ("CR_ENGINE", "PYTHONPATH", "CLASH_DECK")}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=C.REPO)
    code = ("import sys, numpy as np\n"
            "from training import common as C\n"
            "assert not hasattr(C.CE, 'LEVEL_START'), C.ENGINE\n"
            "from training.tests.test_obs_compat import _dump\n"
            "np.savez(sys.argv[1], **_dump(sys.argv[2:]))\n")
    res = subprocess.run([sys.executable, "-c", code, path] + have, cwd=C.REPO, env=env,
                         capture_output=True, text=True)
    if res.returncode != 0:
        pytest.skip("gen-1 engine (engine/) not usable: " + res.stderr[-300:])
    return dict(np.load(path)), have


def _fresh():
    g = CE(list(C.deck_ids()), list(C.deck_ids()), 3600)
    g.seed(3)
    return np.array(g.get_observation_for_team(0), np.float32)


def test_layout():
    obs = _fresh()
    assert obs.shape == (14459,)
    ad = OC.Adapter()
    out = ad(obs)
    assert isinstance(out, np.ndarray) and out.shape == (OC.GEN1["obs"],) == (13977,)
    assert out.dtype == np.float32
    # Spatial planes other than the building / tower cells are copied.
    sp = C.SPATIAL_SIZE
    keep = np.ones(sp, bool)
    for ch in OC.BUILDING_CH:
        keep[ch * OC.CELLS:(ch + 1) * OC.CELLS] = False
    for cell, _, _ in ad.tower_dps:
        keep[cell] = False
    assert np.array_equal(out[:sp][keep], obs[:sp][keep])
    # Elixir and hand costs; hand one-hots still name the same four cards.
    assert np.array_equal(out[sp:sp + 5], obs[sp:sp + 5])
    hot1 = out[sp + 5:sp + 5 + 4 * 185].reshape(4, 185).argmax(1)
    hot2 = obs[sp + 5:sp + 5 + 4 * CE.NUM_CARD_IDS].reshape(4, CE.NUM_CARD_IDS).argmax(1)
    assert list(hot1) == list(hot2) and sorted(hot1) == sorted(set(hot1) & set(C.deck_ids()))
    # Full towers: King 1.0, Princess 2534 / 4008, ours then theirs.
    p = 2534.0 / 4008.0
    assert out[G1_EXTRA + 3:G1_EXTRA + 9] == pytest.approx([1, p, p, 1, p, p])
    # Torch in, torch out; a batch is the rows.
    t = ad(torch.as_tensor(np.stack([obs, obs])))
    assert isinstance(t, torch.Tensor) and t.shape == (2, 13977)
    assert np.array_equal(t[1].numpy(), out)
    with pytest.raises(ValueError):
        ad(obs[:13977])


def test_tower_values_keep_the_fraction_and_dead_towers_read_zero():
    obs = _fresh()
    ad = OC.Adapter()
    ex = CE.EXTRA_SCALARS_START
    # (cell in the building plane, tower scalar, gen-1 full value) for the 6 towers.
    for cell, scalar, full in ad.towers:
        o = obs.copy()
        assert o[cell] == pytest.approx(1.0) and o[scalar] == pytest.approx(1.0)
        o[cell] = o[scalar] = 0.5
        a = ad(o)
        assert a[cell] == pytest.approx(0.5 * full)
        assert a[G1_EXTRA + (scalar - ex)] == pytest.approx(0.5 * full)
        o[cell] = o[scalar] = 0.0
        a = ad(o)
        assert a[cell] == 0.0 and a[G1_EXTRA + (scalar - ex)] == 0.0
    # A deployed building: same hit points -> same value (hp / 4008), capped at 1.
    cell = OC.BUILDING_CH[0] * OC.CELLS + 10 * C.BOARD_W + 9
    o = obs.copy()
    o[cell] = 0.5
    assert ad(o)[cell] == pytest.approx(0.5 * float(C.MAX_BUILDING_HP) / 4008.0)
    o[cell] = 0.95
    assert ad(o)[cell] == 1.0


def test_level_and_opponent_state_sections_are_dropped():
    obs = _fresh()
    ad = OC.Adapter()
    o = obs.copy()
    o[CE.LEVEL_START:] = np.random.default_rng(0).random(len(o) - CE.LEVEL_START).astype(np.float32)
    assert np.array_equal(ad(o), ad(obs))


def test_cycle_block_is_unfolded_for_the_opponents_deck():
    evo = 125                                   # evo:Skeletons, in the gen-1 registry
    face = int(C.E.face_card_id_of(evo))
    assert face != evo and face < 185
    obs = _fresh()
    n2 = CE.NUM_CARD_IDS
    obs[CE.CYCLE_START + face] = 1.0            # seen
    obs[CE.CYCLE_START + n2 + face] = 0.7       # recency
    plain = OC.Adapter()(obs)
    assert plain[G1_CYCLE + face] == 1.0 and plain[G1_CYCLE + 185 + face] == pytest.approx(0.7)
    assert plain[G1_CYCLE + evo] == 0.0
    un = OC.Adapter(opp_deck=C.deck_ids())(obs)
    assert un[G1_CYCLE + evo] == 1.0 and un[G1_CYCLE + 185 + evo] == pytest.approx(0.7)
    assert un[G1_CYCLE + face] == 0.0 and un[G1_CYCLE + 185 + face] == 0.0
    assert un[G1_CYCLE:].sum() == pytest.approx(1.7)


def test_adapted_observation_equals_the_gen1_engines(gen1):
    g1, _ = gen1
    g2 = _dump()
    assert list(g1["hand0"]) == list(g2["hand0"]) and list(g1["hand1"]) == list(g2["hand1"])
    ad = OC.Adapter(opp_deck=C.deck_ids())
    for key in ("reset0", "reset1"):
        assert g1[key].shape == (13977,)
        assert np.array_equal(ad(g2[key]), g1[key]), key
    elixir = C.SPATIAL_SIZE
    for key in ("t1_0", "t1_1", "t41_0"):
        a, b = ad(g2[key]), g1[key]
        diff = np.flatnonzero(np.abs(a - b) > 1e-6)
        # Only our own elixir: gen 2 regenerates 1/28 per tick, gen 1 0.035.
        assert set(diff.tolist()) <= {elixir}, (key, diff[:10])
        ticks = 1 if key.startswith("t1_") else 41
        assert (a[elixir] - b[elixir]) * 10 == pytest.approx(ticks * (1 / 28 - 0.035), abs=2e-5)
    # Without the opponent's deck the Evolution/Hero plays stay folded on their
    # face cards: the cycle block differs from gen 1 (two blocks, two cells each).
    a = OC.Adapter()(g2["t1_0"])
    assert len(np.flatnonzero(np.abs(a - g1["t1_0"])[G1_CYCLE:] > 1e-6)) in (2, 4)


def test_checkpoints_load_strictly_and_run():
    have = [p for p in CKPTS if os.path.exists(p)]
    if not have:
        pytest.skip("no gen-1 checkpoints in remote_results/")
    obs = torch.as_tensor(_fresh()).unsqueeze(0)
    for path in have:
        pol, info = OC.load_policy(path)
        assert isinstance(pol, OC.Gen1Policy) and info["adapter"] and not info["missing_buffers"]
        assert info["checkpoint_card_ids"] == 185 and info["engine_card_ids"] == CE.NUM_CARD_IDS
        sd, _ = OC.state_dict_of(path)
        mine = pol.net.state_dict()
        params = {k for k, _ in pol.net.named_parameters()}
        assert params <= set(sd)                         # every learned tensor came from the file
        for k in params:
            assert torch.equal(mine[k], sd[k]), k
        pol.eval()
        with torch.no_grad():
            mask = pol.affordability_mask(obs)
            f, emb, sp, hi = pol.extract_features_hires(obs)
            h = (torch.zeros(1, pol.LSTM_HIDDEN), torch.zeros(1, pol.LSTM_HIDDEN))
            cl, a1, a2, _, h = pol.step_lstm_and_card(f, h, mask)
            pl = pol.placement_given_card(h[0], emb, torch.argmax(cl, -1), obs, sp, hires_map=hi)
            assert torch.equal(mask, pol.net.affordability_mask(pol.adapter(obs)))
        assert cl.shape == (1, C.HAND_SIZE + 1) and torch.isfinite(cl[mask.bool()]).all()
        assert pl.shape[0] == 1 and torch.isfinite(pl).any()
        assert a1 is not None                            # the Hero Musketeer's ability head


def test_gen1_policy_gives_the_gen1_engines_logits(gen1):
    g1, have = gen1
    if not have:
        pytest.skip("no gen-1 checkpoints in remote_results/")
    g2 = _dump(have)
    for k, path in enumerate(have):
        assert np.allclose(g1[f"card{k}"], g2[f"card{k}"], atol=1e-5, equal_nan=True), path
        a, b = g1[f"place{k}"], g2[f"place{k}"]
        fin = np.isfinite(a)
        assert np.array_equal(fin, np.isfinite(b)) and np.allclose(a[fin], b[fin], atol=1e-4), path
