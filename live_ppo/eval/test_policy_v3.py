"""live_ppo/policy_v3.py: a generation-2 (NetV3) checkpoint in the live driver.

    .venv/bin/python -m pytest -q live_ppo/eval/test_policy_v3.py

Checkpoint detection, the refusals (wrong engine, tail off, wrong tail names,
wrong deck), what the defaults become for a v3 checkpoint and that they stay
what they were for the older ones, how --temp / --greedy / --when-temp /
--when-bias map onto the net, and the treatment of a play that is still in
flight when the next decision is made (with the real play watch). No engine and
no phone: the rules are plain functions. The tests marked `real checkpoint` read
runs/bc3/g2_big_L0_H10/ep4.pth and the gen-1 default where they are on disk.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "live_ppo")]
import policy_v3 as P  # noqa: E402
from play_watch import BLANK, PlayWatch  # noqa: E402

DECK = [125, 168, 129, 15, 40, 72, 33, 7]
HOG26 = "125,168,129,15,40,72,33,7"                 # training/common.HOG26_DECK
V3_CKPT = REPO / "runs" / "bc3" / "g2_big_L0_H10" / "ep4.pth"
GEN1_CKPT = REPO / "remote_results" / "ppo_league26_ep138708.pth"


def v3_blob(**kw):
    b = {"arch": "v3", "model": {"core.0.weight": 0}, "names": "spells", "decision_ticks": 5,
         "obs_dim": 14607, "base_obs_dim": 14459, "obs_v2": True}
    b.update(kw)
    return b


def ok(**kw):
    a = dict(path="x.pth", engine_gen2=True, obs_v2="auto", names=None, driver_deck=DECK,
             train_deck=HOG26)
    a.update(kw)
    return a


# ------------------------------------------------------------------ detection
def test_arch_by_field_and_by_keys():
    assert P.arch(v3_blob()) == "v3"
    assert P.arch({"arch": "net_v2", "model": {}}) == "net_v2"
    assert P.arch({"model": {"board.stem.weight": 0}, "deck": DECK}) == "net_v2"   # older net_v2 files
    assert P.arch({"board.stem.weight": 0}) == "net_v2"                            # a bare state dict
    assert P.arch({"model": {"conv1.weight": 0}, "deck": DECK}) == "v1"
    assert P.arch({"conv1.weight": 0}) == "v1"
    assert P.arch(None) == "v1"


def test_real_checkpoints_are_detected():
    """real checkpoint"""
    torch = pytest.importorskip("torch")
    if not (V3_CKPT.exists() and GEN1_CKPT.exists()):
        pytest.skip("checkpoints not on disk")
    v3 = torch.load(V3_CKPT, map_location="cpu", weights_only=False)
    g1 = torch.load(GEN1_CKPT, map_location="cpu", weights_only=False)
    assert P.arch(v3) == "v3" and P.arch(g1) != "v3"
    assert "deck" not in v3 and list(g1["deck"]) == DECK
    # what the driver takes from it
    assert (v3["names"], v3["decision_ticks"], v3["obs_dim"], v3["base_obs_dim"]) == ("spells", 5, 14607, 14459)
    P.check(v3, **ok(path=str(V3_CKPT)))
    with pytest.raises(P.V3Error, match="gen-2 engine"):
        P.check(v3, **ok(engine_gen2=False))
    with pytest.raises(P.V3Error, match="not a v3"):
        P.check(g1, **ok())


# ------------------------------------------------------------------ refusals
def test_refuses_the_pinned_engine_and_says_what_to_set():
    with pytest.raises(P.V3Error) as e:
        P.check(v3_blob(), **ok(engine_gen2=False, engine="/repo/engine"))
    assert "CR_ENGINE" in str(e.value) and "/repo/engine" in str(e.value) and "x.pth" in str(e.value)


def test_refuses_the_tail_forced_off():
    with pytest.raises(P.V3Error, match="--obs-v2 off"):
        P.check(v3_blob(), **ok(obs_v2="off"))
    P.check(v3_blob(), **ok(obs_v2="on"))
    P.check(v3_blob(), **ok(obs_v2="auto"))


def test_refuses_tail_names_other_than_the_checkpoints():
    with pytest.raises(P.V3Error) as e:
        P.check(v3_blob(names="spells"), **ok(names="all"))
    assert "'spells'" in str(e.value) and "--obs-v2-names all" in str(e.value)
    P.check(v3_blob(names="spells"), **ok(names="spells"))      # given and equal
    P.check(v3_blob(names="all"), **ok(names=None))             # not given: the checkpoint's is used


def test_deck_stand_in_is_the_training_codes_deck():
    P.check(v3_blob(), **ok())                                   # no `deck` key: HOG26_DECK == TRAIN_DECK
    with pytest.raises(P.V3Error, match="HOG26_DECK"):
        P.check(v3_blob(), **ok(train_deck="26,27,28,29,30,31,32,33"))
    with pytest.raises(P.V3Error, match="checkpoint"):           # a checkpoint that does carry one wins
        P.check(v3_blob(deck=[1, 2, 3, 4, 5, 6, 7, 8]), **ok())
    assert P.parse_deck(HOG26) == DECK and P.parse_deck(DECK) == DECK


# ------------------------------------------------------------------ defaults
def test_defaults_for_older_checkpoints_are_what_they_were():
    for kind in ("v1", "net_v2", None):
        s = P.settings(kind, {"names": "all"})
        assert s["names"] == "spells" and s["leak_floor"] == 10.0
        assert s["tactical"] is True and s["pending_mask"] is False
        assert P.settings(kind, tactical_loop=False)["tactical"] is False
        assert P.settings(kind, tactical_loop=False, tactical_officer=True)["tactical"] is False
        assert P.settings(kind, names="all", leak_floor=0.0)["names"] == "all"
        assert P.settings(kind, names="all", leak_floor=0.0)["leak_floor"] == 0.0


def test_defaults_for_v3_follow_the_checkpoint_and_drop_the_crutches():
    s = P.settings("v3", v3_blob(names="all"))
    assert s["names"] == "all"                      # the checkpoint's
    assert s["leak_floor"] == 0.0                   # off
    assert s["tactical"] is False                   # off even though the loop left it on
    assert s["pending_mask"] is True
    # each one comes back with its flag
    assert P.settings("v3", v3_blob(), tactical_officer=True)["tactical"] is True
    assert P.settings("v3", v3_blob(), leak_floor=9.5)["leak_floor"] == 9.5
    assert P.settings("v3", v3_blob(), pending_mask="off")["pending_mask"] is False
    assert P.settings("v1", pending_mask="on")["pending_mask"] is True


def test_pending_mask_needs_the_play_watch():
    s = P.settings("v3", v3_blob(), play_watch=False)            # auto: off, and flagged
    assert s["pending_mask"] is False and s["pending_mask_lost"] is True
    with pytest.raises(P.V3Error, match="--no-play-watch"):
        P.settings("v3", v3_blob(), pending_mask="on", play_watch=False)
    assert P.settings("v1", play_watch=False)["pending_mask_lost"] is False


# ------------------------------------------------------------------ sampling
def test_sampling_for_older_checkpoints_is_what_it_was():
    for kind in ("v1", "net_v2"):
        assert P.sampling(kind, temp=1.0)["patch"] is None
        s = P.sampling(kind, temp=0.5)
        assert (s["patch"], s["patch_temp"], s["set_sampling"]) == ("tempered", 0.5, None)
        s = P.sampling(kind, temp=0.5, greedy=True)              # --greedy wins over --temp
        assert (s["patch"], s["set_sampling"]) == ("argmax", None)
        with pytest.raises(P.V3Error, match="WHEN"):
            P.sampling(kind, when_bias=0.5)
        with pytest.raises(P.V3Error, match="WHEN"):
            P.sampling(kind, when_temp=1.0)                      # given at all, even at its default


def test_v3_temp_goes_into_the_net_and_the_categorical_is_not_tempered_again():
    s = P.sampling("v3", temp=0.5)
    assert s["patch"] is None
    assert s["set_sampling"] == {"what_temp": 0.5, "where_temp": 0.5, "when_temp": 1.0,
                                 "when_bias": 0.0, "when_internal": True}
    s = P.sampling("v3", temp=0.7, when_temp=0.8, when_bias=-0.3)
    assert s["set_sampling"] == {"what_temp": 0.7, "where_temp": 0.7, "when_temp": 0.8,
                                 "when_bias": -0.3, "when_internal": True}


def test_v3_greedy_is_argmax_what_and_where_with_when_still_a_coin():
    s = P.sampling("v3", temp=0.5, greedy=True, when_bias=0.2)
    assert s["patch"] == "argmax"
    k = s["set_sampling"]
    assert k["when_internal"] is True                # the net draws the coin, the argmax never sees it
    assert (k["what_temp"], k["where_temp"]) == (1.0, 1.0)
    assert (k["when_temp"], k["when_bias"]) == (1.0, 0.2)
    P.sampling("v3", temp=0.0, greedy=True)          # --temp is ignored with --greedy, as before


def test_v3_refuses_temperatures_the_net_cannot_take():
    with pytest.raises(P.V3Error, match="--temp"):
        P.sampling("v3", temp=0.0)
    with pytest.raises(P.V3Error, match="--when-temp"):
        P.sampling("v3", when_temp=0.0)
    with pytest.raises(P.V3Error, match="--when-bias"):
        P.sampling("v3", when_bias=float("inf"))


def test_sampling_kwargs_are_the_nets_own():
    """The mapping names exactly NetV3.set_sampling's parameters (read from the
    source: importing training.net_v3 needs the engine)."""
    import ast
    src = (REPO / "training" / "net_v3.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "set_sampling")
    params = [a.arg for a in fn.args.args if a.arg != "self"]
    assert sorted(P.sampling("v3")["set_sampling"]) == sorted(params)


# ------------------------------------------------------------------ plays in flight
HAND = ["the_log", "skeletons", "hog_rider", "cannon"]
S = 20                                   # a toy observation: 20 board floats, elixir, 4 costs


def obs(elixir, costs):
    o = np.zeros(S + 1 + 4 + 8, np.float32)
    o[S] = elixir / 10.0
    o[S + 1:S + 5] = np.array(costs, np.float32) / 10.0
    return o


def playable(o):
    """python_ai/models/net.py affordability_mask: (cost > 0) & (cost <= elixir + 1e-6)."""
    c = o[S + 1:S + 5]
    return ((c > 0) & (c <= o[S] + 1e-6)).tolist()


def test_open_play_blocks_its_slot_until_the_card_leaves():
    w = PlayWatch()
    w.frame(0.0, HAND, 9.0)
    assert w.in_flight() == []
    w.issue(1.0, 2, card="hog_rider", tile=(16, 15))
    w.cost(4.0)                                                  # a sent tap: the loop books the cost
    assert w.in_flight() == [(2, "hog_rider", 4.0)]
    # 0.5 s later, the next decision: the card is still in its slot
    w.frame(1.4, HAND, 9.0)
    b = P.blocked_slots(w.in_flight(), HAND)
    assert b == {2: False}                                       # blocked; elixir is owed() already
    o = obs(9.0 - w.owed(), [2, 1, 4, 3])
    assert playable(o) == [True, True, True, True]               # without the mask: asked for again
    assert P.inflight_obs(o, S, b) == [(2, 4.0)]
    assert playable(o) == [True, True, False, True]
    assert o[S] == pytest.approx(0.5)                            # not paid twice
    # the slot empties: landed, no longer in flight (the empty-slot rule takes over)
    w.tap_done(1.5)
    w.frame(1.6, HAND[:2] + [BLANK] + HAND[3:], 9.0)
    assert w.plays[0].state == "landed" and w.in_flight() == []


def test_dry_run_play_pays_in_the_observation_and_frees_the_slot_when_given_up():
    w = PlayWatch()
    w.frame(0.0, HAND, 6.0)
    w.issue(1.0, 2, card="hog_rider")                            # dry run: no cost is ever booked
    assert w.owed() == 0.0
    b = P.blocked_slots(w.in_flight(), HAND)
    assert b == {2: True}
    o = obs(6.0, [2, 1, 4, 3])
    P.inflight_obs(o, S, b)
    assert o[S] == pytest.approx(0.2) and playable(o) == [True, True, False, False]
    w.frame(1.5, HAND, 6.0)
    assert w.in_flight() != []
    w.frame(1.0 + PlayWatch.NO_TAP_S + 0.1, HAND, 6.0)           # never left its slot: given up
    assert w.plays[0].state == "failed" and w.in_flight() == []
    assert P.blocked_slots(w.in_flight(), HAND) == {}


def test_dropped_placement_is_not_in_flight():
    w = PlayWatch()
    w.frame(0.0, HAND, 6.0)
    p = w.issue(1.0, 1, card="skeletons")
    w.dropped(p)
    assert w.in_flight() == []


def test_slot_showing_another_card_is_not_blocked():
    fl = [(2, "hog_rider", 4.0)]
    assert P.blocked_slots(fl, HAND) == {2: False}
    assert P.blocked_slots(fl, HAND[:2] + ["ice_golem"] + HAND[3:]) == {}     # the next card is there
    assert P.blocked_slots([(2, None, None)], HAND) == {2: True}              # unnamed: by slot
    assert P.blocked_slots([(2, "hog_rider", 4.0), (2, "hog_rider", None)], HAND) == {2: True}
    assert P.blocked_slots([(7, "x", None)], HAND) == {}
    assert P.blocked_slots(fl) == {2: False}                                   # no hand given


def test_inflight_obs_floors_elixir_and_leaves_the_rest():
    o = obs(3.0, [2, 1, 4, 3])
    before = o.copy()
    assert P.inflight_obs(o, S, {}) == [] and (o == before).all()
    P.inflight_obs(o, S, {2: True, 0: True})
    assert o[S] == 0.0 and o[S + 1] == 0.0 and o[S + 3] == 0.0
    assert o[S + 2] == before[S + 2] and o[S + 4] == before[S + 4]
    assert (o[:S] == before[:S]).all() and (o[S + 5:] == before[S + 5:]).all()


# ------------------------------------------------------------------ the log
def test_card_probs_compose_like_the_net():
    p = P.card_probs(0.0, [0.0, 0.0, float("-inf"), 0.0], True)
    assert p["p_play"] == pytest.approx(0.5)
    assert p["what"] == pytest.approx([1 / 3, 1 / 3, 0.0, 1 / 3])
    assert p["p"] == pytest.approx([1 / 6, 1 / 6, 0.0, 1 / 6, 0.5]) and sum(p["p"]) == pytest.approx(1.0)
    # nothing playable: hold, whatever WHEN says
    assert P.card_probs(5.0, [0.0] * 4, False)["p"] == [0.0, 0.0, 0.0, 0.0, 1.0]
    # the knobs: bias then temperature on WHEN, temperature on WHAT
    k = {"when_bias": 1.0, "when_temp": 2.0, "what_temp": 0.5}
    p = P.card_probs(1.0, [1.0, 0.0, 0.0, 0.0], True, k)
    assert p["p_play"] == pytest.approx(1 / (1 + math.exp(-1.0)))
    assert p["what"][0] == pytest.approx(math.exp(2.0) / (math.exp(2.0) + 3))
    assert P.card_probs(0.0, [0.0] * 4, True, {"when_bias": P.FORCE_PLAY_BIAS})["p_play"] == 1.0


def test_card_probs_match_torch_compose():
    torch = pytest.importorskip("torch")
    F = torch.nn.functional
    when, what = torch.tensor([-1.3]), torch.tensor([[0.2, -0.7, float("-inf"), 1.1]])
    k = {"when_bias": 0.4, "when_temp": 0.8, "what_temp": 0.5}
    w = (when + k["when_bias"]) / k["when_temp"]
    ref = torch.cat((F.logsigmoid(w).unsqueeze(-1) + F.log_softmax(what / k["what_temp"], -1),
                     F.logsigmoid(-w).unsqueeze(-1)), -1).exp()[0].tolist()
    got = P.card_probs(float(when), what[0].tolist(), True, k)["p"]
    assert got == pytest.approx(ref, abs=1e-6)


def test_ability_prob_and_record():
    assert P.ability_prob([0.0, 0.0]) == pytest.approx(0.5)
    assert P.ability_prob([5.0, -5.0]) < 1e-4
    p = P.card_probs(0.0, [0.0, 2.0, float("-inf"), 1.0], True)
    r = P.decision_record(p, ["Hog Rider", "Fireball", "The Log", "Cannon"],
                          ability={"slot": 1, "p": 0.25, "want": False, "sent": False},
                          pending=[(2, 2.0)], knobs={"what_temp": 1.0}, obs_len=14607)
    assert r["p_play"] == 0.5 and r["obs_len"] == 14607 and r["pending"] == [[2, 2.0]]
    assert [c for c, _ in r["what"]] == ["Fireball", "Cannon", "Hog Rider"]      # most likely first, no masked
    assert r["ability"]["sent"] is False and r["forced_play"] is False


# ---------------------------------------------------------------- the launcher (play_live.py)
def _play_live():
    sys.path[:0] = [str(REPO)]
    import play_live
    return play_live


def _launch(*argv, arch="v1"):
    L = _play_live()
    a = L.parse_args(list(argv))
    c = L.build_cmd(a, Path("/tmp/run"), ["--capture", "mirror"], a.checkpoint, arch=arch)
    k = c.index("--")
    return c[3:k], c[k + 1:]


def test_launcher_temperature_follows_the_checkpoint_kind():
    drv, _ = _launch()
    assert drv[drv.index("--temp") + 1] == "0.5"                  # gen-1: as before
    drv, _ = _launch("--checkpoint", str(V3_CKPT), arch="v3")
    assert drv[drv.index("--temp") + 1] == "1.0"                  # the simulator evaluation's
    drv, _ = _launch("--checkpoint", str(V3_CKPT), "--temp", "0.7", arch="v3")
    assert drv[drv.index("--temp") + 1] == "0.7"


def test_launcher_tactical_reaches_a_v3_checkpoint_as_driver_flags():
    drv, loop = _launch("--checkpoint", str(V3_CKPT), arch="v3")
    assert "--tactical-officer" not in drv and "--solvency-gate" not in drv
    assert "--no-tactical" in loop
    drv, loop = _launch("--checkpoint", str(V3_CKPT), "--tactical", arch="v3")
    assert "--tactical-officer" in drv and "--solvency-gate" in drv
    drv, loop = _launch("--tactical")                             # older checkpoints: the loop's
    assert "--tactical-officer" not in drv and "--no-tactical" not in loop
    assert "--act" not in loop


@pytest.mark.skipif(not V3_CKPT.is_file(), reason="real checkpoint not on disk")
def test_launcher_refuses_v3_on_the_pinned_engine(monkeypatch, capsys):
    L = _play_live()
    monkeypatch.setattr(L, "ENGINE", REPO / "engine")
    with pytest.raises(SystemExit) as ex:
        L.check_checkpoint(V3_CKPT)
    assert ex.value.code == 2
    err = capsys.readouterr().err
    assert "CR_ENGINE=" in err and "gen-2 engine" in err
    with pytest.raises(SystemExit):
        L.check_checkpoint(V3_CKPT, stock=True)
    assert "drop --stock" in capsys.readouterr().err
