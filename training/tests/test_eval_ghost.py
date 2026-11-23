"""The ghost evaluation (training/eval_ghost.py) and the replay parity score
(training/parity.py): metric definitions on a hand-made log, a fixed panel,
same seed -> same rows.

    CR_ENGINE=build/gen2/engine .venv/bin/python -m pytest training/tests -q
"""
import os

import numpy as np
import pytest

from training import common as C  # noqa: F401  (sets CLASH_DECK first)
from training import eval_ghost as EG
from training import ghost as G
from training import parity as P

HAVE_STORE = os.path.exists(os.path.join(G.STORE, "meta.json"))
needs_store = pytest.mark.skipif(not HAVE_STORE, reason="no ghost cache (training/ghost_data.py build)")
BC = os.path.join(C.REPO, "remote_results", "bc_hog26.pth")
LOG, FIREBALL, HOG = 33, 7, 15


def _card(name):
    for cid in range(C.CE.NUM_CARD_IDS):
        try:
            if G.card_info(cid)["name"] == name:
                return cid
        except Exception:  # noqa: BLE001
            pass
    raise KeyError(name)


def _hand_made_log():
    deck = C.deck_ids()
    barrel = _card("Goblin Barrel")
    log = G.MatchLog(deck, [barrel, HOG, 1, 2, 3, 4, 5, 6])
    hog_i = deck.index(HOG)
    for t in range(0, 120, 10):
        cd = [0.0] * 8
        cd[hog_i] = 100.0 if t >= 80 else 0.0
        log.grid.append({
            "tick": t, "e0": 10.0 if t <= 40 else 5.0, "e1": 5.0,
            "hand": [LOG, FIREBALL, 40, 72] if t < 70 else [FIREBALL, 40, 72, 125],
            "td0": 100.0 if t >= 80 else 0.0, "td1": 200.0 if t >= 60 else 0.0, "cd": cd, "sd1": 0.0,
            "spent0": 0.0 if t <= 40 else 5.0, "mult": 1.0})
    log.final = log.grid.pop()                       # the closing sample, tick 110
    log.plays[0] = [{"tick": 50, "cid": HOG, "x": 4, "y": 15, "elixir": 10.0},
                    {"tick": 62, "cid": LOG, "x": 9, "y": 10, "elixir": 6.0},
                    {"tick": 90, "cid": HOG, "x": 4, "y": 15, "elixir": 4.0}]
    with_log, without = [LOG, FIREBALL, 40, 72], [FIREBALL, 40, 72, 125]
    log.plays[1] = [{"tick": 5, "cid": barrel, "own_hand": with_log, "own_elixir": 1.0},
                    {"tick": 20, "cid": barrel, "own_hand": with_log, "own_elixir": 10.0},
                    {"tick": 30, "cid": HOG, "own_hand": with_log, "own_elixir": 10.0},
                    {"tick": 40, "cid": barrel, "own_hand": with_log, "own_elixir": 10.0},
                    {"tick": 100, "cid": barrel, "own_hand": without, "own_elixir": 10.0}]
    return log


def test_metrics_on_a_hand_made_log():
    m = EG.match_metrics(_hand_made_log())
    assert m["plays"] == 3 and m["opp_plays"] == 5 and m["minutes"] == pytest.approx(110 / 600)
    assert m["plays_Hog Rider"] == 2 and m["plays_The Log"] == 1 and m["plays_Fireball"] == 0
    assert m["elixir_at_play_sum"] == 20.0
    # 11 one-second samples, the bar at 9+ in five of them (ticks 0-40).
    assert (m["samples"], m["samples_9plus"]) == (11, 5)
    # Full bar for four seconds with nothing spent: 4 x 10 ticks of regeneration lost.
    assert m["leak_1hz"] == pytest.approx(4 * 10 * EG.REGEN, rel=1e-6)
    # The Log sat in hand from tick 0 to the sample at 70; Fireball to the end (110).
    assert m["log_held_max_s"] == 7.0 and m["fireball_held_max_s"] == 11.0
    # Hog at 50: Hog damage and tower damage rise together at 80 -> a hit. Hog at 90: nothing.
    assert (m["hog_plays"], m["hog_hits"]) == (2, 1)
    # Goblin Barrels: 4 seen; 2 with The Log in hand and affordable (ticks 20, 40);
    # our Log at 62 is within 3 s of the one at 40 only, within 5 s of both.
    assert m["trig_goblin_barrel_all"] == 4 and m["trig_goblin_barrel"] == 2
    assert (m["trig_goblin_barrel_3s"], m["trig_goblin_barrel_5s"]) == (1, 2)
    assert m["trig_princess"] == m["trig_skeleton_army"] == m["trig_goblin_gang"] == 0
    # One push (their Hog, 4 elixir): answered by our play at 50, and their
    # troop damage on our towers rose within 25 s.
    assert (m["pushes"], m["push_unanswered"], m["push_reached"], m["push_reached_unanswered"]) == (1, 0, 1, 0)


def test_metrics_ignore_the_human_prefix_of_a_mid_match_start():
    log = _hand_made_log()
    log.start_tick = 60
    m = EG.match_metrics(log)
    assert m["plays"] == 2 and m["plays_Hog Rider"] == 1 and m["opp_plays"] == 1
    assert m["minutes"] == pytest.approx(50 / 600) and m["samples"] == 5


def test_summary_pools_ratios_and_clusters_by_ghost():
    rows = []
    for tag, win, n3, n in (("a", 1, 1, 2), ("a", 1, 0, 2), ("b", 0, 3, 4), ("c", 0, 0, 0)):
        r = dict(EG.match_metrics(_hand_made_log()), tag=tag, win=float(win))
        r["trig_goblin_barrel"], r["trig_goblin_barrel_3s"] = n, n3
        rows.append(r)
    s = EG.summarize(rows)
    assert s["n"] == 4 and s["ghosts"] == 3
    assert s["win"]["mean"] == 0.5 and s["win"]["se"] == pytest.approx(np.std([1, 1, 0, 0], ddof=1) / 2)
    # Ghosts as the unit: cluster means 1, 0, 0 with weights 2, 1, 1.
    g, w = np.array([1.0, 0.0, 0.0]), np.array([2.0, 1.0, 1.0])
    assert s["win"]["se_ghost"] == pytest.approx(np.sqrt(3 / 2 * ((w * (g - 0.5)) ** 2).sum()) / 4)
    assert s["log_3s_goblin_barrel"]["p"] == pytest.approx(4 / 8)        # pooled, not a mean of ratios
    assert s["hog_hit_share"]["p"] == 0.5 and s["share_9plus"]["p"] == pytest.approx(5 / 11)
    assert s["log_3s_princess"]["p"] is None
    assert "league" in EG.table(s, None, "league")


def test_same_rows_ignores_only_timing_fields():
    a = [{"i": 0, "win": 1.0, "secs": 1.0}]
    assert EG.same_rows(a, [{"i": 0, "win": 1.0, "secs": 9.0}]) == (True, None)
    ok, diff = EG.same_rows(a, [{"i": 0, "win": 0.0, "secs": 1.0}])
    assert not ok and diff[:2] == (0, "win")


@needs_store
def test_panel_is_fixed_and_uses_validation_ghosts_that_faced_our_deck():
    st = G.get_store()
    a, b = EG.panel(st, 30, seed=5), EG.panel(st, 30, seed=5)
    assert a == b and EG.panel(st, 60, seed=5)[:30] == a
    assert [j["seed"] for j in a] == list(range(5, 35))
    assert [j["tag"] for j in a] == sorted(j["tag"] for j in a)          # in tag order
    for j in a:
        assert G.is_val_tag(j["tag"]) and st.plays_ours[j["m"], 1 - j["side"]]
    n = len(st.eligible("val", "exact"))
    big = EG.panel(st, n + 3)
    assert (big[n]["m"], big[n]["side"]) == (big[0]["m"], big[0]["side"])  # wraps round
    assert big[n]["seed"] != big[0]["seed"]


@needs_store
def test_human_reference_row():
    st = G.get_store()
    job = EG.panel(st, 3)[2]
    cfg = {"timing": G.DEFAULT_TIMING, "spec": dict(G.DEFAULT_SPEC), "temp": 1.0}
    row, again = EG.human_row(job, cfg), EG.human_row(job, cfg)
    assert EG.same_rows([row], [again]) == (True, None)
    m, us = job["m"], 1 - job["side"]
    assert row["who"] == "human" and row["tag"] == job["tag"]
    assert row["real_win"] + row["real_draw"] + row["real_loss"] == 1.0
    assert row["real_win"] == float(st.result[m] * (1 if us == 0 else -1) > 0)
    assert row["real_crowns_for"] == st.crowns[m, us]
    assert row["win"] + row["draw"] + row["loss"] == 1.0
    assert 0 < row["plays"] <= row["real_plays"] == row["human_recorded"]
    assert row["plays"] == row["human_played"]


@needs_store
@pytest.mark.skipif(not os.path.exists(BC), reason="no remote_results/bc_hog26.pth")
def test_same_seed_same_matches_whatever_the_process_count():
    st = G.get_store()
    jobs = EG.panel(st, 4)
    cfg = {"weights": BC, "temp": 1.0, "timing": G.DEFAULT_TIMING, "spec": dict(G.DEFAULT_SPEC),
           "act_delay": None, "obs_noise": None, "levels": False}
    one = EG.run("ghost", cfg, jobs, 1)
    two = EG.run("ghost", cfg, jobs, 2)                # two spawned processes
    assert EG.same_rows(one, two) == (True, None)
    assert [r["i"] for r in one] == [0, 1, 2, 3] and all(r["plays"] > 0 for r in one)
    other = EG.run("ghost", cfg, EG.panel(st, 1, seed=EG.DEFAULT_SEED + 1000), 1)
    assert other[0]["tag"] == one[0]["tag"]            # same ghost, another seed ...
    assert not EG.same_rows(one[:1], other)[0]         # ... another match


# -------------------------------------------------------------------- parity ---
def test_parity_summary():
    def row(real, sim, cf, rcf, end, rend, by, rec, played, exact):
        return {"tag": "t", "side0": 0, "real": real, "sim": sim, "agree": int(real == sim),
                "crowns_for": cf, "real_crowns_for": rcf, "crowns_against": 0,
                "real_crowns_against": 0, "end_tick": end, "real_end_tick": rend, "ended_by": by,
                "end_class": P._class(rend), "tower_frac_dealt": 0.0, "tower_frac_taken": 0.0,
                **{k: 0 for k in P.PLAY_KEYS}, "recorded": rec, "played": played, "exact": exact,
                "not_reached": rec - played}
    rows = [row(1, 1, 1, 1, 1800, 1800, "engine", 10, 10, 10),
            row(1, -1, 0, 2, 900, 1800, "engine", 10, 5, 4),
            row(-1, -1, 3, 3, 1500, 1500, "engine", 10, 10, 10),
            row(0, 1, 1, 1, 3000, 3040, "engine", 10, 10, 8)]
    s = P.summarize(rows)
    assert s["winner_agreement"]["k"] == 2 and s["winner_agreement"]["n"] == 4
    assert s["winner_agreement_real_decisive"]["n"] == 3
    assert s["crowns_for"]["exact"]["k"] == 3 and s["crowns_for"]["mean_abs_error"] == 0.5
    assert s["sim_ended_early"]["k"] == 2              # 900 vs 1800 and 3000 vs 3040
    assert s["plays"]["share_of_recorded"]["played"] == pytest.approx(35 / 40)
    assert s["plays"]["share_of_recorded"]["not_reached"] == pytest.approx(5 / 40)
    assert s["plays"]["share_of_played"]["exact"] == pytest.approx(32 / 35)
    assert s["by_real_end"]["regulation_time"]["winner_agreement"]["n"] == 2
    assert s["by_real_end"]["king_down_in_regulation"]["winner_agreement"]["p"] == 1.0
    assert s["by_real_end"]["overtime_time"]["winner_agreement"]["k"] == 0
    # Agreement over ALL matches is the headline; the old builder's denominator
    # (matches the simulation decided) is reported next to it.
    assert s["winner_agreement_sim_decided"]["n"] == 4 and "faithful" not in s
    rows[3]["sim"], rows[3]["agree"] = 0, 1                 # a simulated draw of a real draw
    rows[1]["sim"] = 0                                      # a simulated draw of a real win
    s = P.summarize(rows)
    assert (s["winner_agreement"]["k"], s["winner_agreement"]["n"]) == (3, 4)
    assert (s["winner_agreement_sim_decided"]["k"], s["winner_agreement_sim_decided"]["n"]) == (2, 2)


def test_parity_faithful_until_summary():
    def row(fu, why, cfu, cwhy, rend, agree=1, endok=1):
        return {"faithful_until": fu, "diverged": why, "crown_faithful_until": cfu,
                "crown_diverged": cwhy, "real_end_tick": rend, "sim_match_end": 3000, "agree": agree,
                "end_consistent": endok}
    rows = [row(1800, None, 1800, None, 1800),
            row(450, "below_real_final", 900, "tower_down_that_stood", 1800, 0, 0),
            row(600, "pocket_tower_stands", 600, "pocket_tower_stands", 2400, 0, 0),
            row(1500, "ended_early", 1500, "ended_early", 3040, 1, 0)]
    f = P.faithful_summary(rows)
    assert f["never_provably_diverged"]["k"] == 1 and f["end_consistent"]["k"] == 1
    assert [f["reason"][k]["k"] for k in ("none", "below_real_final", "pocket_tower_stands", "ended_early")] \
        == [1, 1, 1, 1]
    # share of the match = faithful ticks / the real end (capped at the engine's match end)
    assert f["share_of_all_match_ticks"] == pytest.approx((1800 + 450 + 600 + 1500) / (1800 + 1800 + 2400 + 3000))
    assert f["faithful_until_s"]["q50"] == 150.0 and f["share_of_match"]["n"] == 4
    assert f["matches_faithful_at_least"]["60s"]["k"] == 3 and f["matches_faithful_at_least"]["180s"]["k"] == 1
    assert f["winner_agreement_never_diverged"]["n"] == 1
    c = f["crowns"]
    assert c["reason"]["tower_down_that_stood"]["k"] == 1 and c["never_provably_diverged"]["k"] == 1
    assert c["share_of_all_match_ticks"] == pytest.approx((1800 + 900 + 600 + 1500) / 9000)


def test_card_bias_recovers_a_planted_bias_and_not_noise():
    rng = np.random.default_rng(3)
    cards = list(range(100, 140))
    strong, weak = 107, 121

    def rows(planted):
        out = []
        for _ in range(4000):
            d0 = [int(c) for c in rng.choice(cards, 8, replace=False)]
            d1 = [int(c) for c in rng.choice(cards, 8, replace=False)]
            real = int(rng.choice((-1, 1)))
            p = 0.5 + (0.25 * ((strong in d0) - (strong in d1) - (weak in d0) + (weak in d1)) if planted else 0)
            out.append({"cards_for": d0, "cards_against": d1, "real": real,
                        "sim": 1 if rng.random() < p else -1})
        return out
    cb = P.card_bias(rows(True), min_held=50, permutations=20)
    assert cb["cards"] == 40 and cb["matches"] == 4000
    assert cb["too_strong"][0]["card"] == strong and cb["too_weak"][0]["card"] == weak
    top = cb["too_strong"][0]
    assert top["lo"] < 0.25 < top["hi"] and top["lo"] > 0.15             # win-share units
    assert cb["permutation"]["real_max_abs_t"] > cb["permutation"]["shuffled_max_abs_t_max"]
    assert cb["split_half"]["coef_correlation"] > 0.5
    none = P.card_bias(rows(False), min_held=50, permutations=20)
    assert none["permutation"]["real_max_abs_t"] < 4.5
    assert none["permutation"]["real_cards_abs_t_over_3"] <= 2
    assert abs(none["split_half"]["coef_correlation"]) < 0.5


@needs_store
def test_parity_runs_on_the_live_engine_and_is_reproducible():
    a = P.parity(C.ENGINE, n=12, procs=1)
    b = P.parity(C.ENGINE, n=12, procs=1)
    assert a["rows"] == b["rows"] and len(a["rows"]) == 12
    assert os.path.samefile(a["meta"]["engine"], C.ENGINE)
    for r in a["rows"]:
        assert r["recorded"] == r["played"] + r["dropped"] + r["not_reached"]
        assert r["real"] in (-1, 0, 1) and r["sim"] in (-1, 0, 1)
    assert a["summary"]["matches"] == 12
    # The convention is ghost.CONVENTION, recorded with the result.
    assert a["meta"]["convention"] == dict(G.CONVENTION._asdict()) and a["meta"]["is_default_convention"]
    assert a["meta"]["start_elixir"] == 6.0 and a["meta"]["offset20"] == 0
    for r in a["rows"]:
        assert 0 <= r["faithful_until"] <= r["crown_faithful_until"] <= min(r["real_end_tick"], G.MATCH_END)
        assert len(r["cards_for"]) == len(r["cards_against"]) == 8
    f = a["summary"]["faithful"]
    assert f["share_of_match"]["n"] == 12 and 0 < f["share_of_all_match_ticks"] <= 1
    # An old convention stays selectable, on the same matches.
    tags = [r["tag"] for r in a["rows"]]
    b = P.parity(C.ENGINE, tags=tags, procs=1, convention="bc")
    assert (b["meta"]["timing"], b["meta"]["illegal"]) == ("bc", "force")
    assert not b["meta"]["is_default_convention"] and [r["tag"] for r in b["rows"]] == tags
    c = P.parity(C.ENGINE, tags=tags, procs=1, illegal="skip", max_wait=0)
    assert (c["meta"]["illegal"], c["meta"]["max_wait"], c["meta"]["timing"]) == ("skip", 0, "clock0")


# ------------------------------------------------- eval_panel's checkpoint load ---
def test_eval_ghost_takes_its_convention_from_ghost():
    cv = EG.convention_of({})
    assert cv is G.CONVENTION
    assert EG.convention_of({"timing": G.DEFAULT_TIMING, "levels": False})[1:] == cv[1:]   # old cfg dicts
    assert EG.convention_of({"convention": "bc"}).illegal == "force"
    assert EG.convention_of({"convention": dict(cv._asdict()), "levels": True}).levels is True


def test_eval_panel_refuses_a_checkpoint_it_cannot_load_whole(tmp_path):
    import torch
    from training import eval_panel as EP
    from training import obs_compat
    net = C.new_net("cpu")
    good = str(tmp_path / "good.pth")
    torch.save({"model": net.state_dict()}, good)
    loaded = EP._load(good)
    assert EP.dropped_tensors(loaded, good) == []
    # One layer in another shape: the flexible loader skips it silently and the
    # net keeps its random initialisation there. eval_panel must not evaluate that.
    sd = {k: v.clone() for k, v in net.state_dict().items()}
    name = [k for k, _ in net.named_parameters()][-1]
    sd[name] = torch.cat([sd[name], sd[name][:1]], 0)
    bad = str(tmp_path / "bad.pth")
    torch.save({"model": sd}, bad)
    with pytest.raises(ValueError, match=name.replace(".", r"\.")):
        EP._load(bad)
    # A layer the checkpoint lacks is as bad.
    sd = {k: v.clone() for k, v in net.state_dict().items() if k != name}
    torch.save({"model": sd}, bad)
    with pytest.raises(ValueError, match="not in the checkpoint"):
        EP._load(bad)
    if os.path.exists(BC) and obs_compat.IS_GEN2:
        # A gen-1 checkpoint on gen-2 goes through the observation adapter, whole.
        pol = EP._load(BC)
        assert isinstance(pol, obs_compat.Gen1Policy) and EP.dropped_tensors(pol, BC) == []


def test_tail_noise_follows_the_checkpoint_unless_told_otherwise():
    from training import eval_ghost as EG

    v3 = {"arch": "v3", "names": "spells", "cfg": {"use_tail": True}}
    assert EG.tail_noise_spec("auto", v3, True) == {"names": "spells"}
    assert EG.tail_noise_spec("auto", v3, False) is None                 # clean board, clean tail
    assert EG.tail_noise_spec("auto", {"arch": "v3", "cfg": {"use_tail": False}}, True) is None
    assert EG.tail_noise_spec("auto", {"arch": "v2"}, True) is None      # older nets do not read the tail
    assert EG.tail_noise_spec("auto", dict(v3, names="all"), True) == {"names": "all"}
    assert EG.tail_noise_spec("off", v3, True) is None
    assert EG.tail_noise_spec("default", v3, False) is True
    assert EG.tail_noise_spec("none", {}, False) == {"names": "none"}
    assert EG.tail_noise_spec("some/config.json", v3, True) == "some/config.json"
