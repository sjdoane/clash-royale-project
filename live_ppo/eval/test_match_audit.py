"""Tests for the pure parts of the match audit (audit_core.py, audit_stats.py).

    .venv/bin/python -m pytest -q live_ppo/eval/test_match_audit.py

No detector, no frames: run lengths, stitching, sampling, burst bookkeeping, the
noise-model rates from synthetic records and the config assembly."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import audit_core as AC  # noqa: E402
import audit_stats as AS  # noqa: E402


# ------------------------------------------------------------------ run lengths
def test_runs():
    assert AC.runs([]) == []
    assert AC.runs([0, 0]) == []
    assert AC.runs([1, 1, 0, 1, 0, 0, 1, 1, 1]) == [(0, 2), (3, 1), (6, 3)]
    assert AC.runs([True]) == [(0, 1)]


def test_internal_gaps_ignore_the_ends():
    # leading and trailing unseen frames are not gaps: nothing says the unit existed then
    assert AC.internal_gaps([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0]) == [(3, 1), (6, 3)]
    assert AC.internal_gaps([0, 0, 0]) == []
    assert AC.internal_gaps([1, 1, 1]) == []
    assert AC.internal_gaps([]) == []


def test_edge_runs():
    assert AC.edge_runs([0, 0, 1, 0, 1, 0]) == (2, 1)
    assert AC.edge_runs([1, 1]) == (0, 0)
    assert AC.edge_runs([0, 0, 0]) == (3, 0)


def test_hist_caps_and_orders():
    assert AC.hist([1, 1, 2, 9, 12, 8], cap=8) == {"1": 2, "2": 1, "8+": 3}
    assert list(AC.hist([3, 0, 1])) == ["0", "1", "3"]
    assert AC.hist([]) == {}


def test_geom_mean_truncated_recovers_the_mean():
    rng = np.random.default_rng(0)
    true_mean = 3.0
    x = rng.geometric(1 / true_mean, 200_000)
    seen = x[x <= 3]                               # the tracker forgets after 3 frames
    assert float(seen.mean()) < 2.0                # the naive mean is badly biased
    assert AC.geom_mean_truncated(seen, 3) == pytest.approx(true_mean, rel=0.03)
    assert AC.geom_mean_truncated([1, 1, 1], 3) == 1.0
    assert AC.geom_mean_truncated([], 3) is None
    assert AC.geom_mean_truncated([1, 2, 3], 3) is None      # flat: the data do not bound it


def test_quantile_list():
    assert AC.quantile_list([], 8) == []
    assert AC.quantile_list([5], 4) == [5, 5, 5, 5]
    q = AC.quantile_list([0] * 75 + [2] * 25, 8)
    assert q == [0, 0, 0, 0, 0, 0, 2, 2]
    assert np.mean(q) == pytest.approx(0.5)


def test_quantile_list_weighted_undoes_length_bias():
    # 100 false tracks: 90 last 1 frame, 10 last 20. Looking at random frames catches each
    # in proportion to its length.
    caught = [1] * 90 + [20] * 200
    q = AC.quantile_list(caught, 10, [1.0 / v for v in caught])
    assert q == [1] * 9 + [20]
    assert AC.quantile_list(caught, 10)[0:3] == [1, 1, 1] and AC.quantile_list(caught, 10)[4] == 20


def test_boot_mean_and_ratio():
    assert AC.boot_mean([]) == (None, None, None)
    m, lo, hi = AC.boot_mean([1, 2, 3, 4, 5], B=500, seed=1)
    assert m == 3.0 and lo < 3.0 < hi
    assert AC.boot_mean([1, 2, 3, 4, 5], B=500, seed=1) == (m, lo, hi)
    r, lo, hi = AC.boot_ratio([1, 0, 2, 1], [10, 10, 10, 10], B=500)
    assert r == pytest.approx(0.1) and lo <= r <= hi
    assert AC.boot_ratio([], []) == (None, None, None)


# ------------------------------------------------------------------ stitching
def _trk(key, first, last, p0, p1, base="skeleton", side=0, seg=0):
    return {"key": key, "seg": seg, "first": first, "last": last, "base": base, "side": side, "p0": p0, "p1": p1}


def test_stitch_links_a_reacquired_unit():
    a = _trk("a", 0, 10, (5, 5), (6, 8))
    b = _trk("b", 16, 30, (6.5, 8.5), (9, 12))       # 5 unseen frames, 0.7 tiles away
    assert AC.stitch([a, b]) == [("a", "b", 5)]


def test_stitch_refuses_other_class_side_segment_distance_and_gap():
    a = _trk("a", 0, 10, (5, 5), (6, 8))
    assert AC.stitch([a, _trk("b", 16, 30, (6, 8), (9, 9), base="goblin")]) == []
    assert AC.stitch([a, _trk("b", 16, 30, (6, 8), (9, 9), side=1)]) == []
    assert AC.stitch([a, _trk("b", 16, 30, (6, 8), (9, 9), seg=1)]) == []
    assert AC.stitch([a, _trk("b", 16, 30, (12, 8), (9, 9))]) == []
    assert AC.stitch([a, _trk("b", 30, 40, (6, 8), (9, 9))]) == []           # 19 frames: too long
    assert AC.stitch([a, _trk("b", 11, 20, (6, 8), (9, 9))]) == []           # adjacent: no gap at all


def test_stitch_is_one_to_one_shortest_gap_first():
    a = _trk("a", 0, 10, (5, 5), (6, 8))
    b = _trk("b", 13, 20, (6, 8), (6, 9))            # gap 2
    c = _trk("c", 15, 22, (6, 8), (6, 9))            # gap 4
    assert AC.stitch([a, c, b]) == [("a", "b", 2)]


# ------------------------------------------------------------------ sampling
def test_sample_frames_is_deterministic_and_seeded():
    frames = list(range(1000, 1400))
    a = AC.sample_frames(frames, 100, seed=0)
    b = AC.sample_frames(list(reversed(frames)), 100, seed=0)
    assert a == b                                                  # same seed, any input order
    assert AC.sample_frames(frames, 100, seed=1)[0] != a[0]
    pick, w, info = a
    assert len(pick) == len(set(pick)) == 100 and pick == sorted(pick) and set(pick) <= set(frames)
    assert set(w.values()) == {4.0} and info["mode"] == "random"
    assert len(AC.sample_frames(frames, 5000)[0]) == 400           # cannot draw more than there are


def test_sample_frames_stratified_weights_restore_the_population():
    frames = list(range(400))
    units = {f: (1 if f < 300 else 4 if f < 380 else 8 if f < 395 else 12) for f in frames}
    pick, w, info = AC.sample_frames(frames, 100, seed=3, strata=units)
    assert (pick, w, info) == AC.sample_frames(frames, 100, seed=3, strata=units)
    assert len(pick) == 100
    by = {}
    for f in pick:
        by.setdefault(AC.stratum_of(units[f]), []).append(f)
    assert len(by[3]) == 5 and len(by[2]) == 15                    # the small bins give all they have
    assert len(by[0]) == len(by[1]) == 40                          # the rest is shared out
    assert sum(w.values()) == pytest.approx(400)                   # weights add back to the match
    assert info["strata"]["0"] == {"population": 300, "n": 40}
    # a weighted mean over the sample estimates the match mean, a plain one does not
    true = np.mean([units[f] for f in frames])
    est = sum(w[f] * units[f] for f in pick) / sum(w.values())
    assert est == pytest.approx(true)
    assert np.mean([units[f] for f in pick]) > 1.5 * true


def test_stratum_of():
    assert [AC.stratum_of(n) for n in (0, 2, 3, 5, 6, 9, 10, 40)] == [0, 0, 1, 1, 2, 2, 3, 3]


def test_pick_bursts():
    segs = [list(range(100, 160)), list(range(300, 305)), list(range(500, 540))]
    a = AC.pick_bursts(segs, 6, 8, seed=0)
    assert a == AC.pick_bursts(segs, 6, 8, seed=0) and a != AC.pick_bursts(segs, 6, 8, seed=5)
    assert len(a) == 6
    flat = [f for b in a for f in b]
    assert len(flat) == len(set(flat)) == 48                       # no overlap
    for b in a:
        assert b == list(range(b[0], b[0] + 8))                    # consecutive, inside one segment
        assert not 300 <= b[0] < 305                               # the 5-frame segment is too short
    only = AC.pick_bursts(segs, 3, 8, seed=0, ok=lambda f: f >= 500)
    assert only and all(b[0] >= 500 for b in only)
    assert AC.pick_bursts([[1, 2, 3]], 2, 8) == []


# ------------------------------------------------------------------ config assembly
DEFAULT = json.loads((REPO / "training" / "obs_noise_default.json").read_text())


def test_assemble_config_takes_only_well_measured_fields():
    meas = {"miss_mean_frames": {"value": 1.9, "n": 62, "ci": [1.56, 2.31], "source": "label-free"},
            "wrong_side": {"value": 0.5, "n": 4, "ci": [0.0, 1.0], "source": "labels"},
            "ghost_hold_frames": {"value": None, "n": 100, "ci": None, "source": "bursts"}}
    cfg, meta = AC.assemble_config(DEFAULT, meas, min_n=30)
    assert list(cfg) == list(DEFAULT)                              # same schema, same order
    assert cfg["miss_mean_frames"] == 1.9 and meta["miss_mean_frames"]["status"] == "measured"
    assert meta["miss_mean_frames"]["n"] == 62 and meta["miss_mean_frames"]["ci"] == [1.56, 2.31]
    assert cfg["wrong_side"] == DEFAULT["wrong_side"]              # 4 samples: keeps the default, says so
    assert meta["wrong_side"]["status"] == "default" and meta["wrong_side"]["measured_value"] == 0.5
    assert "only 4 samples" in meta["wrong_side"]["why"]
    assert cfg["ghost_hold_frames"] == DEFAULT["ghost_hold_frames"]
    assert meta["ghost_hold_frames"]["why"] == "no value could be computed"
    assert meta["scale"]["why"] == "not measured by match_audit"
    assert {k for k, v in cfg.items() if v != DEFAULT[k]} == {"miss_mean_frames"}


def test_assemble_config_rejects_a_field_the_model_does_not_read():
    with pytest.raises(KeyError):
        AC.assemble_config(DEFAULT, {"first_seen_delay": {"value": 1, "n": 99}})


def test_empty_measurement_is_the_default_config():
    cfg, meta = AC.assemble_config(DEFAULT, {})
    assert cfg == DEFAULT and all(v["status"] == "default" for v in meta.values())


def _flicker(n_runs=60, delays=(0,) * 40 + (1,) * 8 + (2,) * 2):
    cond = {"unit_frames": 500, "tracks": 30, "rate": 0.05, "ci": [0.03, 0.07]}
    return {
        "perception_hz": 3.98, "battle_frames": 411,
        "first_seen": {"n": len(delays), "delays": list(delays), "mean_frames": float(np.mean(delays)),
                       "ci": [0.1, 0.4], "share_zero": 0.8},
        "miss_tracker": {
            "within_plus_stitched": {"n": n_runs, "mean": 1.9, "ci": [1.6, 2.3]},
            "within_track": {"n": 53, "mean": 1.49, "geometric_mean_if_truncated": 1.74},
            "longest_within_track_gap_frames": 3,
            "rate_by_condition": {k: cond for k in ("isolated", "early", "overlap", "spell")}},
        "disagreement_tracked": {"any": {"mean_run_frames": 3.6, "runs": 16, "mean_run_ci": [2.0, 5.5],
                                         "runs_starting_at_first_frame": 12}},
    }


def test_measured_fields_label_free_only_and_small_samples():
    m = AS.measured_fields(_flicker(), None, None, DEFAULT)
    assert set(m) == {"perception_hz", "miss_mean_frames", "flip_mean_frames", "first_seen_delay_frames"}
    cfg, meta = AC.assemble_config(DEFAULT, m, 30)
    assert cfg["miss_mean_frames"] == 1.9 and cfg["perception_hz"] == 3.98
    assert cfg["flip_mean_frames"] == DEFAULT["flip_mean_frames"]          # 16 runs: not enough
    assert meta["flip_mean_frames"]["measured_value"] == 3.6
    d = cfg["first_seen_delay_frames"]
    assert len(d) == 16 and d == sorted(d) and np.mean(d) == pytest.approx(0.24, abs=0.07)
    # every rate that needs labels is left alone
    for k in ("miss_isolated", "miss_deploy", "wrong_side", "spurious_per_unit", "pos_jitter_sigma"):
        assert cfg[k] == DEFAULT[k] and meta[k]["status"] == "default"


def test_measured_fields_prefers_labelled_bursts_when_there_are_enough():
    b = {"miss_runs_complete": {"n": 40, "runs": [1] * 20 + [3] * 20},
         "first_seen_delay": {"n": 32, "delays": [1] * 32},
         "ghost_after_death": {"deaths": 50, "prob": 0.4, "hold_frames": [1] * 15 + [2] * 5}}
    m = AS.measured_fields(_flicker(), None, b, DEFAULT)
    assert m["miss_mean_frames"]["value"] == 2.0 and m["miss_mean_frames"]["source"].startswith("labelled bursts")
    assert m["first_seen_delay_frames"]["value"] == [1] * 16
    assert m["ghost_death_prob"]["value"] == 0.4 and m["ghost_hold_frames"]["value"] == [1] * 6 + [2] * 2
    few = dict(b, miss_runs_complete={"n": 5, "runs": [4] * 5})
    assert AS.measured_fields(_flicker(), None, few, DEFAULT)["miss_mean_frames"]["value"] == 1.9


def test_measured_config_loads_in_the_noise_model():
    try:
        sys.path.insert(0, str(REPO))
        from training import obs_noise as N
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"training.obs_noise not importable here: {e}")
    cfg, _ = AC.assemble_config(DEFAULT, AS.measured_fields(_flicker(), None, None, DEFAULT), 30)
    c = N.ObsNoiseConfig(**cfg)
    assert c.miss_mean_frames == 1.9 and len(c.first_seen_delay_frames) == 16
    out = REPO / "training" / "obs_noise_measured.json"
    if out.exists():                                               # the generated proposal is a valid config
        assert set(json.loads(out.read_text())) == set(DEFAULT)
        N.load_config(str(out))


def test_pos_scale_matches_the_median():
    k = AS._pos_scale(DEFAULT, 0.4)
    rng = np.random.default_rng(1)
    n = 100_000
    sig = np.where(rng.random(n) < DEFAULT["pos_tail_prob"], DEFAULT["pos_tail_sigma"], DEFAULT["pos_jitter_sigma"]) * k
    e = rng.normal(size=(n, 2)) * DEFAULT["pos_bias_sigma"] * k + rng.normal(size=(n, 2)) * sig[:, None]
    assert float(np.median(np.hypot(e[:, 0], e[:, 1]))) == pytest.approx(0.4, rel=0.03)


# ------------------------------------------------------------------ bursts
def _nearest(prev, cur, gate=2.0):
    """A stand-in for eval_units.match: greedy nearest within the gate."""
    out, used = [], set()
    for i, a in enumerate(prev):
        best = None
        for j, b in enumerate(cur):
            d = math.hypot(a["tile"][0] - b["tile"][0], a["tile"][1] - b["tile"][1])
            if j not in used and d <= gate and (best is None or d < best[0]):
                best = (d, j)
        if best:
            used.add(best[1])
            out.append((i, best[1]))
    return out


def _u(x, y, cls="hog-rider", side=0):
    return {"cls": cls, "side": side, "tile": [x, y], "flags": []}


def test_link_chains_follows_units_and_starts_new_ones():
    gt = {10: [_u(3, 5), _u(12, 20, "skeleton", 1)],
          11: [_u(12, 20.5, "skeleton", 1), _u(3, 6)],                 # order swapped
          12: [_u(3, 7), _u(9, 9, "cannon")],                           # the skeleton died, a cannon appeared
          13: [_u(3, 8), _u(9, 9, "cannon")]}
    ch = AS.link_chains(gt, [10, 11, 12, 13], _nearest)
    assert sorted(ch) == sorted([[(0, 0), (1, 1), (2, 0), (3, 0)], [(0, 1), (1, 0)], [(2, 1), (3, 1)]])


def test_link_chains_does_not_link_another_side_or_class():
    gt = {1: [_u(3, 5, "goblin", 0)], 2: [_u(3, 5, "goblin", 1)], 3: [_u(3, 5, "knight", 1)], 4: [_u(3, 5, "unknown", 1)]}
    assert AS.link_chains(gt, [1, 2, 3, 4], _nearest) == [[(0, 0)], [(1, 0)], [(2, 0), (3, 0)]]
    # forms of one card are the same unit when the caller says so
    gt = {1: [_u(3, 5, "knight")], 2: [_u(3, 5, "knight-ev1")]}
    assert len(AS.link_chains(gt, [1, 2], _nearest)) == 2
    assert len(AS.link_chains(gt, [1, 2], _nearest, base=lambda c: c.split("-ev")[0])) == 1


def test_burst_stats_runs_first_seen_and_totals():
    T, F = True, False
    chains = [
        [(0, T), (1, F), (2, F), (3, T), (4, T), (5, F), (6, T), (7, T)],   # gaps of 2 and 1, both complete
        [(0, F), (1, F), (2, T), (3, T)],                                    # there from the start: no first-seen delay
        [(3, F), (4, F), (5, T), (6, T), (7, F)],                            # appeared at 3, seen 2 frames later
        [(5, F), (6, F), (7, F)],                                            # appeared at 5, never seen
        [(2, T), (3, T)],                                                    # appeared and seen at once
        [],
    ]
    s = AS.burst_stats(chains, 8)
    assert s["chains"] == 5 and s["unit_frames"] == 22 and s["missed"] == 11
    assert sorted(s["complete_runs"]) == [1, 2]
    assert sorted(s["runs"]) == [1, 1, 2, 2, 2, 3]
    assert s["first_seen"] == [2, None, 0] and s["never_seen"] == 1


# ------------------------------------------------------------------ noise-model rates
def _g(matched=True, flags=(), near=False, cls="hog-rider", side=0, d_cls=None, d_side=None, err=None,
       kinds=("other",), off=(0.0, 0.0), **kw):
    e = {"cls": cls, "side": side, "flags": list(flags), "tile": [5.0, 10.0], "matched": matched, "near": near,
         "kinds": list(kinds)}
    if matched:
        e.update(d_cls=d_cls or cls, d_side=side if d_side is None else d_side, err=err,
                 d_tile=[5.0 + off[0], 10.0 + off[1]], d_int=[int(round(5.0 + off[0])), int(round(10.0 + off[1]))])
    e.update(kw)
    return e


def _rec(gt, fp=(), w=1.0, frame=0):
    return {"frame": frame, "w": w, "gt": gt, "fp": list(fp), "n_gt": len(gt),
            "n_dt": sum(e["matched"] for e in gt) + len(fp)}


def test_noise_metrics_miss_conditions():
    recs = [_rec([_g(), _g(), _g(), _g(matched=False),                              # isolated: 1 of 4 missed
                  _g(flags=["deploy"]), _g(matched=False, flags=["deploy"]),        # deploy: 1 of 2
                  _g(near=True), _g(near=True), _g(near=True), _g(matched=False, near=True),   # overlap: 1 of 4
                  _g(matched=False, flags=["spell"], near=True)])]                  # spell 1 of 1, and overlap
    r, n = AS.noise_metrics(recs)
    assert r["miss_isolated"] == 0.25 and n["n_isolated"] == 4
    assert r["miss_deploy"] == 0.5 and n["n_deploy"] == 2
    assert r["miss_overlap"] == pytest.approx(2 / 5) and n["n_overlap"] == 5
    assert r["miss_spell"] == 1.0 and n["n_spell"] == 1
    assert r["miss_all"] == pytest.approx(4 / 11) and n["gt"] == 11 and n["pres"] == 7


def test_noise_metrics_wrong_side_and_card_kinds():
    recs = [_rec([
        _g(), _g(), _g(d_side=1, life_side=True), _g(d_side=1, life_side=False),
        _g(d_side=1, flags=["sideq"]),                                   # side not certain: not counted
        _g(cls="knight", d_cls="knight-ev1", err="form", kinds=("form", "other"), life_cls=True),
        _g(cls="knight", kinds=("form", "other")),
        _g(cls="knight", d_cls="knight-ev1", err="form", kinds=("form", "other"), flags=["formq"]),   # form unsure: ignored
        _g(cls="goblin", d_cls="spear-goblin", err="lookalike", kinds=("lookalike", "other"), life_cls=False),
        _g(cls="goblin", kinds=("lookalike", "other")),
        _g(cls="giant", d_cls="golem", err="other"),
        _g(cls="unknown"),                                               # no class to be wrong about
    ])]
    r, n = AS.noise_metrics(recs)
    assert n["side_n"] == 11 and r["wrong_side"] == pytest.approx(2 / 11)
    assert r["wrong_side_lifelong"] == 0.5 and n["side_life_n"] == 2
    assert n["has_form"] == 2 and r["wrong_card_form"] == 0.5
    assert n["has_lookalike"] == 2 and r["wrong_card_lookalike"] == 0.5
    assert n["cls_n"] == 11 and n["has_other"] == 11 and r["wrong_card_other"] == pytest.approx(1 / 11)
    assert r["wrong_card_any"] == pytest.approx(3 / 11)
    assert n["cls_life_n"] == 2 and r["wrong_card_lifelong"] == 0.5


def test_noise_metrics_false_boxes_ghosts_and_position():
    fp = [{"cls": "skeleton", "side": 1, "tile": [1, 1], "near_gt": 0.8, "life": 1, "before": 0, "after": 0, "ghost_like": False},
          {"cls": "skeleton", "side": 1, "tile": [2, 2], "near_gt": 6.0, "life": 2, "before": 1, "after": 0, "ghost_like": False},
          {"cls": "giant", "side": 1, "tile": [3, 3], "near_gt": 0.2, "life": 30, "before": 29, "after": 0, "ghost_like": True},
          {"cls": "bat", "side": 0, "tile": [4, 4], "near_gt": None}]
    gt = [_g(off=(0.3, 0.0)), _g(off=(0.0, -0.4)), _g(off=(0.0, 0.0)), _g(off=(0.6, 0.8)),
          _g(off=(3.0, 3.0), flags=["bld"])]                             # buildings are not in the position error
    r, n = AS.noise_metrics([_rec(gt, fp)])
    assert n["fp"] == 4 and n["fp_ghost"] == 1 and n["fp_spur"] == 3 and n["spur_life"] == [1, 2]
    assert r["fp_per_real"] == pytest.approx(4 / 5) and r["fp_per_reported"] == pytest.approx(4 / 9)
    assert r["spurious_per_unit"] == pytest.approx(3 / 5) and r["ghost_like_per_real"] == pytest.approx(1 / 5)
    assert r["spurious_near_unit"] == 0.5 and n["fp_spur_d"] == 2
    assert n["pos_n"] == 4 and r["pos_median"] == pytest.approx(0.35)
    assert r["pos_bias_x"] == pytest.approx(0.225) and r["pos_bias_y"] == pytest.approx(0.1)
    assert r["pos_same_cell"] == 0.75


def test_noise_metrics_uses_the_sampling_weights():
    quiet = _rec([_g(), _g(matched=False)], w=9.0)                    # stands for 9 frames: half missed
    busy = _rec([_g()] * 10, w=1.0)                                    # stands for 1 frame: none missed
    r, n = AS.noise_metrics([quiet, busy])
    assert r["miss_all"] == pytest.approx(9 / 28) and n["gt"] == 12   # the count stays a plain count
    r1, _ = AS.noise_metrics([dict(quiet, w=1.0), busy])
    assert r1["miss_all"] == pytest.approx(1 / 12)


def test_noise_metrics_empty():
    r, n = AS.noise_metrics([])
    assert math.isnan(r["miss_all"]) and math.isnan(r["wrong_side"]) and n["pos_n"] == 0
    assert set(AS.N_OF) <= set(r)


# ------------------------------------------------------------------ small helpers
def test_merge_rows_keeps_matches_apart():
    a = [{"frame": 5, "t": 10.0, "dets": []}, {"frame": 6, "t": 10.25, "dets": []}]
    b = [{"frame": 5, "t": 3.0, "dets": []}, {"frame": 6, "t": 3.25, "dets": []}]
    m = AS.merge_rows([a, [], b])
    assert [r["t"] for r in m] == [0.0, 0.25, 1000.25, 1000.5]
    assert len({r["frame"] for r in m}) == 4 and a[0]["frame"] == 5         # inputs untouched


def test_paired_delta_sees_a_consistent_gain():
    base = [{"x": 0.0}] * 30 + [{"x": 1.0}] * 30
    var = [{"x": v["x"] + 0.1} for v in base]
    lo, hi = AS.paired_delta(base, var, lambda rs: float(np.mean([r["x"] for r in rs])), B=200)
    assert lo == pytest.approx(0.1) and hi == pytest.approx(0.1)       # paired: no sampling noise in the difference
    assert AS.paired_delta([], [], lambda rs: 0.0) is None


def test_label_file_loader_and_prefill_check(tmp_path):
    lab = {"src": "m", "frames": {"7": [{"cls": "hog-rider", "side": "1", "tile": [3, 4]},
                                         {"cls": "zap", "side": 0, "tile": [1, 1], "flags": ["spl"]}],
                                  "9": []}}
    p = tmp_path / "labels_units_m.json"
    p.write_text(json.dumps(lab))
    gt, raw = AS.load_label_file(p)
    assert gt == {7: [{"cls": "hog-rider", "side": 1, "tile": [3.0, 4.0], "flags": []}], 9: []}
    assert raw["src"] == "m"
    assert AS.is_unreviewed(p) is False                                # no pre-fill copy beside it
    (tmp_path / "prefill").mkdir()
    (tmp_path / "prefill" / p.name).write_text(json.dumps(lab))
    assert AS.is_unreviewed(p) is True                                 # untouched: stats must refuse it
    lab["frames"]["9"].append({"cls": "skeleton", "side": 0, "tile": [2, 2], "flags": []})
    p.write_text(json.dumps(lab))
    assert AS.is_unreviewed(p) is False
