"""live_ppo/spell_flight.py: where a thrown opponent spell lands, from its first sightings.

    .venv/bin/python -m pytest -q live_ppo/eval/test_spell_flight.py

  spell_flight_cases.json   the detector's sightings of every opponent Goblin Barrel,
                            Fireball and Giant Snowball: two live runs + ten training
                            videos (kind "live" / "train": what the model was fitted on)
                            and three held-out videos (kind "heldout")

Accuracy is ASSERTED on the fit data only. The held-out throws are run through the
same code and must give an estimate; their accuracy is reported by
eval_spell_flight.py and deliberately not asserted here (no number of this module
may be chosen on them).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

EVAL = Path(__file__).resolve().parent
sys.path[:0] = [str(EVAL.parent), str(EVAL)]
import eval_spell_flight as EV  # noqa: E402
import spell_flight as F  # noqa: E402

THROWS = EV.load()
FIT_KINDS = EV.FIT_KINDS
HZ = 11.0


def synthetic(cls, d, seconds, lag=0.1, hz=HZ, t0=50.0, noise=0.0, seed=0):
    """Sightings (t, cx, cy) of a throw at d, from `lag` s after it left, for `seconds`."""
    rng = np.random.default_rng(seed)
    tau = np.arange(0.0, seconds + 1e-9, 1.0 / hz)
    cx, cy = F.model(cls, d, tau + lag)
    return [(t0 + float(a), float(x) + noise * rng.standard_normal() * F.SIGMA_X,
             float(y) + noise * rng.standard_normal() * F.SIGMA_Y) for a, x, y in zip(tau, cx, cy)]


# ------------------------------------------------------------------ the model
@pytest.mark.parametrize("cls", sorted(F.FIT))
def test_model_leaves_the_king_tower_and_comes_down_on_the_target(cls):
    p, d = F.FIT[cls], (14.0, 5.5)
    T = F.flight_s(cls, d)
    x0, y0 = F.model(cls, d, 0.0)
    x1, y1 = F.model(cls, d, T)
    assert (float(x0), float(y0)) == pytest.approx((F.LAUNCH[0], F.LAUNCH[1] + p["h0"]))
    assert (float(x1), float(y1)) == pytest.approx((d[0], d[1] + p["h1"]))
    assert F.from_launch(float(x0), float(y0))
    # an arc: half-way it is above the straight line
    xm, ym = F.model(cls, d, T / 2)
    assert float(ym) > (float(y0) + float(y1)) / 2


def test_flight_times_are_the_measured_ones():
    """Goblin Barrel at a Princess Tower: 2.87-3.05 s by the labels of live match 2
    (first sighting to the goblins); the simulator's 8.33 tiles/s gives 2.76 s."""
    assert F.flight_s("goblin-barrel", (14.0, 5.5)) == pytest.approx(2.96, abs=0.1)
    assert F.flight_s("goblin-barrel", (8.5, 0.7)) == pytest.approx(3.48, abs=0.1)
    assert F.flight_s("fireball", (14.0, 5.5)) < F.flight_s("goblin-barrel", (14.0, 5.5))
    assert F.FIT["goblin-barrel"]["v"] == pytest.approx(8.33, rel=0.06)       # the engine's speeds
    assert F.FIT["fireball"]["v"] == pytest.approx(12.5, rel=0.06)


def test_side_of_is_the_board_in_thirds():
    assert [F.side_of(x) for x in (0.0, 3.5, 5.7, 5.8, 8.5, 11.2, 11.3, 14.5, 17.0)] == \
        ["left", "left", "left", "king", "king", "king", "right", "right", "right"]


def test_from_launch():
    assert F.from_launch(8.5, 30.2) and F.from_launch(8.5, 28.0) and F.from_launch(10.5, 29.0)
    assert not F.from_launch(14.0, 20.0)        # first seen half-way down the field: not fitted
    assert not F.from_launch(3.0, 6.0)          # the goblins themselves


# ------------------------------------------------------------------ the estimator
def test_no_estimate_for_a_class_without_a_fit_or_one_sighting():
    pts = synthetic("goblin-barrel", (14.0, 5.5), 0.5)
    for cls in ("rocket", "arrows", "the-log", "goblin-barrel-ev1"):
        assert F.estimate(cls, pts) is None
    assert F.estimate("goblin-barrel", pts[:1]) is None
    assert F.estimate("goblin-barrel", pts[:2]) is not None


TARGETS = [(14.0, 5.5), (3.0, 5.5), (8.5, 1.0), (14.0, 12.0), (3.5, 16.0)]


@pytest.mark.parametrize("cls", sorted(F.FIT))
@pytest.mark.parametrize("d", TARGETS)
@pytest.mark.parametrize("lag", [0.0, 0.12, 0.25])
def test_exact_sightings_of_the_first_half_second_give_the_side(cls, d, lag):
    """Exact sightings still leave the launch time open (the first sighting comes up to
    0.3 s after the throw), so the landing point is not exact; the side is."""
    pts = synthetic(cls, d, 0.5, lag=lag)
    e = F.estimate(cls, pts)
    assert e.known and e.side == F.side_of(d[0])
    assert e.n == len(pts) and e.age == pytest.approx(pts[-1][0] - pts[0][0])
    want = pts[0][0] - lag + F.flight_s(cls, d) + F.FIT[cls]["land"]
    assert e.t_impact == pytest.approx(want, abs=0.3)
    if cls == "goblin-barrel":
        assert np.hypot(e.x - d[0], e.y - d[1]) <= 1.5
        assert e.t_impact == pytest.approx(want, abs=0.15)


@pytest.mark.parametrize("d", TARGETS)
@pytest.mark.parametrize("lag", [0.0, 0.12, 0.25])
def test_most_of_a_barrels_flight_gives_its_tile(d, lag):
    pts = synthetic("goblin-barrel", d, 0.8 * F.flight_s("goblin-barrel", d), lag=lag)
    e = F.estimate("goblin-barrel", pts)
    assert np.hypot(e.x - d[0], e.y - d[1]) <= 0.6
    assert e.t_impact == pytest.approx(pts[0][0] - lag + F.flight_s("goblin-barrel", d) + 0.03, abs=0.05)


def test_the_estimate_narrows_as_the_flight_goes_on():
    d = (14.0, 5.5)
    spread, err = [], []
    for seconds in (0.2, 0.5, 1.0, 2.0):
        e = F.estimate("goblin-barrel", synthetic("goblin-barrel", d, seconds, noise=1.0, seed=3))
        spread.append(e.spread)
        err.append(float(np.hypot(e.x - d[0], e.y - d[1])))
    assert spread == sorted(spread, reverse=True)
    assert spread[0] > 3 * spread[-1] and err[-1] <= 0.5


def test_two_sightings_are_never_called_known():
    e = F.estimate("goblin-barrel", synthetic("goblin-barrel", (14.0, 5.5), 0.1))
    assert e.n == 2 and not e.known
    assert F.KNOWN_PTS == 3 and F.KNOWN_P == 0.9


def test_the_clock_offset_does_not_matter():
    a = F.estimate("fireball", synthetic("fireball", (3.0, 8.0), 0.4, t0=0.0))
    b = F.estimate("fireball", synthetic("fireball", (3.0, 8.0), 0.4, t0=171.3))
    assert (a.x, a.y) == pytest.approx((b.x, b.y), abs=1e-6) and a.side == b.side
    assert b.t_impact - a.t_impact == pytest.approx(171.3, abs=1e-6)


def test_noisy_synthetic_barrels_side_within_0_6_s():
    """300 barrels at the three towers, each sighting off by half the scatter the
    estimator assumes (which also covers the model's own error)."""
    right = known = 0
    targets = [(14.0, 5.5), (3.0, 5.5), (8.5, 1.0)]
    for i in range(300):
        d = targets[i % 3]
        pts = synthetic("goblin-barrel", d, 0.6, lag=0.03 * (i % 9), noise=0.5, seed=i)
        k = EV.known({"cls": "goblin-barrel", "pts": [[t - pts[0][0], x, y] for t, x, y in pts]})
        if k is not None:
            known += 1
            right += k[1].side == F.side_of(d[0])
    assert known >= 0.95 * 300 and right == known


# ------------------------------------------------------------------ recorded throws (fit data)
def test_cases_file():
    n = {(k, c): len(EV.pick(THROWS, c, (k,))) for k in ("live", "train", "heldout") for c in F.FIT}
    assert n[("live", "goblin-barrel")] == 8 and n[("train", "goblin-barrel")] == 57
    assert n[("train", "fireball")] == 43 and n[("train", "giant-snowball")] == 30
    assert all(r["kind"] in ("live", "train", "heldout") for r in THROWS)
    # the fit's sample sizes are the fit data's, not more
    assert F.FIT["goblin-barrel"]["n"] == 65 and F.FIT["fireball"]["n"] == 43
    held = {"uVf1EUszTIM", "iVgmnDX-GPY", "QD_1suWFNKo", "rec_dry1", "LekrcVeQjiw", "kaLMeKpsuGY"}
    for r in THROWS:
        assert (r["kind"] == "heldout") == any(h in r["src"] for h in held), r["src"]


def test_goblin_barrel_side_is_known_within_0_6_s():
    """The target: correct side within 0.6 s of the throw at least 95 % of the time.
    Fit data: 64 of 65 known within 0.6 s of the first sighting, all right."""
    rows = EV.pick(THROWS, "goblin-barrel", FIT_KINDS)
    k = EV.known_stats(rows)
    t = np.asarray(k["t"])
    assert k["n"] == 65 and k["known"] == 65
    assert k["right"] == k["with_truth"] == 65
    assert (t <= 0.6 + 1e-6).mean() >= 0.95
    assert np.median(t) <= 0.4
    live = EV.known_stats(EV.pick(THROWS, "goblin-barrel", ("live",)))
    assert live["known"] == live["right"] == 8 and max(live["t"]) <= 0.5


def test_goblin_barrel_side_is_never_wrong_at_any_cut():
    for cut, v in EV.by_cut(EV.pick(THROWS, "goblin-barrel", FIT_KINDS)).items():
        assert v["side_right"] == v["n"], cut


def test_goblin_barrel_tile_and_time_error_fall_with_time():
    by = EV.by_cut(EV.pick(THROWS, "goblin-barrel", FIT_KINDS))
    med = {c: float(np.median(v["err"])) for c, v in by.items()}
    p90 = {c: float(np.percentile(v["err"], 90)) for c, v in by.items()}
    tim = {c: float(np.median(np.abs(v["et"]))) for c, v in by.items()}
    assert med[0.6] <= 1.0 and p90[0.6] <= 1.8
    assert med[1.0] <= 0.7 and p90[1.0] <= 1.2
    assert med[2.0] <= 0.7 and p90[2.0] <= 1.2
    assert med[0.2] > med[0.6] > med[1.0]
    assert tim[0.3] <= 0.15 and tim[0.6] <= 0.10 and tim[1.0] <= 0.08
    assert float(np.percentile(np.abs(by[1.0]["et"]), 90)) <= 0.2


@pytest.mark.parametrize("cls,n,in_time", [("fireball", 43, 34), ("giant-snowball", 30, 28)])
def test_fireball_and_snowball_side(cls, n, in_time):
    """Faster and shorter flights: the side is still never wrong, and known within
    0.6 s for 34 of 43 Fireballs and 28 of 30 Giant Snowballs."""
    rows = EV.pick(THROWS, cls, FIT_KINDS)
    k = EV.known_stats(rows)
    assert k["n"] == n and k["known"] == n and k["right"] == k["with_truth"]
    assert int((np.asarray(k["t"]) <= 0.6 + 1e-6).sum()) >= in_time
    for cut, v in EV.by_cut(rows).items():
        assert v["side_right"] == v["n"], cut


def test_fireball_tile_and_time_error():
    by = EV.by_cut(EV.pick(THROWS, "fireball", FIT_KINDS))
    assert float(np.median(by[0.5]["err"])) <= 0.9 and float(np.median(by[1.0]["err"])) <= 0.9
    assert float(np.median(np.abs(by[0.5]["et"]))) <= 0.1


def test_noise_table_is_what_training_carries():
    """training/obs_noise.READER2's per-age error rows are this table."""
    sys.path[:0] = [str(EVAL.parents[1])]
    from training import obs_noise as N
    for cls, name in (("goblin-barrel", "Goblin Barrel"), ("fireball", "Fireball")):
        want = [r for r in EV.noise_table(THROWS, cls) if r[3] is not None]
        got = N.READER2["flight"][name]["err"]
        assert [r[0] for r in got] == [r[0] for r in want]
        assert np.allclose(np.asarray(got, float), np.asarray(want, float), atol=0.011), cls
        t = sorted(round(k[0], 2) for k in map(EV.known, EV.pick(THROWS, cls, FIT_KINDS)) if k is not None)
        assert sorted(N.READER2["flight"][name]["known_delay_s"]) == pytest.approx(t, abs=0.006)


# ------------------------------------------------------------------ held out (run, not judged)
def test_held_out_throws_run_through_the_same_code():
    rows = [r for r in THROWS if r["kind"] == "heldout" and not r["evolved"] and r["cls"] in F.FIT]
    assert len(rows) == 28
    for r in rows:
        e = F.estimate(r["cls"], r["pts"])
        assert e is not None and e.side in F.SIDES and 0.0 <= e.x <= 17.0
