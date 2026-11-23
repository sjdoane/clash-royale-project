"""Screen-reading faults of live match runs/live/20261120-185026 (the second real match).

    .venv/bin/python -m pytest -q live_ppo/eval/test_live_match2.py

What went wrong in that match and is checked here:

* our Princess towers were fed 0.04-0.10 for seconds while really at 0.5-1.0: an enemy
  goblin's red level badge ("11") stood on our numeral and was read as digits
  (3052 as 312, 2677 as 217, 2448 as 218, 1000 as 110), and the filter took the low read;
* our King was fed 1.00 while it fell from 4574 to 57: its numeral was never read;
* every Goblin Barrel was two opponent plays (the sprite over their King Tower, then
  the goblins on our side).
* a Goblin Barrel's target was not in the tail until 4 s after the throw (it had
  landed by then): now from the first frames of its flight (live_ppo/spell_flight.py).

Fixtures (all from this match only, none from the held-out videos):

  labels_own_towers_live_20261120-185026.json        by-eye HP of our three towers per frame
  fixture_own_tower_reads_live_20261120-185026.json  the readers' per-frame bar / numeral reads,
                                                     before and after the reader fix
  fixture_tower_crops_live_20261120-185026.npz       ten 76x42 / 100x44 px crops around a bar
  fixture_goblin_barrels_live_20261120-185026.json   the detector's Goblin Barrel / Goblin boxes

The last test replays the recorded frames themselves and is skipped where the run
directory is not on disk.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
EVAL = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO / "live_ppo")]
import hud_live as H  # noqa: E402
import opp_plays as P  # noqa: E402

RUN = "20261120-185026"
LABELS = json.loads((EVAL / f"labels_own_towers_live_{RUN}.json").read_text())
READS = json.loads((EVAL / f"fixture_own_tower_reads_live_{RUN}.json").read_text())
BARRELS = json.loads((EVAL / f"fixture_goblin_barrels_live_{RUN}.json").read_text())
CROPS = np.load(EVAL / f"fixture_tower_crops_live_{RUN}.npz")
PRINCESS = ("own_princess_left", "own_princess_right")
TOL = 0.05                       # "the fed value is right": within 5 % of the tower's max HP
OWN_DECK = [125, 168, 129, 15, 40, 72, 33, 7]     # the deck we played in that match


# ------------------------------------------------------------------ helpers
def truth(key) -> dict:
    """{frame: true HP fraction} for the frames that were labelled."""
    out = {}
    for a, b, hp, _how in LABELS["towers"][key]:
        for fr in range(a, b + 1):
            out[fr] = hp / LABELS["max_hp"][key]
    return out


def expand(rle):
    """The fixture's run-length coded reads as (BarReading | None, NumeralReading | None) per frame."""
    for n, bar, num in rle:
        b = None if bar is None else H.BarReading(*bar)
        m = None
        if num is not None:
            m = H.NumeralReading(*num[:4])
            m.occluded = bool(num[4])
        for _ in range(n):
            yield b, m


def fed(f) -> float:
    """What the driver feeds the policy from one tower's filter."""
    return 0.0 if f.destroyed else (f.frac if f.measured else 1.0)


def errors(key, steps) -> dict:
    """{frame: fed - truth} over the labelled frames; steps yields the filter after each frame."""
    tr = truth(key)
    out = {}
    for i, f in enumerate(steps):
        fr = READS["first_frame"] + i
        if fr in tr:
            out[fr] = fed(f) - tr[fr]
    return out


def princess_steps(key, which):
    f = H.OwnTowers(hz=READS["hz"]).f[key]
    for bar, num in expand(READS[which][key]):
        f.update(bar, num)
        yield f


def king_steps():
    king = H.OwnKing(hz=READS["hz"])
    for bar, num in expand(READS["reads"]["own_king"]):
        yield king.feed(bar, num)


def share(err) -> float:
    return sum(abs(e) <= TOL for e in err.values()) / len(err)


def canvas(*names):
    """A black frame with the named crops pasted where they were cut."""
    meta = {m["name"]: m for m in json.loads(str(CROPS["meta"]))}
    img = np.zeros(tuple(int(v) for v in CROPS["shape"]), np.uint8)
    for n in names:
        c, m = CROPS[n], meta[n]
        img[m["y0"]:m["y0"] + c.shape[0], m["x0"]:m["x0"] + c.shape[1]] = c
    return img, [meta[n] for n in names]


# ------------------------------------------------------------------ own Princess towers
@pytest.mark.parametrize("key", PRINCESS)
def test_badge_digits_never_pull_the_fed_value_down(key):
    """The reads of the old reader (the badge read as digits, unflagged) through the
    filter: the fed value is never more than 5 % under the tower's real HP. Before the
    fix it sat 0.73 (left) and 0.81 (right) under it for seconds."""
    err = errors(key, princess_steps(key, "reads_before"))
    low = {fr: round(e, 3) for fr, e in err.items() if e < -TOL}
    assert not low, f"{len(low)} frames fed too low, first {sorted(low.items())[:5]}"
    assert share(err) >= 0.97


@pytest.mark.parametrize("key", PRINCESS)
def test_princess_fed_value_tracks_the_real_hp(key):
    """Today's reads (badges flagged) through the filter: >= 97.5 % of the labelled frames
    within 5 % (measured 98.9 % left, 98.2 % right; the rest are lags of 1-16 frames
    after a hit under a troop), none too low."""
    err = errors(key, princess_steps(key, "reads"))
    assert min(err.values()) >= -TOL
    assert share(err) >= 0.975


def test_reader_does_not_read_a_level_badge_as_a_digit():
    """Crops with a goblin's badge on our numeral: the old reader gave 312 (for 3052),
    217 (2677), 5111 (501), 218 (2448) and 110 (1000) with confidence 0.46-0.76."""
    clean = {"own_princess_right": "own_princess_right_f300", "own_princess_left": "own_princess_left_f1000"}
    badge = ["own_princess_right_f360", "own_princess_right_f372", "own_princess_right_f1413",
             "own_princess_left_f830", "own_princess_left_f1665"]
    for name in badge:
        key = name.rsplit("_", 1)[0]
        reader = H.OwnTowerReader()
        img, (m0,) = canvas(clean[key])
        _bar, num = reader.read(img)[key]
        assert num.value == m0["hp"] and not getattr(num, "occluded", False)   # a clear numeral reads as before
        img, (m,) = canvas(name)
        _bar, num = reader.read(img)[key]
        assert num.value is None and getattr(num, "occluded", False), (name, num)   # hidden, and said so
        assert m["hp"] not in (None, 0)


# ------------------------------------------------------------------ own King
def test_king_numeral_is_read_off_the_bar():
    king = H.OwnKing()
    for name in ("own_king_f1805", "own_king_f2000", "own_king_f2037"):
        img, (m,) = canvas(name)
        bar = H.find_own_king_bar(img)
        assert bar is not None, name
        num = king.read_numeral(img, bar)
        assert num.value == m["hp"] and num.confidence >= H.TowerHPFilter.MIN_CONF, (name, num)


def test_king_is_not_held_at_full_while_its_number_falls():
    """Our King fell 4824 -> 57 over the last 22 s; the engine's bar reader fed 1.00
    at 159-165.6 s and 175.1-179.3 s of that match. The numeral reader follows it."""
    err = errors("own_king", king_steps())
    assert share(err) >= 0.99
    tr = truth("own_king")
    falling = [fr for fr in err if tr[fr] < 0.9]
    assert len(falling) > 200
    assert all(tr[fr] + err[fr] < 0.95 for fr in falling)        # never "full" on a damaged King


def test_king_reader_keeps_the_stock_value_until_it_has_a_measurement():
    king = H.OwnKing()
    assert not king.feed(None, None).measured                    # asleep at full HP: no bar drawn


# ------------------------------------------------------------------ Goblin Barrel
def barrel_plays(hz=None, **kw):
    m = P.OppModel(OWN_DECK, spells=True, **kw)
    last = -1e9
    for _frame, t, dets in BARRELS["frames"]:
        if hz is not None and t - last < 1.0 / hz - 1e-3:
            continue
        last = t
        m.update([{"cls": BARRELS["classes"][c], "side": side, "conf": conf, "box": [x1, y1, x2, y2],
                   "tile": [tx, ty]} for c, side, conf, x1, y1, x2, y2, tx, ty in dets], t)
    return [p for p in m.log if p["name"] == "Goblin Barrel"]


@pytest.mark.parametrize("hz", [None, 4.0])
def test_six_goblin_barrels_are_six_plays(hz):
    """Six real barrels; 11 plays before the fix (12 in the live log), at the recorded
    frame rate and at the 4 Hz the evaluation scripts run at."""
    plays = barrel_plays(hz)
    assert len(plays) == len(BARRELS["barrels"]) == 6
    t_of = {fr: t for fr, t, _d in BARRELS["frames"]}
    for p, fr in zip(plays, BARRELS["barrels"]):
        assert abs(p["t"] - t_of[fr]) <= 2.0                      # dated to the throw, seen in flight
    # the cast point is where the barrel came down (our half), not their King Tower
    assert sum(p["xy"][1] <= 16 for p in plays) >= 5


def test_goblin_barrel_was_two_plays_without_the_throw_rule():
    assert len(barrel_plays(spell_evidence=False)) > 6


def _throw(t0, x=14.0, y=6.0, flight=3.0):
    """A Goblin Barrel at 4 Hz: the sprite over their King Tower for 0.5 s, nothing in
    the air, then three goblins on our side `flight` s after the first sighting."""
    fr = {round(t0 + 0.25 * k, 2): [_det("goblin-barrel", 8.5, 31.0 - k, half=20)] for k in range(3)}
    for k in range(8):
        fr[round(t0 + flight + 0.25 * k, 2)] = [_det("goblin", x + dx, y + dy)
                                                for dx, dy in ((-0.6, 0), (0.6, 0), (0, 0.7))]
    return fr


_GRID = None


def _det(cls, tx, ty, conf=0.9, side=1, half=12):
    global _GRID
    import opp_spells as S
    if _GRID is None:
        _GRID = [(float(a), float(b), x, y) for x in range(0, 549, 4) for y in range(0, 976, 4)
                 for a, b in [S.ref_to_tile(x, y)]]
    x, y = min(_GRID, key=lambda g: abs(g[0] - tx) + abs(g[1] - ty))[2:]
    return {"cls": cls, "side": side, "conf": conf, "tile": (tx, ty),
            "box": [x - half, y - half, x + half, y + half]}


def _model_plays(frames, **kw):
    m = P.OppModel(OWN_DECK, spells=True, **kw)
    t, end = 0.0, max(frames) + 4.0
    while t <= end + 1e-9:
        m.update(frames.get(round(t, 2), []), t)
        t = round(t + 0.25, 2)
    return [p for p in m.log if p["name"] == "Goblin Barrel"]


def test_launch_and_landing_of_one_throw_are_one_play():
    plays = _model_plays(_throw(2.0))
    assert len(plays) == 1
    assert plays[0]["t"] == pytest.approx(2.0, abs=0.3)           # when the card was played
    assert plays[0]["xy"][1] < 12                                 # where it landed
    assert len(_model_plays(_throw(2.0), spell_evidence=False)) == 2


def test_two_throws_are_two_plays():
    frames = _throw(2.0)
    frames.update(_throw(12.0, x=3.0))
    assert len(_model_plays(frames)) == 2


# ------------------------------------------------------------------ the barrel's target, in flight
OPP_LABELS = json.loads((EVAL / f"labels_opp_plays_live_{RUN}.json").read_text())
THROWN = [p for p in OPP_LABELS["plays"] if p["card"] == "Goblin Barrel"]
SIDE_IN_S = 0.6                  # the target: the side is in the tail this soon after the throw


def barrel_tail():
    """The six barrels through the reader and the live tail, frame by frame as live:
    per labelled barrel, [(s since the labelled throw, OPP_SPELLS 2 x 5, the barrel's
    OPP_PLAYS entry or None)] for the 4.5 s after it."""
    import obs_tail as LT
    from training import obs_v2 as V
    m = P.OppModel(OWN_DECK, spells=True)
    tail = LT.LiveTail(P.E)
    t_of = {fr: t for fr, t, _d in BARRELS["frames"]}
    out = {g["frame"]: [] for g in THROWN}
    for _frame, t, dets in BARRELS["frames"]:
        new = m.update([{"cls": BARRELS["classes"][c], "side": side, "conf": conf, "box": [x1, y1, x2, y2],
                         "tile": [tx, ty]} for c, side, conf, x1, y1, x2, y2, tx, ty in dets], t)
        tail.see_opp(m, new, t)
        vec = tail.build(t)
        for g in THROWN:
            t0 = t_of[g["frame"]]
            if -0.3 <= t - t0 <= 4.5:
                play = next((q for q in V.decode(vec, t)["opp"] if abs(q.t - t0) <= 1.0), None)
                out[g["frame"]].append((t - t0, vec[V.OPP_SPELLS].reshape(2, 5).copy(), play))
    return out, V


def test_barrel_fixture_and_labels_are_the_same_six_throws():
    assert [g["form"] for g in THROWN] == ["base", "base", "evo", "base", "base", "evo"]
    t_of = {fr: t for fr, t, _d in BARRELS["frames"]}
    # one clock offset between the labels (match seconds) and the fixture
    off = [t_of[g["frame"]] - g["t"] for g in THROWN]
    assert max(off) - min(off) < 0.02


def test_plain_barrels_side_is_in_the_tail_within_0_6_s():
    """Match 2's four plain Goblin Barrels (right tower, left, left, our King Tower):
    the tail's OPP_SPELLS entry appears 0.26-0.44 s after the labelled throw, on the
    right side from its first frame to its last, within a tile and a half of where the
    goblins came down, and counts down to the labelled landing within 0.15 s (0.3 s for
    the one at the King Tower)."""
    import spell_flight as F
    rows, V = barrel_tail()
    delays = []
    for g in (g for g in THROWN if g["form"] == "base"):
        shown = [(dt, sp[0]) for dt, sp, _p in rows[g["frame"]] if sp[0, V.S_VALID] > 0.5]
        assert shown, g["t"]
        first = shown[0][0]
        delays.append(first)
        assert first <= SIDE_IN_S, (g["t"], first)
        gx, gy = g["tile_engine"]
        flight = g["impact_t"] - g["t"]
        for dt, e in shown:
            x, y = e[V.S_X] * (V.BOARD_W - 1), e[V.S_Y] * (V.BOARD_H - 1)
            assert F.side_of(x) == F.side_of(gx), (g["t"], dt)                # never the wrong side
            assert abs(x - gx) <= 1.0 and abs(y - gy) <= 1.5, (g["t"], dt, x, y)
            # the landing time: within 0.10 s at a Princess Tower; the one at our King
            # Tower (a longer flight than any the model was fitted on live) up to 0.28 s late
            tol = 0.3 if F.side_of(gx) == "king" else 0.15
            assert abs(dt + e[V.S_TTI] * V.TTI_CAP_S - flight) <= tol, (g["t"], dt)
            assert e[V.S_CODE] == pytest.approx(V.card_code(P._BY_NAME["goblinbarrel"]))
        # one spell in the air, shown until it lands
        assert all(sp[1, V.S_VALID] == 0 for _dt, sp, _p in rows[g["frame"]])
        assert shown[-1][0] >= flight - 0.45
        # the play entry carries the same tile from the same frame on
        for dt, sp, play in rows[g["frame"]]:
            if first <= dt <= shown[-1][0]:
                e = sp[0]
                assert play is not None and play.x is not None
                assert play.x == round(e[V.S_X] * (V.BOARD_W - 1)), (g["t"], dt)
            elif dt < first and play is not None:
                assert play.x is None                                         # no tile before the estimate
    assert max(delays) <= 0.5 and len(delays) == 4


def test_evolved_barrels_get_no_estimate():
    """An evolved Goblin Barrel is two barrels, drawn as ONE widening box on the centre
    line and lost after about 1.9 s: nothing can be said about its targets, so nothing
    is (no OPP_SPELLS entry; the tile only after the old 4 s)."""
    rows, V = barrel_tail()
    for g in (g for g in THROWN if g["form"] == "evo"):
        r = rows[g["frame"]]
        assert r and not any(sp[:, V.S_VALID].any() for _dt, sp, _p in r), g["t"]
        seen = [(dt, play) for dt, _sp, play in r if play is not None]
        assert seen and seen[0][0] <= 0.5                                     # the play itself is seen at once
        assert all(play.x is None for dt, play in seen if dt < 3.5), g["t"]


def test_no_estimate_without_the_flight_model(monkeypatch):
    """The reader as before: a class without a fit (what Rocket and Arrows still are)."""
    import spell_flight as F
    monkeypatch.setattr(F, "FIT", {})
    rows, V = barrel_tail()
    assert not any(sp[:, V.S_VALID].any() for r in rows.values() for _dt, sp, _p in r)


def _fly(cls, d, hz=11.0, lag=0.1, start=None, base=None):
    """A made-up throw of class `cls` at tile d on the flight model of `base` (default:
    its own), fed to the reader frame by frame; `start`: the first seconds of the
    flight the detector misses. -> ([(s since the first sighting, the play's .land)],
    the reader's play)."""
    import spell_flight as F
    base = base or cls
    T = F.flight_s(base, d)
    m = P.OppModel(OWN_DECK, spells=True)
    t0, out, play = 30.0, [], None
    k0 = int(round((start or 0.0) * hz))
    for k in range(k0, int((T - lag) * hz)):
        tau = k / hz
        cx, cy = F.model(base, d, tau + lag)
        new = m.update([_det(cls, float(cx), float(cy), half=20)], t0 + tau)
        play = play or next((p for p in new if getattr(p, "spell", False)), None)
        out.append((tau - k0 / hz, None if play is None else getattr(play, "land", None)))
    return out, play


@pytest.mark.parametrize("d,side", [((14.0, 5.5), "right"), ((3.0, 5.5), "left"), ((8.5, 1.0), "king")])
def test_a_made_up_barrel_gets_its_side_in_flight(d, side):
    got, play = _fly("goblin-barrel", d)
    assert play.name == "Goblin Barrel"
    first = next((tau, e) for tau, e in got if e is not None)
    assert first[0] <= 0.6 and first[1].side == side and first[1].known
    assert all(e.side == side for _tau, e in got if e is not None)
    last = got[-1][1]
    assert abs(last.x - d[0]) <= 0.7 and abs(last.y - d[1]) <= 0.7


def test_a_fireball_gets_its_side_in_flight():
    got, play = _fly("fireball", (3.0, 8.0))
    assert play.name == "Fireball"
    first = next((tau, e) for tau, e in got if e is not None)
    assert first[0] <= 0.6 and first[1].side == "left"


@pytest.mark.parametrize("cls,kw", [
    ("rocket", {"base": "goblin-barrel"}),               # no fit: the old behaviour
    ("arrows", {"base": "fireball"}),
    ("goblin-barrel-ev1", {"base": "goblin-barrel"}),    # the evolved card: two barrels
    ("goblin-barrel", {"start": 1.2}),                   # first seen half-way down the field
])
def test_no_estimate_for_what_the_model_does_not_cover(cls, kw):
    got, play = _fly(cls, (14.0, 5.5), **kw)
    assert play is not None                              # the throw itself is a play, as before
    assert len(got) > 10 and all(e is None for _tau, e in got)


# ------------------------------------------------------------------ the recorded frames
@pytest.mark.skipif(not (REPO / "runs" / "live" / RUN / "raw" / "frames.jsonl").exists(),
                    reason=f"runs/live/{RUN} is not on disk")
def test_recorded_frames_replay_within_5_percent():
    """Readers + filters over the match's 1933 in-game frames (about 30 s)."""
    import cv2
    raw = REPO / "runs" / "live" / RUN / "raw"
    own, king = H.OwnTowers(hz=READS["hz"]), H.OwnKing(hz=READS["hz"])
    tr = {k: truth(k) for k in (*PRINCESS, "own_king")}
    err = {k: [] for k in tr}
    for line in open(raw / "frames.jsonl"):
        r = json.loads(line)
        if not r.get("in_game") or not r.get("file"):
            continue
        img = cv2.imread(str(raw / r["file"]))
        fs = dict(own.update(img))
        fs["own_king"] = king.update(img)
        for k, f in fs.items():
            if r["frame"] in tr[k]:
                err[k].append(fed(f) - tr[k][r["frame"]])
    for k, e in err.items():
        assert min(e) >= -TOL, k
        assert sum(abs(v) <= TOL for v in e) / len(e) >= (0.99 if k == "own_king" else 0.975), k
