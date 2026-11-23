"""hud_live.TowerHPFilter on the three misreads of the first live match
(runs/live/20261120-093139, our own Princesses; reads copied from its frames).

    .venv/bin/python -m pytest -q live_ppo/eval/test_tower_filter.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "live_ppo")]
from hud_live import BarReading, NumeralReading, TowerHPFilter  # noqa: E402

FULL = BarReading(110, 604, 59, 1, 9)


def num(text, conf=0.7):
    return NumeralReading(int(text) if text else None, text, conf, len(text))


def bar(fill, track=None, height=9):
    return BarReading(110, 604, fill, 59 - fill if track is None else track, height)


def tower(hp=None, max_hp=3052):
    f = TowerHPFilter()
    f.MISSING_TO_DESTROY, f.UP_NEED = 16, 5          # 1.5 s and 0.5 s at 11 Hz
    f.update(FULL, num(str(max_hp)))
    f.update(FULL, num(str(max_hp)))
    assert f.hp == max_hp
    for v in (hp,) if hp is not None else ():
        for _ in range(3):
            f.update(bar(round(59 * v / max_hp)), num(str(v), 0.8))
        assert f.hp == v
    return f


def test_badge_over_a_middle_digit_is_a_dropped_digit():
    # a goblin's level badge covered the 0 of 3013: "313" at margin 0.55-0.64, 4 reads
    f = tower(3013)
    for c in (0.58, 0.64, 0.55, 0.62):
        f.update(bar(58), num("313", c))
    assert f.hp == 3013
    assert TowerHPFilter._dropped_digits(4296, 429) and TowerHPFilter._dropped_digits(3337, 337)
    assert not TowerHPFilter._dropped_digits(3013, 2013) and not TowerHPFilter._dropped_digits(313, 313)


def test_no_destroyed_latch_over_an_unreadable_numeral_of_a_healthy_tower():
    f = tower(3013)
    for c in (0.58, 0.64, 0.55):
        f.update(bar(58), num("313", c))
    for _ in range(40):                               # 3.6 s of goblins on the numeral
        f.update(bar(13, 0), num(""))
    assert not f.destroyed and f.hp == 3013
    for v in ("1771", "1771", "1521", "1482", "1482", "1482"):
        f.update(bar(28), num(v, 0.8))
    assert f.hp == 1482


def test_big_fall_from_unclear_reads_is_not_taken():
    # troops on a 1482 numeral: "172" at margin 0.23, three reads
    f = tower(1482)
    for _ in range(3):
        f.update(bar(28), num("172", 0.23))
    assert f.hp == 1482
    f.update(bar(20), num("1107", 0.85))
    assert f.hp == 1107
    f.update(bar(17), num("904", 0.52))
    f.update(bar(17), num("904", 0.66))
    assert f.hp == 904


def test_higher_reading_that_holds_replaces_a_misread():
    f = tower(1482)
    f.hp, f.frac = 111, 111 / 3052                    # as the old filter left it
    for i in range(5):
        assert f.hp == 111
        f.update(bar(17, 43, 4 if i % 2 else 7), num("904", 0.5))   # the bar is clean on every other frame
    assert f.hp == 904 and f.log[-1][:3] == ("up", 111, 904)


def test_one_higher_read_changes_nothing():
    f = tower(904)
    f.update(bar(17), num("984", 0.9))
    f.update(bar(17), num("904", 0.9))
    for _ in range(4):
        f.update(bar(17), num("984", 0.9))            # the row was broken by a read of the held value
    assert f.hp == 904


def test_destroyed_latch_is_undone_by_a_numeral_that_holds():
    f = tower(313)                                    # the old filter's state before its latch
    for _ in range(16):
        f.update(bar(13, 0), num(""))
    assert f.destroyed and f.hp == 0
    for _ in range(9):
        f.update(bar(28, 32, 6), num("1482", 0.8))
    assert f.destroyed                                # twice the reads to undo a latch
    f.update(bar(28, 32, 6), num("1482", 0.8))
    assert not f.destroyed and f.hp == 1482


def test_rubble_stays_destroyed():
    f = tower(201)
    for _ in range(16):
        f.update(None, None)
    assert f.destroyed
    for _ in range(50):                               # low-margin garbage and no bar
        f.update(bar(0, 0, 0), num("1511", 0.3))
        f.update(None, None)
    assert f.destroyed
