"""Final tower health of every recorded live match vs the official battle log, both sides.

    .venv/bin/python live_ppo/eval/tower_check.py                       # the tower re-read (tower_reread.py)
    .venv/bin/python live_ppo/eval/tower_check.py --source store [--store data/ghost/live3]
    options: --tol 40 (HP per tower; the old towers_agree tolerance)  --hit 320 (the "within one hit" column)
             --dir runs/tower_check/reread (where tower_reread.py wrote)  --quiet (mismatches only)

Exit code 1 when any comparable tower differs by more than --tol, except the listed known exceptions
(EXCEPTIONS: printed with their reason, not counted); 0 otherwise.

What is compared, per side:
  * Princess Towers: the log lists the STANDING ones (a destroyed tower is absent); compared as sorted lists
    (same count, each within tol). The recording's destroyed towers read 0.
  * our King (OwnKing reader; an unread King is full).
  * The log lists no Princess Towers for a side whose King fell (3 crowns against it, or a forfeit), and the
    recording ends with the King's fall: that side is "n/a".
  * the opponent's King (OppKing reader, tower_reread.py; an unread King is full). For the store it is the
    stock reading the live loop recorded (hud_live.OPP_KING_READER is off live). A King that fell (log 0)
    is n/a. Counted on its own line, and in the opponent side's agreement.
Health in the store is a share of the tower's max; it is turned into HP with the max the re-read latched for
that tower, else the battle log's tower troop (TROOP_MAX), else 3052.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO)]
from training import live_battlelog as BL  # noqa: E402

KING_MAX = 4824
# King max HP by the battle log's tower troop level (an unread King is full; a read one uses the max the
# screen showed). Every King in the local log is level 11.
KING_MAX_BY_LEVEL = {11: 4824}
# Known failures that cannot be fixed by reading: (recorder run/match, side) -> reason. Store matches are
# named by the same run/match (the tower re-read that overlaps them).
EXCEPTIONS = {
    ("20261124-094533/m1", "own"): "store m26: iPhone Mirroring dropped ('iPhone in Use') while our left "
                                   "tower read ~627; the log says it fell after the recording ended",
    ("20261124-094533/m3", "opp"): "store m28: forfeit; the log lists the opponent King at 0 and no towers",
}
# Princess-slot max HP by the battle log's tower troop (supportCards name, level). Tower Princess level 11
# is the engine's value; Royal Chef is the max the reader latched (read on a full bar) in live3 match 14.
TROOP_MAX = {("Tower Princess", 11): 3052, ("Royal Chef", 3): 2703}
STORE_ORDER = ("own_king", "own_princess_left", "own_princess_right", "opp_king", "opp_princess_left",
               "opp_princess_right")


def king_max(player):
    s = (player.get("supportCards") or [{}])[0]
    return KING_MAX_BY_LEVEL.get(s.get("level"), KING_MAX)


def troop_max(player):
    s = (player.get("supportCards") or [{}])[0]
    return TROOP_MAX.get((s.get("name"), s.get("level")))


def side_check(log_pr, log_king, rec_pr, rec_king, tol):
    """-> (princess ok or None for n/a, king ok or None)."""
    if log_king == 0:
        return None, None
    r = sorted(int(x) for x in rec_pr if x > 0)
    pr = len(r) == len(log_pr) and all(abs(a - c) <= tol for a, c in zip(sorted(log_pr), r))
    k = None if rec_king is None else abs(int(rec_king) - int(log_king)) <= tol
    return pr, k


def final_hp(st, full):
    if st is None:
        return None
    if st["dead"]:
        return 0
    return st["hp"] if st["measured"] and st["hp"] is not None else full


def rows_reread(d, log):
    out = []
    for p in sorted(glob.glob(os.path.join(d, "*", "m*.json"))):
        J = json.load(open(p))
        b = log.get(J.get("battle"))
        name = f"{J['run']}/m{J['match']}"
        if b is None:
            out.append(dict(name=name, battle=None))
            continue
        f = J["final"]
        omax = troop_max(b["opponent"][0]) or 3052
        mmax = troop_max(b["team"][0]) or 3052
        opp = [final_hp(f["opp_princess_left"], (f["opp_princess_left"] or {}).get("max") or omax),
               final_hp(f["opp_princess_right"], (f["opp_princess_right"] or {}).get("max") or omax)]
        own = [final_hp(f["own_princess_left"], mmax), final_hp(f["own_princess_right"], mmax)]
        ok_ = f.get("opp_king")
        out.append(dict(name=name, battle=b, opp=opp, own=own, own_king=final_hp(f.get("own_king"), KING_MAX),
                        opp_king=None if ok_ is None else final_hp(ok_, ok_.get("max") or king_max(b["opponent"][0])),
                        ended=J.get("ended"), tail=J.get("tail_frames"), wall=J["wall"], run=J["run"]))
    return out


def _overlap(a, b):
    return min(a[1], b[1]) - max(a[0], b[0])


def rows_store(store, d, log):
    meta = json.load(open(REPO / store / "meta.json"))["live"]
    tw = np.load(REPO / store / "towers.npy", allow_pickle=True)
    out = []
    for m, L in enumerate(meta):
        best, most = None, 0.0
        for p in glob.glob(str(REPO / "runs" / "live" / L["run"] / "reread" / "m*.json")):
            J = json.load(open(p))
            ov = _overlap(L["t"], J["wall"])
            if ov > most:
                best, most = J, ov
        tr, most = None, 0.0                   # the tower re-read of the same match: its latched max
        for p in glob.glob(os.path.join(d, L["run"], "m*.json")):
            J = json.load(open(p))
            ov = _overlap(L["t"], J["wall"])
            if ov > most:
                tr, most = J, ov
        b = log.get((best or {}).get("battle", {}).get("time"))
        key = None if tr is None else f"{tr['run']}/m{tr['match']}"
        if b is None:
            out.append(dict(name=f"store m{m}", battle=None, key=key))
            continue
        fin = np.asarray(tw[m], float)[:, -1]
        omax = troop_max(b["opponent"][0]) or 3052
        mmax = troop_max(b["team"][0]) or 3052

        def mx(k, dflt):
            v = ((tr or {}).get("final") or {}).get(k) or {}
            return v.get("max") or dflt
        opp = [int(round(fin[4] * mx("opp_princess_left", omax))), int(round(fin[5] * mx("opp_princess_right", omax)))]
        own = [int(round(fin[1] * mx("own_princess_left", mmax))), int(round(fin[2] * mx("own_princess_right", mmax)))]
        out.append(dict(name=f"store m{m}", key=key, battle=b, opp=opp, own=own, own_king=int(round(fin[0] * KING_MAX)),
                        opp_king=int(round(fin[3] * mx("opp_king", king_max(b["opponent"][0])))), ended=(tr or {}).get("ended"),
                        tail=(tr or {}).get("tail_frames"), run=L["run"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("reread", "store"), default="reread")
    ap.add_argument("--store", default="data/ghost/live3")
    ap.add_argument("--dir", default=str(REPO / "runs" / "tower_check" / "reread"))
    ap.add_argument("--tol", type=int, default=40)
    ap.add_argument("--hit", type=int, default=320, help="'within one hit' tolerance (one Hog Rider hit = 317)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--strict", action="store_true", help="a recording with no battle in the log is a mismatch")
    ap.add_argument("--json", help="write the table here")
    a = ap.parse_args()
    log = {b["battleTime"]: b for b in BL.battles()}
    R = rows_reread(a.dir, log) if a.source == "reread" else rows_store(a.store, a.dir, log)
    hdr = (f"{'match':<22} {'opp Princess log':<18} {'rec':<14} {'ok':<4} {'opp King log/rec':<16} {'ok':<4} "
           f"{'own Princess log':<18} {'rec':<14} {'ok':<4} {'own King log/rec':<16} {'ok':<4} note")
    print(hdr)
    # ok@tol, ok@hit, comparable, n/a. "opp" = the opponent's Princesses, "own" = our Princesses and King
    # (as before), "opp_king" = the opponent's King
    counts = {"opp": [0, 0, 0, 0], "own": [0, 0, 0, 0], "opp_king": [0, 0, 0, 0]}
    bad = unmatched = 0
    excepted = []
    table = []
    for r in R:
        if r["battle"] is None:
            # no battle ends within 20 s of the recording (a partial segment, or the log not pulled yet)
            print(f"{r['name']:<22} not checked: no battle in the log ends with this recording")
            unmatched += 1
            continue
        b = r["battle"]
        lo, lm = b["opponent"][0], b["team"][0]
        okk = {}
        key = r.get("key") or r["name"]
        side_bad = {}
        for side, lp, rec, rk in (("opp", lo, r["opp"], r.get("opp_king")), ("own", lm, r["own"], r["own_king"])):
            pr, k = side_check(lp.get("princessTowersHitPoints") or [], lp.get("kingTowerHitPoints"), rec, rk, a.tol)
            prh, kh = side_check(lp.get("princessTowersHitPoints") or [], lp.get("kingTowerHitPoints"), rec, rk, a.hit)
            okk[side] = (pr, k, prh, kh)
            if side == "opp":                  # the opponent King on its own line
                c = counts["opp_king"]
                if k is None:
                    c[3] += 1
                else:
                    c[0] += k
                    c[1] += kh
                    c[2] += 1
                k, kh = None, None
            if pr is None:
                counts[side][3] += 1
            else:
                ok_t = pr and (k is not False)
                ok_h = prh and (kh is not False)
                counts[side][0] += ok_t
                counts[side][1] += ok_h
                counts[side][2] += 1
            side_bad[side] = pr is False or okk[side][1] is False
            if (key, side) in EXCEPTIONS:
                excepted.append((r["name"], side, side_bad[side], EXCEPTIONS[(key, side)]))
            elif side_bad[side]:
                bad += 1
        fmt = lambda x: "n/a" if x is None else ("ok" if x else "BAD")  # noqa: E731
        note = []
        if any(v is False for v in okk["opp"][:2] + okk["own"][:2]) and r.get("ended"):
            note.append(f"recording end: {r['ended']} ({r.get('tail')} tail frames)")
        for side in ("opp", "own"):
            if (key, side) in EXCEPTIONS:
                note.append(f"known exception ({side}): {EXCEPTIONS[(key, side)]}")
        line = (f"{r['name']:<22} {str(sorted(lo.get('princessTowersHitPoints') or [])):<18} {str(r['opp']):<14} "
                f"{fmt(okk['opp'][0]):<4} {str(lo.get('kingTowerHitPoints')) + '/' + str(r.get('opp_king')):<16} "
                f"{fmt(okk['opp'][1]):<4} {str(sorted(lm.get('princessTowersHitPoints') or [])):<18} {str(r['own']):<14} "
                f"{fmt(okk['own'][0]):<4} {str(lm.get('kingTowerHitPoints')) + '/' + str(r['own_king']):<16} "
                f"{fmt(okk['own'][1]):<4} {'; '.join(note)}")
        table.append(dict(name=r["name"], log_opp=lo.get("princessTowersHitPoints"), rec_opp=r["opp"],
                          log_own=lm.get("princessTowersHitPoints"), rec_own=r["own"],
                          log_own_king=lm.get("kingTowerHitPoints"), rec_own_king=r["own_king"],
                          log_opp_king=lo.get("kingTowerHitPoints"), rec_opp_king=r.get("opp_king"),
                          ok=okk, note=note))
        if not a.quiet or "BAD" in line:
            print(line)
    for side in ("opp", "own", "opp_king"):
        c = counts[side]
        what = {"opp": "opp Princesses", "own": "own Princesses + King", "opp_king": "opp King"}[side]
        print(f"{what}: {c[0]} of {c[2]} comparable matches agree within {a.tol} HP, {c[1]} within one hit "
              f"({a.hit} HP); {c[3]} n/a (that side's King fell)")
    for name, side, failed, why in excepted:
        print(f"known exception {name} {side}: {'fails' if failed else 'passes'} (not counted in the exit code): {why}")
    if unmatched:
        print(f"{unmatched} recording(s) not checked (no battle in the log: pull it with training/live_battlelog.py; "
              f"--strict counts them as mismatches)")
        bad += unmatched if a.strict else 0
    if a.json:
        json.dump({"rows": table, "counts": counts}, open(a.json, "w"), indent=1, default=str)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
