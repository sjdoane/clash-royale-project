"""Look at one recorded live match next to its both-sides replay.

  towers M                 tower health every 5 s, recording vs replay, and every burst of
                           tower damage on each side (replay: by attacking card)
  scene  M T0 T1           each second: all replay units, and the detections of the recording
  frames M T0 T1 OUT.jpg   a sheet of recorded frames (--n tiles, --box x0,y0,x1,y1 as shares
                           of the frame, e.g. 0.5,0.1,1,0.6 = right lane, their side)

M is the match index in the store (default data/ghost/live1); times are match-clock seconds,
the same clock as the store's plays. Needs CR_ENGINE.

  CR_ENGINE=$PWD/build/gen2_e/engine PYTHONDONTWRITEBYTECODE=1 .venv/bin/python evals/tools/live_case.py towers 12
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from training import ghost as G  # noqa: E402

NAMES = ["own king", "own left", "own right", "opp king", "opp left", "opp right"]
MAXHP = [4824, 3052, 3052] * 2


def slot(team, x):
    return team * 3 + (0 if abs(x - 8.5) < 2 else (1 if x < 8.5 else 2))


def name(c):
    try:
        return G.card_name(c)
    except ValueError:
        return f"spawned{c}"


class Case:
    def __init__(self, store, m):
        self.st = G.GhostStore(store)
        self.m = m
        self.L = next(x for x in self.st.meta["live"] if x["tag"] == str(self.st.tag[m]))
        self.run = self.L["run"]
        best, most = None, 0
        for p in sorted(glob.glob(f"runs/live/{self.run}/reread/m*.json")):
            J = json.load(open(p))
            ov = min(self.L["t"][1], J["wall"][1]) - max(self.L["t"][0], J["wall"][0])
            if ov > most:
                best, most = J, ov
        self.J = best
        self.c = best["clock"]["c"]                   # wall seconds = c + match clock
        self.real = np.asarray(np.load(os.path.join(store, "towers.npy"), allow_pickle=True)[m], float) \
            * np.array(MAXHP)[:, None]

    def replay(self, watch=None, until=None):
        st, m = self.st, self.m
        s0, s1 = st.side(m, 0, 0), st.side(m, 1, 1)
        game = G.new_game(s0.deck, s1.deck, s0.seed)
        game.set_tower_hit_log(True)
        self.tower_ids = {u["id"]: slot(u["team"], u["x"]) for u in game.get_units() if u["tower"]}
        cv = G.get_convention(None)
        a, b = G.Script(s0, cv.max_wait, illegal=cv.illegal), G.Script(s1, cv.max_wait, illegal=cv.illegal)
        a.install(game)
        b.install(game)
        G.set_start_elixir(game, st.start_elixir)
        end = min(G.MATCH_END, s0.end_tick) if until is None else int(until * 10) + 10
        G.run_scripts(game, [a, b], end, None, watch, None, 10)
        self.sides, self.scripts = (s0, s1), (a, b)
        return game


def bursts(ev, gap=4.0):
    """[(t, label, amount)] -> [(t0, t1, {label: amount})] joined when <= gap apart."""
    out = []
    for t, lab, a in sorted(ev):
        if out and t - out[-1][1] <= gap:
            out[-1][1] = t
            out[-1][2][lab] += a
        else:
            out.append([t, t, collections.Counter({lab: a})])
    return out


def towers(C):
    class W:
        every = 10

        def __init__(self):
            self.s = {}

        def check(self, game, t):
            if t % 10 == 0:
                hp = [0.0] * 6
                for u in game.get_units():
                    if u["tower"]:
                        hp[slot(u["team"], u["x"])] = u["hp"]
                self.s[t // 10] = hp
    w = W()
    game = C.replay(w)
    n = min(C.real.shape[1], max(w.s) + 1)
    sim = np.array([w.s.get(i, w.s[min(w.s)]) for i in range(n)]).T
    print(f"match {C.m} ({C.L['split']}), run {C.run}, {n} s recorded; opponent deck {C.L['deck']}")
    print("opponent plays in the store:", len(C.sides[1].plays), " executed in the replay:", len(C.scripts[1].done),
          "| unexplained opponent elixir:", C.L["report"].get("unexplained_elixir"))
    print("   t   RECORDING " + " ".join(f"{x:>9s}" for x in NAMES) + "   REPLAY " + " ".join(f"{x:>9s}" for x in NAMES))
    for i in list(range(0, n, 5)) + [n - 1]:
        print(f"{i:4d}             " + " ".join(f"{C.real[k, i]:9.0f}" for k in range(6))
              + "          " + " ".join(f"{sim[k, i]:9.0f}" for k in range(6)))
    hits = game.get_tower_hits()
    for j in range(6):
        ev = [(t / 10, name(c), a) for t, _at, c, _aid, _tt, tid, a in hits if C.tower_ids.get(tid) == j]
        d = -np.diff(C.real[j, :n])
        rv = [(float(i + 1), "damage", float(x)) for i, x in enumerate(d) if x >= 15]
        print(f"\n{NAMES[j]}: recording lost {C.real[j, 0] - C.real[j, n - 1]:.0f}, replay lost {sim[j, 0] - sim[j, n - 1]:.0f}")
        print("  recording bursts:", "; ".join(f"{a:.0f}-{b:.0f}s {sum(c.values()):.0f}" for a, b, c in bursts(rv)))
        print("  replay bursts:   ", "; ".join(f"{a:.0f}-{b:.0f}s " + "+".join(f"{k} {v:.0f}" for k, v in c.items())
                                               for a, b, c in bursts(ev)))


def scene(C, T0, T1):
    J, c = C.J, C.c
    print("own plays as re-read (landing time):", [(round(o["t"] + 1.2, 1), o["card"], o["xy"]) for o in J["own"] if T0 - 12 <= o["t"] + 1.2 <= T1])
    print("opponent plays as re-read (32-row frame):", [(round(o["t"], 1), o["card"], o.get("form"), o["xy"], o["kind"]) for o in J["plays"] if T0 - 15 <= o["t"] <= T1])

    class W:
        every = 10

        def check(self, game, t):
            if T0 * 10 <= t <= T1 * 10 and t % 10 == 0:
                us = game.get_units()
                tw = " ".join(f"{NAMES[slot(u['team'], u['x'])]} {u['hp']:.0f}," for u in us if u["tower"])
                tr = " | ".join(f"{'OPP' if u['team'] else 'us'} {u['name']} ({u['x']:.1f},{u['y']:.1f}) hp {u['hp']:.0f}" for u in us if not u["tower"])
                print(f" replay {t / 10:5.1f}  {tw}\n      units: {tr}")
    C.replay(W(), until=T1)
    s0, s1 = C.sides
    print("own plays in the store (engine tiles, 34 rows, we are at the bottom, y small):", [(p[0] / 10, name(p[1]), p[2], p[3]) for p in s0.plays if T0 - 12 <= p[0] / 10 <= T1])
    print("opponent plays in the store:", [(p[0] / 10, name(p[1]), p[2], p[3]) for p in s1.plays if T0 - 15 <= p[0] / 10 <= T1])
    print("RECORDING, detections once a second (class, column, row converted to the engine's 34 rows, confidence; 'us'/'OPP' is the detector's guess of the side):")
    nxt = T0
    for l in open(f"runs/live/{C.run}/raw.yolo.jsonl"):
        y = json.loads(l)
        t = y["t"] - c
        if t < nxt:
            continue
        if t > T1:
            break
        nxt = t + 1.0
        print(f" recording {t:5.1f} frame {y['frame']} " + " | ".join(
            f"{'OPP' if d.get('side') else 'us'} {d['cls']} ({d['tile'][0]},{G.tile_row_to_engine(d['tile'][1])}) {d['conf']:.2f}" for d in y["dets"] if d.get("tile")))


def frames(C, T0, T1, out, n, box):
    from PIL import Image, ImageDraw  # noqa: PLC0415
    fs = []
    for l in open(f"runs/live/{C.run}/raw.yolo.jsonl"):
        y = json.loads(l)
        t = y["t"] - C.c
        if T0 <= t <= T1 and os.path.exists(f"runs/live/{C.run}/raw/f{y['frame']:06d}.jpg"):
            fs.append((t, y["frame"]))
    if not fs:
        sys.exit("no frames in that window")
    pick = [fs[int(round(i))] for i in np.linspace(0, len(fs) - 1, min(n, len(fs)))]
    ims = []
    for t, f in pick:
        im = Image.open(f"runs/live/{C.run}/raw/f{f:06d}.jpg")
        w, h = im.size
        im = im.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)))
        sc = 640 / im.size[1]
        ims.append(im.resize((max(1, int(im.size[0] * sc)), 640), Image.LANCZOS))
    sheet = Image.new("RGB", (sum(im.size[0] for im in ims), 660), "black")
    d = ImageDraw.Draw(sheet)
    x = 0
    for (t, f), im in zip(pick, ims):
        sheet.paste(im, (x, 20))
        d.text((x + 4, 4), f"{t:.1f}s f{f}", fill="white")
        x += im.size[0]
    sheet.save(out, quality=90)
    print(out, sheet.size, [round(t, 1) for t, _ in pick])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("towers", "scene", "frames"))
    ap.add_argument("m", type=int)
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--store", default="data/ghost/live1")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--box", default="0,0,1,0.82")
    a = ap.parse_args()
    C = Case(a.store, a.m)
    if a.what == "towers":
        towers(C)
    elif a.what == "scene":
        scene(C, float(a.rest[0]), float(a.rest[1]))
    else:
        frames(C, float(a.rest[0]), float(a.rest[1]), a.rest[2], a.n, [float(x) for x in a.box.split(",")])
