"""What a checkpoint would do on a live run's recorded observations: p(play), WHAT, and WHERE for one card.

usage: CR_ENGINE=$PWD/build/gen2_c/engine .venv/bin/python evals/tools/live_where_probe.py RUN_DIR CARD T0,T1[,T0,T1 ...] CKPT [CKPT ...]
           [--merge 2,3] [--match N] [--good FILE.json] [--mirror]
T0,T1 are seconds from the start of match N (1-based, after --merge). For every decision in a window the
tool prints p(play) and the WHAT share of CARD (named as in the `hand` field, e.g. "Cannon"); over the window
it prints where the WHERE head (temperature 1.0 and 0.5) puts CARD: mass by tile column and row, the top tiles,
and with --good the mass on the tiles listed there (a JSON {"x,y": [pulled, tower_loss]} from a simulator
grid; tiles with tower_loss below the worst count as good; --mirror flips x for the other lane).
The run must have been recorded with --dump-obs.
"""
import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "evals/tools")
import live_session as L  # noqa: E402
from training import obs_compat  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("card")
    ap.add_argument("windows")
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--merge", nargs="*", default=[])
    ap.add_argument("--match", type=int, default=1)
    ap.add_argument("--good")
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    d = a.run.rstrip("/")
    R, P, seg, off = L.load(d, merge=[tuple(int(x) for x in m.split(",")) for m in a.merge])
    o = np.fromfile(d + "/obs.f32", dtype=np.float32).reshape(len(R), -1)
    t = np.fromfile(d + "/obs_tail.f32", dtype=np.float32).reshape(len(R), -1)
    O = np.concatenate([o, t], 1)
    index = {id(r): i for i, r in enumerate(R)}
    S = seg[a.match - 1]
    t0 = S[0]["t"]
    w = [float(x) for x in a.windows.split(",")]
    wins = list(zip(w[::2], w[1::2]))
    good = None
    if a.good:
        g = json.load(open(a.good))
        worst = max(v[1] for v in g.values() if v[1] is not None)
        good = set()
        for k, v in g.items():
            x, y = (int(s) for s in k.split(","))
            if v[1] is not None and v[1] < worst:
                good.add(((17 - x) if a.mirror else x, y))
    for ck in a.ckpts:
        net, info = obs_compat.load_policy(ck)
        print(f"\n=== {ck}")
        for (a0, a1) in wins:
            rows = [r for r in S if a0 <= r["t"] - t0 <= a1]
            acc = {1.0: {}, 0.5: {}}
            pp, ws, n = [], [], 0
            for r in rows:
                net.set_sampling(what_temp=1.0, where_temp=1.0, when_temp=1.0, when_bias=0.0)
                ob = torch.as_tensor(O[index[id(r)]]).unsqueeze(0)
                with torch.no_grad():
                    mask = net.affordability_mask(ob)
                    f, emb, sp, hi = net.extract_features_hires(ob)
                    h = (torch.zeros(1, net.LSTM_HIDDEN), torch.zeros(1, net.LSTM_HIDDEN))
                    cl, _, _, _, h = net.step_lstm_and_card(f, h, mask)
                    pr = torch.softmax(cl, -1)[0]
                    p_play = 1.0 - float(pr[-1])
                    hand = r.get("hand") or []
                    slot = hand.index(a.card) if a.card in hand else None
                    share = float(pr[slot] / max(1e-9, 1.0 - float(pr[-1]))) if slot is not None and bool(mask[0, slot]) else float("nan")
                    pp.append(p_play)
                    ws.append(share)
                    if not a.quiet:
                        print(f"  t {r['t'] - t0:6.1f} elixir {r['elixir']:4.1f} p(play) {p_play:.3f} {a.card} share {share:.2f} (logged p {r['v3']['p_play']:.3f}, did: {(r.get('decision') or {}).get('why')})")
                    if slot is None or not bool(mask[0, slot]):
                        continue
                    pl = net.placement_given_card(h[0], emb, torch.tensor([slot]), ob, sp, hires_map=hi)[0]
                    n += 1
                    for T in (1.0, 0.5):
                        q = torch.softmax(pl / T, -1).numpy()
                        for c in np.nonzero(q > 1e-4)[0]:
                            x, y = net.cell_to_xy(torch.tensor([int(c)]))
                            key = (int(float(x)), int(float(y)))
                            acc[T][key] = acc[T].get(key, 0.0) + float(q[c])
            print(f"  window {a0:.0f}-{a1:.0f} s: {len(rows)} decisions, mean p(play) {np.mean(pp):.3f}, mean {a.card} share {np.nanmean(ws) if n else float('nan'):.2f}, {n} with {a.card} playable")
            for T in (1.0, 0.5):
                if not n:
                    continue
                m = {k: v / n for k, v in acc[T].items()}
                top = sorted(m.items(), key=lambda kv: -kv[1])[:6]
                cols = {}
                for (x, y), v in m.items():
                    cols[x] = cols.get(x, 0.0) + v
                line = f"    WHERE T={T}: top " + ", ".join(f"({x},{y}) {v:.2f}" for (x, y), v in top)
                line += " | by column " + " ".join(f"{x}:{cols[x]:.2f}" for x in sorted(cols) if cols[x] >= 0.02)
                if good is not None:
                    line += f" | on good tiles {sum(v for k, v in m.items() if k in good):.2f}"
                print(line)


if __name__ == "__main__":
    main()
