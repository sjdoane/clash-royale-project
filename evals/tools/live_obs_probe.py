"""Recompute p(play) from a replay's dumped observations (obs.f32 + obs_tail.f32) and test what moves it.

usage: CR_ENGINE=$PWD/build/gen2_c/engine .venv/bin/python evals/tools/live_obs_probe.py RUN_DIR CKPT [quick]
Each named intervention edits one group of inputs for every decision of every match, then the whole match
is run through the network again (the LSTM carries state), and p(play) is reported for the decisions at
8+ elixir with one or more of our towers down.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from training import common as C  # noqa: E402
from training import obs_compat  # noqa: E402

S, X = C.SPATIAL_SIZE, int(C.EXTRA)
CE = C.CE


def load(d):
    rows = [json.loads(l) for l in open(d + "/perception.jsonl")]
    n = np.fromfile(d + "/obs_tail.f32", dtype=np.float32).size // 148
    if n != len(rows):                       # rows the driver passed without a policy step have no observation
        rows = [r for r in rows if r.get("v3")]
        assert n == len(rows), (n, len(rows))
    o = np.fromfile(d + "/obs.f32", dtype=np.float32).reshape(len(rows), -1)
    t = np.fromfile(d + "/obs_tail.f32", dtype=np.float32).reshape(len(rows), -1)
    O = np.concatenate([o, t], 1)
    ms, cur = [], []
    for i, r in enumerate(rows):
        if cur and r["t"] - rows[cur[-1]]["t"] > 8:
            ms.append(cur)
            cur = []
        cur.append(i)
    ms.append(cur)
    return rows, O, ms


def run(net, O, ms):
    p = np.zeros(len(O))
    with torch.no_grad():
        for idx in ms:
            h = (torch.zeros(1, net.LSTM_HIDDEN), torch.zeros(1, net.LSTM_HIDDEN))
            for i in idx:
                o = torch.as_tensor(O[i]).unsqueeze(0)
                mask = net.affordability_mask(o)
                f, emb, sp, hi = net.extract_features_hires(o)
                cl, a1, a2, _, h = net.step_lstm_and_card(f, h, mask)
                p[i] = 1.0 - float(torch.softmax(cl, -1)[0, C.NOOP])
    return p


def main():
    d, ck = sys.argv[1].rstrip("/"), sys.argv[2]
    rows, O, ms = load(d)
    net, info = obs_compat.load_policy(ck)
    net.set_sampling(what_temp=1.0, where_temp=1.0, when_temp=1.0, when_bias=0.0)
    down = np.array([sum(1 for k, t in r["towers"].items() if k.startswith("own") and t["dead"]) for r in rows])
    el = np.array([r["elixir"] for r in rows])
    logged = np.array([(r.get("v3") or {}).get("p_play", np.nan) for r in rows])
    base = run(net, O, ms)
    ok = ~np.isnan(logged)
    print(f"{len(rows)} decisions in {len(ms)} matches; recomputed vs logged p(play): mean {base[ok].mean():.3f} vs "
          f"{logged[ok].mean():.3f}, correlation {np.corrcoef(base[ok], logged[ok])[0, 1]:.3f}")
    sel = (el >= 8) & (down >= 1)
    sel0 = (el >= 8) & (down == 0)
    print(f"own towers at tower-down rows (obs K/L/R): min {O[sel][:, X+3:X+6].min(0)}, max {O[sel][:, X+3:X+6].max(0)}")

    def rep(name, O2):
        p = run(net, O2, ms)
        per = " ".join(f"m{k+1} {p[np.array(idx)][sel[np.array(idx)]].mean():.3f}" if sel[np.array(idx)].any() else f"m{k+1} -"
                       for k, idx in enumerate(ms))
        print(f"  {name:44s} 8+ elixir, tower down: {p[sel].mean():.3f} (n={int(sel.sum())}) | no tower down: {p[sel0].mean():.3f} | {per}")

    rep("as recorded", O)
    if len(sys.argv) > 3 and sys.argv[3] == "quick":     # the per-snapshot regression line only
        return
    for name, lo, hi, val in (
            ("clock -> 60 s", X, X + 1, None),
            ("spend scalars -> 0", X + 1, X + 3, 0.0),
            ("own towers -> full", X + 3, X + 6, 1.0),
            ("enemy towers -> full", X + 6, X + 9, 1.0),
            ("multiplier -> 1x", X + 9, X + 10, 1.0 / 3.0),
            ("cycle block -> 0", CE.CYCLE_START, CE.CYCLE_START + CE.CYCLE_BLOCK_SIZE, 0.0),
            ("opponent state block -> 0", CE.OPP_STATE_START, CE.OPP_STATE_START + CE.OPP_STATE_SIZE, 0.0),
            ("tail -> 0", O.shape[1] - 148, O.shape[1], 0.0),
            ("board -> empty", 0, S, 0.0)):
        O2 = O.copy()
        if val is None:
            O2[:, lo:hi] = O[:, lo:hi].min() + 60.0 / 300.0 * 0 + 0.1667
        else:
            O2[:, lo:hi] = val
        rep(name, O2)
    # the opponent's towers, cells of the building channel (7) and scalars together (a state the simulator can
    # produce): all three at a fraction of their health, and one princess tower gone
    nb0 = C.BOARD_H * C.BOARD_W
    cells = {"king": (8, 30), "left": (3, 27), "right": (14, 27)}
    sc = {"king": X + 6, "left": X + 7, "right": X + 8}

    def tower(O2, name, f):
        x, y = cells[name]
        i = 7 * nb0 + y * C.BOARD_W + x
        O2[:, i] = O[:, i] * f
        O2[:, sc[name]] = O[:, sc[name]] * f

    for f in (0.75, 0.5, 0.25):
        O2 = O.copy()
        for nm in cells:
            tower(O2, nm, f)
        rep(f"enemy towers x{f} (cells and scalars)", O2)
    for nm in ("left", "right"):
        O2 = O.copy()
        tower(O2, nm, 0.0)
        rep(f"enemy {nm} princess tower gone", O2)
    for nm in cells:
        O2 = O.copy()
        x, y = cells[nm]
        O2[:, 7 * nb0 + y * C.BOARD_W + x] = 0.0
        rep(f"enemy {nm} tower cell only -> 0", O2)
    # each first-match row's group written over the same-index row of later matches is not meaningful; instead
    # per-channel board ablations
    nb = C.BOARD_H * C.BOARD_W
    for ch in range(CE.NUM_CHANNELS):
        O2 = O.copy()
        O2[:, ch * nb:(ch + 1) * nb] = 0.0
        rep(f"board channel {ch} -> 0", O2)


if __name__ == "__main__":
    main()
