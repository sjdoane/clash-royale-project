"""Every recorded own Hog Rider play in every re-read live run: the recorded track of the Hog
(detection box, engine coordinates) by placement tile. No simulator needed.

  .venv/bin/python -I evals/tools/hog_path2.py [--out evals/hog_path2.json]

x: engine x = 0.03825031 * box centre x - 1.97494532 (pixel fit; mirror-symmetric landings confirm
it for a walking Hog; the deploying sprite reads 0.3 right). y: box bottom -> detector row ->
engine row (tile_map "tower"); a jumping Hog's y reads high.
"""
import argparse, glob, json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from live_ppo import tile_map as TM

def ex(b): return 0.03825031 * (b[0] + b[2]) / 2 - 1.97494532
def ey(b): return float(TM.det_to_engine_y(-0.04766845 * b[3] + 35.70811099))

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="evals/hog_path2.json"); a = ap.parse_args()
out = []
for rd in sorted(glob.glob("runs/live/2026*")):
    ms = sorted(glob.glob(f"{rd}/reread/m*.json"))
    if not ms or not os.path.exists(f"{rd}/raw.yolo.jsonl"): continue
    fr = []
    for l in open(f"{rd}/raw.yolo.jsonl"):
        y = json.loads(l)
        fr.append((y["t"], [(ex(d["box"]), ey(d["box"]), d.get("side"), d["conf"]) for d in y["dets"] if d["cls"] == "hog-rider"]))
    T = np.array([f[0] for f in fr])
    for p in ms:
        J = json.load(open(p)); c = J["clock"]["c"]
        for q in J["own"]:
            if q["card"] != "Hog Rider": continue
            px, py = q["xy"]; tw = c + q["t"]
            i0, i1 = np.searchsorted(T, tw - 1.6), np.searchsorted(T, tw + 9.0)
            tr = []; last = None
            for t, hs in fr[i0:i1]:
                if last is None:
                    cand = [h for h in hs if abs(h[0] - 0.3 - px) < 0.9 and -1.0 < h[1] - py < 3.5]
                else:
                    cand = [h for h in hs if abs(h[0] - last[0]) < 1.6 and -1.0 < h[1] - last[1] < 3.0]
                if cand:
                    h = min(cand, key=lambda h: (h[0] - (last or (px + 0.3, py))[0]) ** 2)
                    tr.append((round(t - tw, 2), round(h[0], 2), round(h[1], 2), h[2])); last = h
            out.append({"run": os.path.basename(rd), "m": os.path.basename(p), "t": q["t"], "place": [px, py], "real": tr})
json.dump(out, open(a.out, "w"))

def stat(tr, lo, hi):
    v = [x for dt, x, y, s in tr if lo <= y < hi]
    return float(np.median(v)) if len(v) >= 2 else None
G = collections.defaultdict(list)
for o in out: G[tuple(o["place"])].append(o)
print("place      n | x at rows 13-15 (approach) | x at rows 18.3-19.5 | rows 19.5-21 | rows 21-23   (median over plays; each play's median)")
for k in sorted(G, key=lambda k: (-(k[1] == 15), k[0], k[1])):
    row = f"({k[0]:2d},{k[1]:2d}) {len(G[k]):3d} |"
    for lo, hi in ((13, 15.2), (18.3, 19.5), (19.5, 21), (21, 23)):
        v = [s for s in (stat(o["real"], lo, hi) for o in G[k]) if s is not None]
        row += f"  {np.median(v):5.2f} [{np.percentile(v,25):5.2f},{np.percentile(v,75):5.2f}] n{len(v):<3d}|" if v else "            -          |"
    print(row)
