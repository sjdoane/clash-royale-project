"""Group harvested tracks by appearance so whole clusters can be named by eye.

    .venv-det/bin/python video_dataset/detector/cluster.py --videos 4s1o1sk-hPM ... --tag train

Enemy units have no tray identity, so each track (harvest.py) is embedded
(KataCR detector-1 backbone, layers 0-9, global-avg-pooled over the RGBA
cutout composited on grey, mean over the track's crops), tracks are clustered
(agglomerative, cosine), and each cluster gets a montage
data/det/clusters/<tag>/m_<k>.jpg (crops from different tracks + the KataCR
class vote and the tray card when the track is ours). Names are then written
by hand to video_dataset/detector/cluster_names_<tag>.json (cluster -> class or
"junk"/"mixed"), checked against data/card_art/cards-150.

Our tracks are clustered too: a cluster that mixes our named tracks exposes
tray form errors, and their known names anchor the enemy tracks next to them.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import torch

from common import DET, best_device

HARV = DET / "harvest"
OUT = DET / "clusters"


def backbone():
    from katacr_run import WEIGHTS, _import_katacr  # noqa: PLC0415
    YOLO_CR, _, _ = _import_katacr()
    m = YOLO_CR(str(WEIGHTS[0])).model
    layers = m.model[:10]
    dev = best_device()
    return layers.eval().float().to(dev), dev


@torch.no_grad()
def embed(layers, dev, crops, size=128):
    xs = []
    for rgba in crops:
        a = rgba[..., 3:4].astype(np.float32) / 255
        im = (rgba[..., :3] * a + 114 * (1 - a)).astype(np.uint8)
        h, w = im.shape[:2]
        s = size / max(h, w)
        im = cv2.resize(im, (max(int(w * s), 1), max(int(h * s), 1)))
        pad = np.full((size, size, 3), 114, np.uint8)
        pad[(size - im.shape[0]) // 2:(size - im.shape[0]) // 2 + im.shape[0],
            (size - im.shape[1]) // 2:(size - im.shape[1]) // 2 + im.shape[1]] = im
        xs.append(pad[..., ::-1].copy())
    x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).float().div(255).contiguous().to(dev)
    y = x
    for layer in layers:
        y = layer(y)
    return torch.nn.functional.normalize(y.mean((2, 3)), dim=1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dist", type=float, default=0.25)
    ap.add_argument("--min-crops", type=int, default=2)
    a = ap.parse_args()
    from sklearn.cluster import AgglomerativeClustering  # noqa: PLC0415
    layers, dev = backbone()
    recs = []
    for vid in a.videos:
        tp = HARV / vid / "tracks.parquet"
        if not tp.exists():
            continue
        tr = pd.read_parquet(tp)
        for (ev, ti), g in tr.groupby(["event", "track"]):
            g = g[g.crop != ""]
            if len(g) < a.min_crops:
                continue
            kv = Counter(c for c, p in zip(g.kcls, g.kconf) if c is not None and p is not None and p >= 0.5)
            recs.append({"vid": vid, "event": int(ev), "track": int(ti), "side": int(g.side.iloc[0]),
                         "card": g.card.iloc[0], "crops": list(g.crop),
                         "kvote": kv.most_common(1)[0][0] if kv else None,
                         "kfrac": kv.most_common(1)[0][1] / len(g) if kv else 0.0,
                         "area": float(np.median(g.area))})
    print("tracks", len(recs))
    embs = []
    for i in range(0, len(recs), 64):
        batch = recs[i:i + 64]
        crops, owner = [], []
        for j, r in enumerate(batch):
            for c in r["crops"]:
                im = cv2.imread(str(HARV / r["vid"] / "crops" / c), cv2.IMREAD_UNCHANGED)
                if im is not None and im.ndim == 3 and im.shape[2] == 4:
                    crops.append(im)
                    owner.append(j)
        e = embed(layers, dev, crops)
        owner = np.array(owner)
        for j in range(len(batch)):
            v = e[owner == j].mean(0)
            embs.append(v / (np.linalg.norm(v) + 1e-8))
    X = np.stack(embs)
    # pooled backbone features share a large common component; centre first
    X = X - X.mean(0, keepdims=True)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    cl = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                 distance_threshold=a.dist).fit_predict(X)
    out = OUT / a.tag
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "crops"} | {"crop0": r["crops"][len(r["crops"]) // 2],
                        "cluster": int(c)} for r, c in zip(recs, cl)])
    df.to_parquet(out / "tracks.parquet")
    np.save(out / "emb.npy", X)
    sizes = df.cluster.value_counts()
    print("clusters", len(sizes), "with >=3 tracks", int((sizes >= 3).sum()),
          "tracks in them", int(sizes[sizes >= 3].sum()))
    # montages
    rng = np.random.default_rng(0)
    for k, n in sizes.items():
        g = df[df.cluster == k]
        pick = g.iloc[rng.permutation(len(g))[:12]]
        tiles = []
        for r in pick.itertuples():
            im = cv2.imread(str(HARV / r.vid / "crops" / r.crop0), cv2.IMREAD_UNCHANGED)
            a_ = im[..., 3:4].astype(np.float32) / 255
            im = (im[..., :3] * a_ + 60 * (1 - a_)).astype(np.uint8)
            s = 96 / max(im.shape[:2])
            im = cv2.resize(im, None, fx=min(s, 2.0), fy=min(s, 2.0))
            t = np.full((110, 100, 3), 60, np.uint8)
            t[:im.shape[0], :im.shape[1]] = im[:110, :100]
            cv2.putText(t, ("O" if r.side == 0 else "E"), (2, 108), 0, 0.35, (255, 255, 255), 1)
            tiles.append(t)
        while len(tiles) % 6:
            tiles.append(np.full((110, 100, 3), 60, np.uint8))
        rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
        m = np.vstack(rows)
        kv = Counter(g.kvote.dropna()).most_common(3)
        cards = Counter(g.card.dropna()).most_common(3)
        head = np.full((34, m.shape[1], 3), 255, np.uint8)
        cv2.putText(head, f"c{k} n={n} ours={int((g.side == 0).sum())}", (3, 13), 0, 0.45, (0, 0, 0), 1)
        cv2.putText(head, f"K:{kv} T:{cards}"[:95], (3, 29), 0, 0.33, (0, 0, 120), 1)
        cv2.imwrite(str(out / f"m_{k:04d}.jpg"), np.vstack([head, m]))
    sizes.to_csv(out / "sizes.csv")


if __name__ == "__main__":
    main()
