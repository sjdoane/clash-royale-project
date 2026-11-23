"""Side classifier (ours = 0 / enemy = 1) for a detected unit box.

The detector predicts unit identity only; side comes from this small CNN on the
box crop extended upward (where the unit's level badge / HP bar sits: blue for
ours, red for the enemy) plus the team colour of the sprite itself.

    .venv-det/bin/python video_dataset/detector/side.py train \
        --sets data/det/yolo/crdd data/det/yolo/real_train
    (labels/<stem>.sides next to the YOLO txt give each box's side)

Weights: data/det/runs/side.pt. Use: SideNet.load(); predict(img, boxes).
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from common import DET, best_device

WEIGHTS = DET / "runs" / "side.pt"
S = 64


def side_crop(img, box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    X1, X2 = int(max(x1 - 0.15 * w, 0)), int(min(x2 + 0.15 * w, img.shape[1]))
    Y1, Y2 = int(max(y1 - 0.6 * h - 8, 0)), int(min(y2 + 2, img.shape[0]))
    if X2 <= X1 + 1 or Y2 <= Y1 + 1:
        return np.zeros((S, S, 3), np.uint8)
    return cv2.resize(img[Y1:Y2, X1:X2], (S, S), interpolation=cv2.INTER_AREA)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(3, 24), blk(24, 48), blk(48, 96), blk(96, 128), nn.AdaptiveAvgPool2d(1))
        self.h = nn.Linear(128, 2)

    def forward(self, x):
        return self.h(self.f(x).flatten(1))


def to_tensor(crops):
    x = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float() / 255
    return x


class SideNet:
    def __init__(self, net, device):
        self.net, self.device = net.eval(), device

    @classmethod
    def load(cls, path=WEIGHTS, device=None):
        device = device or best_device()
        net = Net()
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return cls(net.to(device), device)

    @torch.no_grad()
    def predict(self, img, boxes):
        """img: part2 BGR; boxes: list of xyxy part2 px -> (side array, p_enemy array)."""
        if len(boxes) == 0:
            return np.zeros(0, int), np.zeros(0)
        x = to_tensor([side_crop(img, b) for b in boxes]).to(self.device)
        p = torch.softmax(self.net(x), 1)[:, 1].cpu().numpy()
        return (p > 0.5).astype(int), p


def collect(sets, max_per_set=60000, seed=0):
    rng = random.Random(seed)
    items = []
    for s in sets:
        s = Path(s)
        its = []
        for lab in sorted((s / "labels").glob("*.txt")):
            sp = lab.with_suffix(".sides")
            if not sp.exists():
                continue
            sides = [ln for ln in sp.read_text().split("\n") if ln.strip()]
            lines = [ln for ln in lab.read_text().split("\n") if ln.strip()]
            if len(sides) != len(lines):
                continue
            for ln, sd in zip(lines, sides):
                its.append((s / "images" / (lab.stem + ".jpg"), [float(v) for v in ln.split()[1:5]], int(sd)))
        rng.shuffle(its)
        items += its[:max_per_set]
    return items


def crops_for(items):
    out, cache = [], {}
    for p, (cx, cy, w, h), sd in items:
        if p not in cache:
            if len(cache) > 200:
                cache.clear()
            cache[p] = cv2.imread(str(p))
        img = cache[p]
        H, W = img.shape[:2]
        box = ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)
        out.append((side_crop(img, box), sd))
    return out


def train(sets, epochs=6):
    device = best_device()
    items = collect(sets)
    items.sort(key=lambda t: str(t[0]))
    data = crops_for(items)
    random.Random(1).shuffle(data)
    nval = max(len(data) // 20, 1)
    val, tr = data[:nval], data[nval:]
    print("side train", len(tr), "val", len(val), "enemy frac", np.mean([d[1] for d in tr]))
    net = Net().to(device)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    for ep in range(epochs):
        net.train()
        random.shuffle(tr)
        for i in range(0, len(tr), 256):
            b = tr[i:i + 256]
            ims = [c if random.random() < 0.5 else c[:, ::-1] for c, _ in b]
            x = to_tensor([np.ascontiguousarray(c) for c in ims]).to(device)
            x = x * (0.8 + 0.4 * torch.rand(len(b), 1, 1, 1, device=device))
            y = torch.tensor([s for _, s in b], device=device)
            loss = nn.functional.cross_entropy(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            x = to_tensor([c for c, _ in val]).to(device)
            pred = net(x).argmax(1).cpu().numpy()
        acc = (pred == np.array([s for _, s in val])).mean()
        print(f"epoch {ep} loss {loss.item():.3f} val acc {acc:.4f}", flush=True)
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.cpu().state_dict(), WEIGHTS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    a = ap.parse_args()
    train(a.sets, a.epochs)
