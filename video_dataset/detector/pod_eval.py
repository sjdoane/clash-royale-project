"""Held-out unit scoring without the video / engine stack (runs on the training pod).

    python pod_eval.py export                       # Mac: write data/det/pod_verify/ from the 30 verify frames
    python pod_eval.py score --out metrics.json \
        --model new=/root/out/train/weights/best.pt:ours:1024 \
        --model v2=data/det/runs/v2/weights/best.pt:ours:896 \
        --model sn_v10=data/ext/sudhishneel/model/v10/best.pt:theirs:800

export: for each key of data/det/verify/<vid>/pred.json, the part2 crop in our
framing (katacr_run.part2_crop, 568x922 on this channel) and in SudhishNeel's
568x896 framing (ext_eval.THEIR_FROM_OURS), plus the hand truth and side.pt.
score: name=weights:framing:imgsz. Boxes at --conf, agnostic NMS, side from our
SideNet on the our-framing crop (boxes from the their-framing crop are mapped
back), names through ext_classmap.load_map(extend=True) when not already ours.
Metrics = verify.score's unit block (count match per (class, side) per frame),
plus the hero-collapsed variant from ext_eval.
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PV = REPO / "data" / "det" / "pod_verify"
VID = "uVf1EUszTIM"


def export():
    import shutil  # noqa: PLC0415

    from common import DET, RAW, geo_path, load_geo  # noqa: PLC0415
    from ext_eval import THEIR_FROM_OURS, THEIR_SIZE  # noqa: PLC0415
    from katacr_run import part2_crop  # noqa: PLC0415
    src = json.loads((DET / "verify" / VID / "pred.json").read_text())
    geo = load_geo(geo_path(VID))
    cap = cv2.VideoCapture(str(RAW / f"{VID}.mp4"))
    for f in ("ours", "theirs"):
        (PV / f).mkdir(parents=True, exist_ok=True)
    for key, o in src.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, o["frame"])
        ok, img = cap.read()
        p2, _ = part2_crop(img, geo)
        cv2.imwrite(str(PV / "ours" / f"{key}.png"), p2)
        cv2.imwrite(str(PV / "theirs" / f"{key}.png"),
                    cv2.warpAffine(p2, THEIR_FROM_OURS, THEIR_SIZE, flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE))
    shutil.copy(HERE / f"verify_labels_{VID}.json", PV / "labels.json")
    shutil.copy(DET / "runs" / "side.pt", PV / "side.pt")
    (PV / "affine.json").write_text(json.dumps({"their_from_ours": THEIR_FROM_OURS.tolist()}))
    print("exported", len(src), "frames ->", PV)


def _side_net(device):
    if "common" not in sys.modules:       # side.py imports common, which needs the engine checkout
        try:
            import common  # noqa: F401, PLC0415
        except Exception:  # noqa: BLE001
            sys.modules["common"] = types.SimpleNamespace(DET=REPO / "data" / "det", best_device=lambda: device)
    from side import SideNet  # noqa: PLC0415
    return SideNet.load(PV / "side.pt", device=device)


def _counts(items):
    c = Counter()
    for it in items:
        cls, side = it.rsplit(":", 1)
        c[(cls, int(side))] += 1
    return c


def score_units(lab, pred, collapse_hero=False):
    """verify.score's unit metrics (same definitions); pred: key -> [{"cls","side"}]."""
    base = (lambda c: c[:-5] if c.endswith("-hero") else c) if collapse_hero else (lambda c: c)
    tp, fp, fn = Counter(), Counter(), Counter()
    ag, wrong = [0, 0, 0], 0
    for key, L in lab.items():
        if key.startswith("_") or key not in pred or "units" not in L:
            continue
        pc = _counts(f"{base(d['cls'])}:{d['side']}" for d in pred[key])
        tc = _counts(f"{base(u.rsplit(':', 1)[0])}:{u.rsplit(':', 1)[1]}" for u in L["units"])
        for k in set(pc) | set(tc):
            m = min(pc[k], tc[k])
            tp[k[0]] += m
            fp[k[0]] += pc[k] - m
            fn[k[0]] += tc[k] - m
        for sd in (0, 1):
            fps_ = sum(pc[k] - min(pc[k], tc[k]) for k in pc if k[1] == sd and k[0] != "unknown")
            fnk_ = sum(tc[k] - min(pc[k], tc[k]) for k in tc if k[1] == sd and k[0] != "unknown")
            wrong += min(fps_, fnk_)
            ps = sum(v for k, v in pc.items() if k[1] == sd)
            ts = sum(v for k, v in tc.items() if k[1] == sd)
            ag[0] += min(ps, ts)
            ag[1] += max(ps - ts, 0)
            ag[2] += max(ts - ps, 0)
    T, F, N = sum(tp.values()), sum(fp.values()), sum(fn.values())
    return {"tp": T, "fp": F, "fn": N, "precision": round(T / max(T + F, 1), 3),
            "recall": round(T / max(T + N, 1), 3),
            "recall_known": round(T / max(T + N - fn["unknown"], 1), 3),
            "class_acc": round(T / max(T + wrong, 1), 3), "wrong_class": wrong,
            "detect_side_only": {"precision": round(ag[0] / max(ag[0] + ag[1], 1), 3),
                                 "recall": round(ag[0] / max(ag[0] + ag[2], 1), 3)},
            "per_class": {c: {"tp": tp[c], "fp": fp[c], "fn": fn[c]}
                          for c in sorted(set(tp) | set(fp) | set(fn), key=lambda c: -(tp[c] + fn[c]))}}


def run_model(weights, framing, imgsz, conf, device, side):
    from ultralytics import YOLO  # noqa: PLC0415

    from ext_classmap import load_map, target_classes  # noqa: PLC0415
    model = YOLO(weights)
    names = model.names
    ours = set(target_classes(extend=True))
    cmap = load_map(extend=True)
    A = np.array(json.loads((PV / "affine.json").read_text())["their_from_ours"])
    pred = {}
    for p in sorted((PV / "ours").glob("*.png")):
        img_ours = cv2.imread(str(p))
        img = img_ours if framing == "ours" else cv2.imread(str(PV / "theirs" / p.name))
        r = model.predict(img, conf=conf, iou=0.6, imgsz=imgsz, device=device, verbose=False, agnostic_nms=True)[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        keep = []
        for i, k in enumerate(r.boxes.cls.cpu().numpy()):
            n = names[int(k)]
            n = n if n in ours else cmap.get(n)
            if n is not None:
                keep.append((i, n))
        b = boxes[[i for i, _ in keep]].reshape(-1, 4)
        if framing == "theirs" and len(b):     # back to our framing for the side crop
            b = np.stack([(b[:, 0] - A[0, 2]) / A[0, 0], (b[:, 1] - A[1, 2]) / A[1, 1],
                          (b[:, 2] - A[0, 2]) / A[0, 0], (b[:, 3] - A[1, 2]) / A[1, 1]], 1)
        sides = side.predict(img_ours, b)[0] if len(b) else []
        pred[p.stem] = [{"cls": n, "side": int(s)} for (_, n), s in zip(keep, sides)]
    return pred


def score(models, out, conf, device):
    import torch  # noqa: PLC0415
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    lab = json.loads((PV / "labels.json").read_text())
    side = _side_net(device)
    res = {"heldout": VID, "frames": len([k for k in lab if not k.startswith("_")]), "conf": conf, "models": {}}
    for spec in models:
        name, rest = spec.split("=", 1)
        weights, framing, imgsz = rest.rsplit(":", 2)
        if not Path(weights).exists():
            res["models"][name] = {"error": f"missing {weights}"}
            continue
        pred = run_model(weights, framing, int(imgsz), conf, device, side)
        r = score_units(lab, pred)
        r["hero_collapsed"] = {k: v for k, v in score_units(lab, pred, True).items() if k != "per_class"}
        r.update(weights=weights, framing=framing, imgsz=int(imgsz))
        res["models"][name] = r
        print(name, {k: r[k] for k in ("recall_known", "recall", "precision", "class_acc")},
              "hero-collapsed recall_known", r["hero_collapsed"]["recall_known"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    print("->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "score"])
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--out", default=str(PV / "metrics.json"))
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    if a.cmd == "export":
        export()
    else:
        score(a.model, a.out, a.conf, a.device)


if __name__ == "__main__":
    main()
