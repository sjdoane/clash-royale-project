"""Opponent-play reading for any unit-detector weights: the whole table of LIVE.md "Detector".

    live_ppo/eval/opp_plays_ab.sh <weights.pt> [more weights...]
    .venv/bin/python live_ppo/eval/opp_plays_ab.py [name=]<weights.pt> ... [--conf 0.35]
                                                   [--rules both|new|old|troop|all] [--out DIR]

For each weights file it

1. builds the four side-car caches it needs (`<src>.yolo.jsonl` for rec_dry1, vid_QD_1suWFNKo,
   vid_uVf1EUszTIM, vid_iVgmnDX-GPY) exactly as the live side-car produces boxes: UnitDetector
   defaults at conf 0.1, identity geometry, UI classes dropped, every frame through the
   driver's JPEG q92 round trip. One detector process at a time, under .venv-det, resumable,
   kept in runs/opp_plays_ab/cache/<name>-<sha1 of the weights>/ (so rewritten weights get a
   new cache and an old one is never reused by mistake);
2. replays live_ppo/eval/eval_opp_plays.py in the driver's configuration (stopwatch reader
   on, spell reader on, detector cut 0.35, tolerances 1.0 s / 1.5 tiles) twice: with the
   readers' evidence rules of 2026-11-19 ("new", OppModel(evidence=True), the default) and
   without them ("old", evidence=False); `--rules all` adds "troop" (the troop rules
   without the spell-cast ones, OppModel(spell_evidence=False));
3. prints every metric of the table with its counts, for the development sources
   (rec_dry1, vid_QD_1suWFNKo), the held-out sources (vid_uVf1EUszTIM, vid_iVgmnDX-GPY) and
   all four, and writes the same to <out>/table.md and <out>/table.json
   (default runs/opp_plays_ab/).

It only reads the weights. Nothing here trains, taps or touches the network.

Reading the table: spell casts are labelled in rec_dry1 m0 and vid_uVf m0/m1 only, so spell
recall / precision are held-out numbers; the labels record no Evolution / Hero forms (every
labelled play is "base"), so "form correct" counts a correctly read Evolution as an error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DET_PY = REPO / ".venv-det" / "bin" / "python"
OUT = REPO / "runs" / "opp_plays_ab"
DEV = ("rec_dry1", "vid_QD_1suWFNKo")
TEST = ("vid_uVf1EUszTIM", "vid_iVgmnDX-GPY")
SRCS = DEV + TEST
CACHE_CONF = 0.1          # the side-car's own threshold (live_ppo/yolo_server.py)
DET_CONF = 0.35           # the driver's cut (state_pipeline.DEFAULT_FLAGS det_conf)
TOL_T, TOL_D = 1.0, 1.5   # s, tiles: eval_opp_plays.py defaults
JPEG_Q = 92               # driver.Yolo.__call__
UI = {"king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower",
      "tower-bar", "king-tower-bar", "bar", "bar-level", "clock"}
SPLITS = (("development", DEV), ("held-out", TEST), ("all four", SRCS))


# ------------------------------------------------------------------ names / cache
def parse_spec(spec: str) -> tuple[str, Path]:
    """`[name=]path` -> (name, absolute path). The default name is the run directory:
    data/det/runs/v2/weights/best.pt -> v2, runs/det_v3_pod/ab/B/weights/last.pt -> B,
    runs/det_r3_pod/train/weights/last.pt -> det_r3_pod/train."""
    name, _, path = spec.rpartition("=")
    p = Path(path).expanduser()
    p = p if p.is_absolute() else (Path.cwd() / p)
    if not name:
        parts = list(p.resolve().parts)
        if "weights" in parts:
            parts = parts[: len(parts) - 1 - parts[::-1].index("weights")]
        else:
            parts = parts[:-1] + [p.stem]
        name = parts[-1]
        if name in ("train", "last", "best") and len(parts) >= 2:
            name = f"{parts[-2]}/{name}"
    return name, p


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def cache_dir(name: str, weights: Path, root: Path) -> Path:
    return root / "cache" / f"{name.replace('/', '_')}-{sha1(weights)[:12]}"


def frame_rows(src: str) -> list[dict]:
    """The frame list of a source (the stock side-car file gives the frame keys)."""
    return [json.loads(l) for l in open(REPO / "runs" / "live" / f"{src}.yolo.jsonl")]


def cache_done(d: Path, src: str) -> bool:
    p = d / f"{src}.yolo.jsonl"
    if not p.exists():
        return False
    with open(p) as f:
        n = sum(1 for _ in f)
    with open(REPO / "runs" / "live" / f"{src}.yolo.jsonl") as f:
        return n == sum(1 for _ in f)


def build_cache(weights: Path, out: Path, srcs=SRCS, limit: int = 0) -> None:
    """Runs under .venv-det (cv2, torch, ultralytics). Appends, so it resumes."""
    sys.path.insert(0, str(REPO / "video_dataset" / "detector"))
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from detector import UnitDetector  # noqa: PLC0415  (puts engine/perception on the path)
    from capture.wda import IosGeometry  # noqa: PLC0415
    out.mkdir(parents=True, exist_ok=True)
    geo = IosGeometry((1.0, 1.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), 1.0)
    det = UnitDetector(weights=str(weights), conf=CACHE_CONF)
    det(np.zeros((976, 549, 3), np.uint8), geo)
    (out / "meta.json").write_text(json.dumps(
        {"weights": str(weights), "sha1": sha1(weights), "conf": CACHE_CONF, "jpeg_q": JPEG_Q,
         "classes": len(det.names), "device": str(det.device)}, indent=1))
    print(f"[cache] {weights}: device {det.device}, {len(det.names)} classes -> {out}", flush=True)
    for src in srcs:
        rows = frame_rows(src)
        if limit:
            rows = [r for r in rows if r.get("battle", True)][:limit]
        p = out / f"{src}.yolo.jsonl"
        done = 0
        if p.exists():
            # a run killed mid-line leaves a torn last row: keep the whole lines only
            lines = [l for l in open(p).read().split("\n") if l]
            good = []
            for l in lines:
                try:
                    json.loads(l)
                except json.JSONDecodeError:
                    break
                good.append(l)
            if len(good) != len(lines) or not open(p).read().endswith("\n"):
                p.write_text("".join(g + "\n" for g in good))
            done = len(good)
        t0, ms = time.perf_counter(), []
        with open(p, "a") as fo:
            for k, r in enumerate(rows):
                if k < done:
                    continue
                f = int(r["frame"])
                row = {"frame": f, "t": r.get("t"), "battle": r.get("battle", True), "dets": []}
                fp = REPO / "runs" / "live" / src / f"f{f:06d}.jpg"
                if not fp.exists():
                    fp = REPO / "runs" / "live" / src / f"f{f:05d}.jpg"
                if row["battle"] and fp.exists():
                    bgr = cv2.imread(str(fp))
                    _ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
                    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    t = time.perf_counter()
                    ds = [d for d in det(img, geo) if d["cls"] not in UI]
                    ms.append((time.perf_counter() - t) * 1000)
                    row["dets"] = [{"cls": d["cls"], "side": d["side"], "conf": round(d["conf"], 3),
                                    "box": [round(v, 1) for v in d["box"]], "tile": list(d["tile"])}
                                   for d in ds]
                fo.write(json.dumps(row) + "\n")
                if k % 2000 == 0:
                    print(f"[cache] {src} {k}/{len(rows)} {time.perf_counter() - t0:.0f}s "
                          f"{np.mean(ms) if ms else 0:.1f} ms/frame", flush=True)
        print(f"[cache] {src} done ({len(rows)} frames)", flush=True)


def ensure_cache(name: str, weights: Path, root: Path) -> Path:
    d = cache_dir(name, weights, root)
    if all(cache_done(d, s) for s in SRCS):
        return d
    if not DET_PY.exists():
        raise SystemExit(f"{DET_PY} is missing: the cache builder needs cv2 + ultralytics")
    # one detector process at a time: this call blocks until the cache is complete
    r = subprocess.run([str(DET_PY), str(Path(__file__).resolve()), "--build", str(weights),
                        "--cache-dir", str(d)])
    if r.returncode != 0 or not all(cache_done(d, s) for s in SRCS):
        raise SystemExit(f"cache build failed for {weights} (exit {r.returncode})")
    return d


# ------------------------------------------------------------------ scoring
def load_cache(d: Path, src: str, conf: float) -> dict:
    Y = {}
    for line in open(d / f"{src}.yolo.jsonl"):
        r = json.loads(line)
        Y[r["frame"]] = [x for x in r["dets"] if x["conf"] >= conf]
    return Y


def split_of(rows, fps, elix, cyc, srcs):
    """The part of a 4-source score that belongs to `srcs` (sources are replayed
    independently, so this equals scoring them alone)."""
    pre = tuple(f"{s}:" for s in srcs)
    return ([r for r in rows if r["src"] in srcs], [f for f in fps if f["src"] in srcs],
            [e for e in elix if e[0].startswith(pre)], [c for c in cyc if c[0].startswith(pre)])


# rule set -> OppModel kwargs (the default OppModel is "new")
RULES = {"old": {"evidence": False}, "troop": {"spell_evidence": False}, "new": {}}


def measure(d: Path, conf: float, rules: str = "new") -> dict:
    """{split: eval_opp_plays.report()} for one cache and one rule set."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import eval_opp_plays as EV  # noqa: PLC0415
    EV.load_yolo = lambda src: load_cache(d, src, conf)
    kw = RULES[rules]
    rows, fps, elix, cyc = EV.score(
        list(SRCS), lambda s, w: EV.replay(s, w, spells=True, watch=True, model_kw=kw), TOL_T, TOL_D, "all")
    out = {}
    for label, srcs in SPLITS:
        o = EV.report(*split_of(rows, fps, elix, cyc, srcs), TOL_T, TOL_D)
        o["false_plays"] = [{k: f[k] for k in ("match", "t", "pred", "spell", "psrc")}
                            for f in split_of(rows, fps, elix, cyc, srcs)[1]]
        out[label] = o
    return out


def cells(o: dict) -> dict:
    """The table's rows for one report, as strings with counts."""
    def frac(v):
        return f"{v['p']:.3f} ({v['k']}/{v['n']})" if v["n"] else "- (0/0)"
    a, pr, ps = o["all"], o["precision"], o["precision_spell"]
    el = o.get("opp_elixir")
    return {
        "play recall": frac(a["recall"]),
        "play precision": frac(pr),
        "false plays": str(pr["n"] - pr["k"]),
        "  of them spell casts": str(ps["n"] - ps["k"]),
        "card correct": frac(a["card"]),
        "form correct": frac(a["form"]),
        "strict (time, tile, card)": frac(a["strict"]),
        "spell recall": frac(o["spell"]["recall"]),
        "spell precision": frac(ps),
        "opponent elixir MAE": f"{el['mae']:.2f} ({el['frames']} frames)" if el else "-",
    }


def render(results: dict, names: list[str], rules: list[str], conf: float) -> str:
    """Markdown: one table per split, a column per (weights, rules)."""
    lines = [f"Opponent-play reading, driver configuration (stopwatch + spell readers on, "
             f"detector cut {conf}, tolerances {TOL_T} s / {TOL_D} tiles). "
             f"`old` = OppModel(evidence=False), `new` = the default"
             + (", `troop` = OppModel(spell_evidence=False)." if "troop" in rules else "."), ""]
    for label, srcs in SPLITS:
        cols = [(n, r) for n in names for r in rules]
        tab = {c: cells(results[c[0]][c[1]][label]) for c in cols}
        lines.append(f"**{label}** ({', '.join(srcs)})")
        lines.append("")
        lines.append("| | " + " | ".join(f"{n} {r}" for n, r in cols) + " |")
        lines.append("|---|" + "---|" * len(cols))
        for k in next(iter(tab.values())):
            lines.append(f"| {k.strip() if not k.startswith('  ') else '&nbsp;&nbsp;' + k.strip()} | "
                         + " | ".join(tab[c][k] for c in cols) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("weights", nargs="+", help="[name=]path/to/weights.pt")
    ap.add_argument("--conf", type=float, default=DET_CONF, help="detector cut of the readers")
    ap.add_argument("--rules", choices=("both", "new", "old", "troop", "all"), default="both",
                    help="both = old + new; all = old + troop + new")
    ap.add_argument("--out", default=str(OUT), help="cache/ and table.* go here")
    ap.add_argument("--build", action="store_true", help="(internal) build the cache of one weights file")
    ap.add_argument("--cache-dir", default=None, help="(internal) with --build")
    ap.add_argument("--limit", type=int, default=0, help="(internal) with --build: first N battle frames")
    a = ap.parse_args(argv)
    root = Path(a.out)
    if a.build:
        name, w = parse_spec(a.weights[0])
        build_cache(w, Path(a.cache_dir) if a.cache_dir else cache_dir(name, w, root), limit=a.limit)
        return
    specs = [parse_spec(s) for s in a.weights]
    for name, w in specs:
        if not w.exists():
            raise SystemExit(f"no such weights file: {w}")
    if len({n for n, _ in specs}) != len(specs):
        raise SystemExit("two weights share a name: give them as name=path")
    rules = {"both": ["old", "new"], "all": ["old", "troop", "new"]}.get(a.rules, [a.rules])
    results = {}
    for name, w in specs:
        d = ensure_cache(name, w, root)
        results[name] = {}
        for r in rules:
            t0 = time.perf_counter()
            results[name][r] = measure(d, a.conf, r)
            print(f"[score] {name} {r}: {time.perf_counter() - t0:.0f}s", file=sys.stderr, flush=True)
    md = render(results, [n for n, _ in specs], rules, a.conf)
    print(md)
    root.mkdir(parents=True, exist_ok=True)
    (root / "table.md").write_text(md + "\n")
    (root / "table.json").write_text(json.dumps(
        {"conf": a.conf, "weights": {n: str(w) for n, w in specs}, "results": results},
        indent=1, default=float))
    print(f"written: {root / 'table.md'}  {root / 'table.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
