"""The human BC dataset rebuilt at a finer decision step (default 2 ticks = 0.2 s).

    .venv/bin/python -m training.il_replay_bc_5hz build --workers 12
    .venv/bin/python -m training.il_replay_bc_5hz finalize
    .venv/bin/python -m training.il_replay_bc_5hz build --limit-matches 2 --shards 0:1 --out-suffix _probe

This is video_dataset/il_replay_bc.py unchanged, with four module globals
rebound in every process (spawned Pool workers included):

  SKIP           ticks per decision, 10 -> --skip (2)
  OUT / OUT_OPP  -> data/bc/hog26_5hz{,_opp}  (data/bc/hog26* is never written)
  TMP            -> data/bc/_tmp_5hz
  SHARD_MATCHES  200 -> 50, so a shard's dense float16 obs (~5x the rows per
                 match) stays ~1.6 GB for the CSR conversion in bcdata.

and one line of `run_shard` replaced, the memmap capacity, which assumes one
row per second (`ceil(duration) + 2`). The replacement is asserted, so a change
to the builder fails loudly instead of overflowing the memmap.

Everything else (hand reconstruction, legality, top-ups, tower guards, the
validation rule sha1(tag) % 20) is the 1 Hz builder's, so the held-out matches
are the same 424.
"""
import argparse
import inspect
import json
import os
import sys
import time
import textwrap
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "video_dataset"))

import il_replay_bc as IL  # noqa: E402

DEFAULT_SKIP = 2
SHARD_MATCHES = 50
_CAP_LINE = 'cap += int(math.ceil(rec["duration"])) + 2'
_CAP_NEW = 'cap += int(math.ceil(rec["duration"] * 10.0 / SKIP)) + 2'


def patch(skip=DEFAULT_SKIP, suffix=None):
    """Rebind the builder's globals for a `skip`-tick step. Idempotent."""
    suffix = suffix if suffix is not None else f"_{10 // skip}hz" if 10 % skip == 0 else f"_{skip}t"
    IL.SKIP = int(skip)
    IL.SHARD_MATCHES = SHARD_MATCHES
    IL.OUT = {"primary": IL.ROOT / "data" / "bc" / f"hog26{suffix}",
              "secondary": IL.ROOT / "data" / "bc" / f"hog26_secondary{suffix}"}
    IL.OUT_OPP = {"primary": IL.ROOT / "data" / "bc" / f"hog26{suffix}_opp",
                  "secondary": IL.ROOT / "data" / "bc" / f"hog26_secondary{suffix}_opp"}
    IL.TMP = IL.ROOT / "data" / "bc" / f"_tmp{suffix}"
    for d in list(IL.OUT.values()) + list(IL.OUT_OPP.values()):
        assert not str(d).endswith(("hog26", "hog26_opp", "hog26_secondary", "hog26_secondary_opp")), d
    if not getattr(IL, "_run_shard_5hz", False):
        src = textwrap.dedent(inspect.getsource(IL.run_shard))
        assert src.count(_CAP_LINE) == 1, "il_replay_bc.run_shard changed: memmap capacity line not found"
        ns = vars(IL)
        exec(compile(src.replace(_CAP_LINE, _CAP_NEW), IL.__file__ + ":run_shard_5hz", "exec"), ns)
        IL._run_shard_5hz = True
    return IL.OUT, IL.OUT_OPP


def _worker_init(skip, suffix):
    patch(skip, suffix)


def _run(job):
    return IL.run_shard(job)


def cmd_build(a):
    patch(a.skip, a.out_suffix)
    plan, _ = IL.shard_plan(a.set)
    ids = list(range(len(plan)))
    if a.shards:
        lo, hi = (int(v) for v in a.shards.split(":"))
        ids = [i for i in ids if lo <= i < hi]
    if a.limit_matches:
        plan = [p[: a.limit_matches] for p in plan]
    jobs = [(a.set, i, plan[i], not a.no_opp) for i in ids]
    print(f"skip {IL.SKIP} ticks -> {IL.OUT[a.set]}; {len(jobs)} shards of <= {IL.SHARD_MATCHES}, "
          f"{sum(len(plan[i]) for i in ids)} sides, {a.workers} workers", flush=True)
    t0 = time.time()
    if a.workers <= 1:
        for j in jobs:
            print(_run(j), f"{time.time() - t0:.0f}s", flush=True)
        return
    ctx = get_context("spawn")
    with ctx.Pool(a.workers, initializer=_worker_init, initargs=(a.skip, a.out_suffix),
                  maxtasksperchild=8) as pool:
        for res in pool.imap_unordered(_run, jobs):
            print(res, f"{time.time() - t0:.0f}s", flush=True)


def cmd_finalize(a):
    patch(a.skip, a.out_suffix)
    IL.cmd_finalize(a)
    out = IL.OUT[a.set]
    meta = json.loads((out / "meta.json").read_text())
    meta["decision_ticks"] = IL.SKIP
    meta["built_by"] = "training/il_replay_bc_5hz.py (video_dataset/il_replay_bc.py with SKIP rebound)"
    meta["time_axis_note"] = (f"as time_axis, with 10 replaced by {IL.SKIP}: a play goes to step "
                              f"floor(tick/{IL.SKIP}) and executes at tick {IL.SKIP}*step")
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "finalize"):
        p = sub.add_parser(name)
        p.add_argument("--set", default="primary", choices=["primary", "secondary"])
        p.add_argument("--skip", type=int, default=DEFAULT_SKIP)
        p.add_argument("--out-suffix", default=None)
        if name == "build":
            p.add_argument("--workers", type=int, default=8)
            p.add_argument("--shards", default="")
            p.add_argument("--limit-matches", type=int, default=0)
            p.add_argument("--no-opp", action="store_true")
    a = ap.parse_args()
    {"build": cmd_build, "finalize": cmd_finalize}[a.cmd](a)


if __name__ == "__main__":
    main()
