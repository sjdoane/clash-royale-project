"""Which engine and CR_P0088 switches a checkpoint was trained with (2026-11-27, gen2_l live prep).

A PPO run folder's launch.txt is a list of blocks, one per (re)launch, each starting with a
`date YYYY-MM-DD HH:MM:SS` line (local time) and holding `engine PATH` and `sw LIST` (also
`CR_P0088=LIST`). A snapshot belongs to the block that was running when it was written:

  * its episode (`ep0152502` in the file name) is after that block's resume point, the
    episode count the block started from. The resume point is read from progress.jsonl: the
    first record written after the block's date, `episodes - episodes_this_update`.
  * cross-check: the file's mtime is not before that block's date (a block cannot have
    written a file older than itself).

The profile is the block's LAST engine / sw lines. When the block cannot be told (no episode
in the name, no progress.jsonl, the two tests disagree) and the candidate blocks differ,
pick() refuses unless an override names the build: CR_LIVE_TRAINED_ON=gen2_l (or
build/gen2_l/engine, or the driver's --trained-on), which takes the last block on that engine.
An override that disagrees with a block that CAN be told is refused too.

A checkpoint whose folder has no launch.txt (runs/g2_ppo_s2r, the main run; imitation models)
has no profile: pick() returns None and the caller keeps its behaviour from before this file.

Used by live_ppo/live_session.sh (`--shell`, which also checks CR_LIVE_ENGINE / CR_P0088
against the profile) and by live_ppo/driver.py (which checks the engine it loaded).

    python live_ppo/engine_pick.py CKPT            # print the profile
    python live_ppo/engine_pick.py --shell CKPT    # shell assignments; exit 2 on a refusal
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MTIME_SLACK_S = 120.0           # a file may be stamped this much before its block's date line


class PickError(Exception):
    pass


@dataclass
class Block:
    date: str
    t: float                                  # unix time of the date line (local time)
    engine: str | None = None
    sw: str | None = None
    p0088: str | None = None
    resume: float | None = None               # episodes the block started from (None: unknown)

    def switches(self) -> tuple:
        raw = self.sw if self.sw is not None else self.p0088
        return split_sw(raw or "")


@dataclass
class Profile:
    engine: Path                              # resolved engine root
    sw: tuple                                 # switch names, in the launch.txt order
    launch: Path
    block_date: str
    how: str                                  # how the block was chosen
    notes: list = field(default_factory=list)

    @property
    def p0088(self) -> str:
        return ",".join(self.sw)

    def has(self, name: str) -> bool:
        return name in self.sw

    @property
    def build(self) -> str:
        return build_name(self.engine)

    def describe(self) -> str:
        return (f"{self.build} ({self.engine}), CR_P0088={self.p0088 or '(empty)'}; "
                f"launch.txt block {self.block_date}, chosen by {self.how}")


def split_sw(raw: str) -> tuple:
    return tuple(s.strip() for s in raw.split(",") if s.strip())


def same_sw(a, b) -> bool:
    a = split_sw(a) if isinstance(a, str) else tuple(a)
    b = split_sw(b) if isinstance(b, str) else tuple(b)
    return set(a) == set(b)


def build_name(engine) -> str:
    p = Path(engine)
    return p.parent.name if p.name == "engine" else p.name


def engine_path(s: str) -> Path:
    """'gen2_l' | 'build/gen2_l' | 'build/gen2_l/engine' | an absolute path -> resolved engine root."""
    p = Path(s)
    if not p.is_absolute():
        p = REPO / ("build/" + s if "/" not in s else s)
    if p.name != "engine" and (p / "engine").is_dir():
        p = p / "engine"
    return p.resolve()


def run_dir(ckpt: Path) -> Path | None:
    """The PPO run folder of a checkpoint: its own folder, or the parent of snapshots/."""
    d = ckpt.parent
    for run in (d, d.parent if d.name == "snapshots" else None):
        if run is not None and (run / "launch.txt").is_file():
            return run
    return None


def parse_launch(path: Path) -> list:
    blocks = []
    for ln in path.read_text().splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        m = re.match(r"date (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)$", s)
        if m:
            blocks.append(Block(m.group(1), time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))))
            continue
        if not blocks:
            continue                          # lines before the first date line belong to no block
        b = blocks[-1]
        if s.startswith("engine "):
            b.engine = s.split(None, 1)[1].strip()
        elif s.startswith("sw "):
            b.sw = s.split(None, 1)[1].strip()
        elif s == "sw":
            b.sw = ""
        elif s.startswith("CR_P0088="):
            b.p0088 = s.split("=", 1)[1].strip()
    return blocks


def resume_points(blocks: list, progress: Path) -> None:
    """Block.resume from progress.jsonl (first record after the block's date)."""
    if not progress.is_file():
        return
    recs = []
    with open(progress) as fh:
        for ln in fh:
            try:
                r = json.loads(ln)
                recs.append((float(r["t"]), float(r["episodes"]), float(r.get("episodes_this_update", 0))))
            except (ValueError, KeyError, TypeError):
                continue
    recs.sort()
    for b in blocks:
        nxt = next((r for r in recs if r[0] >= b.t), None)
        b.resume = None if nxt is None else nxt[1] - nxt[2]


def _episode(ckpt: Path):
    m = re.search(r"ep0*(\d+)", ckpt.name)
    return int(m.group(1)) if m else None


def _profile_of(b: Block, launch: Path, how: str) -> Profile:
    if not b.engine:
        raise PickError(f"{launch}: the block of {b.date} has no engine line")
    if b.sw is not None and b.p0088 is not None and not same_sw(b.sw, b.p0088):
        raise PickError(f"{launch}: the block of {b.date} has sw {b.sw!r} but CR_P0088={b.p0088!r}")
    return Profile(engine=engine_path(b.engine), sw=b.switches(), launch=launch, block_date=b.date, how=how)


def _key(b: Block):
    return (str(engine_path(b.engine)) if b.engine else None, frozenset(b.switches()))


def pick(ckpt, override: str | None = None) -> Profile | None:
    """The checkpoint's training profile; None without a launch.txt. Raises PickError."""
    ckpt = Path(os.path.realpath(ckpt))
    run = run_dir(ckpt)
    if run is None:
        if override:
            raise PickError(f"trained-on override {override!r} for {ckpt}, whose folder has no launch.txt: "
                            "the override only chooses among a run's launch.txt blocks")
        return None
    launch = run / "launch.txt"
    blocks = [b for b in parse_launch(launch) if b.engine]
    if not blocks:
        raise PickError(f"{launch} has no block with an engine line")
    resume_points(blocks, run / "progress.jsonl")

    ep = _episode(ckpt)
    by_ep = None
    if ep is not None and all(b.resume is not None for b in blocks[:1]):
        for i, b in enumerate(blocks):
            if b.resume is not None and ep > b.resume:
                by_ep = i
    mtime = ckpt.stat().st_mtime if ckpt.exists() else None
    by_time = None
    if mtime is not None:
        for i, b in enumerate(blocks):
            if mtime + MTIME_SLACK_S >= b.t:
                by_time = i

    notes = []
    told = None
    lo = 0
    if by_ep is not None:
        if by_time is not None and by_time < by_ep:
            notes.append(f"ep{ep} is after the resume point ({blocks[by_ep].resume:.0f}) of the block of "
                         f"{blocks[by_ep].date}, but the file is older than that block")
            lo = by_time                          # candidates: from the mtime's block to the episode's
            by_time = by_ep
        else:
            told = by_ep
            how = f"episode {ep} > resume point {blocks[by_ep].resume:.0f}"
    else:
        notes.append("no episode in the file name" if ep is None else
                     "no progress.jsonl resume points")

    if override:
        want = engine_path(override)
        cand = [i for i, b in enumerate(blocks) if engine_path(b.engine) == want]
        if not cand:
            raise PickError(f"trained-on override {override!r}: no block of {launch} runs {want}")
        i = cand[-1]
        if told is not None and _key(blocks[told]) != _key(blocks[i]):
            raise PickError(f"trained-on override {override!r} disagrees with {launch}: {ckpt.name} was "
                            f"written by the block of {blocks[told].date} ({how}), on "
                            f"{build_name(engine_path(blocks[told].engine))} with sw {blocks[told].sw}")
        p = _profile_of(blocks[i], launch, f"override {override!r} (last block on that engine)")
        p.notes = notes
        return p
    if told is not None:
        p = _profile_of(blocks[told], launch, how)
        p.notes = notes
        return p
    # cannot tell: fine only when every candidate block has the same profile
    hi = len(blocks) - 1 if by_time is None else by_time
    keys = {_key(b) for b in blocks[lo:hi + 1]}
    if len(keys) == 1:
        p = _profile_of(blocks[hi], launch, "every candidate block has the same engine and switches")
        p.notes = notes
        return p
    builds = sorted({build_name(engine_path(b.engine)) for b in blocks[lo:hi + 1]})
    raise PickError(f"cannot tell which launch.txt block wrote {ckpt} ({'; '.join(notes)}); its candidate "
                    f"blocks differ ({', '.join(builds)}): name the build with CR_LIVE_TRAINED_ON=<build> "
                    f"(driver: --trained-on <build>)")


def env_check(prof: Profile | None, live_engine: str | None, p0088: str | None, root: Path = REPO):
    """live_session.sh: a refusal message when CR_LIVE_ENGINE / CR_P0088 (as set by the caller)
    disagree with the profile, else None. Unset (None) means 'take the profile's'."""
    if prof is None:
        return None
    if live_engine:
        le = Path(os.path.realpath(root / live_engine))
        if le != prof.engine:
            return (f"CR_LIVE_ENGINE={live_engine} ({le}) but the checkpoint was trained on {prof.engine} "
                    f"({prof.how}); unset it or pass the same engine")
    if p0088 is not None and not same_sw(p0088, prof.sw):
        return (f"CR_P0088={p0088!r} but the checkpoint was trained with {prof.p0088!r} ({prof.how}); "
                f"unset it or pass the same switches")
    return None


def _sh(v) -> str:
    import shlex
    return shlex.quote(str(v))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--shell", action="store_true",
                    help="print PICK_* shell assignments (checks CR_LIVE_ENGINE / CR_P0088 from the "
                         "environment); exit 2 with the reason on stderr on a refusal")
    ap.add_argument("--trained-on", default=os.environ.get("CR_LIVE_TRAINED_ON") or None)
    a = ap.parse_args(argv)
    try:
        prof = pick(a.checkpoint, a.trained_on)
    except PickError as ex:
        print(f"engine_pick: {ex}", file=sys.stderr)
        return 2
    if not a.shell:
        print("no launch.txt: no profile (the caller's defaults apply)" if prof is None else prof.describe())
        for n in (prof.notes if prof else ()):
            print(f"  note: {n}")
        return 0
    if prof is None:
        print("PICK_KIND=none")
        print(f"PICK_DESC={_sh('no launch.txt beside ' + os.path.realpath(a.checkpoint) + ': engine and CR_P0088 as before')}")
        return 0
    bad = env_check(prof, os.environ.get("CR_LIVE_ENGINE") or None, os.environ.get("CR_P0088"))
    if bad:
        print(f"engine_pick: {bad}", file=sys.stderr)
        return 2
    print("PICK_KIND=run")
    print(f"PICK_ENGINE={_sh(prof.engine)}")
    print(f"PICK_P0088={_sh(prof.p0088)}")
    print(f"PICK_CLOAKED_ALWAYS={1 if prof.has('cloakboard') else 0}")
    print(f"PICK_DESC={_sh(prof.describe())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
