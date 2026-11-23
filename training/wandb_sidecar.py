"""Stream training metrics from runs/<run>/ to Weights & Biases.

Runs beside the trainers so nothing has to be restarted. For every directory
under --runs it tails:
  progress.jsonl    one line per PPO update        -> train/*, opp/*
  eval.jsonl        eval-panel results             -> eval/*
  bc_metrics.jsonl  BC validation per epoch        -> bc/*
  league.jsonl      self-play opponent mix         -> league/*  (x = update)
  h2h.jsonl         head-to-head vs frozen snaps   -> h2h/vs_ep<N>/*
  events.out.tfevents.*  every TensorBoard scalar  -> tb/<tag>

One W&B run per training run (id = run name, resume="allow"), so re-running
the sidecar continues the same charts. Read offsets are kept in --state.
A separate "balerion-system" run logs GPU and load.

The API key is read from ~/.config/wandb/key at start-up; never pass it on the
command line or write it to the repo.
"""
import argparse
import glob
import json
import math
import os
import re
import subprocess
import time

import wandb

IDLE_FINISH_S = 1800


def clean(name):
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", str(name))


def flatten(d, prefix, out):
    for k, v in d.items():
        key = f"{prefix}/{clean(k)}"
        if isinstance(v, dict):
            flatten(v, key, out)
        elif isinstance(v, bool):
            out[key] = float(v)
        elif isinstance(v, (int, float)):
            if math.isfinite(v):
                out[key] = v
    return out


def progress_row(d):
    row = {}
    for k, v in d.items():
        if k == "opp_wr" and isinstance(v, dict):
            flatten(v, "opp/win", row)
        elif k == "opp_n" and isinstance(v, dict):
            flatten(v, "opp/n", row)
        elif k in ("episodes", "t"):
            continue
        elif isinstance(v, dict):
            flatten(v, f"train/{clean(k)}", row)
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            row[f"train/{clean(k)}"] = v
    return row


def eval_row(d):
    row = {}
    for k, v in d.items():
        if k == "per_opponent" and isinstance(v, dict):
            for opp, s in v.items():
                flatten(s, f"eval_opp/{clean(opp)}", row)
                if "win" in s and "draw" in s:
                    row[f"eval_opp/{clean(opp)}/loss"] = 1.0 - s["win"] - s["draw"]
        elif k in ("episodes", "t"):
            continue
        elif isinstance(v, dict):
            flatten(v, f"eval/{clean(k)}", row)
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            row[f"eval/{clean(k)}"] = v
    return row


def bc_row(d):
    row = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flatten(v, f"bc/{clean(k)}", row)
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            row[f"bc/{clean(k)}"] = v
    return row


KINDS = ("teacher", "mirror", "snapshot")


def league_row(d):
    """league.jsonl: per-update opponent mix; list fields are [teacher, mirror, snapshot]."""
    row = {}
    for k, v in d.items():
        if k in ("t", "update", "snap_eps"):
            continue
        if isinstance(v, list) and len(v) == len(KINDS):
            for kind, x in zip(KINDS, v):
                row[f"league/{k}/{kind}"] = x
        elif isinstance(v, dict):
            flatten(v, f"league/{clean(k)}", row)
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            row[f"league/{clean(k)}"] = v
    eps, wins = d.get("episodes"), d.get("wins")
    if isinstance(eps, list) and isinstance(wins, list):
        for kind, e, w in zip(KINDS, eps, wins):
            if e:
                row[f"league/win_rate/{kind}"] = w / e
    if isinstance(d.get("snap_eps"), dict):
        row["league/snap_distinct"] = len(d["snap_eps"])
    return row


def h2h_row(d, prefix):
    m = re.search(r"ep0*(\d+)", str(d.get("ref", "")))
    ref = f"vs_ep{m.group(1)}" if m else clean(d.get("ref", "ref"))
    row = {}
    for k, v in d.items():
        if k in ("t", "episodes"):
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            row[f"{prefix}/{ref}/{clean(k)}"] = v
    return row


def read_new_lines(path, offset):
    """Return (complete JSON lines after offset, new offset)."""
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read()
    end = data.rfind(b"\n")
    if end < 0:
        return [], offset
    rows = []
    for line in data[: end + 1].splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line.decode().replace("NaN", "null")))
        except json.JSONDecodeError:
            pass
    return rows, offset + end + 1


class Tracker:
    def __init__(self, name, runs_dir, state, args):
        self.name = name
        self.dir = os.path.join(runs_dir, name)
        self.state = state.setdefault(name, {"off": {}, "tb_step": {}, "bc_epoch": 0})
        self.args = args
        self.run = None
        self.last_data = 0.0
        self.tb = None

    def ensure_run(self):
        if self.run is not None:
            return
        kind = "bc" if os.path.exists(os.path.join(self.dir, "bc_metrics.jsonl")) else "ppo"
        self.run = wandb.init(
            entity=self.args.entity, project=self.args.project,
            id=clean(self.name), name=self.name, resume="allow",
            job_type=kind, group=kind, config={"run_dir": self.dir},
            reinit="create_new", settings=wandb.Settings(x_disable_stats=True, console="off"),
        )
        r = self.run
        r.define_metric("episodes")
        r.define_metric("tb_step")
        r.define_metric("bc_epoch")
        r.define_metric("update")
        r.define_metric("league/*", step_metric="update")
        for p in ("train/*", "opp/*", "eval/*", "eval_opp/*", "h2h/*", "h2h_ref/*"):
            r.define_metric(p, step_metric="episodes")
        r.define_metric("tb/*", step_metric="tb_step")
        r.define_metric("bc/*", step_metric="bc_epoch")
        for m in ("eval/win_rate", "eval/win_rate_real_decks", "eval/win_rate_rungs"):
            r.define_metric(m, step_metric="episodes", summary="max")
        r.define_metric("bc/card_match_given_play", step_metric="bc_epoch", summary="max")

    def jsonl(self, fname, to_row, step_key):
        path = os.path.join(self.dir, fname)
        if not os.path.exists(path):
            return 0
        off = self.state["off"].get(fname, 0)
        if os.path.getsize(path) < off:  # file rewritten
            off = 0
        rows, off = read_new_lines(path, off)
        n = 0
        for d in rows:
            row = to_row(d)
            if step_key == "update":
                if d.get("update") is None:
                    continue
                row["update"] = d["update"]
            elif step_key == "bc_epoch":
                self.state["bc_epoch"] += 1
                row["bc_epoch"] = d.get("epoch", self.state["bc_epoch"])
            else:
                if d.get("episodes") is None:
                    continue
                row["episodes"] = d["episodes"]
            if len(row) > 1:
                self.ensure_run()
                self.run.log(row)
                n += 1
        self.state["off"][fname] = off
        return n

    def tensorboard(self):
        if not glob.glob(os.path.join(self.dir, "events.out.tfevents.*")):
            return 0
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        if self.tb is None:
            self.tb = EventAccumulator(self.dir, size_guidance={"scalars": 0})
        self.tb.Reload()
        done = self.state["tb_step"]
        by_step = {}
        for tag in self.tb.Tags().get("scalars", []):
            last = done.get(tag, -1)
            for ev in self.tb.Scalars(tag):
                if ev.step > last and math.isfinite(ev.value):
                    by_step.setdefault(ev.step, {})[f"tb/{tag}"] = ev.value
        n = 0
        for step in sorted(by_step):
            row = by_step[step]
            row["tb_step"] = step
            self.ensure_run()
            self.run.log(row)
            for k in row:
                if k.startswith("tb/"):
                    done[k[3:]] = max(done.get(k[3:], -1), step)
            n += 1
        return n

    def poll(self):
        n = self.jsonl("progress.jsonl", progress_row, "episodes")
        n += self.jsonl("eval.jsonl", eval_row, "episodes")
        n += self.jsonl("bc_metrics.jsonl", bc_row, "bc_epoch")
        n += self.jsonl("league.jsonl", league_row, "update")
        n += self.jsonl("h2h.jsonl", lambda d: h2h_row(d, "h2h"), "episodes")
        # the same h2h panel run on a copy of the baseline run, for comparison
        n += self.jsonl("ref_ppo_bc26_h2h.jsonl", lambda d: h2h_row(d, "h2h_ref"), "episodes")
        n += self.tensorboard()
        now = time.time()
        if n:
            self.last_data = now
        elif self.run is not None and now - self.last_data > IDLE_FINISH_S:
            self.run.finish()
            self.run = None
        return n


def system_row():
    row = {"sys/load1": os.getloadavg()[0]}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        u, mu, mt, temp, pw = [float(x) for x in out.strip().split("\n")[0].split(",")]
        row.update({"sys/gpu_util": u, "sys/gpu_mem_gb": mu / 1024, "sys/gpu_mem_frac": mu / mt,
                    "sys/gpu_temp_c": temp, "sys/gpu_power_w": pw})
    except Exception:
        pass
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.expanduser("~/cr-robot-remote/train/runs"))
    ap.add_argument("--state", default=os.path.expanduser("~/cr-robot-remote/wandb_sidecar_state.json"))
    ap.add_argument("--entity", default="clash-royale")
    ap.add_argument("--project", default="clash-royale-ai")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--skip", default="ppo_tput", help="comma-separated run names to ignore")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    key_file = os.path.expanduser("~/.config/wandb/key")
    if "WANDB_API_KEY" not in os.environ and os.path.exists(key_file):
        os.environ["WANDB_API_KEY"] = open(key_file).read().strip()

    state = json.load(open(args.state)) if os.path.exists(args.state) else {}
    skip = set(filter(None, args.skip.split(",")))
    trackers = {}
    sysrun = wandb.init(entity=args.entity, project=args.project, id="balerion-system",
                        name="balerion-system", resume="allow", job_type="system",
                        reinit="create_new", settings=wandb.Settings(x_disable_stats=True, console="off"))
    sysrun.define_metric("sys/*", step_metric="sys/time_h")
    t0 = state.setdefault("_t0", time.time())
    while True:
        for d in sorted(os.listdir(args.runs)):
            if d in skip or d.startswith("_") or not os.path.isdir(os.path.join(args.runs, d)):
                continue
            if d not in trackers:
                trackers[d] = Tracker(d, args.runs, state, args)
            try:
                n = trackers[d].poll()
                if n:
                    print(f"{time.strftime('%H:%M:%S')} {d}: +{n} rows", flush=True)
            except Exception as e:  # keep the sidecar alive across one bad file
                print(f"{time.strftime('%H:%M:%S')} {d}: error {e!r}", flush=True)
        row = system_row()
        row["sys/time_h"] = (time.time() - t0) / 3600
        sysrun.log(row)
        tmp = args.state + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, args.state)
        if args.once:
            break
        time.sleep(args.interval)
    for t in trackers.values():
        if t.run is not None:
            t.run.finish()
    sysrun.finish()


if __name__ == "__main__":
    main()
