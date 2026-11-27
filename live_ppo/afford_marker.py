"""Ties a checkpoint to --afford-project (2026-11-26 pm, pocket/data owner). A checkpoint trained under the
projected-affordability rule (runs/imit_timing_1126/AFFORD_RULE.txt; sim hook runs/pocket_1126/fix/afford_lead.py) must
play live with --afford-project, and only such a checkpoint may. marker(path, blob) -> a reason string when the
checkpoint carries the rule, None otherwise. Read-only; looks at, in order:
  1. a sidecar <checkpoint>.afford_lead.json (for a copy kept elsewhere; any JSON with "afford_lead": 1 or a dict);
  2. pocket.json beside the checkpoint with an "afford_lead" field (imitation models, runs/bc3/<name>/);
  3. launch.txt of the PPO run folder (the checkpoint's own folder, or the parent of snapshots/) whose last
     "afford_lead" line is 1 (run_g2_pocket_fresh.sh writes it);
  4. the blob's ppo["init"] (every PPO snapshot / state names its init): pocket.json beside that init with afford_lead.
A checkpoint none of these mark returns None: the driver then behaves exactly as before this file."""
import json
import os


def _pocket_has(d):
    pj = os.path.join(d, "pocket.json")
    try:
        return bool(json.load(open(pj)).get("afford_lead")) and pj
    except (OSError, ValueError):
        return False


def _launch_has(d):
    lt = os.path.join(d, "launch.txt")
    try:
        vals = [ln.split()[1] for ln in open(lt) if ln.startswith("afford_lead ") and len(ln.split()) > 1]
    except OSError:
        return False
    return bool(vals) and vals[-1] == "1" and lt


def marker(path, blob=None):
    p = os.path.abspath(path)
    d = os.path.dirname(p)
    sc = p + ".afford_lead.json"
    if os.path.exists(sc):
        try:
            if json.load(open(sc)).get("afford_lead"):
                return f"sidecar {sc}"
        except (OSError, ValueError):
            return f"sidecar {sc} (unreadable: treated as marked)"
    r = _pocket_has(d)
    if r:
        return f"afford_lead in {r}"
    for run in (d, os.path.dirname(d) if os.path.basename(d) == "snapshots" else None):
        r = run and _launch_has(run)
        if r:
            return f"afford_lead 1 in {r}"
    init = ((blob or {}).get("ppo") or {}).get("init") if isinstance(blob, dict) else None
    if init:
        ip = init if os.path.isabs(init) else os.path.join("/Users/samueldoane/cr-robot", init)
        r = _pocket_has(os.path.dirname(ip))
        if r:
            return f"its PPO init {init} has afford_lead in {r}"
    return None


def check(path, blob, afford_project):
    """SystemExit message when the flag and the checkpoint disagree, else None."""
    m = marker(path, blob)
    if m and not afford_project:
        return (f"{path} was trained with projected affordability ({m}): run it with --afford-project "
                f"(driver flag); without it the exact-cost mask lands Cannon / Musketeer late. Refusing to start.")
    if afford_project and not m:
        return (f"--afford-project on {path}, which carries no afford_lead marker (sidecar, pocket.json, run "
                f"launch.txt or its PPO init): the projected rule is only for a checkpoint trained with it. Refusing.")
    return None
