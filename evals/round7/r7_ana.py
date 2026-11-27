"""Round 7 diagnosis. usage: r7_ana.py diag RUN.json [RUN.json ...] | people"""
import collections, contextlib, io, json, sys
import numpy as np
S = "/private/tmp/claude-501/-Users-samueldoane-cr-robot/40e94860-6611-4922-a6d8-a1b149e53030/scratchpad"
sys.path.insert(0, S); sys.path.insert(0, "/Users/samueldoane/cr-robot"); sys.path.insert(0, "/Users/samueldoane/cr-robot/evals/tools")
from training import ghost as G

DROP = {1, 6, 12, 14, 16, 23, 25, 26, 27, 28, 32, 35}


def cost(cid):
    try:
        return float(G.card_info(int(cid))["cost"])
    except Exception:
        return None


def regen(t):          # elixir a second at time t (s)
    return (1 + (t >= 120) + (t >= 240)) / 2.8


def bank_at(plays, t, extra_rate=0.0, start=5.0):
    """Elixir at t from income (cap 10) minus the listed (time s, cost) plays and an even extra spend."""
    b, k, ps = start, 0, sorted(plays)
    dt = 0.1
    for i in range(int(round(t / dt))):
        tt = i * dt
        b = min(10.0, b + regen(tt) * dt - extra_rate * dt)
        while k < len(ps) and ps[k][0] <= tt + dt:
            b -= ps[k][1]; k += 1
        b = max(b, 0.0)
    return b


def shape(times_costs, end):
    """Timing shape of one match: gaps, bursts (chain of plays <= 3 s apart), idle share (gaps >= 6 s),
    plays a minute by phase."""
    tc = sorted(times_costs)
    ts = [t for t, _ in tc]
    gaps = np.diff([0.0] + ts + [end])
    idle = float(sum(g for g in gaps if g >= 6.0)) / end if end > 0 else 0.0
    bursts, cur = [], []
    for t, c in tc:
        if cur and t - cur[-1][0] > 3.0:
            bursts.append(sum(x[1] for x in cur)); cur = []
        cur.append((t, c))
    if cur:
        bursts.append(sum(x[1] for x in cur))
    ph = []
    for lo, hi in ((0, 120), (120, 240), (240, 1e9)):
        span = max(0.0, min(end, hi) - lo)
        ph.append((sum(1 for t in ts if lo <= t < hi), span))
    return dict(gaps=list(np.diff(ts)), idle=idle, bursts=bursts, ph=ph, n=len(ts), end=end)


def agg(shapes, label):
    g = np.concatenate([s["gaps"] for s in shapes])
    b = np.concatenate([s["bursts"] for s in shapes])
    ph = np.array([[s["ph"][i][0] for i in range(3)] for s in shapes], float).sum(0)
    sp = np.array([[s["ph"][i][1] for i in range(3)] for s in shapes], float).sum(0) / 60.0
    print(f"{label}: gaps median {np.median(g):.1f} s, quartiles {np.percentile(g, 25):.1f}/{np.percentile(g, 75):.1f}, "
          f"<=1 s {np.mean(g <= 1):.0%}, >=6 s {np.mean(g >= 6):.0%}; burst elixir mean {b.mean():.1f} median {np.median(b):.1f} "
          f"(plays a burst {sum(s['n'] for s in shapes) / len(b):.2f}); idle share (gaps >= 6 s) {np.mean([s['idle'] for s in shapes]):.0%}; "
          f"plays/min single/double/triple {ph[0] / sp[0]:.1f}/{ph[1] / max(sp[1], 1e-9):.1f}/{ph[2] / max(sp[2], 1e-9):.1f} "
          f"(minutes {sp.round(1).tolist()})")


def dedup(e):
    best = {}
    for o in e["opp_plays"]:
        nm = G.card_name(o[1]).replace("evo:", "").replace("Hero ", "")
        k = (o[0], nm)
        if k not in best or (o[2] >= 0 and best[k][2] < 0):
            best[k] = o
    return sorted(best.values())


CHEAP_MAX = 2.0
ROLES = collections.Counter()
YS = collections.defaultdict(list)


def classify(a):
    """a: TLOG act row with context (t, team, hand, elixir, slot, x, y, kind, threat, role, ours)."""
    _, t, team, hand, el, slot, x, y, kind, threat, role, ours = a
    if role == "wincon" or y >= 14.0:
        return "attack"
    if threat > 400.0:
        return "threat"
    cs = [cost(u[0]) for u in ours]
    if ours and all(c is not None and c <= CHEAP_MAX for c in cs):
        return "cheap"
    return "other"


def diag(path):
    res = json.load(open(path))
    ROLES.clear(); YS.clear()
    cls_el = collections.Counter(); cls_n = collections.Counter(); hogs = 0; banks = []; same = []
    for e in res:
        acts = [a for a in e["tlog"] if a[0] == "a" and len(a) > 9]
        ops = [(o[0] / 10.0, float(o[6])) for o in dedup(e)]
        for b in e["bodies"]:
            if b["card"] != 15 or "x0" not in b or e["end"] - b["t0"] < 100:
                continue
            hogs += 1
            t0 = b["t0"]
            for o in dedup(e):                         # landed plays, each with its latest decision
                if not (t0 - 80 <= o[0] < t0):
                    continue
                nm_ = G.card_name(o[1]).replace("evo:", "").replace("Hero ", "")
                dec = [a for a in acts if a[1] <= o[0] and o[0] - a[1] <= 30
                       and G.card_name(a[3][a[5]]).replace("evo:", "").replace("Hero ", "") == nm_]
                k = classify(dec[-1]) if dec else "build answer / unmatched"
                c = float(o[6])
                cls_el[k] += c; cls_n[k] += 1
                if dec:
                    ROLES[(k, dec[-1][10])] += c
                    YS[k].append(dec[-1][7])
            if b.get("opp_elx0") is not None:
                banks.append(float(b["opp_elx0"]))
                same.append(bank_at(ops, t0 / 10.0, start=float(e["hand"][0][2])))
    tot = sum(cls_el.values())
    print(f"{path.split('/')[-1]}: {hogs} Hog landings; teacher elixir in the 8 s before, per landing {tot / hogs:.2f}: "
          + ", ".join(f"{k} {cls_el[k] / hogs:.2f} ({cls_el[k] / tot:.0%}, {cls_n[k]} plays)" for k in ("cheap", "threat", "attack", "other", "build answer / unmatched")))
    print("   roles", {k: round(v / hogs, 2) for k, v in sorted(ROLES.items(), key=lambda kv: -kv[1])[:10]})
    print("   own-frame y quartiles", {k: np.percentile(v, [25, 50, 75]).round(1).tolist() for k, v in YS.items() if v})
    print(f"   bank at landing: true median {np.median(banks):.1f} mean {np.mean(banks):.2f}; same method (income - landed plays) "
          f"median {np.median(same):.1f} mean {np.mean(same):.2f} (bias {np.mean(same) - np.mean(banks):+.2f})")
    agg([shape([(o[0], o[1]) for o in [(oo[0] / 10.0, float(oo[6])) for oo in dedup(e)]], e["end"] / 10.0) for e in res],
        "   timing")
    mins = sum(e["end"] for e in res) / 600.0
    print(f"   spend/min {sum(e['spent1'] for e in res) / mins:.1f} leak/min {sum(e['leak1'] for e in res) / mins:.2f}")


def people():
    import live_case as LC
    shapes, ub, est = [], [], []
    for m in range(36):
        if m in DROP:
            continue
        C = LC.Case("data/ghost/live3", m)
        J = C.J
        end = float(J["end_s"])
        tc = [(float(p["t"]), cost(p["id"]) or 0.0) for p in J["plays"]]
        shapes.append(shape(tc, end))
        rate = float(J["elixir"]["unexplained"]) / end
        for p in J["own"]:
            if p["card"] == "Hog Rider" and end - p["t"] >= 10:
                t = float(p["t"]) + 1.0
                ub.append(bank_at(tc, t)); est.append(bank_at(tc, t, rate))
    print(f"people: {len(shapes)} matches, {len(ub)} Hog landings; bank at landing upper bound median {np.median(ub):.1f} "
          f"mean {np.mean(ub):.2f}; with the unexplained spread evenly median {np.median(est):.1f} mean {np.mean(est):.2f}")
    agg(shapes, "   timing")


if __name__ == "__main__":
    if sys.argv[1] == "people":
        with contextlib.redirect_stderr(io.StringIO()):
            people()
    else:
        for p in sys.argv[2:]:
            diag(p)
