"""Controlled grid: one factor at a time on a fixed sample of recorded matches.
usage (engine from CR_ENGINE): grid.py TAGS.json OUT.json PROCS CELL [CELL...]
Each cell is name or name:key=value,key=value over the defaults below."""
import sys, os, json, time, collections, math
ROOT = "/Users/samueldoane/cr-robot"
sys.path.insert(0, ROOT)
BASE = dict(path="new", off=0, se=6.0, levels=False, towers=False, illegal="wait", max_wait=10,
            after_end="truncate", forms="exact", seat=0, guards=True, oldmode="plain")

def parse_cell(spec):
    name, _, rest = spec.partition(":")
    cfg = dict(BASE)
    for kv in filter(None, rest.split(",")):
        k, v = kv.split("=")
        cur = BASE[k]
        cfg[k] = (v == "1" or v == "True") if isinstance(cur, bool) else (None if v == "None" else type(cur)(v) if cur is not None else float(v))
    return name, cfg

_S = {}
def store_for(cfg):
    from training import ghost as G
    key = (cfg["off"], cfg["se"], cfg["forms"])
    if key not in _S:
        tn = f"g_{cfg['off']}_{cfg['se']}"
        G.TIMINGS[tn] = (cfg["off"], cfg["se"])
        st = G.GhostStore(timing=tn)
        if cfg["forms"] == "base":            # every card in its base form (no evolution / hero)
            st.ids_exact = st.kbase[st.deck_key]
            st.ok_exact = st._valid(st.ids_exact, "exact")
        _S[key] = st
    return _S[key]

def new_row(cfg, tag, m, s2):
    from training import ghost as G
    st = store_for(cfg)
    side0 = s2 if cfg["seat"] == 0 else 1 - s2
    if not st.ok_exact[m].all():
        return None
    r = G.replay_match(st, m, side0, levels=cfg["levels"], tower_troops=cfg["towers"], illegal=cfg["illegal"],
                       max_wait=cfg["max_wait"], after_end=cfg["after_end"], log=False)
    sgn = 1 if cfg["seat"] == 0 else -1
    sim = 1 if r["reward"] > 0.5 else (-1 if r["reward"] < -0.5 else 0)
    cf, ca = (r["crowns_for"], r["crowns_against"]) if cfg["seat"] == 0 else (r["crowns_against"], r["crowns_for"])
    row = {"tag": tag, "real": int(st.result[m]) * (1 if s2 == 0 else -1), "sim": sim * sgn,
           "rcf": int(st.crowns[m, s2]), "rca": int(st.crowns[m, 1 - s2]), "cf": int(cf), "ca": int(ca),
           "end": int(r["end_tick"]), "rend": int(r["recorded_end_tick"]), "mend": int(G.MATCH_END),
           "fu": int(r["faithful_until"]), "div": r["diverged"], "endok": r["end_consistent"],
           "x26": bool(st.plays_ours[m, s2])}
    tot = collections.Counter()
    for sc in r["scripts"]:
        for k, v in sc.summary().items(): tot[k] += v
    row.update(rec=tot["recorded"], played=tot["played"], exact=tot["exact"], moved=tot["moved_tile"], topup=tot["topup"],
               late=tot["late"], dropped=tot["dropped"], notreached=tot["not_reached"],
               cool=tot["waited:cooldown"] + tot["cooldown_cleared"] + tot["dropped:cooldown"],
               nelix=tot["waited:elixir"] + tot["dropped:elixir"] + (tot["topup"] if cfg["illegal"] == "force" else 0),
               ntile=tot["moved_tile"] + tot["dropped:tile"] + tot["dropped:no_legal_tile"])
    return row

_B = {}
def old_setup():
    if "B" not in _B:
        from training import common as C
        sys.path.insert(0, ROOT + "/video_dataset")
        import il_replay_bc as B
        assert B.E is C.E, "old builder bound to another engine"
        _B["B"] = B; _B["Guard"] = B.TowerGuard; _B["E"] = C.E
    return _B["B"]

class NoGuard:
    def __init__(self, *a, **k): pass
    def tick(self, stats, now): pass

CTX = {}
def old_row(cfg, tag, m, s2):
    from training import ghost as G
    import numpy as np
    B = old_setup(); E = _B["E"]
    st = G.get_store()
    B.TowerGuard = _B["Guard"] if cfg["guards"] else NoGuard
    names = ("team", "opponent")
    rec = {"duration": float(st.duration[m]), "result": None, "decks": {}, "crowns": {}, "fth": {}, "tower": {}, "events": []}
    for i, s in enumerate(names):
        rec["decks"][s] = [(st.keys[int(k)], int(l)) for k, l in zip(st.deck_key[m, i], st.level[m, i])]
        rec["crowns"][s] = int(st.crowns[m, i])
        f = st.fth[m, i]
        rec["fth"][s] = {"king": int(f[0]), "princess_left": int(f[1]), "princess_right": int(f[2])}
        tk = int(st.tower_key[m, i])
        rec["tower"][s] = (st.keys[tk] if tk >= 0 else None, int(st.tower_level[m, i]) or None)
    a, b = int(st.play_off[m]), int(st.play_off[m + 1])
    for i in range(a, b):
        rec["events"].append(("play", names[int(st.play_side[i])], st.keys[int(st.play_key[i])], None, int(st.play_tick20[i]),
                              float(st.play_x[i]) / 1000.0, float(st.play_y[i]) / 1000.0))
    a2, b2 = int(st.abil_off[m]), int(st.abil_off[m + 1])
    for i in range(a2, b2):
        cands = tuple(st.keys[int(c)] for c in st.abil_cand[i] if c >= 0)
        rec["events"].append(("ability", names[int(st.abil_side[i])], None, cands, int(st.abil_tick20[i]), None, None))
    rec["events"].sort(key=lambda e: e[4])
    us = names[s2]
    real = int(st.result[m]) * (1 if s2 == 0 else -1)
    row_in = {"us_side": us, "replay_tag": tag, "crowns_us": int(st.crowns[m, s2]), "crowns_them": int(st.crowns[m, 1 - s2]),
              "side_idx": 0, "won": None if real == 0 else real > 0}
    if cfg["oldmode"] == "lvl":
        def lv(i):
            L = [int(l) for l in st.level[m, i] if l > 0]
            return max(set(L), key=L.count) if L else 11
        cl = lambda v: min(16, max(1, int(v)))
        order = (s2, 1 - s2)
        CTX.update(tt=[rec["tower"][names[i]][0] for i in order], cl=[cl(lv(i)) for i in order],
                   tl=[cl(rec["tower"][names[i]][1] or lv(i)) for i in order])
        Real = E.ClashRoyaleEnv
        class Proxy:
            def __getattr__(self, k): return getattr(E, k)
            def ClashRoyaleEnv(self, d0, d1, mt):
                env = Real(d0, d1, mt, G.tower_troop(CTX["tt"][0]), G.tower_troop(CTX["tt"][1]))
                for t in (0, 1): env.set_team_level(t, CTX["cl"][t], CTX["tl"][t])
                return env
        B.E = Proxy()
    else:
        B.E = E
    stc = collections.Counter()
    got = B.simulate(row_in, rec, stc, want_opp=True)
    B.E = E
    if got is None:
        return None
    rows, fid = got
    sim = 0 if fid["sim_won"] is None else (1 if fid["sim_won"] else -1)
    rcd = sum(1 for e in rec["events"] if e[0] == "play")
    played = stc["plays_team0"] + stc["plays_team1"]
    reloc = sum(int((np.asarray(rows[t]["relocated"]) > 0).sum()) for t in (0, 1))
    topup = sum(int((np.asarray(rows[t]["topup"]) > 0).sum()) for t in (0, 1))
    cool = stc["cooldown_clear_team0"] + stc["cooldown_clear_team1"]
    bad = 0
    for t in (0, 1):
        r_ = rows[t]
        cid = np.asarray(r_["card_id"]); bad += int(((cid >= 0) & ((np.asarray(r_["relocated"]) > 0) | (np.asarray(r_["topup"]) > 0) | (np.asarray(r_["shift"]) > 0))).sum())
    return {"tag": tag, "real": real, "sim": sim, "rcf": int(st.crowns[m, s2]), "rca": int(st.crowns[m, 1 - s2]),
            "cf": int(fid["sim_crowns_us"]), "ca": int(fid["sim_crowns_them"]),
            "end": int(fid["steps"]) * 10, "rend": int(round((rec["duration"] - 4.55) * 10)), "mend": 3200,
            "fu": -1, "div": None, "endok": None, "x26": bool(st.plays_ours[m, s2]),
            "rec": rcd, "played": played, "exact": played - bad - 0, "moved": reloc, "topup": topup,
            "late": stc["plays_deferred_team0"] + stc["plays_deferred_team1"], "dropped": stc["engine_refused_team0"] + stc["engine_refused_team1"],
            "notreached": rcd - played, "cool": cool, "nelix": topup, "ntile": reloc,
            "heals": stc["tower_heal_team0"] + stc["tower_heal_team1"], "forced": stc["tower_forced_destroy_team0"] + stc["tower_forced_destroy_team1"],
            "breach": stc["tower_guard_breach_team0"] + stc["tower_guard_breach_team1"]}

def work(args):
    cfg, items = args
    from training import common as C  # noqa
    out = []
    for tag, m, s2 in items:
        try:
            out.append((old_row if cfg["path"] == "old" else new_row)(cfg, tag, m, s2))
        except Exception as ex:  # noqa
            out.append({"tag": tag, "err": type(ex).__name__ + ": " + str(ex)[:200]})
    return out

def main():
    tags_f, out_f, procs = sys.argv[1], sys.argv[2], int(sys.argv[3])
    cells = [parse_cell(c) for c in sys.argv[4:]]
    from training import common as C
    from training import ghost as G
    import multiprocessing as mp
    st = G.get_store()
    want = json.load(open(tags_f))
    idx = {str(t): i for i, t in enumerate(st.tag)}
    ours = st.plays_variant
    items = []
    for t in want:
        m = idx[t]; items.append((t, m, 0 if ours[m, 0] else 1))
    res = json.load(open(out_f)) if os.path.exists(out_f) else {}
    with mp.get_context("spawn").Pool(procs) as pool:
        for name, cfg in cells:
            t0 = time.time()
            parts = pool.map(work, [(cfg, items[k::procs]) for k in range(procs)])
            rows = [None] * len(items)
            for k, part in enumerate(parts): rows[k::procs] = part
            res[name] = {"cfg": cfg, "engine": C.ENGINE, "rows": rows, "wall": time.time() - t0}
            err = [r for r in rows if r and "err" in r]
            print(name, "done", round(time.time() - t0, 1), "s; none", sum(r is None for r in rows), "errors", len(err), err[:1], flush=True)
            json.dump(res, open(out_f, "w"))

if __name__ == "__main__":
    main()
