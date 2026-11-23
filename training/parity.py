"""Replay parity: do real matches come out the same when both humans' recorded
plays are re-simulated in an engine?

    .venv/bin/python -m training.parity --engine build/gen2/engine --n 1000
    .venv/bin/python -m training.parity --engine build/gen2/engine --engine engine --n 1000 \\
        --out evals/parity_20261121.json
    ... --n 0            # every eligible match
    ... --convention bc  # another replay convention (ghost.CONVENTIONS); the
                         # default is ghost.CONVENTION
    ... --timing bc / --levels / --tower-troops / --illegal skip / --max-wait 0
                         # one field of the convention replaced
    ... --faced any --n 20000 --card-bias     # all decks + per-card bias regression

    from training import parity
    res = parity.parity("build/gen2/engine", n=1000)       # -> dict (summary, rows)

For each sampled Hog 2.6 match (one of the two real players used our eight
cards; `--faced exact` = in our exact forms; `--faced any` = any match, team 0
picked by the tag) both recorded players are replayed open-loop with
training/ghost.py (replay_match under ghost.CONVENTION: no tower guards,
nothing reads the real result), the 2.6 player on team 0, and the engine's
result is compared with the dataset's:

  winner            the engine's result class (win / draw / loss for team 0)
                    equals the real one. At the end of the recording the engine
                    result is its own timeout rule (ghost.timeout_reward).
  crowns            each side's crowns equal the real count; mean absolute error.
  early end         the simulated match is over more than 2 s before the real
                    one (a King Tower fell, or a first tower in overtime, that
                    did not fall in the real match at that time). Real matches
                    that run the whole overtime are recorded as ending a few
                    seconds after tick 3000 (the tiebreaker); the engine stops
                    at its own match end, so "early" is measured against the
                    earlier of the recorded end and the engine's match end.
  plays             of all recorded plays: made on the recorded tick and tile
                    (exact), made late, made on another tile (the recorded one
                    was illegal in the simulated state), made after an elixir
                    top-up, dropped (refused three times), not reached (the
                    engine's match was over before the play's tick).
  faithful until    ghost.Fidelity: the last tick at which the simulation was
                    not yet PROVABLY off the real match (a simulated tower below
                    the real final health, a tower standing where a recorded
                    pocket play proves it was down, the engine's match over
                    early). The dataset has no tower health over time, so this
                    is an upper bound on how long the replay is right.
  winner agreement  is over ALL sampled matches (the simulation's draws count
                    as disagreements unless the real match was a draw);
                    `winner_agreement_sim_decided` is the old builder's
                    denominator (matches the simulation decided).

A match can only be replayed by an engine that has every card of both decks in
the recorded form. With several --engine arguments the first engine is also run
on its own sample, then every engine on one common sample (matches all of them
can load), which is the like-for-like comparison.

Each engine runs in its own Python process (CR_ENGINE is read at import by
training/common.py), with PYTHONDONTWRITEBYTECODE=1 so nothing is written into
an engine tree. This module itself imports no engine until a child process
runs, so it can be imported whatever engine the caller uses.
"""
import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SEED = 20261121
DEFAULT_PROCS = 5
#: A simulated match counts as over early when it ends more than this many ticks
#: before the recording does.
EARLY = 20
PLAY_KEYS = ("recorded", "played", "exact", "late", "moved_tile", "topup", "dropped", "not_reached",
             "waited:elixir", "waited:cooldown", "waited:refused", "refused", "abilities_used",
             "abilities_dropped")


# ------------------------------------------------------------ child process ---
def _eligible(store, faced, min_plays=4):
    """[(tag, match, side0)]: matches this engine can replay; side0 = the 2.6 player."""
    import numpy as np
    import zlib
    if faced == "any":                   # every loadable match; team 0 = a fixed coin flip of the tag
        ok = store.ok_exact.all(1) & (store.n_plays >= min_plays)
        return sorted((str(store.tag[m]), int(m), zlib.crc32(str(store.tag[m]).encode()) & 1)
                      for m in np.flatnonzero(ok))
    ours = store.plays_ours if faced == "exact" else store.plays_variant
    ok = store.ok_exact.all(1) & ours.any(1) & (store.n_plays >= min_plays)
    out = []
    for m in np.flatnonzero(ok):
        out.append((str(store.tag[m]), int(m), 0 if ours[m, 0] else 1))
    out.sort()
    return out


def _class(real_end):
    if real_end < 1800:
        return "king_down_in_regulation"
    if real_end == 1800:
        return "regulation_time"
    if real_end < 2990:
        return "overtime_sudden_death"
    return "overtime_time"


def _row(store, m, side0, cfg):
    from training import ghost as G
    r = G.replay_match(store, m, side0, convention=cfg["convention"], log=False, guard=False)   # never guarded
    real = int(store.result[m]) * (1 if side0 == 0 else -1)
    sim = 1 if r["reward"] > 0.5 else (-1 if r["reward"] < -0.5 else 0)
    row = {"tag": r["tag"], "side0": side0, "real": real, "sim": sim, "agree": int(real == sim),
           "real_crowns_for": int(store.crowns[m, side0]),
           "real_crowns_against": int(store.crowns[m, 1 - side0]),
           "crowns_for": int(r["crowns_for"]), "crowns_against": int(r["crowns_against"]),
           "end_tick": int(r["end_tick"]), "real_end_tick": int(r["recorded_end_tick"]),
           "ended_by": r["ended_by"], "end_class": _class(int(r["recorded_end_tick"])),
           "sim_match_end": int(G.MATCH_END),
           "faithful_until": int(r["faithful_until"]), "diverged": r["diverged"],
           "diverged_at": r["diverged_at"], "end_consistent": int(r["end_consistent"]),
           "crown_faithful_until": int(r["crown_faithful_until"]), "crown_diverged": r["crown_diverged"],
           "exact26": int(bool(store.plays_ours[m, side0])),
           "cards_for": [int(c) for c in store.deck_ids(m, side0)],
           "cards_against": [int(c) for c in store.deck_ids(m, 1 - side0)],
           "tower_frac_dealt": r["tower_frac_dealt"], "tower_frac_taken": r["tower_frac_taken"]}
    for k in PLAY_KEYS:
        row[k] = sum(int(s.summary().get(k, 0)) for s in r["scripts"])
    return row


def _worker(args):
    cfg, items = args
    from training import common as C  # noqa: F401
    from training import ghost as G
    cfg = dict(cfg, convention=G.Convention(**cfg["convention"]))
    store = G.get_store(timing=cfg["convention"].timing)
    return [_row(store, m, s, cfg) for _, m, s in items]


def _child(a):
    from training import common as C
    from training import ghost as G
    cv = G.get_convention(a.convention, timing=a.timing, levels=a.levels, tower_troops=a.tower_troops,
                          illegal="force" if a.force else a.illegal, max_wait=a.max_wait,
                          after_end=a.after_end)
    if cv.levels and not G.HAS_LEVELS:
        cv = G.get_convention(cv, levels=False)
    if cv.tower_troops and not G.HAS_TOWER_TROOPS:
        cv = G.get_convention(cv, tower_troops=False)
    store = G.get_store(timing=cv.timing)
    elig = _eligible(store, a.faced)
    if a.child == "list":
        with open(a.out, "w") as f:
            json.dump({"engine": C.ENGINE, "tags": [t for t, _, _ in elig]}, f)
        return
    with open(a.tags_file) as f:
        want = json.load(f)
    by = {t: (t, m, s) for t, m, s in elig}
    items = [by[t] for t in want]
    cfg = {"convention": dict(cv._asdict())}
    t0 = time.time()
    procs = max(1, min(a.procs, len(items) // 50 or 1))
    if procs == 1:
        rows = _worker((cfg, items))
    else:
        with mp.get_context("spawn").Pool(procs) as pool:
            parts = pool.map(_worker, [(cfg, items[k::procs]) for k in range(procs)])
        rows = [None] * len(items)
        for k, part in enumerate(parts):
            rows[k::procs] = part
    names = {}
    for r in rows:
        for c in r["cards_for"] + r["cards_against"]:
            if c not in names:
                names[c] = G.card_name(c)
    meta = {"engine": C.ENGINE, "convention": dict(cv._asdict()),
            "is_default_convention": cv[1:] == G.CONVENTION[1:],
            "timing": cv.timing, "levels": cv.levels, "tower_troops": cv.tower_troops,
            "illegal": cv.illegal, "force": cv.illegal == "force", "max_wait": cv.max_wait,
            "after_end": cv.after_end, "start_elixir": store.start_elixir, "offset20": store.offset20,
            "faced": a.faced, "wall_s": time.time() - t0, "procs": procs,
            "card_names": {str(k): v for k, v in sorted(names.items())},
            "our_deck": [int(c) for c in store.our_deck],
            "match_end_tick": G.MATCH_END, "cycle_cooldown_ticks": G.COOLDOWN,
            "tower_hp": [float(x) for x in C.TOWER_HP0], "eligible_matches": len(elig)}
    with open(a.out, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f)


# ------------------------------------------------------------------ summary ---
def _p(k, n):
    if not n:
        return {"p": None, "k": int(k), "n": 0}
    p = k / n
    return {"p": p, "se": (p * (1 - p) / n) ** 0.5, "k": int(k), "n": int(n)}


def summarize(rows):
    n = len(rows)
    s = {"matches": n, "winner_agreement": _p(sum(r["agree"] for r in rows), n)}
    dec = [r for r in rows if r["real"] != 0]
    s["winner_agreement_real_decisive"] = _p(sum(r["agree"] for r in dec), len(dec))
    dec = [r for r in rows if r["sim"] != 0]         # the old builder's denominator
    s["winner_agreement_sim_decided"] = _p(sum(r["agree"] for r in dec), len(dec))
    s["real_result_team0"] = {k: sum(1 for r in rows if r["real"] == v) / max(1, n)
                              for k, v in (("win", 1), ("draw", 0), ("loss", -1))}
    s["sim_result_team0"] = {k: sum(1 for r in rows if r["sim"] == v) / max(1, n)
                             for k, v in (("win", 1), ("draw", 0), ("loss", -1))}
    for side in ("for", "against"):
        d = [r[f"crowns_{side}"] - r[f"real_crowns_{side}"] for r in rows]
        s[f"crowns_{side}"] = {"exact": _p(sum(1 for x in d if x == 0), n),
                               "mean_abs_error": sum(abs(x) for x in d) / max(1, n),
                               "mean_error": sum(d) / max(1, n),
                               "sim_mean": sum(r[f"crowns_{side}"] for r in rows) / max(1, n),
                               "real_mean": sum(r[f"real_crowns_{side}"] for r in rows) / max(1, n)}
    s["crowns_both_exact"] = _p(sum(1 for r in rows if r["crowns_for"] == r["real_crowns_for"]
                                    and r["crowns_against"] == r["real_crowns_against"]), n)
    s["ended_by_engine"] = _p(sum(1 for r in rows if r["ended_by"] == "engine"), n)

    def short(r):                        # ticks the simulated match ended before it should have
        return min(r["real_end_tick"], r.get("sim_match_end", 10 ** 9)) - r["end_tick"]

    early = lambda r: short(r) > EARLY  # noqa: E731
    s["sim_ended_early"] = _p(sum(1 for r in rows if early(r)), n)
    s["end_tick"] = {"sim_mean": sum(r["end_tick"] for r in rows) / max(1, n),
                     "real_mean": sum(r["real_end_tick"] for r in rows) / max(1, n),
                     "sim_over_10s_early": _p(sum(1 for r in rows if short(r) > 100), n),
                     "sim_over_30s_early": _p(sum(1 for r in rows if short(r) > 300), n)}
    rec = sum(r["recorded"] for r in rows)
    played = sum(r["played"] for r in rows)
    tot = {k: sum(r[k] for r in rows) for k in PLAY_KEYS}
    s["plays"] = {"recorded": rec, "played": played, "counts": tot,
                  "share_of_recorded": {k: tot[k] / max(1, rec) for k in
                                        ("played", "exact", "late", "moved_tile", "topup", "dropped",
                                         "not_reached")},
                  "share_of_played": {k: tot[k] / max(1, played) for k in
                                      ("exact", "late", "moved_tile", "topup")},
                  "illegal_or_moved_share_of_played":
                      (played - tot["exact"] + tot["dropped"]) / max(1, played + tot["dropped"])}
    s["by_real_end"] = {}
    for c in ("king_down_in_regulation", "regulation_time", "overtime_sudden_death", "overtime_time"):
        g = [r for r in rows if r["end_class"] == c]
        s["by_real_end"][c] = {"winner_agreement": _p(sum(r["agree"] for r in g), len(g)),
                               "sim_ended_early": _p(sum(1 for r in g if early(r)), len(g))}
    # Matches the engine carried to the end of the recording.
    g = [r for r in rows if not early(r)]
    s["winner_agreement_when_not_early"] = _p(sum(r["agree"] for r in g), len(g))
    s["end_gap_s"] = _quantiles([short(r) / 10.0 for r in rows if early(r)])
    if rows and "faithful_until" in rows[0]:
        s["faithful"] = faithful_summary(rows)
    return s


QS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def _quantiles(xs, qs=QS):
    xs = sorted(xs)
    if not xs:
        return {"n": 0}
    out = {"n": len(xs), "mean": sum(xs) / len(xs)}
    for q in qs:
        out[f"q{int(round(100 * q)):02d}"] = xs[min(len(xs) - 1, int(q * len(xs)))]
    return out


def faithful_summary(rows):
    """Distribution of ghost.Fidelity's "faithful until" over replayed matches.

    share = faithful_until / the tick the real match was over (the recorded end,
    capped at the engine's match end): the part of the match whose decision
    states a cloning dataset may keep.
    """
    n = len(rows)

    def end(r):
        return max(1, min(r["real_end_tick"], r.get("sim_match_end", 10 ** 9)))

    share = [min(1.0, r["faithful_until"] / end(r)) for r in rows]
    out = {"never_provably_diverged": _p(sum(1 for r in rows if not r["diverged"]), n),
           "end_consistent": _p(sum(1 for r in rows if r["end_consistent"]), n),
           "faithful_to_end_and_end_consistent":
               _p(sum(1 for r in rows if not r["diverged"] and r["end_consistent"]), n),
           "reason": {k: _p(sum(1 for r in rows if (r["diverged"] or "none") == k), n)
                      for k in ("none", "below_real_final", "pocket_tower_stands", "ended_early")},
           "faithful_until_s": _quantiles([r["faithful_until"] / 10.0 for r in rows]),
           "share_of_match": _quantiles(share),
           "share_of_all_match_ticks": sum(min(r["faithful_until"], end(r)) for r in rows)
                                       / max(1, sum(end(r) for r in rows)),
           "matches_faithful_at_least": {f"{t}s": _p(sum(1 for r in rows if r["faithful_until"] >= 10 * t
                                                         or not r["diverged"]), n)
                                         for t in (30, 60, 90, 120, 150, 180)},
           "winner_agreement_never_diverged":
               _p(sum(r["agree"] for r in rows if not r["diverged"]),
                  sum(1 for r in rows if not r["diverged"]))}
    if "crown_faithful_until" in rows[0]:        # the coarser mark: crowns still possible
        cs = [min(1.0, r["crown_faithful_until"] / end(r)) for r in rows]
        ok = [r for r in rows if not r["crown_diverged"]]
        out["crowns"] = {
            "never_provably_diverged": _p(len(ok), n),
            "reason": {k: _p(sum(1 for r in rows if (r["crown_diverged"] or "none") == k), n)
                       for k in ("none", "tower_down_that_stood", "pocket_tower_stands", "ended_early")},
            "faithful_until_s": _quantiles([r["crown_faithful_until"] / 10.0 for r in rows]),
            "share_of_match": _quantiles(cs),
            "share_of_all_match_ticks": sum(min(r["crown_faithful_until"], end(r)) for r in rows)
                                        / max(1, sum(end(r) for r in rows)),
            "winner_agreement_never_diverged": _p(sum(r["agree"] for r in ok), len(ok))}
    return out


# ---------------------------------------------------------------- card bias ---
def card_bias(rows, names=None, min_held=200, seed=DEFAULT_SEED, permutations=200, top=20):
    """Which cards the simulator makes too strong or too weak.

    One row per replayed match, team 0's view (team 0 must not be chosen by the
    result: use --faced any). y = (sim - real) / 2 in win-share units (win 1,
    draw 1/2, loss 0), regressed by least squares on an intercept and, per card
    (an engine id, so an evolution or hero form is its own card), x = [team 0
    holds it] - [team 1 holds it]. A coefficient of +0.05 says: the side holding
    the card gets 5 points more win share in the simulator than it really got,
    other cards equal. Standard errors are heteroskedasticity-robust (HC1).
    Cards held by fewer than `min_held` sides are pooled into one column.

    Checks that the pattern is not noise: (a) `permutation`: y shuffled over
    matches `permutations` times, the largest |t| of each shuffle against the
    real ones; (b) `split_half`: the fit on two random halves of the matches
    and the correlation of the two coefficient vectors.
    """
    import numpy as np
    held = {}
    for r in rows:
        for c in r["cards_for"] + r["cards_against"]:
            held[c] = held.get(c, 0) + 1
    cards = sorted(c for c, k in held.items() if k >= min_held)
    col = {c: i + 1 for i, c in enumerate(cards)}
    rare = len(cards) + 1
    n = len(rows)
    X = np.zeros((n, len(cards) + 2))
    X[:, 0] = 1.0
    for i, r in enumerate(rows):
        for c in r["cards_for"]:
            X[i, col.get(c, rare)] += 1.0
        for c in r["cards_against"]:
            X[i, col.get(c, rare)] -= 1.0
    y = np.array([(r["sim"] - r["real"]) / 2.0 for r in rows])

    def fit(X, y, robust=True, inv=None):
        XtX_inv = np.linalg.pinv(X.T @ X) if inv is None else inv
        Xty = X.T @ y
        b = XtX_inv @ Xty
        if robust:
            e = y - X @ b
            meat = (X * (e ** 2)[:, None]).T @ X
            cov = XtX_inv @ meat @ XtX_inv * (len(y) / max(1, len(y) - X.shape[1]))
        else:
            cov = XtX_inv * max(0.0, y @ y - b @ Xty) / max(1, len(y) - X.shape[1])
        return b, np.sqrt(np.maximum(np.diag(cov), 0.0))

    b, se = fit(X, y)
    t = np.where(se > 0, b / np.where(se > 0, se, 1.0), 0.0)
    name = (lambda c: (names or {}).get(str(c), str(c)))
    tab = [{"card": int(c), "name": name(c), "held": int(held[c]), "coef": float(b[col[c]]),
            "se": float(se[col[c]]), "lo": float(b[col[c]] - 1.96 * se[col[c]]),
            "hi": float(b[col[c]] + 1.96 * se[col[c]]), "t": float(t[col[c]])} for c in cards]
    tab.sort(key=lambda d: -d["coef"])
    rng = np.random.default_rng(seed)
    tc = np.abs(t[1:len(cards) + 1])
    perm_max, perm_n3 = [], []
    inv = np.linalg.pinv(X.T @ X)
    for _ in range(permutations):
        bp, sp = fit(X, rng.permutation(y), robust=False, inv=inv)
        tp = np.abs(bp[1:len(cards) + 1] / np.where(sp[1:len(cards) + 1] > 0, sp[1:len(cards) + 1], 1.0))
        perm_max.append(float(tp.max()))
        perm_n3.append(int((tp > 3).sum()))
    idx = rng.permutation(n)
    h1, h2 = idx[: n // 2], idx[n // 2:]
    b1, _ = fit(X[h1], y[h1], robust=False)
    b2, _ = fit(X[h2], y[h2], robust=False)
    sl = slice(1, len(cards) + 1)
    w = np.array([held[c] for c in cards], float)
    corr = float(np.corrcoef(b1[sl], b2[sl])[0, 1]) if len(cards) > 2 else None
    big = np.argsort(-np.abs(t[sl]))[:40]
    return {"matches": n, "cards": len(cards), "min_held": min_held,
            "outcome": "(sim - real) / 2, team 0's view, win share",
            "intercept": {"coef": float(b[0]), "se": float(se[0])},
            "rare_cards_pooled": {"coef": float(b[rare]), "se": float(se[rare]),
                                  "cards": len(held) - len(cards)},
            "mean_abs_y": float(np.abs(y).mean()), "y_var": float(y.var()),
            "r2": float(1 - ((y - X @ b) ** 2).sum() / max(1e-12, ((y - y.mean()) ** 2).sum())),
            "table": tab, "too_strong": tab[:top], "too_weak": tab[::-1][:top],
            "permutation": {"n": permutations, "real_max_abs_t": float(tc.max()),
                            "real_cards_abs_t_over_3": int((tc > 3).sum()),
                            "shuffled_max_abs_t_mean": float(np.mean(perm_max)),
                            "shuffled_max_abs_t_max": float(np.max(perm_max)),
                            "shuffled_cards_abs_t_over_3_mean": float(np.mean(perm_n3)),
                            "p_max_t": float((1 + sum(m >= tc.max() for m in perm_max)) / (1 + permutations))},
            "split_half": {"coef_correlation": corr,
                           "coef_correlation_40_largest_t":
                               float(np.corrcoef(b1[sl][big], b2[sl][big])[0, 1]) if len(big) > 2 else None,
                           "sign_agreement_40_largest_t":
                               float((np.sign(b1[sl][big]) == np.sign(b2[sl][big])).mean()) if len(big) else None,
                           "held_weighted_correlation":
                               float(np.cov(b1[sl], b2[sl], aweights=w)[0, 1]
                                     / np.sqrt(np.cov(b1[sl], aweights=w) * np.cov(b2[sl], aweights=w)))
                               if len(cards) > 2 else None}}


def show_card_bias(cb, ours=()):
    print(f"--- per-card bias: {cb['matches']} matches, {cb['cards']} cards held >= {cb['min_held']} times; "
          f"intercept {cb['intercept']['coef']:+.3f} ±{cb['intercept']['se']:.3f}; R2 {cb['r2']:.3f}")
    pm, sh = cb["permutation"], cb["split_half"]
    print(f"  max |t| {pm['real_max_abs_t']:.1f} (shuffled outcomes: mean {pm['shuffled_max_abs_t_mean']:.1f}, "
          f"max {pm['shuffled_max_abs_t_max']:.1f}); cards with |t| > 3: {pm['real_cards_abs_t_over_3']} "
          f"(shuffled {pm['shuffled_cards_abs_t_over_3_mean']:.2f}); split-half coefficient correlation "
          f"{sh['coef_correlation']:.2f} (40 largest |t|: {sh['coef_correlation_40_largest_t']:.2f})")
    line = lambda d: (f"    {d['name']:<28} {d['coef']:+.3f}  [{d['lo']:+.3f}, {d['hi']:+.3f}]  "  # noqa: E731
                      f"held {d['held']}")
    print("  sim too STRONG for the holder:")
    for d in cb["too_strong"]:
        print(line(d))
    print("  sim too WEAK for the holder:")
    for d in cb["too_weak"]:
        print(line(d))
    rank = {d["card"]: i + 1 for i, d in enumerate(cb["table"])}
    if ours:
        print("  our cards (rank from too strong, of %d):" % len(cb["table"]))
        by = {d["card"]: d for d in cb["table"]}
        for c in ours:
            if c in by:
                print(line(by[c]) + f"  rank {rank[c]}")


def _pct(v):
    if v["p"] is None:
        return "-"
    return f"{100 * v['p']:.1f}% ±{100 * v['se']:.1f} ({v['k']}/{v['n']})"


def show(name, s):
    sh, sp = s["plays"]["share_of_recorded"], s["plays"]["share_of_played"]
    print(f"--- {name}: {s['matches']} matches")
    print(f"  winner agreement            {_pct(s['winner_agreement'])} of all matches"
          f"   (of matches the sim decided: {_pct(s['winner_agreement_sim_decided'])})")
    print("  team 0 = the 2.6 player (--faced any: either player, by the tag)")
    print("  team-0 result real / sim    "
          + " ".join(f"{k} {100 * s['real_result_team0'][k]:.1f}/{100 * s['sim_result_team0'][k]:.1f}%"
                     for k in ("win", "draw", "loss")))
    for side, label in (("for", "team 0"), ("against", "team 1")):
        c = s[f"crowns_{side}"]
        print(f"  crowns of {label:<16}  exact {_pct(c['exact'])}  mean abs error "
              f"{c['mean_abs_error']:.2f}  sim {c['sim_mean']:.2f} vs real {c['real_mean']:.2f}")
    print(f"  both crown counts exact     {_pct(s['crowns_both_exact'])}")
    print(f"  sim match over >{EARLY / 10:.0f} s early  {_pct(s['sim_ended_early'])}   "
          f"(>30 s early: {_pct(s['end_tick']['sim_over_30s_early'])})   "
          f"end tick sim {s['end_tick']['sim_mean']:.0f} vs real {s['end_tick']['real_mean']:.0f}")
    print(f"  winner agreement when the sim match was not over early "
          f"{_pct(s['winner_agreement_when_not_early'])}")
    print(f"  plays: {s['plays']['recorded']} recorded; made {100 * sh['played']:.1f}%, never reached "
          f"{100 * sh['not_reached']:.1f}%, dropped {100 * sh['dropped']:.2f}%")
    print(f"  of plays made: exact {100 * sp['exact']:.1f}%, late {100 * sp['late']:.2f}%, tile moved "
          f"{100 * sp['moved_tile']:.2f}%, elixir topped up {100 * sp['topup']:.2f}%; "
          f"illegal or moved {100 * s['plays']['illegal_or_moved_share_of_played']:.2f}%")
    for c, v in s["by_real_end"].items():
        print(f"    real end {c:<26} agreement {_pct(v['winner_agreement'])}   sim over early "
              f"{_pct(v['sim_ended_early'])}")
    f = s.get("faithful")
    if f:
        q, sh = f["faithful_until_s"], f["share_of_match"]
        print(f"  faithful until (s)          q10 {q['q10']:.0f}  q25 {q['q25']:.0f}  median {q['q50']:.0f}  "
              f"q75 {q['q75']:.0f}  q90 {q['q90']:.0f};  share of the match: q25 {sh['q25']:.2f}  median "
              f"{sh['q50']:.2f}  q75 {sh['q75']:.2f};  of all match ticks {f['share_of_all_match_ticks']:.2f}")
        print(f"  never provably diverged     {_pct(f['never_provably_diverged'])}   and end crowns right "
              f"{_pct(f['faithful_to_end_and_end_consistent'])}   end crowns right {_pct(f['end_consistent'])}")
        print("  first proof of divergence   "
              + "  ".join(f"{k} {_pct(v)}" for k, v in f["reason"].items() if k != "none"))
        c = f.get("crowns")
        if c:
            q, sh = c["faithful_until_s"], c["share_of_match"]
            print(f"  crowns possible until (s)   q10 {q['q10']:.0f}  q25 {q['q25']:.0f}  median {q['q50']:.0f}  "
                  f"q75 {q['q75']:.0f}  q90 {q['q90']:.0f};  share of the match: q25 {sh['q25']:.2f}  median "
                  f"{sh['q50']:.2f}  q75 {sh['q75']:.2f};  of all match ticks {c['share_of_all_match_ticks']:.2f}; "
                  f"never diverged {_pct(c['never_provably_diverged'])}")


# ------------------------------------------------------------------- parent ---
def _run_child(engine, mode, out, extra):
    env = dict(os.environ, CR_ENGINE=os.path.abspath(engine), PYTHONDONTWRITEBYTECODE="1")
    env.pop("CLASH_DECK", None)
    cmd = [sys.executable, "-m", "training.parity", "--child", mode, "--engine", engine,
           "--out", out] + extra
    res = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"parity child failed for {engine}:\n{res.stderr[-3000:]}")
    with open(out) as f:
        return json.load(f)


def _sample(tags, n, seed):
    import random
    tags = sorted(tags)
    random.Random(seed).shuffle(tags)
    return tags if not n else tags[:n]


def _extra(faced="variant", procs=DEFAULT_PROCS, convention=None, timing=None, levels=None,
           tower_troops=None, illegal=None, force=False, max_wait=None, after_end=None):
    extra = ["--faced", faced, "--procs", str(procs)]
    if convention is not None:
        extra += ["--convention", convention if isinstance(convention, str) else convention.name]
        if not isinstance(convention, str):          # a Convention object: pass every field
            timing, levels, tower_troops = convention.timing, convention.levels, convention.tower_troops
            illegal, max_wait, after_end = convention.illegal, convention.max_wait, convention.after_end
            extra[-2:] = []
    for flag, v in (("--timing", timing), ("--illegal", illegal), ("--max-wait", max_wait),
                    ("--after-end", after_end)):
        if v is not None:
            extra += [flag, str(v)]
    for flag, v in (("levels", levels), ("tower-troops", tower_troops)):
        if v is not None:
            extra += [f"--{flag}" if v else f"--no-{flag}"]
    extra += ["--force"] if force else []
    return extra


def parity(engine, n=1000, seed=DEFAULT_SEED, timing=None, faced="variant", levels=None,
           force=False, max_wait=None, procs=DEFAULT_PROCS, tags=None, keep_rows=True,
           convention=None, tower_troops=None, illegal=None, after_end=None):
    """Replay parity of one engine on `n` seeded Hog 2.6 matches (0 = all), or
    on the given replay `tags`. -> {"meta", "summary", "rows"}.

    The replay convention is ghost.CONVENTION, or `convention` (a
    ghost.CONVENTIONS name or a Convention), with any of timing / levels /
    tower_troops / illegal / max_wait / after_end replaced when given."""
    extra = _extra(faced, procs, convention, timing, levels, tower_troops, illegal, force, max_wait,
                   after_end)
    with tempfile.TemporaryDirectory() as d:
        if tags is None:
            tags = _sample(_run_child(engine, "list", os.path.join(d, "l.json"), extra)["tags"], n, seed)
        tf = os.path.join(d, "tags.json")
        with open(tf, "w") as f:
            json.dump(list(tags), f)
        res = _run_child(engine, "run", os.path.join(d, "r.json"), extra + ["--tags-file", tf])
    res["meta"].update(n=len(res["rows"]), seed=seed)
    res["summary"] = summarize(res["rows"])
    if not keep_rows:
        res.pop("rows")
    return res


def compare(engines, n=1000, seed=DEFAULT_SEED, **kw):
    """Every engine on one common sample (matches all of them can load), and the
    first engine on a sample of its own. -> dict."""
    extra = _extra(**{k: v for k, v in kw.items() if k not in ("keep_rows", "tags")})
    lists = {}
    with tempfile.TemporaryDirectory() as d:
        for k, e in enumerate(engines):
            lists[e] = set(_run_child(e, "list", os.path.join(d, f"l{k}.json"), extra)["tags"])
    out = {"engines": list(engines), "eligible": {e: len(v) for e, v in lists.items()}}
    out["own"] = parity(engines[0], tags=_sample(lists[engines[0]], n, seed), seed=seed, **kw)
    if len(engines) > 1:
        common = set.intersection(*lists.values())
        tags = _sample(common, n, seed)
        out["eligible_common"] = len(common)
        out["common"] = {e: parity(e, tags=tags, seed=seed, **kw) for e in engines}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", action="append", required=True,
                   help="engine tree (repeat to compare engines on a common sample)")
    p.add_argument("--n", type=int, default=1000, help="matches (0 = every eligible match)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--convention", default=None,
                   help="a ghost.CONVENTIONS name (default: ghost.CONVENTION); the options below "
                        "replace single fields of it")
    p.add_argument("--timing", default=None, choices=("clock0", "bc"))
    p.add_argument("--faced", default="variant", choices=("variant", "exact", "any"),
                   help="variant: a player used our eight cards in any form; exact: our exact deck; "
                        "any: every loadable match")
    p.add_argument("--levels", action=argparse.BooleanOptionalAction, default=None,
                   help="recorded card and tower levels instead of level 11")
    p.add_argument("--tower-troops", action=argparse.BooleanOptionalAction, default=None,
                   help="each side's recorded tower card instead of the engine's default tower")
    p.add_argument("--illegal", default=None, choices=("wait", "force", "skip"),
                   help="a recorded play that is illegal at its tick: wait for it, force it, drop it")
    p.add_argument("--force", action="store_true", help="= --illegal force (the BC pipeline's forcing)")
    p.add_argument("--max-wait", type=int, default=None)
    p.add_argument("--after-end", default=None, choices=("truncate", "idle"))
    p.add_argument("--procs", type=int, default=DEFAULT_PROCS)
    p.add_argument("--card-bias", action="store_true",
                   help="per-card regression of (sim - real) on the first engine's own sample "
                        "(use with --faced any and a large --n)")
    p.add_argument("--min-held", type=int, default=200)
    p.add_argument("--no-rows", action="store_true", help="leave the per-match rows out of --out")
    p.add_argument("--out", default=None)
    p.add_argument("--child", default=None, choices=("list", "run"), help=argparse.SUPPRESS)
    p.add_argument("--tags-file", default=None, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.child:
        a.engine = a.engine[0]
        return _child(a)
    kw = dict(convention=a.convention, timing=a.timing, faced=a.faced, levels=a.levels,
              tower_troops=a.tower_troops, illegal=a.illegal, force=a.force, max_wait=a.max_wait,
              after_end=a.after_end, procs=a.procs)
    t0 = time.time()
    res = compare(a.engine, a.n, a.seed, **kw)
    res["wall_s"] = time.time() - t0
    print(f"eligible matches: " + ", ".join(f"{e}: {k}" for e, k in res["eligible"].items())
          + (f"; common {res['eligible_common']}" if "common" in res else ""))
    show(f"{a.engine[0]} (own sample)", res["own"]["summary"])
    for e, r in res.get("common", {}).items():
        show(f"{e} (common sample)", r["summary"])
    meta = res["own"]["meta"]
    if a.card_bias:
        res["card_bias"] = card_bias(res["own"]["rows"], meta.get("card_names"), a.min_held, a.seed)
        show_card_bias(res["card_bias"], ours=tuple(meta.get("our_deck", ())))
    print(f"convention {meta['convention']}, faced {a.faced}, {res['wall_s']:.0f} s")
    if a.no_rows:
        for r in [res["own"]] + list(res.get("common", {}).values()):
            r.pop("rows", None)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
