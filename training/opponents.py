"""Opponent decks = the full sample of real decks 2.6 players face.

Source: data/decks/il_replay_pool.json (sibling dataset builder; every opponent
deck in the IL_Replay hog26 matches, with `count`), or, until it exists, the
same matches straight from data/il_replay/hog26/matches.parquet.

`build_pool()` validates every deck against this engine (all 8 ids registered,
legal evo/hero slots via validate_deck_slots, and the UtilityTeacher can be
built for it) and writes data/decks/train_pool.json:

    {"decks": [{"name", "card_ids", "count", "bucket"}...], "top_k": K, ...}

Sampling (`RealDeckSampler`), per episode:
  * with prob `p_freq` (default 0.5): a deck drawn by its REAL frequency over the
    whole pool -- i.e. the opponent of a random real 2.6 match, long tail included;
  * otherwise PFSP over K+1 buckets (each of the top-K decks is a bucket; every
    other deck is in the "tail" bucket, drawn by frequency inside it) with the
    engine's own weights deck_pool.pfsp_weights ((1-wr)^2, floor 0.05), so hard
    matchups get extra practice without leaving the real distribution.
Win-rate estimates are kept per bucket (count-weighted EWMA, as the engine does),
so the trainer's per-deck read-out and TensorBoard stay at K+1 series.

The env hooks: RewardEnv passes deck_pool=False to the engine env (so its own
16-deck pool is off) and calls `install_in_worker()`; the sampler lives in
`RealDeckEnvMixin` methods below, mixed into RewardEnv by `make_env`.
"""
import json
import os
import random

import numpy as np

from training import common as C

POOL_PATH = C.data_dir("decks", "train_pool.json")
SIBLING_POOL = C.data_dir("decks", "il_replay_pool.json")
# Current Path of Legends meta from the official API (training/meta_decks_api.py).
META_POOL = C.data_dir("decks", "meta_api_pool.json")
PARQUET = C.data_dir("il_replay", "hog26", "matches.parquet")
TAIL = "tail"


def _raw_decks():
    """[(card_ids, count, keys, substituted)] from the replay pool plus the API meta
    pool, a deck seen in both counted once with both counts summed."""
    raw, src = _replay_decks()
    if not os.path.exists(META_POOL):
        return raw, src
    merged = {tuple(sorted(r[0])): list(r) for r in raw}
    for d in json.load(open(META_POOL))["decks"]:
        if not d.get("enabled", True):
            continue
        sig = tuple(sorted(d["card_ids"]))
        if sig in merged:
            merged[sig][1] += int(d["count"])
        else:
            merged[sig] = [list(map(int, d["card_ids"])), int(d["count"]), d["cards"],
                           bool(d.get("substituted", False))]
    return [tuple(v) for v in merged.values()], f"{src} + {META_POOL}"


def _replay_decks():
    """[(card_ids, count, keys)] from the sibling pool, else the parquet."""
    if os.path.exists(SIBLING_POOL):
        blob = json.load(open(SIBLING_POOL))
        out = []
        for d in blob["decks"]:
            ids = d.get("card_ids")
            if ids and None not in ids and d.get("enabled", True):
                out.append((list(map(int, ids)), int(d.get("count", 1)),
                            d.get("card_keys") or d.get("cards"),
                            bool(d.get("substituted", False))))
        return out, SIBLING_POOL
    import collections
    import pyarrow.parquet as pq
    t = pq.read_table(PARQUET, columns=["deck_them", "keys_them"]).to_pydict()
    c, keys = collections.Counter(), {}
    for ids, k in zip(t["deck_them"], t["keys_them"]):
        if None in ids:
            continue
        sig = tuple(sorted(ids))
        c[sig] += 1
        keys.setdefault(sig, (list(ids), k))
    return [(keys[s][0], n, keys[s][1], False) for s, n in c.most_common()], PARQUET


def _deck_ok(ids):
    """None if the engine and the teacher can field this deck, else the reason."""
    if len(ids) != 8 or len(set(ids)) != 8:
        return "not 8 distinct cards"
    for cid in ids:
        try:
            C.E.get_card_info(int(cid))
        except Exception:
            return f"unknown card id {cid}"
    try:
        C.E.validate_deck_slots(list(ids))
    except Exception as e:  # noqa: BLE001
        return f"slot rules: {e}"
    return None


def build_pool(top_k=24, out=POOL_PATH, probe_teacher=True, verbose=True):
    raw, src = _raw_decks()
    decks, skipped = [], []
    from python_ai.opponents.teacher import UtilityTeacher
    for ids, n, keys, subst in sorted(raw, key=lambda r: -r[1]):
        why = _deck_ok(ids)
        if why is None and probe_teacher:
            try:
                UtilityTeacher(list(ids), team=1)
            except Exception as e:  # noqa: BLE001
                why = f"teacher: {type(e).__name__}: {e}"
        if why:
            skipped.append({"card_ids": ids, "count": n, "keys": keys, "why": why})
            continue
        decks.append({"card_ids": [int(c) for c in ids], "count": int(n),
                      "keys": keys, "substituted": subst})
    total = sum(d["count"] for d in decks)
    for i, d in enumerate(decks):
        d["name"] = f"r{i:04d}"
        d["bucket"] = d["name"] if i < top_k else TAIL
        d["names"] = [C.E.get_card_info(c)["name"] for c in d["card_ids"]]
    cover = sum(d["count"] for d in decks[:top_k]) / max(1, total)
    blob = {"source": src, "top_k": top_k, "n_decks": len(decks), "n_matches": total,
            "top_k_match_share": cover,
            "skipped": skipped, "n_skipped_matches": sum(s["count"] for s in skipped),
            "decks": decks}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(blob, fh, indent=1)
    if verbose:
        print(f"pool: {len(decks)} decks / {total} matches from {src}; top-{top_k} cover "
              f"{cover:.1%}; skipped {len(skipped)} decks "
              f"({blob['n_skipped_matches']} matches) -> {out}")
        reasons = {}
        for s in skipped:
            reasons[s["why"].split(":")[0]] = reasons.get(s["why"].split(":")[0], 0) + 1
        if reasons:
            print("  skip reasons:", reasons)
    return blob


def load_pool(path=None):
    path = path or os.environ.get("CR_TRAIN_POOL") or POOL_PATH
    return json.load(open(path))


class RealDeckSampler:
    def __init__(self, blob, p_freq=0.5, rng=None):
        self.decks = blob["decks"]
        self.p_freq = float(p_freq)
        self.rng = rng or random.Random()
        self.counts = np.asarray([d["count"] for d in self.decks], dtype=np.float64)
        self.cum = np.cumsum(self.counts / self.counts.sum())
        self.buckets = sorted({d["bucket"] for d in self.decks})
        tail_idx = [i for i, d in enumerate(self.decks) if d["bucket"] == TAIL]
        self.tail_idx = np.asarray(tail_idx, dtype=np.int64)
        if len(tail_idx):
            tc = self.counts[self.tail_idx]
            self.tail_cum = np.cumsum(tc / tc.sum())
        self.bucket_index = {d["name"]: i for i, d in enumerate(self.decks)}
        self.stats = {b: 0.5 for b in self.buckets}
        self.n = {b: 0 for b in self.buckets}

    def _by_freq(self, cum, idx=None):
        j = int(np.searchsorted(cum, self.rng.random(), side="right"))
        j = min(j, len(cum) - 1)
        return int(idx[j]) if idx is not None else j

    def sample(self):
        """-> deck dict."""
        if self.rng.random() < self.p_freq:
            return self.decks[self._by_freq(self.cum)]
        from python_ai.opponents.deck_pool import pfsp_weights
        w = pfsp_weights(self.stats)
        names = list(w)
        r, acc = self.rng.random(), 0.0
        pick = names[-1]
        for nm in names:
            acc += w[nm]
            if r <= acc:
                pick = nm
                break
        if pick == TAIL:
            return self.decks[self._by_freq(self.tail_cum, self.tail_idx)]
        return self.decks[self.bucket_index[pick]]

    def record(self, bucket, won):
        if bucket not in self.stats:
            return
        n = self.n[bucket] = self.n[bucket] + 1
        a = max(0.05, 1.0 / (n + 1))
        self.stats[bucket] = (1 - a) * self.stats[bucket] + a * (1.0 if won else 0.0)


def install_in_worker():
    """Nothing to patch globally: sampling is done by RewardEnv itself. Kept as the
    one place a worker-side hook would go."""
    return None
