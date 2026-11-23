"""Behaviour-cloning rows on disk: sharded sparse storage and the batch loader.

    .venv/bin/python -m training.bcdata2 info  --root data/bc2/pilot
    .venv/bin/python -m training.bcdata2 bench --root data/bc2/pilot --batch 4096 --device mps

Written by training/replay_bc.py (the generator). This module has no engine
import: it only knows the files.

Layout of a dataset directory
  dataset.json   format, obs_dim (read off the wrapper when the data was made),
                 values dtype, the replay convention and file fingerprints
  manifest.npy / manifest.json / decks.npy   the sampling plan (replay_bc.plan)
  shards/<split>-<hog|oth>-<nnnn>[-m]/       "-m" = left-right mirrored copies
      indptr.npy   int64 (rows + 1)   CSR row pointers
      indices.npy  uint16             non-zero observation columns
      values.npy   float32            their values
      rows.npy     ROW                one record per decision row
      eps.npy      EP                 one record per re-simulated side
      opp.npy      OPP                the other seat's executed plays
      meta.json

One row = one decision point of the cloned seat = one `env.step` call of the
generator: the observation the wrapper returned BEFORE that step, and what the
human did at that tick.

Label (deck independent)
  slot      0..3 hand slot played, 4 = wait
  tile      row * 18 + col in the player's own frame (row 0 = own back line), -1 = wait
  card      engine card id played, -1 = wait
  ability   bit 0 = the ability of the card in deck position 1, bit 1 = deck
            position 2 (the engine's two ability slots; the same for every deck)
  dt_next / nx_slot / nx_tile / nx_card   ticks to this seat's next executed
            play after this tick (65535 = none) and that play
Row kind (bits): K_PLAY a recorded play is attempted here, K_ABILITY an ability
fires here, K_GRID the tick is on the regular wait grid, K_AUX none of those (a
step boundary only; never served by default).
"""
import argparse
import json
import os
import queue
import shutil
import threading
import time

import numpy as np

FORMAT = "bc2/1"
BOARD_W = 18
WAIT = 4
NO_NEXT = 65535

K_PLAY, K_ABILITY, K_GRID, K_AUX = 1, 2, 4, 8
F_LATE = 1          # executed later than recorded (it was not possible at its tick)
F_MOVED = 2         # recorded tile not legal in the simulation: nearest legal tile
F_TOPUP = 4         # the bar was topped up to make the play possible
F_HAND = 8          # the hand was forced at this tick (recording contradicts the cycle)
F_COOLDOWN = 16     # the cycle-in cooldown was cleared (illegal="force")
F_PENDING = 32      # a recorded play / ability is overdue here: "wait" is not the human's choice
F_REFUSED = 64      # the engine refused the play
F_PAST_DIV = 128    # after the first not-exactly-replayed event of either seat
F_PAST_FAITH = 256  # after the convention's "faithful until" tick
F_AB_LATE = 512     # ability fired later than recorded
F_T0 = 1024         # the tick-0 row of a dataset whose tick-0 observation is the reset one, BEFORE the
#                     recorded hand (the pilot). Data built with dataset.json "t0_hand": true has the
#                     recorded hand in that observation and does not set the flag.
F_DROPPED = 2048    # a recorded play / ability was given up at this tick
F_GUARD = 4096      # the tower guard (ghost.Guard) has changed the board before this row was observed
BAD_PLAY = F_MOVED | F_TOPUP | F_HAND | F_COOLDOWN | F_REFUSED
NOT_EXACT = BAD_PLAY | F_LATE | F_PENDING | F_AB_LATE | F_DROPPED

ROW = np.dtype([
    ("ep", "<i4"), ("tick", "<u2"), ("rec_tick", "<u2"), ("skip", "u1"),
    ("kind", "u1"), ("slot", "u1"), ("tile", "<i2"), ("card", "<i2"), ("orig_tile", "<i2"),
    ("ability", "u1"), ("ab_ready", "u1"), ("flags", "<u2"),
    ("dt_next", "<u2"), ("nx_slot", "u1"), ("nx_tile", "<i2"), ("nx_card", "<i2"),
    ("hand", "<i2", 4), ("queue", "<i2", 4),
    ("elixir", "<f4"), ("elixir_set", "<f4"),
    # tower guard: engine hit points given back so far to the cloned seat's / the other seat's towers
    ("g_own", "<f4"), ("g_opp", "<f4"),
])
#: Fields a shard written before the tower guard does not have (read as 0).
ROW_ADDED = ("g_own", "g_opp")
STATS = ("recorded", "played", "exact", "late", "moved_tile", "topup", "dropped", "refused",
         "hand_inconsistent", "hand_resync", "not_reached", "abilities_used", "abilities_dropped",
         "cooldown_cleared")
EP = np.dtype([
    ("m", "<i4"), ("side", "u1"), ("mirrored", "u1"), ("split", "u1"), ("group", "u1"), ("tags", "u1"),
    ("tag", "<U32"), ("row0", "<i8"), ("nrows", "<i4"),
    ("deck_id", "<i4"), ("opp_deck_id", "<i4"), ("deck", "<i2", 8), ("opp_deck", "<i2", 8),
    ("hand0", "<i2", 4), ("queue0", "<i2", 4),
    ("level", "u1", 2), ("opp_level", "u1", 2),            # (card, tower) level set in the simulation
    ("rec_level", "u1", 2), ("rec_opp_level", "u1", 2),    # recorded (median card, tower card)
    ("tower", "u1"), ("opp_tower", "u1"),                  # engine tower troop in the simulation
    ("result", "i1"), ("crowns", "u1", 2),                 # the REAL match, cloned side's view
    ("sim_reward", "<f4"), ("sim_crowns", "u1", 2), ("sim_end", "<u2"), ("rec_end", "<u2"),
    ("ended_by", "u1"),                                    # 0 recording, 1 engine, 2 stuck
    ("div_tick", "<i4"), ("own_div", "<i4"), ("opp_div", "<i4"),   # first not-exact event; -1 none
    ("faithful", "<i4"), ("faith_reason", "u1"), ("end_consistent", "i1"),
    ("seed", "<u4"), ("own", "<u2", len(STATS)), ("opp", "<u2", len(STATS)),
    ("cpu_ms", "<f4"),
    # tower guard (ghost.Guard.summary()); pairs are (cloned seat, other seat)
    ("guard", "u1"), ("g_first", "<i4"),                   # 1 = guard on; tick of its first intervention, -1 none
    ("g_heals", "<u4", 2), ("g_hp", "<f4", 2),             # times a tower was set back to its floor; hit points given back
    ("g_saved", "<u2", 2),                                 # ... of which the tower would have died in the segment
    ("g_forced", "u1", 2), ("g_leveled", "u1", 2),         # towers destroyed by pocket evidence / the 3:00 rule
    ("g_breach", "u1", 2), ("g_hold", "u1", 2),            # protected towers that died anyway; held past their real fall
    ("king_asleep", "u1", 2), ("king_hit", "<i4", 2),      # recording: King never woke; first tick the simulated King was hit
])
EP_ADDED = ("guard", "g_first", "g_heals", "g_hp", "g_saved", "g_forced", "g_leveled", "g_breach", "g_hold",
            "king_asleep", "king_hit")
OPP = np.dtype([("ep", "<i4"), ("tick", "<u2"), ("rec_tick", "<u2"), ("card", "<i2"),
                ("col", "u1"), ("row", "u1"), ("flags", "u1")])
SPLITS = ("train", "val")
GROUPS = ("other", "hog_variant", "hog_exact")
T_FREQ, T_FLOOR, T_HOG = 1, 2, 4
ENGINE_MATCH_TICKS = 3000      # the engine ends a match at 5:00 (ghost.MATCH_END)
FAITH_REASONS = ("", "below_real_final", "pocket_tower_stands", "ended_early", "first_illegal_play", "other")

#: Integer label columns of a batch, in order.
ICOLS = ("slot", "tile", "card", "ab1", "ab2", "rdy1", "rdy2", "kind", "flags", "tick", "dt_next",
         "nx_slot", "nx_tile", "nx_card", "h0", "h1", "h2", "h3", "deck_id", "opp_deck_id", "won",
         "mirrored", "group", "tags", "m", "side", "shard", "row", "ep")
IC = {n: i for i, n in enumerate(ICOLS)}
#: Float label columns of a batch, in order (new ones are appended).
FCOLS = ("elixir", "g_own", "g_opp")
FC = {n: i for i, n in enumerate(FCOLS)}


class ObsDimMismatch(RuntimeError):
    """The stored observation length is not the one asked for / the wrapper's."""


def shard_name(split, hog, k, mirrored):
    return f"{SPLITS[int(split)]}-{'hog' if hog else 'oth'}-{int(k):04d}{'-m' if mirrored else ''}"


# ------------------------------------------------------------------ writing ---
class ShardWriter:
    """Collects episodes in memory, writes one shard directory atomically."""

    def __init__(self, root, name, obs_dim, values_dtype="float32"):
        if obs_dim >= 65536:
            raise ValueError("observation too long for uint16 column indices")
        self.root, self.name, self.obs_dim = root, name, int(obs_dim)
        self.vdtype = np.dtype(values_dtype)
        self.lens, self.idx, self.val, self.rows, self.eps, self.opp = [], [], [], [], [], []
        self.n = 0

    def add(self, ep, rows, lens, indices, values, opp):
        """ep: EP record (1,), rows: ROW (n,), lens: non-zeros per row, opp: OPP."""
        if len(rows) != len(lens) or int(np.sum(lens)) != len(indices) or len(indices) != len(values):
            raise ValueError("inconsistent episode arrays")
        if len(indices) and int(indices.max()) >= self.obs_dim:
            raise ObsDimMismatch(f"column {int(indices.max())} outside obs_dim {self.obs_dim}")
        e = len(self.eps)
        ep = np.array(ep, dtype=EP).reshape(1).copy()
        ep["row0"], ep["nrows"] = self.n, len(rows)
        rows = np.array(rows, dtype=ROW)
        rows["ep"] = e
        opp = np.array(opp, dtype=OPP)
        opp["ep"] = e
        self.eps.append(ep)
        self.rows.append(rows)
        self.opp.append(opp)
        self.lens.append(np.asarray(lens, np.int64))
        self.idx.append(np.asarray(indices, np.uint16))
        self.val.append(np.asarray(values, self.vdtype))
        self.n += len(rows)

    def close(self, extra=None):
        cat = lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dt)   # noqa: E731
        lens = cat(self.lens, np.int64)
        indptr = np.zeros(len(lens) + 1, np.int64)
        np.cumsum(lens, out=indptr[1:])
        eps = cat(self.eps, EP)
        arrays = {"indptr": indptr, "indices": cat(self.idx, np.uint16), "values": cat(self.val, self.vdtype),
                  "rows": cat(self.rows, ROW), "eps": eps, "opp": cat(self.opp, OPP)}
        final = os.path.join(self.root, "shards", self.name)
        tmp = os.path.join(self.root, "shards", f".tmp-{self.name}-{os.getpid()}")
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        nbytes = 0
        for k, a in arrays.items():
            p = os.path.join(tmp, k + ".npy")
            np.save(p, a)
            nbytes += os.path.getsize(p)
        meta = {"format": FORMAT, "name": self.name, "obs_dim": self.obs_dim, "values": self.vdtype.name,
                "n_rows": int(len(lens)), "n_eps": int(len(eps)), "nnz": int(indptr[-1]), "bytes": int(nbytes),
                "mirrored": bool(self.name.endswith("-m")), "split": self.name.split("-")[0],
                "groups": sorted(int(g) for g in np.unique(eps["group"]))}
        meta.update(extra or {})
        with open(os.path.join(tmp, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        shutil.rmtree(final, ignore_errors=True)
        os.replace(tmp, final)
        return meta


# ------------------------------------------------------------------ reading ---
class Shard:
    __slots__ = ("name", "meta", "indptr", "indices", "values", "rows", "eps", "opp")

    def dense(self, ids):
        """(len(ids), obs_dim) float32 observations of these rows."""
        ids = np.atleast_1d(np.asarray(ids, np.int64))
        out = np.zeros((len(ids), int(self.meta["obs_dim"])), np.float32)
        for k, r in enumerate(ids):
            a, b = int(self.indptr[r]), int(self.indptr[r + 1])
            out[k, np.asarray(self.indices[a:b], np.int64)] = self.values[a:b]
        return out

    def episode(self, e):
        """(EP record, slice of its rows)."""
        ep = self.eps[int(e)]
        return ep, slice(int(ep["row0"]), int(ep["row0"]) + int(ep["nrows"]))


class BCDataset:
    def __init__(self, root, expect_obs_dim=None):
        self.root = root
        p = os.path.join(root, "dataset.json")
        if not os.path.exists(p):
            raise FileNotFoundError(f"no dataset at {root} (dataset.json missing)")
        with open(p) as f:
            self.info = json.load(f)
        if self.info.get("format") != FORMAT:
            raise ValueError(f"{root}: format {self.info.get('format')!r}, this reader is {FORMAT!r}")
        self.obs_dim = int(self.info["obs_dim"])
        if expect_obs_dim is not None and int(expect_obs_dim) != self.obs_dim:
            raise ObsDimMismatch(f"{root} stores observations of length {self.obs_dim}, "
                                 f"{int(expect_obs_dim)} expected: regenerate the data")
        self.metas = []
        sd = os.path.join(root, "shards")
        for name in sorted(os.listdir(sd)) if os.path.isdir(sd) else ():
            mp = os.path.join(sd, name, "meta.json")
            if name.startswith(".") or not os.path.exists(mp):
                continue
            with open(mp) as f:
                meta = json.load(f)
            if int(meta["obs_dim"]) != self.obs_dim:
                raise ObsDimMismatch(f"shard {name} has obs_dim {meta['obs_dim']}, dataset.json says "
                                     f"{self.obs_dim}: mixed generations, regenerate")
            self.metas.append(meta)

    def __len__(self):
        return len(self.metas)

    @property
    def n_rows(self):
        return sum(m["n_rows"] for m in self.metas)

    @property
    def nbytes(self):
        return sum(m["bytes"] for m in self.metas)

    def select(self, split=None, mirrored=None, groups=None):
        """Indices of the shards that can hold rows of this selection."""
        out = []
        for i, m in enumerate(self.metas):
            if split is not None and m["split"] != split:
                continue
            if mirrored is not None and bool(m["mirrored"]) != bool(mirrored):
                continue
            if groups is not None and not set(m["groups"]) & set(groups):
                continue
            out.append(i)
        return out

    def open(self, i, mmap=True, obs=True):
        meta = self.metas[i]
        d = os.path.join(self.root, "shards", meta["name"])
        s = Shard()
        s.name, s.meta = meta["name"], meta
        ld = lambda n: np.load(os.path.join(d, n + ".npy"), mmap_mode="r" if mmap else None)   # noqa: E731
        s.rows, s.eps, s.opp = (np.load(os.path.join(d, n + ".npy")) for n in ("rows", "eps", "opp"))
        s.indptr = np.load(os.path.join(d, "indptr.npy"))
        s.indices, s.values = (ld("indices"), ld("values")) if obs else (None, None)
        return s

    def episodes(self, split=None, mirrored=None, groups=None, obs=True):
        """Yield (shard, EP record, row slice) in stored order (sequence trainers)."""
        for i in self.select(split, mirrored, groups):
            s = self.open(i, obs=obs)
            for e in range(len(s.eps)):
                ep, sl = s.episode(e)
                if groups is None or int(ep["group"]) in groups:
                    yield s, ep, sl

    def loader(self, **kw):
        return BCLoader(self, **kw)


def labels(shard, shard_index):
    """(int32 (n, len(ICOLS)), float32 (n, len(FCOLS))) label matrices of a shard."""
    r, e = shard.rows, shard.eps[shard.rows["ep"]]
    li = np.empty((len(r), len(ICOLS)), np.int32)
    cols = {
        "slot": r["slot"], "tile": r["tile"], "card": r["card"],
        "ab1": r["ability"] & 1, "ab2": (r["ability"] >> 1) & 1,
        "rdy1": r["ab_ready"] & 1, "rdy2": (r["ab_ready"] >> 1) & 1,
        "kind": r["kind"], "flags": r["flags"], "tick": r["tick"], "dt_next": r["dt_next"],
        "nx_slot": r["nx_slot"], "nx_tile": r["nx_tile"], "nx_card": r["nx_card"],
        "h0": r["hand"][:, 0], "h1": r["hand"][:, 1], "h2": r["hand"][:, 2], "h3": r["hand"][:, 3],
        "deck_id": e["deck_id"], "opp_deck_id": e["opp_deck_id"], "won": e["result"],
        "mirrored": e["mirrored"], "group": e["group"], "tags": e["tags"], "m": e["m"], "side": e["side"],
        "shard": np.full(len(r), shard_index), "row": np.arange(len(r)), "ep": r["ep"],
    }
    for n, i in IC.items():
        li[:, i] = cols[n]
    lf = np.zeros((len(r), len(FCOLS)), np.float32)
    for n, i in FC.items():
        if n in r.dtype.names:              # a shard older than the field: 0
            lf[:, i] = r[n]
    return li, lf


class RowFilter:
    """Which rows a loader serves.

    split     "train" | "val" | None (both)
    mirrored  None (both) | False | True
    groups    None | iterable of 0 other, 1 Hog variant, 2 Hog exact form
    kinds     row-kind bits to serve (default plays, abilities, grid waits)
    legal     "any"      everything, as simulated
              "late_ok"  (default) drop plays that needed a top-up, a forced
                         hand, a cleared cooldown, a moved tile or were refused,
                         and waits where a recorded play was overdue
              "exact"    also drop plays / abilities executed late
    faithful  True (default): drop rows after the "faithful until" tick
    undiverged True: drop rows after the first not-exact event of either seat
    decks     None | deck ids (the cloned seat's)
    cards     None | card ids: play rows of other cards are dropped (waits stay)
    tags      None | bit mask over T_FREQ / T_FLOOR / T_HOG
    wait_keep share of wait rows kept (random per epoch)
    t0        False: drop rows flagged F_T0 (a tick-0 observation read before the
              recorded hand; data with "t0_hand" has none)
    guarded   None (default): both | False: only rows before the tower guard's
              first intervention | True: only rows after it
    guard_hp  None | drop rows by which the guard had given back more than this
              many hit points in total (both seats)
    fn        optional fn({column: int32 array}) -> bool mask
    """

    def __init__(self, split="train", mirrored=None, groups=None, kinds=K_PLAY | K_ABILITY | K_GRID,
                 legal="late_ok", faithful=True, undiverged=False, decks=None, cards=None, tags=None,
                 wait_keep=1.0, t0=False, fn=None, guarded=None, guard_hp=None):
        if legal not in ("any", "late_ok", "exact"):
            raise ValueError(legal)
        self.split, self.mirrored = split, mirrored
        self.groups = None if groups is None else sorted(int(g) for g in groups)
        self.kinds, self.legal, self.faithful, self.undiverged = int(kinds), legal, faithful, undiverged
        self.decks = None if decks is None else np.asarray(sorted(decks), np.int64)
        self.cards = None if cards is None else np.asarray(sorted(cards), np.int64)
        self.tags, self.wait_keep, self.t0, self.fn = tags, float(wait_keep), t0, fn
        self.guarded, self.guard_hp = guarded, guard_hp

    def mask(self, li, rng=None, lf=None):
        kind, flags = li[:, IC["kind"]], li[:, IC["flags"]]
        acts = (kind & (K_PLAY | K_ABILITY)) != 0
        play = (kind & K_PLAY) != 0
        ok = (kind & self.kinds) != 0
        if self.legal != "any":
            ok &= ~(play & ((flags & BAD_PLAY) != 0))
            ok &= ~(~play & ((flags & F_PENDING) != 0))
            if self.legal == "exact":
                ok &= (flags & (F_LATE | F_AB_LATE)) == 0
        if self.faithful:
            ok &= (flags & F_PAST_FAITH) == 0
        if self.undiverged:
            ok &= (flags & F_PAST_DIV) == 0
        if not self.t0:
            ok &= (flags & F_T0) == 0
        if self.groups is not None:
            ok &= np.isin(li[:, IC["group"]], self.groups)
        if self.mirrored is not None:
            ok &= li[:, IC["mirrored"]] == int(bool(self.mirrored))
        if self.decks is not None:
            ok &= np.isin(li[:, IC["deck_id"]], self.decks)
        if self.cards is not None:
            ok &= ~play | np.isin(li[:, IC["card"]], self.cards)
        if self.tags is not None:
            ok &= (li[:, IC["tags"]] & int(self.tags)) != 0
        if self.guarded is not None:
            ok &= ((flags & F_GUARD) != 0) == bool(self.guarded)
        if self.guard_hp is not None:
            if lf is None:
                raise ValueError("guard_hp needs the float label matrix (mask(li, rng, lf))")
            ok &= (lf[:, FC["g_own"]] + lf[:, FC["g_opp"]]) <= float(self.guard_hp)
        if self.fn is not None:
            ok &= np.asarray(self.fn({n: li[:, i] for n, i in IC.items()}), bool)
        if self.wait_keep < 1.0:
            rng = rng or np.random.default_rng(0)
            ok &= acts | (rng.random(len(li)) < self.wait_keep)
        return ok


class Batch(dict):
    """Tensors of one batch; fields are attributes too. obs (B, obs_dim) float32;
    slot, tile, card, kind, flags, tick, dt_next, nx_slot, nx_tile, nx_card,
    deck_id, opp_deck_id, won, mirrored, group, tags, m, side (B,) int64;
    ability, ab_ready (B, 2); hand (B, 4); ref (B, 3) = shard, row, episode;
    elixir, g_own, g_opp (B,) float32 (hit points the tower guard had given back
    to the cloned seat's / the other seat's towers; flags & F_GUARD says whether
    it had intervened at all); weight (B,) or None."""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def table_weights(table, key="deck_id"):
    """A `weight_fn` for BCLoader: weight = table[batch[key]] (key: "deck_id" for
    per-deck, "card" for per-card reweighting; waits index the table's last
    entry through card = -1). Off unless passed to the loader."""
    import torch

    cache = {}

    def fn(batch):
        idx = batch[key]
        t = cache.get(idx.device)
        if t is None:
            t = cache[idx.device] = torch.as_tensor(np.asarray(table, np.float32), device=idx.device)
        return t[idx.clamp(min=-1, max=len(t) - 1)]

    return fn


class _Piece:
    __slots__ = ("n", "ln", "cols", "vals", "li", "lf")


class BCLoader:
    """Shuffled batches, dense on `device`.

    Shards are read `block` at a time (whole, into RAM), their kept rows are
    permuted together, and batches are cut from that permutation; a background
    thread does the reading and the sparse gather, the caller's thread builds
    the dense tensor on the device. Peak RAM is about 2 x block x shard size.

    label_transform(batch) -> batch   after the tensors exist, before noise
    noise(obs, batch) -> obs          per-row observation noise (on device)
    weight_fn(batch) -> (B,) weights  off by default (see table_weights)
    epochs: None = forever. dense=False yields CPU pieces (bench / tests).
    """

    def __init__(self, ds, batch_size=4096, device="cpu", flt=None, shuffle=True, seed=0, block=8,
                 epochs=1, drop_last=False, noise=None, label_transform=None, weight_fn=None,
                 prefetch=6, dense=True):
        self.ds, self.B, self.device = ds, int(batch_size), device
        self.flt = flt or RowFilter()
        self.shuffle, self.seed, self.block, self.epochs = shuffle, int(seed), int(block), epochs
        self.drop_last, self.noise, self.label_transform, self.weight_fn = drop_last, noise, label_transform, weight_fn
        self.prefetch, self.dense = int(prefetch), dense
        self.shard_ids = ds.select(self.flt.split, self.flt.mirrored, self.flt.groups)
        if not self.shard_ids:
            raise ValueError("no shard matches the filter")
        self.rows_served = 0

    # -- CPU side ---------------------------------------------------------------
    def _load_block(self, ids, rng):
        ptr, idx, val, li, lf, off = [np.zeros(1, np.int64)], [], [], [], [], 0
        for i in ids:
            s = self.ds.open(i, mmap=False)
            a, b = labels(s, i)
            ptr.append(s.indptr[1:] + off)
            off += int(s.indptr[-1])
            idx.append(s.indices)
            val.append(s.values)
            li.append(a)
            lf.append(b)
        blk = _Piece()
        blk.ln = np.concatenate(ptr)          # CSR pointers of the block
        blk.cols, blk.vals = np.concatenate(idx), np.concatenate(val)
        blk.li, blk.lf = np.concatenate(li), np.concatenate(lf)
        keep = np.flatnonzero(self.flt.mask(blk.li, rng, blk.lf))
        if self.shuffle:
            rng.shuffle(keep)
        return blk, keep

    @staticmethod
    def _gather(blk, r):
        lo = blk.ln[r]
        ln = blk.ln[r + 1] - lo
        tot = int(ln.sum())
        src = np.arange(tot, dtype=np.int64) + np.repeat(lo - (np.cumsum(ln) - ln), ln)
        p = _Piece()
        p.n, p.ln, p.cols, p.vals = len(r), ln, blk.cols[src], blk.vals[src]
        p.li, p.lf = blk.li[r], blk.lf[r]
        return p

    @staticmethod
    def _merge(a, b):
        p = _Piece()
        p.n = a.n + b.n
        for k in ("ln", "cols", "vals", "li", "lf"):
            setattr(p, k, np.concatenate([getattr(a, k), getattr(b, k)]))
        return p

    def _finish(self, p):
        D = self.ds.obs_dim
        p.cols = np.repeat(np.arange(p.n, dtype=np.int64), p.ln) * D + p.cols.astype(np.int64)
        if p.vals.dtype != np.float32:
            p.vals = p.vals.astype(np.float32)
        return p

    def pieces(self):
        """CPU batches (flat scatter index + values + label matrices)."""
        epoch = 0
        while self.epochs is None or epoch < self.epochs:
            rng = np.random.default_rng([self.seed, epoch])
            order = list(self.shard_ids)
            if self.shuffle:
                rng.shuffle(order)
            carry = None
            for b0 in range(0, len(order), self.block):
                blk, ids = self._load_block(order[b0:b0 + self.block], rng)
                pos = 0
                if carry is not None:
                    take = ids[:self.B - carry.n]
                    pos = len(take)
                    carry = self._merge(carry, self._gather(blk, take))
                    if carry.n == self.B:
                        yield self._finish(carry)
                        carry = None
                while pos + self.B <= len(ids):
                    yield self._finish(self._gather(blk, ids[pos:pos + self.B]))
                    pos += self.B
                if pos < len(ids):
                    carry = self._gather(blk, ids[pos:])
            if carry is not None and not self.drop_last:
                yield self._finish(carry)
            epoch += 1

    # -- device side --------------------------------------------------------------
    def to_batch(self, p):
        import torch

        dev, D = self.device, self.ds.obs_dim
        obs = torch.zeros(p.n * D, dtype=torch.float32, device=dev)
        obs[torch.from_numpy(p.cols).to(dev)] = torch.from_numpy(p.vals).to(dev)
        li = torch.from_numpy(p.li).to(dev).long()
        lf = torch.from_numpy(p.lf).to(dev)
        c = lambda n: li[:, IC[n]]   # noqa: E731
        b = Batch(obs=obs.view(p.n, D), weight=None)
        for n in ("slot", "tile", "card", "kind", "flags", "tick", "dt_next", "nx_slot", "nx_tile",
                  "nx_card", "deck_id", "opp_deck_id", "won", "mirrored", "group", "tags", "m", "side"):
            b[n] = c(n)
        b["ability"] = li[:, IC["ab1"]:IC["ab2"] + 1]
        b["ab_ready"] = li[:, IC["rdy1"]:IC["rdy2"] + 1]
        b["hand"] = li[:, IC["h0"]:IC["h3"] + 1]
        b["ref"] = li[:, IC["shard"]:IC["ep"] + 1]
        for n, i in FC.items():
            b[n] = lf[:, i]
        if self.label_transform is not None:
            b = self.label_transform(b) or b
        if self.noise is not None:
            b["obs"] = self.noise(b["obs"], b)
        if self.weight_fn is not None:
            b["weight"] = self.weight_fn(b)
        return b

    def __iter__(self):
        q, stop = queue.Queue(maxsize=max(1, self.prefetch)), threading.Event()

        def work():
            try:
                for p in self.pieces():
                    while not stop.is_set():
                        try:
                            q.put(p, timeout=0.2)
                            break
                        except queue.Full:
                            continue
                    if stop.is_set():
                        return
                q.put(None)
            except BaseException as ex:  # noqa: BLE001  (hand it to the consumer)
                q.put(ex)

        th = threading.Thread(target=work, daemon=True)
        th.start()
        try:
            while True:
                p = q.get()
                if p is None:
                    return
                if isinstance(p, BaseException):
                    raise p
                self.rows_served += p.n
                yield self.to_batch(p) if self.dense else p
        finally:
            stop.set()


# ------------------------------------------------------------------ summary ---
def summary(ds, split=None, mirrored=None, groups=None):
    """Counts that need no engine: rows by kind and flag, legality, divergence."""
    out = {"shards": 0, "eps": 0, "rows": 0, "nnz": 0, "bytes": 0, "cpu_s": 0.0}
    own, opp = np.zeros(len(STATS), np.int64), np.zeros(len(STATS), np.int64)
    c = {k: 0 for k in ("play", "ability", "wait", "aux", "served_default", "past_div", "past_faith",
                        "play_exact", "play_late_only", "play_bad", "wait_pending", "t0",
                        "eps_diverged", "eps_unfaithful", "eps_end_consistent", "eps_engine_end",
                        "eps_early_end", "eps_cut_short", "eps_end_at_1800_early", "ticks_recorded", "ticks_simulated",
                        "ticks_faithful", "rows_guarded", "eps_guard", "eps_guard_intervened", "guard_heals",
                        "guard_forced", "guard_leveled", "guard_breach", "eps_guard_breach", "guard_hp_own",
                        "guard_hp_opp", "sides_king_asleep", "sides_king_woke_wrongly")}
    flt = RowFilter(split=None, groups=None if groups is None else list(groups))
    for i in ds.select(split, mirrored, groups):
        s = ds.open(i, obs=False)
        eps = s.eps
        if groups is not None:
            keep_ep = np.isin(eps["group"], list(groups))
            rows = s.rows[keep_ep[s.rows["ep"]]]
            eps = eps[keep_ep]
        else:
            rows = s.rows
        k, f = rows["kind"], rows["flags"]
        play, ab = (k & K_PLAY) != 0, (k & K_ABILITY) != 0
        wait = ((k & K_GRID) != 0) & ~play & ~ab
        out["shards"] += 1
        out["eps"] += len(eps)
        out["rows"] += len(rows)
        out["bytes"] += s.meta["bytes"]
        out["nnz"] += int(s.meta["nnz"])
        out["cpu_s"] += float(eps["cpu_ms"].sum()) / 1000.0
        own += eps["own"].sum(axis=0, dtype=np.int64)
        opp += eps["opp"].sum(axis=0, dtype=np.int64)
        c["play"] += int(play.sum())
        c["ability"] += int(ab.sum())
        c["wait"] += int(wait.sum())
        c["aux"] += int(((k & K_AUX) != 0).sum())
        c["t0"] += int(((f & F_T0) != 0).sum())
        c["past_div"] += int(((f & F_PAST_DIV) != 0).sum())
        c["past_faith"] += int(((f & F_PAST_FAITH) != 0).sum())
        c["rows_guarded"] += int(((f & F_GUARD) != 0).sum())
        if "guard" in eps.dtype.names:
            on = eps["guard"] == 1
            g = eps[on]
            c["eps_guard"] += int(on.sum())
            c["eps_guard_intervened"] += int((g["g_first"] >= 0).sum())
            for key, name in (("guard_heals", "g_heals"), ("guard_forced", "g_forced"),
                              ("guard_leveled", "g_leveled"), ("guard_breach", "g_breach")):
                c[key] += int(g[name].sum(dtype=np.int64))
            c["eps_guard_breach"] += int((g["g_breach"].sum(axis=1) > 0).sum())
            c["guard_hp_own"] += float(g["g_hp"][:, 0].sum(dtype=np.float64))
            c["guard_hp_opp"] += float(g["g_hp"][:, 1].sum(dtype=np.float64))
            c["sides_king_asleep"] += int(g["king_asleep"].sum())
            c["sides_king_woke_wrongly"] += int(((g["king_asleep"] == 1) & (g["king_hit"] >= 0)).sum())
        c["play_exact"] += int((play & ((f & NOT_EXACT) == 0)).sum())
        c["play_late_only"] += int((play & ((f & BAD_PLAY) == 0) & ((f & F_LATE) != 0)).sum())
        c["play_bad"] += int((play & ((f & BAD_PLAY) != 0)).sum())
        c["wait_pending"] += int((wait & ((f & F_PENDING) != 0)).sum())
        c["eps_diverged"] += int((eps["div_tick"] >= 0).sum())
        c["eps_unfaithful"] += int((eps["faith_reason"] != 0).sum())
        c["eps_end_consistent"] += int((eps["end_consistent"] == 1).sum())
        c["eps_engine_end"] += int((eps["ended_by"] == 1).sum())
        gap = eps["rec_end"].astype(np.int64) - eps["sim_end"].astype(np.int64)
        c["eps_early_end"] += int((gap > 10).sum())                  # simulation over > 1 s before the real match
        # ... before the part of it the engine can play (a real match can run past the engine's 5:00)
        c["eps_cut_short"] += int((np.minimum(eps["rec_end"].astype(np.int64), ENGINE_MATCH_TICKS)
                                   - eps["sim_end"].astype(np.int64) > 10).sum())
        c["eps_end_at_1800_early"] += int(((gap > 10) & (eps["sim_end"] == 1800)).sum())
        c["ticks_recorded"] += int(eps["rec_end"].sum(dtype=np.int64))
        c["ticks_simulated"] += int(eps["sim_end"].sum(dtype=np.int64))
        c["ticks_faithful"] += int(np.minimum(eps["faithful"], eps["sim_end"]).sum(dtype=np.int64))
        li, lf = labels(s, i)
        c["served_default"] += int(flt.mask(li, None, lf).sum())      # rows the default loader filter keeps
    out.update(c)
    out["own"] = dict(zip(STATS, (int(v) for v in own)))
    out["opp"] = dict(zip(STATS, (int(v) for v in opp)))
    return out


def bench(root, batch=4096, device="mps", batches=150, block=8, split="train"):
    import torch

    ds = BCDataset(root)
    res = {"root": root, "obs_dim": ds.obs_dim, "batch": batch, "device": device, "block": block}
    ld = BCLoader(ds, batch_size=batch, flt=RowFilter(split=split), epochs=None, block=block, dense=False)
    t0, n = time.perf_counter(), 0
    for k, p in enumerate(ld):
        n += p.n
        if k + 1 >= batches:
            break
    res["cpu_rows_per_s"] = n / (time.perf_counter() - t0)
    ld = BCLoader(ds, batch_size=batch, device=device, flt=RowFilter(split=split), epochs=None, block=block)
    sync = (lambda: torch.mps.synchronize()) if str(device).startswith("mps") else (lambda: None)
    t0, n, chk = None, 0, 0.0
    for k, b in enumerate(ld):
        if k == 5:                       # warm-up: first block load, MPS allocation
            sync()
            t0, n = time.perf_counter(), 0
        chk += float(b.obs.sum().item())   # reads every value back: the batch exists, densely, on the device
        n += len(b.slot)
        if k + 1 >= batches + 5:
            break
    sync()
    res["dense_rows_per_s"] = n / (time.perf_counter() - t0)
    res["rows_timed"] = n
    import resource
    import sys

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    res["peak_rss_gb"] = rss / (1e9 if sys.platform == "darwin" else 1e6)     # bytes on macOS, KB on Linux
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("info", "bench"))
    ap.add_argument("--root", required=True)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batches", type=int, default=150)
    ap.add_argument("--block", type=int, default=8)
    args = ap.parse_args()
    if args.cmd == "info":
        ds = BCDataset(args.root)
        print(json.dumps({"obs_dim": ds.obs_dim, "shards": len(ds), "rows": ds.n_rows, "bytes": ds.nbytes,
                          "summary": summary(ds)}, indent=1))
    else:
        print(json.dumps(bench(args.root, args.batch, args.device, args.batches, args.block), indent=1))


if __name__ == "__main__":
    main()
