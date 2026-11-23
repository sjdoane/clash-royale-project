"""The human BC dataset (data/bc/hog26/shard-*.npz), held in RAM as sparse rows.

A dense float16 obs row is 13,977 values and only ~0.7% are non-zero, so the
full ~2M-row dataset is ~56 GB dense but ~1.5 GB as CSR. Each shard is converted
once and cached in data/bc_csr/ (the sibling builder owns data/bc/; we never
write there). Batches are densified on the training device.

Episode = one real match from the 2.6 player's side, rows in time order, one
row per 1-s decision (DATASET_SCHEMA in engine/python_ai/trainers/bc_pretrain.py
plus extras: t, won, elixir, ability, ability_ready, replay_tag, relocated ...).
"""
import glob
import hashlib
import json
import os
import threading

import numpy as np
import torch

from training import common as C

SRC = C.data_dir("bc", "hog26")
CACHE = C.data_dir("bc_csr", "hog26")
EXTRA_KEYS = ("card", "cell", "episode", "t", "won", "ability", "ability_ready",
              "replay_tag", "relocated", "topup")


def is_val_tag(tag):
    """The sibling's split rule (meta.json 'split'): sha1(replay_tag) % 20 == 0.
    Tags are stored as ints in the shards; meta.json's explicit list wins."""
    return int(hashlib.sha1(str(tag).encode()).hexdigest(), 16) % 20 == 0


def _convert(src, dst):
    z = np.load(src)
    obs = z["obs"]
    n, d = obs.shape
    indptr = np.zeros(n + 1, np.int64)
    idx_parts, val_parts = [], []
    step = 2048
    for s in range(0, n, step):
        blk = obs[s:s + step]
        r, c = np.nonzero(blk)
        counts = np.bincount(r, minlength=len(blk))
        indptr[s + 1:s + 1 + len(blk)] = counts
        idx_parts.append(c.astype(np.int16))
        val_parts.append(blk[r, c].astype(np.float16))
    indptr = np.cumsum(indptr)
    out = {"indptr": indptr, "indices": np.concatenate(idx_parts),
           "data": np.concatenate(val_parts), "obs_dim": np.int64(d)}
    for k in EXTRA_KEYS:
        if k in z.files:
            out[k] = z[k]
    tmp = dst + ".tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, dst)


def shard_files(src=SRC):
    return sorted(f for f in glob.glob(os.path.join(src, "shard-*.npz")) if ".tmp" not in f)


def ensure_cached(src=SRC, cache=CACHE, verbose=True):
    os.makedirs(cache, exist_ok=True)
    out = []
    for f in shard_files(src):
        dst = os.path.join(cache, os.path.basename(f))
        if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(f):
            try:
                _convert(f, dst)
            except Exception as e:  # noqa: BLE001  (a shard still being written)
                if verbose:
                    print(f"  skip {os.path.basename(f)} ({type(e).__name__})", flush=True)
                continue
            if verbose:
                print(f"  cached {os.path.basename(f)}", flush=True)
        out.append(dst)
    return out


class Episode:
    __slots__ = ("indptr", "indices", "data", "card", "cell", "won", "ability",
                 "ability_ready", "tag", "T", "returns", "shard", "episode", "relocated")


class BCData:
    """All episodes of the cached shards. `val_tags` from meta.json if present."""

    def __init__(self, files=None, meta_path=None, max_shards=None, verbose=True):
        files = files if files is not None else ensure_cached(verbose=verbose)
        if max_shards:
            files = files[:max_shards]
        meta_path = meta_path or os.path.join(SRC, "meta.json")
        # The sibling's split: meta.json lists the held-out EPISODE ids (row of
        # matches.parquet); shards store replay_tag as an index into
        # replay_tags.json, so the sha1 fallback hashes the tag string.
        self.val_tags = None
        self.tag_names = None
        if os.path.exists(meta_path):
            try:
                meta = json.load(open(meta_path))
                ve = (meta.get("split") or {}).get("val_episodes")
                if ve is not None:
                    self.val_tags = set(int(x) for x in ve)
            except Exception:  # noqa: BLE001
                pass
        tn = os.path.join(os.path.dirname(meta_path), "replay_tags.json")
        if os.path.exists(tn):
            self.tag_names = json.load(open(tn))
        self.obs_dim = None
        self.train, self.val = [], []
        for f in files:
            z = np.load(f)
            self.obs_dim = int(z["obs_dim"])
            indptr, indices, data = z["indptr"], z["indices"], z["data"]
            ep = z["episode"]
            starts = np.r_[0, np.nonzero(np.diff(ep))[0] + 1, len(ep)]
            for a, b in zip(starts[:-1], starts[1:]):
                e = Episode()
                e.indptr = indptr[a:b + 1] - indptr[a]
                e.indices = indices[indptr[a]:indptr[b]]
                e.data = data[indptr[a]:indptr[b]]
                e.card = z["card"][a:b].astype(np.int64)
                e.cell = z["cell"][a:b].astype(np.int64)
                e.won = int(z["won"][a]) if "won" in z.files else 0
                e.ability = z["ability"][a:b].astype(np.int64) if "ability" in z.files else None
                e.ability_ready = (z["ability_ready"][a:b].astype(np.int64)
                                   if "ability_ready" in z.files else None)
                e.tag = int(z["replay_tag"][a]) if "replay_tag" in z.files else int(ep[a])
                e.T = b - a
                e.relocated = (z["relocated"][a:b].astype(np.float32)
                               if "relocated" in z.files else np.zeros(b - a, np.float32))
                e.returns = None
                e.shard = os.path.basename(f)
                e.episode = int(ep[a])
                if self.tag_names is not None:
                    is_val_rule = is_val_tag(self.tag_names[e.tag])
                else:
                    is_val_rule = is_val_tag(e.tag)
                # Episodes newer than meta.json's list fall back to the rule.
                is_val = is_val_rule if self.val_tags is None else (
                    e.episode in self.val_tags or is_val_rule)
                (self.val if is_val else self.train).append(e)
                continue
                (self.val if is_val else self.train).append(e)
        if verbose:
            n_tr = sum(e.T for e in self.train)
            n_va = sum(e.T for e in self.val)
            print(f"BC data: {len(files)} shards, {len(self.train)} train / {len(self.val)} val "
                  f"matches, {n_tr} / {n_va} rows, obs_dim {self.obs_dim}, split "
                  f"{'meta.json' if self.val_tags is not None else 'sha1(tag)%20'}", flush=True)

    @staticmethod
    def dense_obs(ep, rows=None):
        """(T, D) float32 numpy for an episode (or a row range)."""
        T = ep.T
        a, b = (0, T) if rows is None else rows
        out = np.zeros((b - a, ep_dim(ep)), np.float32)
        lo, hi = ep.indptr[a], ep.indptr[b]
        r = np.repeat(np.arange(b - a), np.diff(ep.indptr[a:b + 1]))
        out[r, ep.indices[lo:hi].astype(np.int64)] = ep.data[lo:hi].astype(np.float32)
        return out

    def compute_returns(self, spec):
        """Critic targets: the discounted REAL outcome only, gamma^(T-1-t) * (+-1).

        Not the sim tower-shaped return: the replayed sim destroys ~3x fewer
        towers than the real matches did (data/bc/hog26/REPORT.md, fidelity),
        so tower HP along these trajectories is not the real game's."""
        for e in self.train + self.val:
            outcome = 1.0 if e.won else -1.0
            e.returns = (outcome * spec.gamma ** np.arange(e.T - 1, -1, -1)).astype(np.float32)


_OBS_DIM = {}


def ep_dim(ep):
    return _OBS_DIM.get("d", 13977)


def set_obs_dim(d):
    _OBS_DIM["d"] = int(d)


def batch_tensors(episodes, t0, L, device, obs_dim):
    """Dense (L, B, D) obs + label tensors for rows [t0, t0+L) of each episode;
    rows past an episode's end are padding (valid=0)."""
    B = len(episodes)
    rows_r, rows_c, vals = [], [], []
    card = np.full((L, B), C.NOOP, np.int64)
    cell = np.zeros((L, B), np.int64)
    valid = np.zeros((L, B), np.float32)
    won = np.zeros((L, B), np.float32)
    ab = np.zeros((L, B), np.int64)
    ab_ready = np.zeros((L, B), np.float32)
    ret = np.zeros((L, B), np.float32)
    reloc = np.zeros((L, B), np.float32)
    for j, e in enumerate(episodes):
        a, b = t0, min(t0 + L, e.T)
        if b <= a:
            continue
        n = b - a
        lo, hi = e.indptr[a], e.indptr[b]
        r = np.repeat(np.arange(n), np.diff(e.indptr[a:b + 1]))
        rows_r.append(r * B + j)            # flat index (l * B + j)
        rows_c.append(e.indices[lo:hi])
        vals.append(e.data[lo:hi])
        card[:n, j] = e.card[a:b]
        cell[:n, j] = e.cell[a:b]
        valid[:n, j] = 1.0
        won[:n, j] = float(e.won)
        if e.ability is not None:
            ab[:n, j] = (e.ability[a:b] >= 0).astype(np.int64)
            ab_ready[:n, j] = (e.ability_ready[a:b] > 0) if e.ability_ready is not None else 0
        if e.returns is not None:
            ret[:n, j] = e.returns[a:b]
        reloc[:n, j] = e.relocated[a:b]
    obs = torch.zeros(L * B, obs_dim, dtype=torch.float32, device=device)
    if rows_r:
        rr = torch.from_numpy(np.concatenate(rows_r).astype(np.int64)).to(device)
        cc = torch.from_numpy(np.concatenate(rows_c).astype(np.int64)).to(device)
        vv = torch.from_numpy(np.concatenate(vals).astype(np.float32)).to(device)
        obs[rr, cc] = vv
    t = lambda x: torch.from_numpy(x).to(device)  # noqa: E731
    return {"obs": obs.view(L, B, obs_dim), "card": t(card), "cell": t(cell),
            "valid": t(valid), "won": t(won), "ability": t(ab), "ability_ready": t(ab_ready),
            "returns": t(ret), "relocated": t(reloc)}


class Prefetcher:
    """Background thread building the next batch while the device trains."""

    def __init__(self, gen):
        self.gen = gen
        self.q = []
        self.lock = threading.Condition()
        self.done = False
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        for item in self.gen:
            with self.lock:
                while len(self.q) >= 3:
                    self.lock.wait()
                self.q.append(item)
                self.lock.notify_all()
        with self.lock:
            self.done = True
            self.lock.notify_all()

    def __iter__(self):
        while True:
            with self.lock:
                while not self.q and not self.done:
                    self.lock.wait()
                if not self.q and self.done:
                    return
                item = self.q.pop(0)
                self.lock.notify_all()
            yield item
