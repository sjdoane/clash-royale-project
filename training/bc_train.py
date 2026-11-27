"""Behaviour cloning for NetV3 (training/net_v3.py) on a bcdata2 dataset.

    CR_ENGINE=build/gen2/engine .venv/bin/python -m training.bc_train \
        --data data/bc2/hog_g2 --name l0_h10 --lead 0 --what-h 10 --epochs 3

Targets (per decision row; `Targets`, built from the ROW table alone):

  WHEN   on grid rows (one per decision interval D = the dataset's grid): 1 if
         a human play is assigned to the row. A play commanded at tick tau is
         assigned to the last grid tick t <= tau - L, i.e. tau in [t + L, t + L
         + D): L = `--lead` ticks is what the bot's own latency (0.3 - 0.7 s
         from frame to effect) asks it to anticipate. One play per row; when
         the row is taken or the card is not playable there (not in hand / not
         affordable at t), the play moves to the following grid rows that are
         still <= tau; if none will do it is "lost" (reported) and its natural
         row is ignored instead of being called a wait. Rows where a recorded
         play is overdue (F_PENDING) and plays the simulator had to bend
         (BAD_PLAY) are ignored too.
  WHAT / WHERE   the assigned play's slot and tile on the WHEN = 1 rows (the
         states the net will act in), plus the rows selected by `--what-h`:
         "play" = the play rows themselves (the observation at the command
         tick), an integer H = every row whose next play is at most H ticks
         away. `--no-what-on-when` leaves only the second set (the ablation).
  ability   as WHEN, per engine slot, on rows where the ability is ready.
  auxiliary  bucketed time to the next own play; which slot it will be; the
         match result as the value target.

Noise (on by default): the board noise (training/obs_noise.ObsNoise, measured
defaults) and the tail noise (noise_tail_rows, names policy "spells") are both
stateful per episode, and the loader serves shuffled rows, so each episode is
noised once in time order and the CHANGED entries are cached per shard
(`NoiseCache`, under runs/bc3/_cache/, never inside the dataset); the loader's
noise hook scatters them into the batch.

Only the training split is ever read for training: the validation matches are
also the evaluation opponents (training/eval_ghost.py).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter

import numpy as np

from training import common as C  # noqa: F401  (engine path + CLASH_DECK first)

import torch
import torch.nn.functional as F

from training import bcdata2 as D
from training import net_v3 as N3
from training import obs_v2 as OV

HOG_GROUPS = (1, 2)
RUNS = os.path.join(C.REPO, "runs", "bc3")
NO_DT = 32767
LOSS_W = {"when": 1.0, "what": 1.0, "where": 1.0, "ability": 0.25, "value": 0.1, "dt": 0.2, "next": 0.2}
WHEN_BINS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0001)


def cost_table():
    """Elixir cost per engine card id (0 for holes in the id range)."""
    n = int(C.CE.NUM_CARD_IDS)
    out = np.zeros(n + 1, np.float32)
    for c in range(n):
        try:
            out[c] = float(C.E.get_card_info(c)["cost"])
        except Exception:  # noqa: BLE001
            pass
    return out


def card_id_by_name(name):
    for c in range(int(C.CE.NUM_CARD_IDS)):
        try:
            i = C.E.get_card_info(c)
        except Exception:  # noqa: BLE001
            continue
        if i["name"] == name and not (i.get("is_evolution") or i.get("is_hero")):
            return c
    raise KeyError(name)


def dataset_grid(ds):
    fp = ds.info.get("fingerprint")
    if isinstance(fp, str):
        fp = json.loads(fp)
    return int(fp["grid"])


# ------------------------------------------------------------------ targets ---
def build_targets(rows, eps, lead, interval, cost, stats=None):
    """Per-row targets of one shard (see the module docstring). Arrays over the
    shard's rows: when (-1 ignore / 0 / 1), w_slot, w_tile, w_dt (the assigned
    play), u_dt, u_slot, u_tile (the next play at or after the row; u_slot -1
    when it is unusable), ab (n, 2) (-1 / 0 / 1 per engine ability slot)."""
    n = len(rows)
    L, Dt = int(lead), int(interval)
    st = stats if stats is not None else Counter()
    T = {"when": np.full(n, -1, np.int8), "w_slot": np.full(n, -1, np.int8),
         "w_tile": np.full(n, -1, np.int16), "w_dt": np.zeros(n, np.int16),
         "u_dt": np.full(n, NO_DT, np.int16), "u_slot": np.full(n, -1, np.int8),
         "u_tile": np.full(n, -1, np.int16), "ab": np.full((n, 2), -1, np.int8)}
    tick_a = rows["tick"].astype(np.int64)
    kind_a, flags_a = rows["kind"].astype(np.int64), rows["flags"].astype(np.int64)
    hand_a, elix_a = rows["hand"].astype(np.int64), rows["elixir"].astype(np.float64)
    card_a, slot_a, tile_a = (rows[k].astype(np.int64) for k in ("card", "slot", "tile"))
    abil_a, rdy_a = rows["ability"].astype(np.int64), rows["ab_ready"].astype(np.int64)
    for e in range(len(eps)):
        r0, nr = int(eps["row0"][e]), int(eps["nrows"][e])
        if nr == 0:
            continue
        sl = slice(r0, r0 + nr)
        tick, kind, flags, hand = tick_a[sl], kind_a[sl], flags_a[sl], hand_a[sl]
        is_play = (kind & D.K_PLAY) != 0
        grid = np.nonzero((kind & D.K_GRID) != 0)[0]
        gt = tick[grid]
        when = np.where((flags[grid] & D.F_PENDING) != 0, -1, 0).astype(np.int8)
        w_slot = np.full(len(grid), -1, np.int8)
        w_tile = np.full(len(grid), -1, np.int16)
        w_dt = np.zeros(len(grid), np.int16)
        taken = np.zeros(len(grid), bool)
        plays = np.nonzero(is_play)[0]
        for pi in plays:
            if not len(grid):
                break
            tau, card = int(tick[pi]), int(card_a[r0 + pi])
            bad = (int(flags[pi]) & D.BAD_PLAY) != 0 or card < 0
            nat = j = max(int(np.searchsorted(gt, tau - L, side="right")) - 1, 0)
            if gt[nat] > tau:
                st["play_before_grid"] += 1
                continue
            st["plays"] += 1
            placed = False
            while j < len(grid) and gt[j] <= tau:
                if not taken[j]:
                    if bad:
                        when[j], taken[j], placed = -1, True, True
                        st["play_bad"] += 1
                        break
                    gi = grid[j]
                    s = np.nonzero(hand[gi] == card)[0]
                    if len(s) and elix_a[r0 + gi] + 1e-4 >= cost[card]:
                        when[j], taken[j], placed = 1, True, True
                        w_slot[j], w_tile[j], w_dt[j] = s[0], tile_a[r0 + pi], tau - gt[j]
                        st["play_assigned" if j == nat else "play_moved"] += 1
                        break
                j += 1
            if not placed:
                st["play_lost_taken" if taken[nat] else "play_lost_infeasible"] += 1
                if when[nat] == 0:
                    when[nat] = -1
        T["when"][r0 + grid] = when
        T["w_slot"][r0 + grid] = w_slot
        T["w_tile"][r0 + grid] = w_tile
        T["w_dt"][r0 + grid] = w_dt

        if len(plays):                               # the next play at or after each row
            pos = np.searchsorted(plays, np.arange(nr), side="left")
            has = pos < len(plays)
            up = plays[np.minimum(pos, len(plays) - 1)]
            u_card, u_slot = card_a[r0 + up], slot_a[r0 + up]
            ok = has & ((flags[up] & D.BAD_PLAY) == 0) & (u_slot >= 0) & (u_slot < hand.shape[1])
            ok &= hand[np.arange(nr), np.clip(u_slot, 0, hand.shape[1] - 1)] == u_card
            T["u_dt"][sl] = np.where(has, np.minimum(tick[up] - tick, NO_DT - 1), NO_DT)
            T["u_slot"][sl] = np.where(ok, u_slot, -1)
            T["u_tile"][sl] = np.where(ok, tile_a[r0 + up], -1)

        rdy = rdy_a[sl]
        ab = np.full((nr, 2), -1, np.int8)
        for b in (0, 1):
            ready = ((rdy >> b) & 1) != 0
            ab[grid[ready[grid]], b] = 0
            for ai in np.nonzero(((kind & D.K_ABILITY) != 0) & (((abil_a[sl] >> b) & 1) != 0))[0]:
                if not len(grid):
                    break
                tau = int(tick[ai])
                j = max(int(np.searchsorted(gt, tau - L, side="right")) - 1, 0)
                while j < len(grid) and gt[j] <= tau:
                    if ab[grid[j], b] == 0:
                        ab[grid[j], b] = 1
                        st["ability_assigned"] += 1
                        break
                    j += 1
                else:
                    st["ability_lost"] += 1
        T["ab"][sl] = ab
    return T


class Targets:
    """The loader's `label_transform`: adds t_when, t_w_slot, t_w_tile, t_w_dt,
    t_u_dt, t_u_slot, t_u_tile (B,) and t_ab (B, 2) to each batch."""

    def __init__(self, ds, lead, interval=None):
        self.ds, self.L = ds, int(lead)
        self.D = int(interval or dataset_grid(ds))
        self.cost = cost_table()
        self.stats = Counter()
        self._sh = {}

    def shard(self, i):
        t = self._sh.get(i)
        if t is None:
            sh = self.ds.open(int(i), mmap=True, obs=False)
            t = self._sh[i] = build_targets(np.asarray(sh.rows), np.asarray(sh.eps), self.L, self.D,
                                            self.cost, self.stats)
        return t

    def __call__(self, batch):
        ref = batch["ref"].cpu().numpy()
        dev = batch["obs"].device
        B = len(ref)
        out = {}
        for s in np.unique(ref[:, 0]):
            sel = np.nonzero(ref[:, 0] == s)[0]
            t = self.shard(int(s))
            for k, a in t.items():
                if k not in out:
                    out[k] = np.empty((B,) + a.shape[1:], a.dtype)
                out[k][sel] = a[ref[sel, 1]]
        for k, a in out.items():
            batch["t_" + k] = torch.from_numpy(a).to(dev).long()
        return batch


# -------------------------------------------------------------------- noise ---
def _episode_seed(seed, ep):
    return int((int(seed) * 1000003 + int(ep["m"]) * 4 + int(ep["side"]) * 2 + int(ep["mirrored"]))
               % (2 ** 31 - 1))


def noise_shard(root, shard_i, out_path, seed=0, names="spells", board=True, tail=True,
                upto_faithful=True):
    """Noise every episode of one shard in time order and store the changed
    entries: indptr (rows + 1), idx (uint16 column), val (float32)."""
    from training import obs_noise as ON  # noqa: PLC0415
    from training import reward as RW  # noqa: PLC0415

    ds = D.BCDataset(root)
    sh = ds.open(int(shard_i), mmap=True, obs=True)
    Dm, base = int(ds.obs_dim), N3.engine_obs_dim()
    rows, eps = np.asarray(sh.rows), np.asarray(sh.eps)
    n = len(rows)
    tol = 2e-3 if str(sh.meta.get("values")) == "float16" else 1e-6
    model = ON.ObsNoise(ON.load_config(True), seed=seed, match_tol=tol) if board else None
    lens = np.zeros(n, np.int64)
    idx_parts, val_parts = [], []
    for e in range(len(eps)):
        ep = eps[e]
        r0, nr = int(ep["row0"]), int(ep["nrows"])
        R = rows[r0:r0 + nr]
        if upto_faithful:
            good = np.nonzero((R["flags"].astype(np.int64) & D.F_PAST_FAITH) == 0)[0]
            nr = int(good[-1]) + 1 if len(good) else 0
        if nr == 0:
            continue
        a, b = int(sh.indptr[r0]), int(sh.indptr[r0 + nr])
        rl = np.diff(np.asarray(sh.indptr[r0:r0 + nr + 1]))
        clean = np.zeros((nr, Dm), np.float32)
        clean[np.repeat(np.arange(nr), rl), np.asarray(sh.indices[a:b], np.int64)] = sh.values[a:b]
        noisy = clean.copy()
        es = _episode_seed(seed, ep)
        tick = R["tick"][:nr].astype(np.float64)
        if model is not None:
            prefer = [int(c) for c in list(ep["deck"]) + list(ep["opp_deck"]) if int(c) >= 0]
            pend = []
            for i in range(nr):
                if i == 0:
                    noisy[0] = model.reset(clean[0], seed=es, prefer=prefer)
                else:
                    noisy[i] = model.apply(clean[i], float(tick[i] - tick[i - 1]), spells=pend)
                pend = []
                card, tile = int(R["card"][i]), int(R["tile"][i])
                if (int(R["kind"][i]) & D.K_PLAY) and card >= 0 and tile >= 0:
                    rad = RW._spell_radius(card)
                    if rad > 0:
                        pend = [(tile % C.BOARD_W + 0.5, tile // C.BOARD_W + 0.5, rad)]
        if tail:
            noisy = ON.noise_tail_rows(noisy, base_size=base, cfg={"names": names}, seed=es + 9974,
                                       times=tick / OV.TPS)
        rr, cc = np.nonzero(noisy != clean)
        lens[r0:r0 + nr] = np.bincount(rr, minlength=nr)
        idx_parts.append(cc.astype(np.uint16))
        val_parts.append(noisy[rr, cc].astype(np.float32))
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(lens, out=indptr[1:])
    idx = np.concatenate(idx_parts) if idx_parts else np.zeros(0, np.uint16)
    val = np.concatenate(val_parts) if val_parts else np.zeros(0, np.float32)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + f".tmp{os.getpid()}.npz"
    np.savez(tmp, indptr=indptr, idx=idx, val=val)
    os.replace(tmp, out_path)
    return int(shard_i), n, int(indptr[-1])


def _noise_job(args):
    return noise_shard(*args)


class NoiseCache:
    """The loader's `noise` hook. `ensure(shards)` computes what is missing (in
    `procs` processes); the call scatters the cached changes into a batch."""

    def __init__(self, ds, seed=0, names="spells", board=True, tail=True, p=1.0, root=None):
        self.ds, self.seed, self.names, self.board, self.tail, self.p = ds, int(seed), names, board, tail, float(p)
        tag = f"noise_{'b' if board else ''}{'t-' + names if tail else ''}_s{self.seed}"
        self.dir = os.path.join(root or os.path.join(RUNS, "_cache"), str(ds.info["fp_sha"]), tag)
        self._sh = {}
        self.rng = np.random.default_rng(self.seed + 17)

    def path(self, i):
        return os.path.join(self.dir, self.ds.metas[int(i)]["name"] + ".npz")

    def ensure(self, shard_ids, procs=5, log=print):
        todo = [int(i) for i in shard_ids if not os.path.exists(self.path(i))]
        if not todo:
            return 0
        os.makedirs(self.dir, exist_ok=True)
        jobs = [(self.ds.root, i, self.path(i), self.seed, self.names, self.board, self.tail) for i in todo]
        t0 = time.time()
        if procs <= 1 or len(jobs) == 1:
            res = [_noise_job(j) for j in jobs]
        else:
            import multiprocessing as mp  # noqa: PLC0415
            with mp.get_context("spawn").Pool(min(procs, len(jobs))) as pool:
                res = []
                for k, r in enumerate(pool.imap_unordered(_noise_job, jobs)):
                    res.append(r)
                    if log and (k + 1) % 10 == 0:
                        log(f"  noise cache {k + 1}/{len(jobs)} shards, {time.time() - t0:.0f} s")
        if log:
            log(f"noise cache: {len(jobs)} shards, {sum(r[1] for r in res)} rows, "
                f"{sum(r[2] for r in res)} changed entries, {time.time() - t0:.0f} s -> {self.dir}")
        return len(jobs)

    def _load(self, i):
        t = self._sh.get(i)
        if t is None:
            with np.load(self.path(i)) as z:
                t = self._sh[i] = (z["indptr"], z["idx"], z["val"])
        return t

    def __call__(self, obs, batch):
        ref = batch["ref"].cpu().numpy()
        B, Dm = obs.shape
        fi, fv = [], []
        for s in np.unique(ref[:, 0]):
            ip, idx, val = self._load(int(s))
            br = np.nonzero(ref[:, 0] == s)[0]
            if self.p < 1.0:
                br = br[self.rng.random(len(br)) < self.p]
            r = ref[br, 1]
            st = ip[r]
            ln = ip[r + 1] - st
            tot = int(ln.sum())
            if tot == 0:
                continue
            off = np.repeat(st - (np.cumsum(ln) - ln), ln) + np.arange(tot)
            fi.append(np.repeat(br, ln) * Dm + idx[off].astype(np.int64))
            fv.append(val[off])
        if fi:
            flat = obs.reshape(-1)
            flat[torch.from_numpy(np.concatenate(fi)).to(obs.device)] = torch.from_numpy(
                np.concatenate(fv)).to(obs.device)
        return obs


# ------------------------------------------------------------------- losses ---
def _what_rows(batch, what_h, what_on_when):
    """(row mask, slot target, tile target) of the WHAT / WHERE loss."""
    a = batch["t_when"] == 1
    u_ok = batch["t_u_slot"] >= 0
    if what_h == "play":
        b = u_ok & (batch["t_u_dt"] == 0) & ((batch["kind"] & D.K_PLAY) != 0)
    elif what_h in (None, "none"):
        b = torch.zeros_like(a)
    else:
        b = u_ok & (batch["t_u_dt"] <= int(what_h))
    if not what_on_when:
        a = torch.zeros_like(a)
    slot = torch.where(a, batch["t_w_slot"], batch["t_u_slot"])
    tile = torch.where(a, batch["t_w_tile"], batch["t_u_tile"])
    sel = (a | b) & (slot >= 0) & (tile >= 0)
    return sel, slot, tile


def _mmean(x, m):
    """Mean of x over the rows of mask m (0 when empty). No boolean indexing:
    on MPS that is a nonzero + sync per call."""
    return torch.where(m, x, torch.zeros_like(x)).sum() / m.sum().clamp(min=1)


def when_kept(obs, free_elixir=None, free_behind=False):
    """Rows that keep their WHEN reference (bool, obs: (..., obs_dim)).
    free_elixir: rows reading that much elixir or more lose it (a full bar).
    free_behind: rows where more of our Princess Towers are down than of the opponent's lose it. The human
    matches include players who stop playing once a match looks lost; live, 2026-11-23, the policy sat at a
    full bar with p(play) 0.004 whenever our towers were down and theirs untouched."""
    keep = torch.ones(obs.shape[:-1], dtype=torch.bool, device=obs.device)
    if free_elixir is not None:
        keep = keep & (obs[..., C.SPATIAL_SIZE] * 10.0 < float(free_elixir) - 1e-4)
    if free_behind:
        x = int(C.EXTRA)
        own = (obs[..., x + 4] <= 0.0).long() + (obs[..., x + 5] <= 0.0).long()
        opp = (obs[..., x + 7] <= 0.0).long() + (obs[..., x + 8] <= 0.0).long()
        keep = keep & (own <= opp)
    return keep


def forward_losses(net, batch, what_h=10, what_on_when=True, weights=None, detail=False,
                   when_free_elixir=None, when_free_behind=False):
    """One forward pass and every loss term. Returns (total, parts dict[, detail])."""
    W = dict(LOSS_W, **(weights or {}))
    obs = batch["obs"]
    hs = net.hand_size
    mask = net.affordability_mask(obs)
    f, emb, sp, hi = net.extract_features_hires(obs)
    o = net.heads(f, mask)
    parts = {}

    grid = (batch["kind"] & D.K_GRID) != 0
    wm = grid & (batch["t_when"] >= 0) & o["has_legal"]
    if when_free_elixir is not None or when_free_behind:
        # no WHEN target at a full bar / behind on towers (ppo_pool --anchor-free-elixir, --anchor-free-behind)
        wm = wm & when_kept(obs, when_free_elixir, when_free_behind)
    y_when = (batch["t_when"] == 1).float()
    parts["when"] = _mmean(F.binary_cross_entropy_with_logits(o["when"], y_when, reduction="none"), wm)

    sel, slot, tile = _what_rows(batch, what_h, what_on_when)
    slot_c = slot.clamp(min=0)
    legal = sel & mask[:, :hs].gather(1, slot_c.unsqueeze(1)).squeeze(1)
    parts["what"] = _mmean(F.cross_entropy(o["what"], slot_c, reduction="none"), legal)
    rows = torch.nonzero(sel).squeeze(1)
    place = ok = None
    if rows.numel():
        place = net._placement_raw(o["hx"][rows], emb[rows], slot[rows], obs[rows], sp[rows],
                                   hires_map=hi[rows], cycle_feat=o["head_in"][rows, -net.skip_dim:])
        ce = F.cross_entropy(place, tile[rows], reduction="none")
        ok = torch.isfinite(ce)                       # the human's tile is legal on this row
        parts["where"] = _mmean(ce, ok)
    else:
        parts["where"] = obs.new_zeros(())

    ab_loss = obs.new_zeros(())
    for k in (0, 1):
        t = batch["t_ab"][:, k]
        ab_loss = ab_loss + _mmean(F.cross_entropy(o["ability"][k], t.clamp(min=0), reduction="none"),
                                   grid & (t >= 0))
    parts["ability"] = ab_loss
    # bcdata2 "won" is the real match result from the cloned side: -1 / 0 / 1
    parts["value"] = F.mse_loss(o["value"].squeeze(1), batch["won"].float().clamp(-1.0, 1.0))
    parts["dt"] = F.cross_entropy(o["dt"], N3.dt_bucket(batch["dt_next"]))
    nx = batch["nx_slot"]
    nxc = nx.clamp(0, hs - 1)
    nm = (nx >= 0) & (nx < hs) & (batch["hand"].gather(1, nxc.unsqueeze(1)).squeeze(1) == batch["nx_card"])
    parts["next"] = _mmean(F.cross_entropy(o["next"], nxc, reduction="none"), nm)
    total = sum(W[k] * v for k, v in parts.items())
    if not detail:
        return total, parts
    return total, parts, {"o": o, "mask": mask, "wm": wm, "y_when": y_when, "sel": sel, "legal": legal,
                          "slot": slot, "tile": tile, "rows": rows, "place": place, "place_ok": ok}


# ------------------------------------------------------------ label metrics ---
def _opp_recent(obs, base, card_id, max_age_s):
    """Rows whose tail shows an opponent play of `card_id` at most max_age_s old."""
    plays = obs[:, base + OV.OPP_PLAYS.start:base + OV.OPP_PLAYS.stop].reshape(len(obs), OV.K_OPP, OV.ENTRY)
    idx = (plays[..., OV.E_CODE] * OV.CODE_DIV).round().long()
    hit = (plays[..., OV.E_VALID] > 0.5) & (idx == card_id + 1)
    hit &= plays[..., OV.E_AGE] * OV.AGE_CAP_S <= max_age_s + 1e-6
    return hit.any(dim=1)


@torch.no_grad()
def _per_card_by_name(per_card):
    """Per-card counters keyed by display name. A card and its evolution share
    a name (Skeletons, Cannon, Ice Spirit), so ids are summed, not overwritten."""
    by = {}
    for c, e in sorted(per_card.items()):
        if c >= 0:
            by.setdefault(C.E.get_card_info(c)["name"], Counter()).update(e)
    return {name: {"n": int(e["n"]), "top1": e["top1"] / max(e["n"], 1), "p_when": e["p_when"] / max(e["n"], 1),
                   "p_what": e["p_what"] / max(e["n"], 1)} for name, e in by.items()}


def label_metrics(net, loader, what_h=10, max_batches=None):
    """Held-out label metrics (see format_label_metrics). `loader` must carry the
    Targets hook (and the noise hook, to measure what the net will see)."""
    net.eval()
    log_id, fb_id = card_id_by_name("The Log"), card_id_by_name("Fireball")
    try:
        gb_id = card_id_by_name("Goblin Barrel")
    except KeyError:
        gb_id = None
    nb = len(WHEN_BINS) - 1
    edges = torch.tensor(WHEN_BINS[1:-1])
    acc = Counter()
    bins = np.zeros((nb, 3))
    per_card, tile_err = {}, []
    trig = {"log_target": Counter(), "fireball_target": Counter(), "gb_recent": Counter()}
    for k, batch in enumerate(loader):
        if max_batches and k >= max_batches:
            break
        _, parts, d = forward_losses(net, batch, what_h=what_h, what_on_when=True, detail=True)
        gb_m = (_opp_recent(batch["obs"], net.base_obs_dim, gb_id, 5.0).cpu() if gb_id is not None else None)
        cpu = lambda v: v.cpu() if torch.is_tensor(v) else v   # noqa: E731
        o = {k: cpu(v) for k, v in d["o"].items() if k in ("when", "what")}
        d = {k: cpu(v) for k, v in d.items() if k != "o"}
        batch = {k: cpu(v) for k, v in batch.items() if k != "obs"}
        batch["n"] = len(d["wm"])
        for name, v in parts.items():
            acc["loss_" + name] += float(v)
        acc["batches"] += 1
        acc["rows"] += batch["n"]
        p = torch.sigmoid(o["when"])
        wm = d["wm"]
        pw, yw = p[wm], d["y_when"][wm]
        b = torch.bucketize(pw, edges, right=True).numpy()
        for j in range(nb):
            m = b == j
            bins[j] += (m.sum(), float(pw[m].sum()), float(yw[m].sum()))
        acc["when_rows"] += len(pw)
        acc["when_pos"] += float(yw.sum())
        acc["when_psum"] += float(pw.sum())
        acc["when_bce"] += float(F.binary_cross_entropy(pw.clamp(1e-7, 1 - 1e-7), yw, reduction="sum"))

        hand = batch["hand"]
        a = (batch["t_when"] == 1) & d["legal"]
        log_p = F.log_softmax(o["what"], dim=-1)
        top = o["what"].argmax(-1)
        tgt_card = hand.gather(1, d["slot"].clamp(min=0).unsqueeze(1)).squeeze(1)
        for name, m in (("when_rows", a), ("play_rows", d["legal"] & ((batch["kind"] & D.K_PLAY) != 0)
                                          & (batch["t_u_dt"] == 0) & (batch["t_when"] != 1))):
            acc[f"what_{name}_n"] += int(m.sum())
            acc[f"what_{name}_top1"] += int((top[m] == d["slot"][m]).sum())
        for c in tgt_card[a].unique().tolist():
            m = a & (tgt_card == c)
            e = per_card.setdefault(int(c), Counter())
            e["n"] += int(m.sum())
            e["top1"] += int((top[m] == d["slot"][m]).sum())
            e["p_when"] += float(p[m].sum())
            e["p_what"] += float(log_p[m].gather(1, d["slot"][m].unsqueeze(1)).exp().sum())
        for name, cid in (("log_target", log_id), ("fireball_target", fb_id)):
            m = a & (tgt_card == cid)
            t = trig[name]
            t["n"] += int(m.sum())
            t["recall_top1"] += int((top[m] == d["slot"][m]).sum())
            t["p_when"] += float(p[m].sum())
            t["p_joint"] += float((p[m] * log_p[m].gather(1, d["slot"][m].unsqueeze(1)).exp().squeeze(1)).sum())
        if gb_id is not None:
            in_hand = hand == log_id
            log_slot = in_hand.float().argmax(1)
            can = in_hand.any(1) & d["mask"][:, :net.hand_size].gather(1, log_slot.unsqueeze(1)).squeeze(1)
            m = wm & can & gb_m
            t = trig["gb_recent"]
            t["n"] += int(m.sum())
            t["human_log"] += int((m & (batch["t_when"] == 1) & (tgt_card == log_id)).sum())
            t["p_log"] += float((p[m] * log_p[m].gather(1, log_slot[m].unsqueeze(1)).exp().squeeze(1)).sum())
            t["p_when"] += float(p[m].sum())
            t["human_when"] += int((m & (batch["t_when"] == 1)).sum())

        if d["place"] is not None:                    # tile error on the WHEN = 1 rows
            rows, ok = d["rows"], d["place_ok"]
            keep = ok & (batch["t_when"][rows] == 1)
            if keep.any():
                pred = d["place"][keep].argmax(-1)
                t = d["tile"][rows][keep]
                dx = (pred % C.BOARD_W - t % C.BOARD_W).float()
                dy = (pred // C.BOARD_W - t // C.BOARD_W).float()
                tile_err.append(torch.sqrt(dx * dx + dy * dy).numpy())
    nbt = max(acc["batches"], 1)
    out = {"rows": int(acc["rows"]), "losses": {k[5:]: v / nbt for k, v in acc.items() if k.startswith("loss_")}}
    out["when"] = {
        "rows": int(acc["when_rows"]), "base_rate": acc["when_pos"] / max(acc["when_rows"], 1),
        "mean_p": acc["when_psum"] / max(acc["when_rows"], 1),
        "bce": acc["when_bce"] / max(acc["when_rows"], 1),
        "reliability": [{"lo": WHEN_BINS[j], "hi": min(WHEN_BINS[j + 1], 1.0), "n": int(bins[j, 0]),
                         "mean_p": bins[j, 1] / max(bins[j, 0], 1), "rate": bins[j, 2] / max(bins[j, 0], 1)}
                        for j in range(nb)]}
    n_tot = max(bins[:, 0].sum(), 1)
    out["when"]["ece"] = float(sum(abs(bins[j, 1] - bins[j, 2]) for j in range(nb)) / n_tot)
    out["what"] = {"top1_when_rows": acc["what_when_rows_top1"] / max(acc["what_when_rows_n"], 1),
                   "n_when_rows": int(acc["what_when_rows_n"]),
                   "top1_play_rows": acc["what_play_rows_top1"] / max(acc["what_play_rows_n"], 1),
                   "n_play_rows": int(acc["what_play_rows_n"]),
                   "per_card": _per_card_by_name(per_card)}
    te = np.concatenate(tile_err) if tile_err else np.zeros(0)
    out["where"] = {"n": int(len(te)), "mean_tiles": float(te.mean()) if len(te) else None,
                    "median_tiles": float(np.median(te)) if len(te) else None,
                    "exact": float((te == 0).mean()) if len(te) else None,
                    "within_1.5": float((te <= 1.5).mean()) if len(te) else None}
    out["triggers"] = {}
    for name in ("log_target", "fireball_target"):
        t = trig[name]
        nn_ = max(t["n"], 1)
        out["triggers"][name] = {"n": int(t["n"]), "recall_top1": t["recall_top1"] / nn_,
                                 "p_when": t["p_when"] / nn_, "p_joint": t["p_joint"] / nn_}
    t = trig["gb_recent"]
    nn_ = max(t["n"], 1)
    out["triggers"]["gb_recent_log_playable"] = {
        "n": int(t["n"]), "human_log_rate": t["human_log"] / nn_, "model_p_log": t["p_log"] / nn_,
        "human_play_rate": t["human_when"] / nn_, "model_p_when": t["p_when"] / nn_}
    return out


def format_label_metrics(d):
    if not d or "when" not in d:
        return f"label metrics: {d}"
    w = d["when"]
    lines = [f"label metrics on {d['rows']} held-out rows ({d.get('dataset', '')}, lead {d.get('lead')}, "
             f"interval {d.get('interval')}, noise {d.get('noise')})",
             f"  WHEN  rows {w['rows']}  human rate {w['base_rate']:.4f}  mean p {w['mean_p']:.4f}  "
             f"BCE {w['bce']:.4f}  ECE {w['ece']:.4f}",
             "    p bin          n      mean p   human rate"]
    for r in w["reliability"]:
        lines.append(f"    {r['lo']:.2f}-{r['hi']:.2f}  {r['n']:8d}   {r['mean_p']:.3f}    {r['rate']:.3f}")
    wh = d["what"]
    lines.append(f"  WHAT  top-1 card given a play: {wh['top1_when_rows']:.3f} on {wh['n_when_rows']} decision rows, "
                 f"{wh['top1_play_rows']:.3f} on {wh['n_play_rows']} play rows")
    for name, e in wh["per_card"].items():
        lines.append(f"    {name:16s} n {e['n']:6d}  top-1 {e['top1']:.3f}  p(card|play) {e['p_what']:.3f}  "
                     f"p(when) {e['p_when']:.3f}")
    wr = d["where"]
    if wr["n"]:
        lines.append(f"  WHERE tile error {wr['mean_tiles']:.2f} tiles mean, {wr['median_tiles']:.2f} median, "
                     f"exact {wr['exact']:.3f}, within 1.5 tiles {wr['within_1.5']:.3f} (n {wr['n']})")
    for name in ("log_target", "fireball_target"):
        t = d["triggers"][name]
        lines.append(f"  {name:16s} n {t['n']:6d}  recall (top-1 card) {t['recall_top1']:.3f}  p(when) {t['p_when']:.3f}  "
                     f"p(when and card) {t['p_joint']:.3f}")
    t = d["triggers"]["gb_recent_log_playable"]
    lines.append(f"  Goblin Barrel seen <= 5 s, Log playable: n {t['n']}  human Log / row {t['human_log_rate']:.3f}  "
                 f"model {t['model_p_log']:.3f}  (any play: human {t['human_play_rate']:.3f}, model {t['model_p_when']:.3f})")
    return "\n".join(lines)


# ------------------------------------------------------------------ loaders ---
def pick_device(device=None):
    if device:
        return device
    return "mps" if torch.backends.mps.is_available() else "cpu"


def make_loader(ds, split, lead, device, groups=HOG_GROUPS, mirrored=None, noise=True, names="spells",
                noise_seed=0, noise_p=1.0, batch_size=4096, shuffle=True, seed=0, procs=5, log=print,
                interval=None, targets=None):
    """(BCLoader, Targets, NoiseCache | None) for one split."""
    flt = D.RowFilter(split=split, mirrored=mirrored, groups=None if groups is None else list(groups))
    tg = targets or Targets(ds, lead, interval)
    cache = None
    if noise:
        cache = NoiseCache(ds, seed=noise_seed, names=names, p=noise_p)
        cache.ensure(ds.select(split, mirrored, flt.groups), procs=procs, log=log)
    ld = D.BCLoader(ds, batch_size=batch_size, device=device, flt=flt, shuffle=shuffle, seed=seed,
                    noise=cache, label_transform=tg)
    return ld, tg, cache


def estimate_rows(ds, split, groups, mirrored=None):
    n = 0
    for i in ds.select(split, mirrored, None if groups is None else list(groups)):
        sh = ds.open(i, mmap=True, obs=False)
        fl, kd = np.asarray(sh.rows["flags"]).astype(np.int64), np.asarray(sh.rows["kind"]).astype(np.int64)
        ok = ((fl & D.F_PAST_FAITH) == 0) & ((kd & (D.K_PLAY | D.K_ABILITY | D.K_GRID)) != 0)
        if groups is not None:
            ok &= np.isin(np.asarray(sh.eps["group"])[np.asarray(sh.rows["ep"]).astype(np.int64)], list(groups))
        n += int(ok.sum())
    return n


def label_metrics_cli(weights, dataset_root, noise=True, device=None, max_batches=None, log=print):
    """Held-out label metrics of a v3 checkpoint on the validation split of
    `dataset_root` (unmirrored Hog rows), with the checkpoint's lead and names."""
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    if not N3.is_v3(ck):
        return {"skipped": "label metrics need a v3 checkpoint"}
    device = pick_device(device)
    net, meta = N3.load_v3(ck, device)
    ds = D.BCDataset(dataset_root, expect_obs_dim=net.obs_dim)
    ld, tg, _ = make_loader(ds, "val", int(meta["lead"]), device, mirrored=False, noise=noise,
                            names=meta.get("names") or "spells", shuffle=False, log=log)
    out = label_metrics(net, ld, what_h=meta.get("train", {}).get("what_h", 10), max_batches=max_batches)
    out.update(dataset=os.path.relpath(os.path.abspath(dataset_root), C.REPO), dataset_fp=ds.info["fp_sha"],
               lead=int(meta["lead"]), interval=tg.D, noise=bool(noise), targets=dict(tg.stats),
               trained_on_fp=meta.get("dataset_fp"))
    return out


# -------------------------------------------------------------------- train ---
def train(a, log=print):
    device = pick_device(a.device)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    cfg = {k: getattr(a, k) for k in ("tail_dim", "core_hidden", "core_layers") if getattr(a, k) is not None}
    cfg["use_tail"] = not a.no_tail
    if getattr(a, "init", None):
        # Continue from a v3 checkpoint (e.g. any-deck pretraining, then the Hog
        # rows): the net config is the checkpoint's, the CLI's net options are ignored.
        net, init_meta = N3.load_v3(torch.load(a.init, map_location="cpu", weights_only=False), device)
        for q in net.parameters():
            q.requires_grad_(True)
        log(f"init from {a.init} (cfg {net.cfg}, trained on {init_meta.get('dataset')})")
    else:
        net = N3.new_net_v3(device, **cfg)
    ds = D.BCDataset(a.data, expect_obs_dim=net.obs_dim)
    interval = dataset_grid(ds)
    groups = None if a.groups in ("", "all") else [int(g) for g in a.groups.split(",")]
    what_h = a.what_h if a.what_h in ("play", "none") else int(a.what_h)
    out_dir = os.path.join(RUNS, a.name)
    os.makedirs(out_dir, exist_ok=True)
    params = N3.param_table(net)
    log(f"NetV3 {params['total']:,} parameters, device {device}; data {a.data} (grid {interval}, "
        f"fp {ds.info['fp_sha']}); lead {a.lead}, what_h {what_h}, noise {not a.no_noise}")

    targets = Targets(ds, a.lead, interval)
    cache = None
    if not a.no_noise:
        cache = NoiseCache(ds, seed=a.noise_seed, names=a.names, p=a.noise_p)
        cache.ensure(ds.select("train", None, groups), procs=a.procs, log=log)
    vld, vtg, _ = make_loader(ds, "val", a.lead, device, groups=groups, mirrored=False, noise=not a.no_noise,
                              names=a.names, noise_seed=a.noise_seed, shuffle=False, procs=a.procs, log=log)
    est = estimate_rows(ds, "train", groups)
    steps_total = max(1, int(math.ceil(est / a.batch_size)) * a.epochs)
    if a.max_steps:
        steps_total = min(steps_total, a.max_steps)
    log(f"about {est:,} training rows per epoch, {steps_total} steps planned")
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    def lr_at(step):
        if step < a.warmup:
            return a.lr * (step + 1) / a.warmup
        x = min(1.0, (step - a.warmup) / max(1, steps_total - a.warmup))
        return a.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * x)))

    train_cfg = {"lead": a.lead, "what_h": what_h, "what_on_when": not a.no_what_on_when, "epochs": a.epochs,
                 "batch_size": a.batch_size, "lr": a.lr, "weight_decay": a.weight_decay, "seed": a.seed,
                 "noise": not a.no_noise, "noise_p": a.noise_p, "noise_seed": a.noise_seed,
                 "groups": groups, "interval": interval, "loss_weights": LOSS_W}
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"args": vars(a), "net_cfg": net.cfg, "params": params, "train": train_cfg,
                   "dataset_fp": ds.info["fp_sha"], "code_hash": N3.code_hash()}, f, indent=1)

    def save(path, **extra):
        return N3.save_v3(net, path, names=a.names if not a.no_noise else "all", decision_ticks=interval,
                          lead=int(a.lead), dataset=os.path.relpath(os.path.abspath(a.data), C.REPO),
                          dataset_fp=ds.info["fp_sha"], split="train", train=train_cfg,
                          targets=dict(targets.stats), **extra)

    step, t0, rows_seen = 0, time.time(), 0
    hist = []
    logf = open(os.path.join(out_dir, "log.jsonl"), "a")
    done = False
    for epoch in range(a.epochs):
        flt = D.RowFilter(split="train", groups=groups)
        assert flt.split == "train"
        ld = D.BCLoader(ds, batch_size=a.batch_size, device=device, flt=flt, shuffle=True,
                        seed=a.seed * 1000 + epoch, noise=cache, label_transform=targets, drop_last=True)
        net.train()
        run = Counter()
        for batch in ld:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            total, parts = forward_losses(net, batch, what_h=what_h, what_on_when=not a.no_what_on_when)
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), a.clip)
            opt.step()
            step += 1
            rows_seen += len(batch["obs"])
            if step % a.log_every == 0 or step == 1:
                for k, v in parts.items():
                    run[k] += float(v.detach())
                run["n"] += 1
            if step % (a.log_every * 10) == 0:
                el = time.time() - t0
                msg = " ".join(f"{k} {run[k] / max(run['n'], 1):.4f}" for k in LOSS_W)
                log(f"ep {epoch} step {step}/{steps_total} lr {lr_at(step):.2e} {msg} | "
                    f"{rows_seen / el:,.0f} rows/s, {el / 60:.1f} min")
                run = Counter()
            if step >= steps_total:
                done = True
                break
        vm = label_metrics(net, vld, what_h=what_h, max_batches=a.val_batches or None)
        el = time.time() - t0
        rec = {"epoch": epoch, "step": step, "rows": rows_seen, "minutes": el / 60,
               "rows_per_s": rows_seen / el, "val": vm, "targets": dict(targets.stats)}
        hist.append(rec)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        log(f"== epoch {epoch} done: {rows_seen:,} rows in {el / 60:.1f} min ({rows_seen / el:,.0f} rows/s)")
        log(format_label_metrics(dict(vm, dataset=a.data, lead=a.lead, interval=interval, noise=not a.no_noise)))
        save(os.path.join(out_dir, f"ep{epoch}.pth"), epoch=epoch, step=step, val=vm,
             train_minutes=el / 60, rows_per_s=rows_seen / el)
        if done:
            break
    blob = save(os.path.join(out_dir, "last.pth"), epoch=hist[-1]["epoch"], step=step, val=hist[-1]["val"],
                train_minutes=(time.time() - t0) / 60, rows_per_s=rows_seen / (time.time() - t0))
    logf.close()
    log(f"targets: {dict(targets.stats)}")
    log(f"wrote {os.path.join(out_dir, 'last.pth')} (code {blob['code_hash']})")
    return os.path.join(out_dir, "last.pth")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--name", required=True, help="checkpoints go to runs/bc3/<name>/")
    p.add_argument("--init", default=None, help="a v3 checkpoint to continue from (its net config is used)")
    p.add_argument("--lead", type=int, default=0, help="L: ticks between the decision row and the human play")
    p.add_argument("--what-h", default="10", help="'play', 'none' or H ticks (WHAT / WHERE rows)")
    p.add_argument("--no-what-on-when", action="store_true")
    p.add_argument("--groups", default="1,2", help="bcdata2 groups (1,2 = Hog 2.6 sides; 'all')")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-noise", action="store_true")
    p.add_argument("--noise-p", type=float, default=1.0, help="share of rows that get their cached noise")
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--names", default="spells", choices=("all", "spells", "none"), help="tail names policy")
    p.add_argument("--no-tail", action="store_true", help="ablation: the net does not read the tail")
    p.add_argument("--tail-dim", type=int, default=None)
    p.add_argument("--core-hidden", type=int, default=None)
    p.add_argument("--core-layers", type=int, default=None)
    p.add_argument("--procs", type=int, default=5)
    p.add_argument("--device", default=None)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--val-batches", type=int, default=0)
    a = p.parse_args(argv)
    train(a)


if __name__ == "__main__":
    main()
