"""Behaviour cloning of net_v2 on a BC dataset at any decision step, and an
evaluation that is comparable across decision rates.

    # train (GPU)
    .venv/bin/python -m training.bc_v2 train --run bc_v2_5hz --data hog26_5hz --minutes 90
    # held-out comparison of any checkpoints, each on its own dataset
    .venv/bin/python -m training.bc_v2 eval --weights checkpoints/bc_hog26.pth --data hog26 \
        --weights checkpoints/bc_v2_5hz.pth --data hog26_5hz --out runs/bc_v2_5hz/compare.json

Training is bc.py's recipe (whole matches in order, TBPTT with the LSTM state
carried, the PPO masks, win-weighting, relocated labels down-weighted, critic on
the discounted REAL outcome) with these differences:

  * the decision step comes from the dataset's meta.json (`decision_ticks`,
    10 if absent) and the per-decision constants are re-derived from it:
    gamma = 0.999 ** (ticks/10) (same horizon in seconds), BPTT = 50 s of rows;
  * with net_v2's wait/act gate the card loss is split: a gate BCE on every
    decision row (no-op rows weighted by w_noop = p_play/p_noop, the class
    balance bc.py uses) plus a card CE on play rows only (label smoothing over
    the LEGAL cards), so the no-op weight never touches which card is chosen;
  * the post-hoc no-op shift starts at -log(w_noop) and is then re-fitted (one
    scalar, 1-D search on the gate NLL) on a CALIBRATION split carved from the
    training matches (sha1(tag) % 40 == 1, disjoint from the validation rule
    % 20 == 0), and model selection uses the same split, so the 424 held-out
    matches are only ever used for the final report.

Evaluation (`evaluate_common`) runs any net with the shared interface on its own
dataset, and reports
  per human play: card-given-play accuracy and NLL, exact / within-one-tile cell
                  and cell NLL (teacher-forced on the human card), per card id;
  per decision row: mean P(play) vs human rate, ECE, AUC (native rate);
  per 1-s window: P(>= 1 play in the window) = 1 - prod(1 - p_k) over the
                  window's rows up to and including the human's first play
                  there (the rows after it are off the human path), against
                  "the human played in this window": NLL, ECE, AUC, and the
                  expected plays per second (sum of p_k) against the human's.
At 1 Hz a window is one row, so those reduce to the per-row numbers.
"""
import argparse
import hashlib
import json
import math
import os
import time

from training import common as C

import numpy as np
import torch
import torch.nn.functional as F

from training import bc as BC
from training import bcdata as D
from training import net_v2 as N2
from training.reward import RewardSpec

REPO = C.REPO
W = C.BOARD_W
NOOP = C.NOOP
TICKS_PER_SECOND = 10        # engine tick = 0.1 s (bcdata/il_replay_bc time axis)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def dataset_meta(name):
    p = C.data_dir("bc", name, "meta.json")
    meta = json.load(open(p)) if os.path.exists(p) else {}
    ticks = int(meta.get("decision_ticks", (meta.get("action") or {}).get("skip_frames", 10)))
    return meta, ticks


def load_data(name, max_shards=None, verbose=True):
    src = C.data_dir("bc", name)
    files = D.ensure_cached(src=src, cache=C.data_dir("bc_csr", name), verbose=verbose)
    data = D.BCData(files=files, meta_path=os.path.join(src, "meta.json"), max_shards=max_shards,
                    verbose=verbose)
    D.set_obs_dim(data.obs_dim)
    return data


def is_calib(data, e):
    tag = data.tag_names[e.tag] if data.tag_names is not None else str(e.tag)
    return int(hashlib.sha1(str(tag).encode()).hexdigest(), 16) % 40 == 1


def per_decision_constants(ticks):
    """Constants that are per decision at 1 Hz, re-derived for `ticks`."""
    k = ticks / TICKS_PER_SECOND
    gamma = 0.999 ** k
    # GAE: keep the per-second decay of gamma*lambda (0.999*0.9 at 1 Hz).
    gl = (0.999 * 0.9) ** k
    return {"decision_ticks": ticks, "rows_per_second": TICKS_PER_SECOND / ticks,
            "gamma": gamma, "gae_lambda": gl / gamma,
            "horizon_seconds": k / (1 - gamma), "gae_lookahead_seconds": k / (1 - gl),
            "bptt_rows_50s": int(round(50 / k)),
            "card_rotation_rows": 30 / k}


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def v2_losses(net, b, out, smooth=0.05, cell_smooth=0.1, w_noop=1.0, lost_weight=1.0,
              value_coef=0.5, relocated_weight=0.3):
    if not getattr(net, "gate", False):
        return BC.bc_losses(net, b, out, smooth=smooth, cell_smooth=cell_smooth, w_noop=w_noop,
                            lost_weight=lost_weight, value_coef=value_coef, ability_coef=0.0,
                            relocated_weight=relocated_weight)
    card, decision = out["card"], out["decision"]
    won = b["won"].reshape(-1)
    row_w = torch.where(won > 0.5, torch.ones_like(won), torch.full_like(won, lost_weight))
    st = {}
    loss = torch.zeros((), device=card.device)
    dec = decision.nonzero(as_tuple=True)[0]
    if dec.numel():
        lp = out["card_logits"][dec].float()
        mask = out["card_mask"][dec]
        lp_noop = lp[:, NOOP]
        lp_cards = lp[:, :NOOP]
        lp_play = torch.logsumexp(lp_cards, -1)
        y = (card[dec] != NOOP)
        nll_gate = -torch.where(y, lp_play, lp_noop)
        wg = row_w[dec] * torch.where(y, torch.ones_like(nll_gate), torch.full_like(nll_gate, w_noop))
        loss_gate = (nll_gate * wg).sum() / wg.sum().clamp(min=1e-6)
        loss = loss + loss_gate
        st["loss_gate"] = float(loss_gate.detach())
        if y.any():
            pr = y.nonzero(as_tuple=True)[0]
            lc = (lp_cards[pr] - lp_play[pr, None])
            legal = mask[pr, :NOOP]
            lc = lc.masked_fill(~legal, 0.0)
            n_leg = legal.sum(-1, keepdim=True).float()
            tgt = F.one_hot(card[dec][pr], NOOP).float() * (1 - smooth) + smooth * legal.float() / n_leg
            ce = -(tgt * lc).sum(-1)
            wc = row_w[dec][pr]
            loss_card = (ce * wc).sum() / wc.sum().clamp(min=1e-6)
            loss = loss + loss_card
            st["loss_card"] = float(loss_card.detach())
    pr = out["play_rows"]
    if pr.numel():
        pl = out["place_logits"][pr].float()
        legal = torch.isfinite(pl)
        cell = b["cell"].reshape(-1)[pr]
        ok = legal.gather(1, cell.view(-1, 1)).squeeze(1)
        if ok.any():
            pl, legal, cell = pl[ok], legal[ok], cell[ok]
            rw = row_w[pr][ok]
            rel = b["relocated"].reshape(-1)[pr][ok]
            rw = rw * (1.0 - (1.0 - relocated_weight) * rel)
            nb = BC._neighbour_target(cell, legal)
            nb_sum = nb.sum(-1, keepdim=True)
            tgt = F.one_hot(cell, pl.shape[1]).float()
            tgt = torch.where(nb_sum > 0, tgt * (1 - cell_smooth) + cell_smooth * nb / nb_sum.clamp(min=1), tgt)
            logp = torch.log_softmax(pl, -1).masked_fill(~legal, 0.0)
            ce = -(tgt * logp).sum(-1)
            loss_place = (ce * rw).sum() / rw.sum().clamp(min=1e-6)
            loss = loss + loss_place
            st["loss_place"] = float(loss_place.detach())
    v = out["valid"]
    if v.any():
        loss_v = F.mse_loss(out["values"][v].float(), b["returns"].reshape(-1)[v])
        loss = loss + value_coef * loss_v
        st["loss_value"] = float(loss_v.detach())
    return loss, st


# ---------------------------------------------------------------------------
# Evaluation, comparable across decision rates
# ---------------------------------------------------------------------------
def _auc(p, y):
    p, y = np.asarray(p, np.float64), np.asarray(y, bool)
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    ps = p[order]
    # average ranks for ties
    i = 0
    r = np.arange(1, len(p) + 1, dtype=np.float64)
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[i]:
            j += 1
        r[i:j + 1] = (i + j + 2) / 2.0
        i = j + 1
    ranks[order] = r
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _ece(p, y, bins=10):
    p, y = np.asarray(p, np.float64), np.asarray(y, np.float64)
    bi = np.minimum((p * bins).astype(int), bins - 1)
    e = 0.0
    for k in range(bins):
        m = bi == k
        if m.any():
            e += abs(p[m].sum() - y[m].sum())
    return float(e / max(1, len(p)))


def _bce(p, y):
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    y = np.asarray(y, np.float64)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


@torch.no_grad()
def collect(net, episodes, device, obs_dim, rows_per_second, B=32, noop_shift=0.0, amp=False):
    """Per-row arrays over `episodes` (whole matches, LSTM carried)."""
    net.eval()
    L = int(round(50 * rows_per_second))
    rng = np.random.default_rng(0)
    rec = {k: [] for k in ("ep", "t", "dec", "p_play", "y", "card", "card_pred", "card_nll", "cid",
                           "cell", "cell_pred", "cell_nll", "w")}
    hidden = None
    for b, first in _chunk_batches_tagged(episodes, B, L, rng, device, obs_dim):
        Lb, Bb = b["card"].shape
        if first:
            hidden = BC.zero_hidden(net, Bb, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            out, hidden = BC.forward_chunk(net, b, hidden)
        lp = torch.log_softmax(out["card_logits"].float(), -1)
        if noop_shift:
            lp = lp.clone()
            lp[:, NOOP] = lp[:, NOOP] + noop_shift
            lp = torch.log_softmax(lp, -1)
        dec = out["decision"]
        valid = out["valid"]
        p_play = (1 - lp[:, NOOP].exp()).clamp(0, 1)
        card = out["card"]
        lc = lp[:, :NOOP].masked_fill(~out["card_mask"][:, :NOOP], float("-inf"))
        lc = lc - torch.logsumexp(lc, -1, keepdim=True)
        card_pred = lc.argmax(-1)
        card_nll = -lc.gather(1, card.clamp(max=NOOP - 1).view(-1, 1)).squeeze(1)
        ids = net.hand_card_ids(out["flat_obs"])
        cid = ids.gather(1, card.clamp(max=NOOP - 1).view(-1, 1)).squeeze(1)
        n = Lb * Bb
        cell_pred = torch.full((n,), -1, dtype=torch.long, device=device)
        cell_nll = torch.full((n,), float("nan"), device=device)
        pr = out["play_rows"]
        if pr.numel():
            pl = torch.log_softmax(out["place_logits"][pr].float(), -1)
            cell = b["cell"].reshape(-1)[pr]
            cell_pred[pr] = pl.argmax(-1)
            cell_nll[pr] = -pl.gather(1, cell.view(-1, 1)).squeeze(1)
        # episode id / time per row: rows are (l, j) flattened l*B + j
        t_idx = (torch.arange(Lb, device=device).view(-1, 1).expand(Lb, Bb)).reshape(-1)
        rec["dec"].append(dec.cpu().numpy())
        rec["p_play"].append(p_play.cpu().numpy())
        rec["y"].append((card != NOOP).cpu().numpy())
        rec["card"].append(card.cpu().numpy())
        rec["card_pred"].append(card_pred.cpu().numpy())
        rec["card_nll"].append(card_nll.cpu().numpy())
        rec["cid"].append(cid.cpu().numpy())
        rec["cell"].append(b["cell"].reshape(-1).cpu().numpy())
        rec["cell_pred"].append(cell_pred.cpu().numpy())
        rec["cell_nll"].append(cell_nll.cpu().numpy())
        rec["w"].append(valid.cpu().numpy())
        rec["t"].append(t_idx.cpu().numpy() + b["_t0"])
        rec["ep"].append(np.tile(b["_ep"], Lb))
    net.train()
    return {k: np.concatenate(v) for k, v in rec.items()}


def _chunk_batches_tagged(episodes, B, L, rng, device, obs_dim, shuffle=False):
    """bc.chunk_batches plus the episode index and chunk start on each batch."""
    order = np.arange(len(episodes))
    groups = [order[i:i + B * 8] for i in range(0, len(order), B * 8)]
    for g in groups:
        g = sorted(g, key=lambda i: episodes[i].T)
        for idx in [g[i:i + B] for i in range(0, len(g), B)]:
            eps = [episodes[i] for i in idx]
            T = max(e.T for e in eps)
            for t0 in range(0, T, L):
                b = D.batch_tensors(eps, t0, L, device, obs_dim)
                last = np.zeros((L, len(eps)), np.float32)
                for j, e in enumerate(eps):
                    if t0 <= e.T - 1 < t0 + L:
                        last[e.T - 1 - t0, j] = 1.0
                b["last"] = torch.from_numpy(last).to(device)
                b["_ep"] = np.asarray(idx, np.int64)
                b["_t0"] = t0
                yield b, t0 == 0


def summarize(r, rows_per_second, card_names=None):
    v = r["w"] > 0
    dec = r["dec"] & v
    m = {}
    # ---- per decision row (native rate)
    p, y = r["p_play"][dec], r["y"][dec]
    m["rows_per_second"] = rows_per_second
    m["n_decision_rows"] = int(dec.sum())
    m["row_mean_p_play"] = float(p.mean())
    m["row_human_play_rate"] = float(y.mean())
    m["row_ece"] = _ece(p, y)
    m["row_auc"] = _auc(p, y)
    m["row_play_nll"] = _bce(p, y)
    # ---- per 1-s window
    rps = int(round(rows_per_second))
    ep, t = r["ep"][dec], r["t"][dec]
    win = t // rps
    key = ep * 100000 + win
    order = np.lexsort((t, key))
    key_s, p_s, y_s, t_s = key[order], p[order], y[order], t[order]
    starts = np.r_[0, np.nonzero(np.diff(key_s))[0] + 1]
    ends = np.r_[starts[1:], len(key_s)]
    pw, pw_trunc, yw, ew, hw = [], [], [], [], []
    for a, bnd in zip(starts, ends):
        yy = y_s[a:bnd]
        pp = p_s[a:bnd]
        first = np.nonzero(yy)[0]
        if len(first):
            j = first[0]
            # P(>= 1 play in the window) needs the hazard on the steps after the
            # human's play had they NOT played, which is unobserved (the rows
            # after it are post-play states). Truncating at the play biases play
            # windows low (AUC < 0.5 at 5 Hz); instead hold the hazard at the
            # play step's value for the rest of the window. A no-op at 1 Hz.
            rest = rps - 1 - int(t_s[a + j] % rps)
            q = np.prod(1 - pp[:j + 1]) * (1 - pp[j]) ** rest
            pw_trunc.append(1 - np.prod(1 - pp[:j + 1]))
        else:
            q = np.prod(1 - pp)
            pw_trunc.append(1 - q)
        pw.append(1 - q)
        yw.append(bool(len(first)))
        ew.append(pp.sum())
        hw.append(yy.sum())
    pw, yw = np.asarray(pw), np.asarray(yw)
    m["sec_auc_truncated"] = _auc(np.asarray(pw_trunc), yw)
    m["n_windows"] = int(len(pw))
    m["sec_mean_p_play"] = float(pw.mean())
    m["sec_human_play_rate"] = float(yw.mean())
    m["sec_ece"] = _ece(pw, yw)
    m["sec_auc"] = _auc(pw, yw)
    m["sec_play_nll"] = _bce(pw, yw)
    m["sec_expected_plays"] = float(np.mean(ew))
    m["sec_human_plays"] = float(np.mean(hw))
    # ---- per human play
    play = dec & r["y"]
    m["n_plays"] = int(play.sum())
    m["card_given_play"] = float((r["card_pred"][play] == r["card"][play]).mean())
    m["card_nll_given_play"] = float(np.nanmean(r["card_nll"][play]))
    cp = r["cell_pred"][play]
    ok = cp >= 0
    cell = r["cell"][play][ok]
    cp = cp[ok]
    exact = cp == cell
    within = (np.abs(cp // W - cell // W) <= 1) & (np.abs(cp % W - cell % W) <= 1)
    m["n_cell"] = int(ok.sum())
    m["cell_exact"] = float(exact.mean())
    m["cell_within1"] = float(within.mean())
    m["cell_nll"] = float(np.nanmean(r["cell_nll"][play][ok]))
    m["per_play_nll_card_plus_cell"] = m["card_nll_given_play"] + m["cell_nll"]
    # timing NLL per play (window-level), so the three add to a per-play total
    m["sec_play_nll_per_play"] = m["sec_play_nll"] * len(pw) / max(1, m["n_plays"])
    # Timing NLL per human play at 0.2 s resolution, needing no counterfactual: the
    # native per-row likelihood, plus log(5 / rows_per_second) per play for a
    # coarser model (a 1 Hz hazard spread evenly over the second's five 0.2 s
    # steps puts ~1/5 of a play window's mass on the step the human used).
    m["timing_nll_per_play_native"] = m["row_play_nll"] * m["n_decision_rows"] / max(1, m["n_plays"])
    m["timing_nll_per_play_at_0p2s"] = m["timing_nll_per_play_native"] + float(np.log(5.0 / rows_per_second))
    cid = r["cid"][play]
    per = {}
    for c in np.unique(cid):
        mc = cid == c
        mo = mc[ok]
        name = card_names.get(int(c), str(int(c))) if card_names else str(int(c))
        per[name] = {"n": int(mc.sum()),
                     "card_recall": float((r["card_pred"][play][mc] == r["card"][play][mc]).mean()),
                     "cell_exact": float(exact[mo].mean()) if mo.any() else None,
                     "cell_within1": float(within[mo].mean()) if mo.any() else None,
                     "cell_nll": float(np.nanmean(r["cell_nll"][play][ok][mo])) if mo.any() else None}
    m["per_card"] = per
    return m


def card_names():
    out = {}
    for cid in C.deck_ids():
        try:
            out[int(cid)] = C.E.get_card_info(int(cid))["name"]
        except Exception:  # noqa: BLE001
            pass
    return out


def evaluate_common(net, episodes, device, obs_dim, rows_per_second, noop_shift=0.0, amp=False, B=32):
    r = collect(net, episodes, device, obs_dim, rows_per_second, B=B, noop_shift=noop_shift, amp=amp)
    return summarize(r, rows_per_second, card_names()), r


def fit_noop_shift(r, lo=-3.0, hi=3.0, n=121):
    """Scalar s added to log-odds(noop) minimising the per-row gate NLL."""
    v = (r["w"] > 0) & r["dec"]
    p = np.clip(r["p_play"][v].astype(np.float64), 1e-7, 1 - 1e-7)
    y = r["y"][v]
    logit = np.log(p) - np.log1p(-p)
    best = (None, 1e9)
    for s in np.linspace(lo, hi, n):
        q = 1 / (1 + np.exp(-(logit - s)))
        nll = _bce(q, y)
        if nll < best[1]:
            best = (float(s), nll)
    return best


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(a):
    device = torch.device(a.device)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    if device.type == "cuda":
        # Off by default: the placement head runs on the play rows only, a different
        # count every batch, and benchmark mode re-autotunes every new shape.
        torch.backends.cudnn.benchmark = a.cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    log_dir = os.path.join(REPO, "runs", a.run)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(REPO, "checkpoints"), exist_ok=True)
    mlog = open(os.path.join(log_dir, "bc_metrics.jsonl"), "a")
    _, ticks = dataset_meta(a.data)
    K = per_decision_constants(ticks)
    rps = K["rows_per_second"]
    bptt = a.bptt or K["bptt_rows_50s"]
    print("per-decision constants:", json.dumps(K), "| bptt", bptt, flush=True)

    data = load_data(a.data, max_shards=a.max_shards)
    calib = [e for e in data.train if is_calib(data, e)]
    train_eps = [e for e in data.train if not is_calib(data, e)]
    print(f"train {len(train_eps)} / calib {len(calib)} / val {len(data.val)} matches", flush=True)
    spec = RewardSpec(gamma=K["gamma"])
    data.compute_returns(spec)

    n_noop = sum(int((e.card == NOOP).sum()) for e in train_eps)
    n_all = sum(e.T for e in train_eps)
    p_noop = n_noop / n_all
    w_noop = a.w_noop if a.w_noop is not None else float(min(1.0, (1 - p_noop) / p_noop))
    shift0 = BC.card_noop_shift(w_noop)
    print(f"train rows {n_all}, no-op frac {p_noop:.4f}; w_noop {w_noop:.4f} (analytic shift {shift0:+.3f})",
          flush=True)

    cfg = json.loads(a.cfg) if a.cfg else {}
    net = N2.new_net_v2(device, **cfg)
    if device.type == "cuda" and a.channels_last:
        net = net.to(memory_format=torch.channels_last)
    print("params:", json.dumps(N2.param_table(net)), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    loss_kw = dict(smooth=a.smooth, cell_smooth=a.cell_smooth, w_noop=w_noop, lost_weight=a.lost_weight,
                   value_coef=a.value_coef)
    amp = device.type == "cuda" and not a.no_amp
    deadline = time.time() + a.minutes * 60
    total = a.minutes * 60
    ck = os.path.join(REPO, "checkpoints", f"{a.run}.pth")
    step, epoch, best = 0, 0, float("inf")
    done = False
    t_start = time.time()
    rows_seen = 0
    while not done:
        epoch += 1
        hidden = None
        t_ep = time.time()
        n_rows = 0
        gen = BC.chunk_batches(train_eps, a.batch, bptt, rng, device, data.obs_dim)
        for b, first in D.Prefetcher(gen):
            if first:
                hidden = BC.zero_hidden(net, b["obs"].shape[1], device)
            else:
                hidden = (hidden[0].detach(), hidden[1].detach())
            frac = min(1.0, (time.time() - t_start) / total)
            warm = min(1.0, step / 200.0)
            for g in opt.param_groups:
                g["lr"] = a.lr * warm * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out, hidden = BC.forward_chunk(net, b, hidden)
                loss, st = v2_losses(net, b, out, **loss_kw)
            hidden = (hidden[0].float(), hidden[1].float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            step += 1
            nr = int(b["valid"].sum())
            n_rows += nr
            rows_seen += nr
            if step % 50 == 0:
                print(f"ep {epoch} step {step} loss {float(loss):.4f} " +
                      " ".join(f"{k} {v:.4f}" for k, v in st.items() if k.startswith("loss")) +
                      f" | {n_rows / (time.time() - t_ep):.0f} rows/s", flush=True)
            last = time.time() > deadline
            if step % a.eval_every == 0 or last:
                m, r = evaluate_common(net, calib, device, data.obs_dim, rps, noop_shift=shift0, amp=amp)
                score = m["per_play_nll_card_plus_cell"] + m["sec_play_nll_per_play"]
                m.update(step=step, epoch=epoch, rows_seen=rows_seen, wall=time.time(), split="calib",
                         score=score)
                mlog.write(json.dumps({k: v for k, v in m.items() if k != "per_card"}) + "\n")
                mlog.flush()
                print(f"  CALIB step {step} ep {epoch}: card|play {m['card_given_play']:.3f} cell "
                      f"{m['cell_exact']:.3f} w1 {m['cell_within1']:.3f} p_play/s {m['sec_mean_p_play']:.3f} "
                      f"vs {m['sec_human_play_rate']:.3f} sec_auc {m['sec_auc']:.3f} "
                      f"nll/play card {m['card_nll_given_play']:.3f} cell {m['cell_nll']:.3f} "
                      f"timing {m['sec_play_nll_per_play']:.3f} score {score:.4f}", flush=True)
                if score < best:
                    best = score
                    N2.save_v2(net, ck, {"noop_shift_analytic": shift0, "w_noop": w_noop, "step": step,
                                         "calib_at_save": {k: v for k, v in m.items() if k != "per_card"},
                                         "constants": K, "args": vars(a), "deck": C.deck_ids()})
                if last:
                    done = True
                    break
        print(f"epoch {epoch} done in {time.time() - t_ep:.0f}s ({n_rows / (time.time() - t_ep):.0f} rows/s)",
              flush=True)
        if epoch >= a.epochs:
            break
    # Re-fit the shift on calib with the best weights, bake it in, report val.
    net = N2.load_any(ck, device)
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    _, r = evaluate_common(net, calib, device, data.obs_dim, rps, noop_shift=0.0, amp=amp)
    s_fit, nll_fit = fit_noop_shift(r)
    print(f"no-op shift: analytic {shift0:+.3f}; fitted on calib (on top of raw) {s_fit:+.3f} "
          f"(gate nll {nll_fit:.4f})", flush=True)
    with torch.no_grad():
        net.card_head.bias[NOOP] += s_fit
    N2.save_v2(net, ck, {k: v for k, v in blob.items() if k not in ("arch", "cfg", "model")} |
               {"noop_shift_baked": s_fit})
    m, _ = evaluate_common(net, data.val, device, data.obs_dim, rps, amp=amp)
    m.update(final=True, split="val", noop_shift_baked=s_fit)
    mlog.write(json.dumps(m) + "\n")
    mlog.flush()
    json.dump(m, open(os.path.join(log_dir, "val_final.json"), "w"), indent=1)
    print("FINAL VAL:", json.dumps({k: v for k, v in m.items() if k != "per_card"}), flush=True)


def cmd_eval(a):
    device = torch.device(a.device)
    res = {}
    cache = {}
    for w, d in zip(a.weights, a.data):
        if d not in cache:
            cache[d] = load_data(d, max_shards=a.max_shards)
        data = cache[d]
        _, ticks = dataset_meta(d)
        rps = TICKS_PER_SECOND / ticks
        net = N2.load_any(w, device)
        eps = data.val[:a.max_eps] if a.max_eps else data.val
        t0 = time.time()
        m, _ = evaluate_common(net, eps, device, data.obs_dim, rps, amp=device.type == "cuda" and not a.no_amp)
        m["eval_s"] = time.time() - t0
        m["n_matches"] = len(eps)
        res[f"{os.path.basename(w)}@{d}"] = m
        print(w, d, json.dumps({k: v for k, v in m.items() if k != "per_card"}), flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--run", default="bc_v2_5hz")
    t.add_argument("--data", default="hog26_5hz")
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    t.add_argument("--minutes", type=float, default=90)
    t.add_argument("--epochs", type=int, default=1000)
    t.add_argument("--batch", type=int, default=16, help="matches per batch")
    t.add_argument("--bptt", type=int, default=None, help="rows; default 50 s of rows")
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--wd", type=float, default=1e-4)
    t.add_argument("--smooth", type=float, default=0.05)
    t.add_argument("--cell-smooth", type=float, default=0.1)
    t.add_argument("--w-noop", type=float, default=None)
    t.add_argument("--lost-weight", type=float, default=0.5)
    t.add_argument("--value-coef", type=float, default=0.5)
    t.add_argument("--eval-every", type=int, default=1000)
    t.add_argument("--max-shards", type=int, default=None)
    t.add_argument("--cfg", default=None, help="net_v2 config overrides, JSON")
    t.add_argument("--channels-last", action="store_true")
    t.add_argument("--cudnn-benchmark", action="store_true")
    t.add_argument("--no-amp", action="store_true")
    t.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("eval")
    e.add_argument("--weights", action="append", required=True)
    e.add_argument("--data", action="append", required=True)
    e.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    e.add_argument("--max-eps", type=int, default=None)
    e.add_argument("--max-shards", type=int, default=None)
    e.add_argument("--no-amp", action="store_true")
    e.add_argument("--out", default=None)
    a = p.parse_args()
    train(a) if a.cmd == "train" else cmd_eval(a)


if __name__ == "__main__":
    main()
