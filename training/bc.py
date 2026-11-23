"""Recurrent behaviour cloning of human Hog 2.6 play onto the PPO network.

    .venv/bin/python -m training.bc --run bc_hog26 --minutes 40 [--device cuda]

What it trains (MicroRoyaleNet, the exact net PPO builds; same masks as PPO):
  * whole matches, in order, as truncated-BPTT chunks of `--bptt` rows; the LSTM
    state is carried across chunks of the same match (detached), zeroed at the
    match start -- the policy learns WHEN to play, not just what;
  * card head: cross-entropy on decision rows (>1 legal arm under
    net.affordability_mask), label smoothing `--smooth` spread over the LEGAL
    arms only, no-op rows down-weighted by `--w-noop` (class balance) and the
    bias shift log(1/w_noop) added back to the no-op logit at save time, so the
    saved policy's P(play) is calibrated to the human rate (prior-shift
    correction); both the raw and the corrected play rate are reported;
  * placement head: teacher-forced on the human card, CE over the legal cells
    (net.placement_mask) with `--cell-smooth` mass on the 8 legal neighbours;
  * ability head (Hero Musketeer): CE on rows where the ability was ready;
  * value head: MSE to the MC return of the minimal reward (training/reward.py)
    on the human outcome -- a warm critic for PPO;
  * `--lost-weight`: weight on rows of matches the 2.6 player lost (1 = plain
    BC, <1 = win-weighted). Pick by validation (`--sweep-lost` runs both).

Validation = meta.json's held-out replay tags (sha1(tag) % 20 == 0).
Outputs: checkpoints/<run>.pth (bare state_dict, loads as CLASH_WEIGHTS),
checkpoints/<run>.meta.json, runs/<run>/bc_metrics.jsonl, runs/<run>.log.
"""
import argparse
import json
import math
import os
import time

from training import common as C

import numpy as np
import torch
import torch.nn.functional as F

from training import bcdata as D
from training.reward import RewardSpec

REPO = C.REPO
W = C.BOARD_W


# ---------------------------------------------------------------------------
# Forward pass + losses (also used by PPO's BC-auxiliary term)
# ---------------------------------------------------------------------------
def _neighbour_target(cell, legal):
    """(N, cells) soft target: 1 at `cell`, plus 8-neighbours, restricted to legal."""
    N, P = legal.shape
    r, c = cell // W, cell % W
    tgt = torch.zeros(N, P, device=cell.device)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            ok = (rr >= 0) & (rr < P // W) & (cc >= 0) & (cc < W)
            idx = (rr.clamp(0, P // W - 1) * W + cc.clamp(0, W - 1))
            tgt[torch.arange(N, device=cell.device), idx] += ok.float()
    tgt = tgt * legal.float()
    return tgt


def forward_chunk(net, b, hidden):
    """Run one (L, B) chunk. Returns (outputs dict, new hidden)."""
    obs = b["obs"]
    L, B, Dm = obs.shape
    flat = obs.reshape(L * B, Dm)
    feats, embeds, spatial, hires = net.extract_features_hires(flat)
    card_mask = net.affordability_mask(flat)
    n_legal = card_mask.sum(-1)
    card = b["card"].reshape(-1)
    valid = b["valid"].reshape(-1) > 0
    label_legal = card_mask.gather(1, card.view(-1, 1)).squeeze(1)
    is_play = card != C.NOOP
    decision = valid & (n_legal > 1) & label_legal
    play_rows = (decision & is_play).nonzero(as_tuple=True)[0]
    # reset_seq: 1 inside a match, 0 after its last row (state zeroed after).
    reset = b["valid"].clone()
    if "last" in b:
        reset = reset * (1 - b["last"])
    fwd = net.forward_sequence(
        feats.view(L, B, -1), embeds.view(L, B, net.hand_size + 1, -1),
        spatial.view(L, B, *spatial.shape[1:]), obs, card_mask.view(L, B, -1),
        b["card"], reset, hidden, hires_seq=hires.view(L, B, *hires.shape[1:]),
        active_rows=play_rows, with_ability=bool(net.num_ability_slots))
    out = {"card_logits": fwd[0].reshape(L * B, -1), "place_logits": fwd[1].reshape(L * B, -1),
           "values": fwd[2].reshape(-1), "card_mask": card_mask, "decision": decision,
           "play_rows": play_rows, "valid": valid, "card": card, "flat_obs": flat,
           "n_legal": n_legal, "label_legal": label_legal}
    if len(fwd) > 6 and fwd[6]:
        out["ability_logits"] = fwd[6][0].reshape(L * B, 2)
    return out, fwd[4]


def bc_losses(net, b, out, smooth=0.05, cell_smooth=0.1, w_noop=1.0, lost_weight=1.0,
              value_coef=0.5, ability_coef=0.5, relocated_weight=0.3):
    """Scalar loss + a dict of detached stats for one chunk."""
    card, decision, play_rows = out["card"], out["decision"], out["play_rows"]
    won = b["won"].reshape(-1)
    row_w = torch.where(won > 0.5, torch.ones_like(won), torch.full_like(won, lost_weight))
    st = {}
    # ---- card head
    dec = decision.nonzero(as_tuple=True)[0]
    loss = torch.zeros((), device=card.device)
    if dec.numel():
        logits = out["card_logits"][dec]
        mask = out["card_mask"][dec]
        logp = torch.log_softmax(logits, -1).masked_fill(~mask, 0.0)
        n_leg = mask.sum(-1, keepdim=True).float()
        tgt = F.one_hot(card[dec], net.hand_size + 1).float() * (1 - smooth) + smooth * mask.float() / n_leg
        ce = -(tgt * logp).sum(-1)
        w = row_w[dec] * torch.where(card[dec] == C.NOOP, torch.full_like(ce, w_noop), torch.ones_like(ce))
        loss_card = (ce * w).sum() / w.sum().clamp(min=1e-6)
        loss = loss + loss_card
        st["loss_card"] = float(loss_card.detach())
        st["n_dec"] = int(dec.numel())
    # ---- placement head
    if play_rows.numel():
        pl = out["place_logits"][play_rows]
        legal = torch.isfinite(pl)
        cell = b["cell"].reshape(-1)[play_rows]
        ok = legal.gather(1, cell.view(-1, 1)).squeeze(1)
        st["cell_illegal_frac"] = float(1 - ok.float().mean())
        if ok.any():
            pl, legal, cell = pl[ok], legal[ok], cell[ok]
            rows_w = row_w[play_rows][ok]
            if "relocated" in b:
                # Legal-but-moved labels (the real cell was illegal in the sim).
                rel = b["relocated"].reshape(-1)[play_rows][ok]
                rows_w = rows_w * (1.0 - (1.0 - relocated_weight) * rel)
            nb = _neighbour_target(cell, legal)
            nb_sum = nb.sum(-1, keepdim=True)
            tgt = F.one_hot(cell, pl.shape[1]).float()
            tgt = torch.where(nb_sum > 0, tgt * (1 - cell_smooth) + cell_smooth * nb / nb_sum.clamp(min=1), tgt)
            logp = torch.log_softmax(pl, -1).masked_fill(~legal, 0.0)
            ce = -(tgt * logp).sum(-1)
            loss_place = (ce * rows_w).sum() / rows_w.sum().clamp(min=1e-6)
            loss = loss + loss_place
            st["loss_place"] = float(loss_place.detach())
    # ---- ability head
    if "ability_logits" in out:
        ready = (b["ability_ready"].reshape(-1) > 0) & out["valid"]
        r = ready.nonzero(as_tuple=True)[0]
        if r.numel():
            lab = b["ability"].reshape(-1)[r]
            ce = F.cross_entropy(out["ability_logits"][r], lab, reduction="none")
            # Activations are rare (~1 in 10 ready rows): balance the two arms.
            pos = lab.float()
            wa = torch.where(pos > 0.5, torch.full_like(ce, 3.0), torch.ones_like(ce))
            loss_ab = (ce * wa).sum() / wa.sum()
            loss = loss + ability_coef * loss_ab
            st["loss_ability"] = float(loss_ab.detach())
    # ---- value head
    v = out["valid"]
    if v.any():
        loss_v = F.mse_loss(out["values"][v], b["returns"].reshape(-1)[v])
        loss = loss + value_coef * loss_v
        st["loss_value"] = float(loss_v.detach())
    return loss, st


def chunk_batches(episodes, B, L, rng, device, obs_dim, shuffle=True):
    """Yield (batch dict, is_first_chunk) over groups of B whole matches."""
    order = np.arange(len(episodes))
    if shuffle:
        rng.shuffle(order)
    # Group similar lengths so padding stays small.
    groups = [order[i:i + B * 8] for i in range(0, len(order), B * 8)]
    for g in groups:
        g = sorted(g, key=lambda i: episodes[i].T)
        sub = [g[i:i + B] for i in range(0, len(g), B)]
        if shuffle:
            rng.shuffle(sub)
        for idx in sub:
            eps = [episodes[i] for i in idx]
            T = max(e.T for e in eps)
            for t0 in range(0, T, L):
                b = D.batch_tensors(eps, t0, L, device, obs_dim)
                last = np.zeros((L, len(eps)), np.float32)
                for j, e in enumerate(eps):
                    if t0 <= e.T - 1 < t0 + L:
                        last[e.T - 1 - t0, j] = 1.0
                b["last"] = torch.from_numpy(last).to(device)
                yield b, t0 == 0


def zero_hidden(net, B, device):
    h = torch.zeros(B, net.LSTM_HIDDEN, device=device)
    return (h, h.clone())


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(net, episodes, device, obs_dim, B=64, L=50, modal=None, noop_shift=0.0,
             max_eps=None, loss_kw=None):
    net.eval()
    rng = np.random.default_rng(0)
    eps = episodes[:max_eps] if max_eps else episodes
    acc = dict(n_dec=0, card_hit=0, n_play=0, card_hit_play=0, cell_hit=0, within1=0,
               cell_n=0, modal_hit=0, p_play_sum=0.0, human_play=0, argmax_play=0,
               ab_n=0, ab_hit=0, ab_pos=0, ab_pred=0, noop_base_hit=0)
    losses = {}
    nb = 0
    bins = np.zeros((10, 3))  # sum p, sum y, n
    hidden = None
    for b, first in chunk_batches(eps, B, L, rng, device, obs_dim, shuffle=False):
        if first:
            hidden = zero_hidden(net, b["obs"].shape[1], device)
        out, hidden = forward_chunk(net, b, hidden)
        if loss_kw is not None:
            _, st = bc_losses(net, b, out, **loss_kw)
            for k, v in st.items():
                if k.startswith("loss"):
                    losses[k] = losses.get(k, 0.0) + v
            nb += 1
        dec = out["decision"].nonzero(as_tuple=True)[0]
        if dec.numel():
            lg = out["card_logits"][dec].clone()
            lg[:, C.NOOP] += noop_shift
            p = torch.softmax(lg, -1)
            lab = out["card"][dec]
            pred = p.argmax(-1)
            acc["n_dec"] += dec.numel()
            acc["card_hit"] += int((pred == lab).sum())
            acc["noop_base_hit"] += int((lab == C.NOOP).sum())
            pp = 1 - p[:, C.NOOP]
            y = (lab != C.NOOP).float()
            acc["p_play_sum"] += float(pp.sum())
            acc["human_play"] += int(y.sum())
            acc["argmax_play"] += int((pred != C.NOOP).sum())
            bi = (pp * 10).clamp(max=9.999).long().cpu().numpy()
            np.add.at(bins[:, 0], bi, pp.cpu().numpy())
            np.add.at(bins[:, 1], bi, y.cpu().numpy())
            np.add.at(bins[:, 2], bi, 1)
            # Card choice given that the human played (which card, not when).
            playm = lab != C.NOOP
            if playm.any():
                lp = lg[playm].clone()
                lp[:, C.NOOP] = float("-inf")
                acc["n_play"] += int(playm.sum())
                acc["card_hit_play"] += int((lp.argmax(-1) == lab[playm]).sum())
        pr = out["play_rows"]
        if pr.numel():
            pl = out["place_logits"][pr]
            cell = b["cell"].reshape(-1)[pr]
            pred = pl.argmax(-1)
            acc["cell_n"] += pr.numel()
            acc["cell_hit"] += int((pred == cell).sum())
            acc["within1"] += int((((pred // W) - (cell // W)).abs() <= 1)
                                  .logical_and(((pred % W) - (cell % W)).abs() <= 1).sum())
            if modal is not None:
                ids = net.hand_card_ids(out["flat_obs"][pr])
                cid = ids.gather(1, out["card"][pr].view(-1, 1)).squeeze(1).cpu().numpy()
                mc = np.array([modal.get(int(c), -1) for c in cid])
                acc["modal_hit"] += int((mc == cell.cpu().numpy()).sum())
        if "ability_logits" in out:
            r = ((b["ability_ready"].reshape(-1) > 0) & out["valid"]).nonzero(as_tuple=True)[0]
            if r.numel():
                lab = b["ability"].reshape(-1)[r]
                pa = out["ability_logits"][r].argmax(-1)
                acc["ab_n"] += r.numel()
                acc["ab_hit"] += int((pa == lab).sum())
                acc["ab_pos"] += int(lab.sum())
                acc["ab_pred"] += int(pa.sum())
    net.train()
    n = max(1, acc["n_dec"])
    m = {
        "card_match": acc["card_hit"] / n,
        "card_match_baseline_always_noop": acc["noop_base_hit"] / n,
        "card_match_given_play": acc["card_hit_play"] / max(1, acc["n_play"]),
        "cell_match": acc["cell_hit"] / max(1, acc["cell_n"]),
        "cell_within1": acc["within1"] / max(1, acc["cell_n"]),
        "cell_match_baseline_modal": acc["modal_hit"] / max(1, acc["cell_n"]) if modal else None,
        "human_play_rate": acc["human_play"] / n,
        "bc_mean_p_play": acc["p_play_sum"] / n,
        "bc_argmax_play_rate": acc["argmax_play"] / n,
        "calibration_ece_play": float(np.abs(bins[:, 0] - bins[:, 1]).sum() / max(1, bins[:, 2].sum())),
        "calibration_bins": [[float(bins[i, 0] / bins[i, 2]), float(bins[i, 1] / bins[i, 2]), int(bins[i, 2])]
                             for i in range(10) if bins[i, 2] > 0],
        "ability_acc": acc["ab_hit"] / max(1, acc["ab_n"]),
        "ability_human_rate": acc["ab_pos"] / max(1, acc["ab_n"]),
        "ability_pred_rate": acc["ab_pred"] / max(1, acc["ab_n"]),
        "n_decision_rows": acc["n_dec"], "n_play_rows": acc["cell_n"],
    }
    for k, v in losses.items():
        m["val_" + k] = v / max(1, nb)
    return m


def modal_cells(episodes, net, device, obs_dim, max_eps=2000):
    """Most common human cell per card id over training matches (baseline)."""
    from collections import Counter, defaultdict
    cnt = defaultdict(Counter)
    for e in episodes[:max_eps]:
        play = np.nonzero(e.card != C.NOOP)[0]
        if not len(play):
            continue
        obs = torch.from_numpy(D.BCData.dense_obs(e)[play]).to(device)
        ids = net.hand_card_ids(obs).cpu().numpy()
        for j, t in enumerate(play):
            cid = ids[j, e.card[t]]
            cnt[int(cid)][int(e.cell[t])] += 1
    return {c: k.most_common(1)[0][0] for c, k in cnt.items()}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def card_noop_shift(w_noop):
    return float(-math.log(w_noop)) if w_noop < 1 else 0.0


def apply_noop_shift(net, shift):
    with torch.no_grad():
        net.card_head.bias[C.NOOP] += shift


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    log_dir = os.path.join(REPO, "runs", args.run)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(REPO, "checkpoints"), exist_ok=True)
    mlog = open(os.path.join(log_dir, "bc_metrics.jsonl"), "a")

    data = D.BCData(max_shards=args.max_shards)
    D.set_obs_dim(data.obs_dim)
    spec = RewardSpec(gamma=args.gamma, k=args.k, leak_worst=args.leak_worst)
    t0 = time.time()
    data.compute_returns(spec)
    print(f"returns computed in {time.time() - t0:.0f}s", flush=True)

    # No-op weight: by default bring decision-row no-op mass to ~= play mass.
    n_noop = sum(int((e.card == C.NOOP).sum()) for e in data.train)
    n_all = sum(e.T for e in data.train)
    if args.w_noop is None:
        p_noop = n_noop / n_all
        args.w_noop = float(min(1.0, (1 - p_noop) / max(p_noop, 1e-6) * args.noop_balance))
    print(f"train rows {n_all}, no-op frac {n_noop / n_all:.3f}; w_noop {args.w_noop:.3f} "
          f"(post-hoc no-op logit shift {card_noop_shift(args.w_noop):+.3f})", flush=True)

    net = C.new_net(device)
    if args.init:
        C.load_weights(net, args.init)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    modal = modal_cells(data.train, net, device, data.obs_dim)
    loss_kw = dict(smooth=args.smooth, cell_smooth=args.cell_smooth, w_noop=args.w_noop,
                   lost_weight=args.lost_weight, value_coef=args.value_coef,
                   ability_coef=args.ability_coef)
    eval_kw = dict(smooth=0.0, cell_smooth=0.0, w_noop=1.0, lost_weight=1.0,
                   value_coef=args.value_coef)

    deadline = time.time() + args.minutes * 60
    step, epoch, best = 0, 0, float("inf")
    ck_path = os.path.join(REPO, "checkpoints", f"{args.run}.pth")
    shift = card_noop_shift(args.w_noop)
    total_steps_est = None
    done = False
    while not done and epoch < args.epochs:
        epoch += 1
        hidden = None
        t_ep = time.time()
        n_rows = 0
        for b, first in chunk_batches(data.train, args.batch, args.bptt, rng, device, data.obs_dim):
            if first:
                hidden = zero_hidden(net, b["obs"].shape[1], device)
            else:
                hidden = (hidden[0].detach(), hidden[1].detach())
            # Cosine LR over the time budget.
            frac = min(1.0, 1 - (deadline - time.time()) / (args.minutes * 60))
            for g in opt.param_groups:
                g["lr"] = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
            out, hidden = forward_chunk(net, b, hidden)
            loss, st = bc_losses(net, b, out, **loss_kw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            step += 1
            n_rows += int(b["valid"].sum())
            if step % 50 == 0:
                print(f"ep {epoch} step {step} loss {float(loss):.4f} " +
                      " ".join(f"{k} {v:.4f}" for k, v in st.items() if k.startswith("loss")) +
                      f" | {n_rows / (time.time() - t_ep):.0f} rows/s", flush=True)
            if step % args.eval_every == 0 or time.time() > deadline:
                m = evaluate(net, data.val, device, data.obs_dim, modal=modal, noop_shift=shift,
                             max_eps=args.val_eps, loss_kw=eval_kw)
                m.update(step=step, epoch=epoch, wall=time.time(), w_noop=args.w_noop,
                         lost_weight=args.lost_weight)
                mlog.write(json.dumps(m) + "\n")
                mlog.flush()
                score = m.get("val_loss_card", 0) + m.get("val_loss_place", 0)
                print(f"  VAL step {step}: card {m['card_match']:.3f} (noop base "
                      f"{m['card_match_baseline_always_noop']:.3f}) given-play "
                      f"{m['card_match_given_play']:.3f} cell {m['cell_match']:.3f} "
                      f"within1 {m['cell_within1']:.3f} modal {m['cell_match_baseline_modal']} "
                      f"p_play {m['bc_mean_p_play']:.3f} vs human {m['human_play_rate']:.3f} "
                      f"ece {m['calibration_ece_play']:.3f} ab {m['ability_acc']:.3f} "
                      f"val card+place NLL {score:.4f}", flush=True)
                if score < best:
                    best = score
                    save(net, ck_path, shift, args, m)
                if time.time() > deadline:
                    done = True
                    break
        print(f"epoch {epoch} done in {time.time() - t_ep:.0f}s", flush=True)
    # Final full-val metrics of the best checkpoint.
    best_net = C.new_net(device)
    best_net.load_state_dict(torch.load(ck_path, map_location=device))
    m = evaluate(best_net, data.val, device, data.obs_dim, modal=modal, noop_shift=0.0,
                 loss_kw=eval_kw)
    m.update(final=True, best_val_card_place_nll=best, w_noop=args.w_noop, lost_weight=args.lost_weight)
    mlog.write(json.dumps(m) + "\n")
    meta = json.load(open(ck_path.replace(".pth", ".meta.json")))
    meta["final_val"] = m
    json.dump(meta, open(ck_path.replace(".pth", ".meta.json"), "w"), indent=1)
    print("FINAL VAL (saved, calibrated):", json.dumps({k: v for k, v in m.items()
                                                        if k != "calibration_bins"}), flush=True)
    return m


def save(net, path, shift, args, m):
    sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    sd["card_head.bias"][C.NOOP] += shift      # calibrated play rate in the saved policy
    tmp = path + ".tmp"
    torch.save(sd, tmp)
    os.replace(tmp, path)
    json.dump({"args": vars(args), "noop_shift": shift, "val_at_save": m,
               "deck": C.deck_ids(), "format": "bare state_dict (CLASH_WEIGHTS legacy load)"},
              open(path.replace(".pth", ".meta.json"), "w"), indent=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="bc_hog26")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--minutes", type=float, default=40)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch", type=int, default=32, help="matches per batch")
    p.add_argument("--bptt", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--smooth", type=float, default=0.05)
    p.add_argument("--cell-smooth", type=float, default=0.1)
    p.add_argument("--w-noop", type=float, default=None)
    p.add_argument("--noop-balance", type=float, default=1.0,
                   help="w_noop = balance * p_play/p_noop when --w-noop is unset")
    p.add_argument("--lost-weight", type=float, default=1.0)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--ability-coef", type=float, default=0.0,
                   help="0 by default: the dataset only has ability rows where the human "
                        "activated (ability_ready is set on event rows only), so every "
                        "'ready' label is positive and the head would learn 'always fire'")
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--k", type=float, default=0.0)
    p.add_argument("--leak-worst", type=float, default=0.2)
    p.add_argument("--eval-every", type=int, default=400)
    p.add_argument("--val-eps", type=int, default=200, help="val matches per periodic eval")
    p.add_argument("--max-shards", type=int, default=None)
    p.add_argument("--init", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
