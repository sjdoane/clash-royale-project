"""Gen-2 PPO: fine-tune a NetV3 clone against the opponent pool, anchored to the clone.

    CR_ENGINE=build/gen2/engine .venv/bin/python -m training.ppo_pool \
        --run g2_ppo --init runs/bc3/g2_big_L0_H10/last.pth \
        --clone-ckpt runs/bc3/g2_big_L0_H10/last.pth runs/bc3/g2_big_L5_H10/last.pth \
        --num-envs 12 --device mps --act-device cpu

Re-running the same command resumes from runs/<run>/state.pth (`--fresh` starts
over). Stop with SIGTERM: the update in flight finishes, then everything is saved.

What it is
  net        training/net_v3.NetV3 built from the init checkpoint's own `cfg`
             (TrainNetV3 below: same parameters and state-dict keys).
  envs       training/pool_env.PoolEnv workers at `decision_ticks` engine ticks
             per decision (CLI, else the checkpoint's, else 5), with actuation
             delay (3,6,1), board noise and tail noise on by default.
  loop       the engine's BaseTrainer (rollout, buffer) and its PPO update, the
             latter through ppo_bc.AnchoredUpdater (one injected loss hook).
             --device is where the update runs; the rollout's forward passes
             (a handful of rows per step) run on --act-device, by default the
             CPU, on a copy of the net refreshed before every rollout.
  reward     training/reward.py unchanged (ghost.DEFAULT_SPEC); only its `gamma`
             follows the per-decision discount, as potential shaping requires.
  anchors    frozen reference = the init checkpoint.
             * critic warm-up: the first --critic-warmup updates train only
               `value_head`; the policy stays the clone, so those rollouts are
               also the clone's own play-rate baseline for the guard;
             * KL between the policy and the clone on rollout states, per head
               (WHEN, WHAT given play, WHERE given the taken card, abilities):
               (1 - m) KL(pi || clone) + m KL(clone || pi), m = --kl-mix;
               coefficient --kl-start -> --kl-end (log) over --anchor-decay
               episodes, then held at --kl-end;
             * the cloning loss (bc_train.forward_losses) on human training rows,
               one batch per PPO minibatch, --bc-start -> --bc-end likewise.
  pool       training/pool_gen2_default.json merged with --pool-override. The
             `clone` entry holds exactly the --clone-ckpt paths (none: its weight
             moves to --clone-fallback and every log line says so). The `self`
             entry is runs/<run>/latest.pth (rewritten every update; workers
             reload it by mtime) plus older ones: the JSON's own entries and
             runs/<run>/snapshots/epNNNNNNN.pth, one every --snapshot-every
             episodes.
  guard      on every rollout the clone is evaluated on the SAME states: play
             ratio = E[p_play] / E_clone[p_play], and per card share ratio =
             (card's share of expected plays) / (the clone's). A breach is a
             play ratio outside [--guard-play-lo, --guard-play-hi], a card
             share ratio below --guard-share-lo, or sampled plays per minute
             outside --guard-ppm of the warm-up baseline. --guard-patience
             consecutive breaches trip it; it is always logged and stops the
             run only with --guard-stop.
  files      runs/<run>/progress.jsonl (one line per update), latest.pth (v3
             policy checkpoint: obs_compat.load_policy / eval_ghost / live),
             state.pth (resume), snapshots/, tb/.

Per-decision constants inherited from gen 1 (10 ticks per decision), see
`convert_constants`: gamma and the GAE lambda are given per game second and
raised to dt/10; the rollout and segment lengths are given in game seconds.
"""
import argparse
import json
import math
import os
import random
import shutil
import signal
import sys
import time

#: gen 1's decision interval: what every inherited per-decision constant assumed.
REF_TICKS = 10
TICKS_PER_SECOND = 10


def _parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", default="g2_ppo")
    p.add_argument("--out", default=None, help="run directory (default runs/<run>)")
    p.add_argument("--init", default=None, help="NetV3 clone checkpoint: start weights and frozen reference")
    p.add_argument("--fresh", action="store_true", help="ignore an existing state.pth")
    p.add_argument("--device", default="cpu", help="device of the PPO update: cpu | mps | cuda")
    p.add_argument("--act-device", default="cpu",
                   help="device of the rollout's forward passes (a copy of the net, synced every update)")
    p.add_argument("--num-envs", type=int, default=12)
    p.add_argument("--torch-threads", type=int, default=4,
                   help="learner CPU threads during the rollout, when the workers want the cores (0 = torch's default)")
    p.add_argument("--update-threads", type=int, default=0,
                   help="learner CPU threads during the update, when the workers idle (0 = as in the rollout)")
    p.add_argument("--vec", default="async", choices=("async", "sync"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-episodes", type=int, default=100_000)
    p.add_argument("--max-updates", type=int, default=0, help="0 = no limit")
    # environment
    p.add_argument("--decision-ticks", type=int, default=None,
                   help="engine ticks per decision (default: the checkpoint's, else 5)")
    p.add_argument("--act-delay", default="3,6,1", help="lo,hi,jitter ticks; '' = off")
    p.add_argument("--no-board-noise", action="store_true")
    p.add_argument("--obs-noise-scale", default="")
    p.add_argument("--board-hp-live", action="store_true",
                   help="board noise writes every unit at full health with the card table's attributes, "
                        "as the live driver does (troop health is not read live): obs_noise `hp_as_live`. "
                        "The privileged critic still sees the true health")
    p.add_argument("--no-tail-noise", action="store_true")
    p.add_argument("--tail-preset", default="",
                   help="tail-noise preset of training/obs_noise.py (reader2 = the live reader since "
                        "2026-11-22: landing estimates for thrown spells in the opponent-spell block, "
                        "fewer missed follow-up plays). Default: the reader of 2026-11-20, which shows "
                        "no opponent spell in flight.")
    # 2026-11-27 (gen2_l switch, runs/sim_0091/SWITCH_PLAN.md): default off; unset = the tail noise as before.
    p.add_argument("--tail-evo-no-tile", default="", metavar="IDS",
                   help="comma list of Evolution card ids whose EVOLVED cast never gets a landing estimate in "
                        "the tail (obs_noise TailNoiseConfig.evo_no_tile; 132 = Evo Goblin Barrel, live 0 of 9). "
                        "Needs --tail-preset reader2 and an engine with get_evolution_state (build/gen2_l). With "
                        "132 the preset's flight['Goblin Barrel']['fail'] goes 0.07 -> 0.02: 0.07 was the evolved "
                        "throws as a rate; live plain throws without an estimate are 0 of 71 (0.02 = the file's "
                        "ASSUMED floor).")
    # pool
    p.add_argument("--pool", default=os.path.join("training", "pool_gen2_default.json"))
    p.add_argument("--pool-override", default="", help="JSON merged over --pool")
    p.add_argument("--clone-ckpt", nargs="*", default=[], help="human-clone opponents (v3 checkpoints)")
    p.add_argument("--clone-fallback", default="ghost", choices=("ghost", "teacher", "self"))
    p.add_argument("--snapshot-every", type=int, default=2000, help="episodes")
    p.add_argument("--self-latest-share", type=float, default=0.5)
    p.add_argument("--self-max-older", type=int, default=8)
    # PPO; times in game seconds (see convert_constants)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup-lr", type=float, default=5e-4, help="value head, during critic warm-up")
    p.add_argument("--gamma", type=float, default=0.999, help="discount per game SECOND")
    p.add_argument("--gae-lambda", type=float, default=0.9, help="GAE lambda per game SECOND")
    p.add_argument("--rollout-seconds", type=float, default=500.0, help="game seconds per env per update")
    p.add_argument("--chunk-seconds", type=float, default=50.0, help="minibatch segment length")
    p.add_argument("--rollout-steps", type=int, default=0, help="override, in decisions")
    p.add_argument("--chunk-steps", type=int, default=0, help="override, in decisions")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--eps-clip", type=float, default=0.2)
    p.add_argument("--ent-when", type=float, default=0.0)
    p.add_argument("--ent-what", type=float, default=0.0)
    p.add_argument("--ent-where", type=float, default=0.0)
    # anchors
    p.add_argument("--critic-warmup", type=int, default=6, help="updates")
    p.add_argument("--kl-start", type=float, default=1.0)
    p.add_argument("--kl-end", type=float, default=0.3)
    p.add_argument("--kl-mix", type=float, default=0.5,
                   help="0 = KL(pi||clone) only, 1 = KL(clone||pi) only")
    p.add_argument("--kl-head-weights", default="1,1,1,1", help="when,what,where,ability")
    p.add_argument("--bc-head-weights", default="1,1,1",
                   help="when,what,where: multiplies the cloning loss weights of these heads (default: as trained)")
    p.add_argument("--kl-max", type=float, default=0.0,
                   help="> 0: raise the KL coefficient while the pre-update KL exceeds this")
    p.add_argument("--bc-start", type=float, default=0.5)
    p.add_argument("--kl-final", type=float, default=0.0,
                   help="second stage of the KL anchor: after --kl-final-from episodes the coefficient "
                        "goes on (log) from --kl-end to this value, reached at --kl-final-at episodes "
                        "and held after. 0 = off (held at --kl-end). The cloning loss is not touched.")
    p.add_argument("--kl-final-from", type=float, default=0.0, help="episodes")
    p.add_argument("--kl-final-at", type=float, default=0.0, help="episodes")
    p.add_argument("--bc-end", type=float, default=0.2)
    p.add_argument("--anchor-decay", type=float, default=40000, help="episodes")
    p.add_argument("--anchor-free-elixir", type=float, default=None, metavar="E",
                   help="decisions where the learner reads E elixir or more carry no WHEN reference: "
                        "their WHEN KL to the clone is left out of the loss and of the --kl-max "
                        "measurement, and the cloning rows at E or more carry no WHEN target. The "
                        "human matches include players who stop playing when a match looks lost and "
                        "sit at a full bar (2026-11-23: 41%% of 3-crown losses hold 10 s or more of "
                        "it), and the anchor held the learner to that; the reward decides there instead. "
                        "Default: off")
    p.add_argument("--anchor-free-behind", action="store_true",
                   help="decisions where more of the learner's Princess Towers are down than of the "
                        "opponent's carry no WHEN reference either (same treatment as "
                        "--anchor-free-elixir). Live, 2026-11-23: p(play) 0.004 at 8+ elixir once our "
                        "towers were down and theirs untouched (0.22 with theirs shown at half health), "
                        "the human habit of giving up a match that looks lost. Default: off")
    p.add_argument("--bc-data", default=None, help="bcdata2 root (default: the checkpoint's dataset)")
    p.add_argument("--bc-batch", type=int, default=512, help="human rows per PPO minibatch")
    p.add_argument("--bc-block", type=int, default=2, help="shards read per block")
    p.add_argument("--bc-procs", type=int, default=5)
    p.add_argument("--bc-allow-unready", action="store_true")
    # 2026-11-27 (imit_shift agent): both default off; unset = the behaviour before they existed.
    p.add_argument("--anchor-ckpt", default=None, metavar="PATH",
                   help="NetV3 checkpoint for the frozen KL reference (ref_net) instead of the init checkpoint. "
                        "The init (fresh-run weights, meta, the cloning rows' dataset/lead/train cfg) is "
                        "unchanged; PATH must match its cfg, names, lead, dataset_fp, decision_ticks.")
    p.add_argument("--bc-noise-root", default=None, metavar="DIR",
                   help="root of the cloning rows' noise cache (bc_train.NoiseCache root; default "
                        "runs/bc3/_cache). Every needed shard must already be under DIR: the trainer "
                        "refuses to start rather than build missing shards with the default reader.")
    # guard
    p.add_argument("--guard-play-lo", type=float, default=0.6)
    p.add_argument("--guard-play-hi", type=float, default=1.6)
    p.add_argument("--guard-share-lo", type=float, default=0.5)
    p.add_argument("--guard-ppm", default="0.5,2.0")
    p.add_argument("--guard-patience", type=int, default=3)
    p.add_argument("--guard-stop", action="store_true")
    # sim-to-real randomization (training/sim2real.py) and the privileged critic
    # (training/priv_critic.py): all off unless asked for
    p.add_argument("--sim2real", default="", help="a preset ('v1') or a JSON path")
    p.add_argument("--sim2real-scale", default="",
                   help="strength: 'x', or 'lo:hi' drawn per episode (v1: '0:1.5')")
    p.add_argument("--sim2real-only", default="", help="comma list: only these randomizations")
    p.add_argument("--sim2real-off", default="", help="comma list: all but these randomizations")
    p.add_argument("--sim2real-elixir-est", action="store_true",
                   help="with --sim2real: the elixir digit goes through the live driver's estimator "
                        "(live_ppo/elixir_estimate.py) and the policy sees its fractional estimate")
    p.add_argument("--sim2real-elixir-est-share", type=float, default=1.0,
                   help="with --sim2real-elixir-est: share of integer-digit episodes that use the estimator "
                        "(the rest show the integer, so the policy also runs with the estimator off)")
    p.add_argument("--sim2real-widen", type=float, default=0.0,
                   help="episodes over which the upper end of a 'lo:hi' scale grows from lo to hi "
                        "(0 = the full range from the start)")
    p.add_argument("--sim2real-anchor-clean", action="store_true",
                   help="with --sim2real: the frozen reference (the clone) of the KL anchor and of the "
                        "diagnostics reads the TRUE values of what the layer changed, so the learner is "
                        "pulled towards what the clone would do knowing them, not towards its reaction to "
                        "readings it was never trained on")
    p.add_argument("--priv-critic", action="store_true",
                   help="advantages from V_net(obs) + g(true state); the policy net is unchanged")
    p.add_argument("--priv-critic-lr", type=float, default=3e-4)
    return p.parse_args(argv)


# --------------------------------------------------- pure helpers (no engine) ---
def per_decision(per_second, dt):
    """A per-game-second factor (gen 1: per 10-tick decision) per dt-tick decision."""
    return float(per_second) ** (float(dt) / REF_TICKS)


def convert_constants(dt, gamma=0.999, gae_lambda=0.9, rollout_seconds=500.0, chunk_seconds=50.0,
                      rollout_steps=0, chunk_steps=0):
    """Gen 1's per-decision constants at `dt` ticks per decision.

    gamma, gae_lambda: per game second -> per decision (x ** (dt / 10)), so the
    discount horizon and the GAE horizon 1 / (1 - gamma * lambda) are the same
    number of game seconds at any interval. rollout / chunk: game seconds ->
    decisions. At dt = 10 every value is gen 1's (0.999, 0.9, 500, 50)."""
    dt = int(dt)
    if dt < 1:
        raise ValueError("decision_ticks must be >= 1")
    T = int(rollout_steps) or int(round(rollout_seconds * TICKS_PER_SECOND / dt))
    L = int(chunk_steps) or int(round(chunk_seconds * TICKS_PER_SECOND / dt))
    if T < 1 or L < 1 or T % L:
        raise ValueError(f"rollout of {T} decisions is not a multiple of the {L}-decision segment")
    g, lam = per_decision(gamma, dt), per_decision(gae_lambda, dt)
    return {"decision_ticks": dt, "seconds_per_decision": dt / TICKS_PER_SECOND,
            "gamma": g, "gae_lambda": lam, "update_timestep": T, "bptt_chunk": L,
            "rollout_seconds": T * dt / TICKS_PER_SECOND, "chunk_seconds": L * dt / TICKS_PER_SECOND,
            "gae_horizon_seconds": dt / TICKS_PER_SECOND / (1.0 - g * lam)}


def per_minute(count, steps, dt):
    """`count` events over `steps` decisions of dt ticks -> events per game minute."""
    return float(count) / max(1e-9, steps * dt / (60.0 * TICKS_PER_SECOND))


def thin(items, k):
    """At most k of `items`, evenly spaced, keeping the first and the last."""
    items = list(items)
    if k <= 0:
        return []
    if len(items) <= k:
        return items
    if k == 1:
        return [items[-1]]
    idx = sorted({int(round(i * (len(items) - 1) / (k - 1))) for i in range(k)})
    return [items[i] for i in idx]


def self_entries(latest, older=(), latest_share=0.5, max_older=8, seeds=()):
    """The pool's `self.ckpts`: the latest policy with `latest_share` of the
    weight, the rest uniform over `seeds` (always kept) and at most `max_older`
    of `older` (snapshots, oldest first, evenly thinned). Entries are
    {path, weight, name}; with nothing older the latest takes it all."""
    def _e(e):
        e = {"path": e} if isinstance(e, str) else dict(e)
        e.setdefault("name", os.path.splitext(os.path.basename(e["path"]))[0])
        return e
    old = [_e(e) for e in seeds] + [_e(e) for e in thin(older, max_older)]
    if not old:
        return [{"path": latest, "weight": 1.0, "name": "latest"}]
    share = min(1.0, max(0.0, float(latest_share)))
    out = [{"path": latest, "weight": share, "name": "latest"}] if share > 0 else []
    w = (1.0 - share) / len(old) if share > 0 else 1.0 / len(old)
    return out + [dict(e, weight=w) for e in old]


def ckpt_name(path):
    """A checkpoint's pool name: its file name, or its directory for last.pth etc."""
    base = os.path.splitext(os.path.basename(path))[0]
    if base in ("last", "best", "latest", "state"):
        return os.path.basename(os.path.dirname(os.path.abspath(path))) or base
    return base


def guard_reasons(play_ratio, share_ratios, ppm_ratio, play_lo=0.6, play_hi=1.6, share_lo=0.5,
                  ppm=(0.5, 2.0)):
    """Why this update breaches the guard ([] = it does not). play_ratio:
    E[p_play] / E_clone[p_play] on the rollout's states; share_ratios: {card:
    share of expected plays / the clone's}; ppm_ratio: sampled plays per minute
    over the warm-up baseline (None = no baseline yet)."""
    out = []
    if play_ratio is not None and math.isfinite(play_ratio):
        if play_ratio < play_lo:
            out.append(f"play ratio {play_ratio:.2f} < {play_lo}")
        elif play_ratio > play_hi:
            out.append(f"play ratio {play_ratio:.2f} > {play_hi}")
    for card, r in sorted((share_ratios or {}).items()):
        if math.isfinite(r) and r < share_lo:
            out.append(f"{card} share ratio {r:.2f} < {share_lo}")
    if ppm_ratio is not None and math.isfinite(ppm_ratio) and not ppm[0] <= ppm_ratio <= ppm[1]:
        out.append(f"plays per minute {ppm_ratio:.2f}x the clone's baseline")
    return out


ARGS = _parse() if __name__ == "__main__" else None

# ppo_bc sets the engine's environment switches (coverage terms and scenario
# play off) before it imports the engine; importing it first keeps that order.
from training import ppo_bc as PB  # noqa: E402
from training import common as C  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import gymnasium as gym  # noqa: E402

from python_ai.rl import base_trainer as bt  # noqa: E402
from python_ai.rl import gae as gae_mod  # noqa: E402
from python_ai.engine_constants import card_name  # noqa: E402

from training import reward as R  # noqa: E402
from training import ghost as G  # noqa: E402
from training import net_v3 as N3  # noqa: E402
from training import pool_env as PE  # noqa: E402
from training import bc_train as BT  # noqa: E402
from training import bcdata2 as D2  # noqa: E402

REPO = C.REPO
log_interp = PB.log_interp
parse_delay = PB.parse_delay


def tail_evo_spec(ids, preset):
    """--tail-evo-no-tile (2026-11-27, default off): the keys it adds to the envs' obs_v2_noise dict;
    {} when unset, so the dict (and every worker's TailNoise) is exactly as before. Workers get the
    keys through the pickled env thunk, not from a module default."""
    ids = sorted({int(x) for x in str(ids or "").replace(",", " ").split()})
    if not ids:
        return {}
    import copy  # noqa: PLC0415
    from training import obs_noise  # noqa: PLC0415
    if preset not in obs_noise.TAIL_PRESETS:
        raise SystemExit(f"--tail-evo-no-tile needs --tail-preset reader2 (got {preset!r})")
    if not hasattr(C.E.ClashRoyaleEnv, "get_evolution_state"):
        raise SystemExit(f"--tail-evo-no-tile needs an engine with get_evolution_state (build/gen2_l); "
                         f"CR_ENGINE is {C.ENGINE}")
    out = {"evo_no_tile": ids}
    if 132 in ids:
        fl = copy.deepcopy(obs_noise.TAIL_PRESETS[preset]["flight"])
        if fl.get("Goblin Barrel", {}).get("fail") != 0.07:
            raise SystemExit(f"--tail-evo-no-tile 132: preset {preset} Goblin Barrel fail is "
                             f"{fl.get('Goblin Barrel', {}).get('fail')!r}, expected 0.07 (preset changed?)")
        fl["Goblin Barrel"]["fail"] = 0.02
        out["flight"] = fl
    return out


def tail_noise_spec(a, names):
    """The envs' obs_v2_noise value (None = clean tail): names, --tail-preset, --tail-evo-no-tile."""
    evo = getattr(a, "tail_evo_no_tile", "")
    if a.no_tail_noise:
        if evo:
            raise SystemExit("--tail-evo-no-tile needs the tail noise (drop --no-tail-noise)")
        return None
    spec = {"names": names}
    if getattr(a, "tail_preset", ""):
        spec["preset"] = a.tail_preset
    if evo:
        spec.update(tail_evo_spec(evo, getattr(a, "tail_preset", "")))
    return spec


# ------------------------------------------------------ factorised policy ---
def card_factors(card_logp):
    """The composed card log-probabilities (..., hand + 1) of NetV3, split back
    into its heads -> (log p_play, log p_wait, log pi(slot | play), legal, has).
    Rows with no playable card (has False) are forced waits."""
    slots, lwait = card_logp[..., :-1], card_logp[..., -1]
    legal = torch.isfinite(slots)
    has = legal.any(-1)
    safe = torch.where(legal, slots, torch.full_like(slots, -1e30))
    lplay = torch.logsumexp(safe, dim=-1)
    lwhat = torch.where(legal, slots - lplay.unsqueeze(-1), torch.zeros_like(slots))
    return lplay, lwait, lwhat, legal, has


def kl_heads(lp, lq):
    """KL(P || Q) of two composed card distributions, by head -> (kl_when,
    kl_what, p_play): kl_what is the KL of the slot choice GIVEN a play, and the
    KL of the composed distributions is kl_when + p_play * kl_what."""
    lplay_p, lwait_p, lwhat_p, legal, has = card_factors(lp)
    lplay_q, lwait_q, lwhat_q, legal_q, has_q = card_factors(lq)
    zero = torch.zeros_like(lplay_p)
    pp = torch.where(has, lplay_p.exp(), zero)
    # WHEN is compared only where BOTH can play something. With
    # --sim2real-anchor-clean the clone's affordability mask comes from the
    # true values and the learner's from its readings, so one of them can be a
    # forced wait where the other is not; that row carries no WHEN preference
    # to copy, and its log p_play of about -1e30 must not reach the loss
    # (2026-11-22: it gave a WHEN KL of 1e24 and drove --kl-max to its cap).
    both_has = has & has_q
    kw = (pp * (lplay_p - torch.where(both_has, lplay_q, lplay_p))
          + lwait_p.exp() * (lwait_p - torch.where(both_has, lwait_q, lwait_p)))
    kw = torch.where(both_has, kw, zero)
    both = legal & legal_q
    kh = (lwhat_p.exp() * torch.where(both, lwhat_p - lwhat_q, torch.zeros_like(lwhat_p))).sum(-1)
    return kw, kh, pp


def kl_cat(lp, lq):
    """KL(P || Q) over the last axis of two log-probability tensors that share
    their -inf (illegal) entries."""
    ok = torch.isfinite(lp) & torch.isfinite(lq)
    return (lp.exp() * torch.where(ok, lp - lq, torch.zeros_like(lp))).sum(-1)


def head_entropies(card_logp):
    """(H_when, H_what given play, p_play), in nats, per row."""
    lplay, lwait, lwhat, legal, has = card_factors(card_logp)
    zero = torch.zeros_like(lplay)
    pp = torch.where(has, lplay.exp(), zero)
    hw = torch.where(has, -(pp * lplay + lwait.exp() * lwait), zero)
    hh = -(lwhat.exp() * torch.where(legal, lwhat, torch.zeros_like(lwhat))).sum(-1)
    return hw, hh, pp


class TrainNetV3(N3.NetV3):
    """NetV3 for the PPO update. Same parameters and state-dict keys (it saves
    and loads as a plain NetV3); two differences in what the engine's update
    asks of it:

    * the placement of the no-op "card" is a constant (uniform over its legal
      cells). The engine scores a placement on every decision row, wait rows
      included (about nine in ten); as a network output it would send the PPO
      gradient of a placement nobody makes through the shared trunk, and its
      drift would enter the probability ratio. The rollout uses this class too,
      so stored and recomputed log-probabilities agree.
    * the second placement pass for the engine's coverage term (coefficient 0
      here) is not computed."""

    def _placement_raw(self, hx, card_embeds, card_idx, *args, **kwargs):
        out = super()._placement_raw(hx, card_embeds, card_idx, *args, **kwargs)
        wait = (card_idx == self.hand_size).unsqueeze(-1)
        const = torch.where(torch.isfinite(out), torch.zeros_like(out), out).detach()
        return torch.where(wait, const, out)

    def forward_sequence(self, *args, extra_card_idx_seq=None, **kwargs):
        out = super().forward_sequence(*args, extra_card_idx_seq=None, **kwargs)
        if extra_card_idx_seq is None:
            return out
        return out[:5] + (torch.zeros_like(out[1]),) + out[6:]


@torch.no_grad()
def policy_tables(net, batch, chunk=2048):
    """One pass of `net` over a rollout batch -> {"card": (T, N, hand + 1)
    composed log-probabilities, "place": (T, N, cells) log-probabilities of the
    placement of the TAKEN card on rows that placed one (zeros elsewhere),
    "ability": (T, N, k, 2) logits or None}."""
    obs_all = batch["obs"]
    T, N = obs_all.shape[:2]
    flat_all = obs_all.reshape(T * N, -1)
    cards = batch["card_actions"].reshape(-1)
    placed = (batch["decision"].reshape(-1) > 0) & (cards != net.hand_size)
    card = flat_all.new_zeros(T * N, net.hand_size + 1)
    place = flat_all.new_zeros(T * N, net.placement_cells)
    k = int(net.num_ability_slots or 0)
    ab = flat_all.new_zeros(T * N, k, 2) if k else None
    for s in range(0, T * N, chunk):
        flat = flat_all[s:s + chunk]
        f, emb, sp, hi = net.extract_features_hires(flat)
        o = net.heads(f, net.affordability_mask(flat))
        card[s:s + chunk] = net.compose_card_logits(o["when"], o["what"], o["has_legal"])
        rows = placed[s:s + chunk].nonzero(as_tuple=True)[0]
        if rows.numel():
            pl = N3.NetV3._placement_raw(net, o["hx"][rows], emb[rows], cards[s:s + chunk][rows],
                                         flat[rows], sp[rows], hires_map=hi[rows],
                                         cycle_feat=net._split_cycle(f)[rows])
            place[s + rows] = torch.log_softmax(pl, dim=-1)
        if k:
            ab[s:s + chunk] = torch.stack([a for a in net._active_ability(o["ability"]) if a is not None], 1)
    return {"card": card.view(T, N, -1), "place": place.view(T, N, -1),
            "ability": None if ab is None else ab.view(T, N, k, 2)}


# ---------------------------------------------------------------- pool ---
def build_pool(base=None, over=None, clone_ckpts=(), clone_fallback="ghost"):
    """(pool config, note). `base`: dict | JSON path | None, merged over
    pool_env.DEFAULT_POOL; `over`: dict merged on top. The clone entry's
    checkpoints are exactly `clone_ckpts`, whatever the JSON holds; with none,
    the clone weight moves to `clone_fallback` and the note says so. Training
    draws ghosts from the training split only."""
    if isinstance(base, str) and not os.path.isabs(base):
        base = os.path.join(REPO, base)
    pool = PE.load_pool_config(base)
    if over:
        pool = PE.merge_pool(pool, over)
    if pool["ghost"].get("split") != "train":
        raise ValueError("the trainer only draws from the training split (the validation split "
                         "is the evaluation set)")
    note = ""
    pool["clone"]["ckpts"] = [{"path": os.path.abspath(p), "name": ckpt_name(p)} for p in clone_ckpts]
    w = pool["weights"]
    moved = float(w.get("clone", 0.0))
    if not clone_ckpts and moved > 0:
        w[clone_fallback] = float(w.get(clone_fallback, 0.0)) + moved
        w["clone"] = 0.0
        note = (f"no --clone-ckpt: the clone weight {moved:g} was moved to {clone_fallback} "
                f"(now {w[clone_fallback]:g})")
    return pool, note


def _col(infos, key, n, default, dtype=np.float64):
    v = infos.get(key)
    if v is None:
        return np.full(n, default, dtype=dtype)
    out = np.asarray(v).astype(dtype)
    m = infos.get("_" + key)
    if m is not None:
        out = np.where(np.asarray(m, dtype=bool), out, default).astype(dtype)
    return out


def _kind_row():
    return {"episodes": 0, "wins": 0, "draws": 0, "losses": 0, "dmg_dealt": 0.0, "dmg_taken": 0.0,
            "crowns_for": 0, "crowns_against": 0, "midmatch": 0, "steps": 0, "leak": 0.0}


def kind_summary(rows):
    """Per-kind sums -> the per-episode figures the log shows."""
    out = {}
    for kind, r in rows.items():
        n = r["episodes"]
        if not n:
            continue
        out[kind] = {"episodes": n, "win_rate": round(r["wins"] / n, 4), "draw_rate": round(r["draws"] / n, 4),
                     "tower_dmg_dealt": round(r["dmg_dealt"] / n, 4),
                     "tower_dmg_taken": round(r["dmg_taken"] / n, 4),
                     "crowns_for": round(r["crowns_for"] / n, 3),
                     "crowns_against": round(r["crowns_against"] / n, 3),
                     "midmatch": round(r["midmatch"] / n, 3), "decisions": round(r["steps"] / n, 1),
                     "elixir_leaked": round(r["leak"] / n, 3)}
    return out


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# -------------------------------------------------------------- trainer ---
class PoolTrainer(bt.BaseTrainer):
    pipeline_name = "pool"
    replay_prefix = "pool"
    #: every PoolEnv ending is terminal (a ghost recording's end too): no V(final) bootstrap
    uses_truncation_bootstrap = False

    def __init__(self, args, envs=None):
        self.args = a = args
        self.run_dir = os.path.abspath(a.out or os.path.join(REPO, "runs", a.run))
        self.weight_path = os.path.join(self.run_dir, "state.pth")
        self.latest_path = os.path.join(self.run_dir, "latest.pth")
        self.snap_dir = os.path.join(self.run_dir, "snapshots")
        self.log_dir = os.path.join(self.run_dir, "tb")
        self.progress_path = os.path.join(self.run_dir, "progress.jsonl")
        self._envs_override = envs
        self._blob = None
        if os.path.exists(self.weight_path) and not a.fresh:
            self._blob = torch.load(self.weight_path, map_location="cpu", weights_only=False)
        elif os.path.isdir(self.run_dir):
            for name in ("progress.jsonl", "latest.pth", "state.pth"):
                if os.path.exists(os.path.join(self.run_dir, name)):
                    os.remove(os.path.join(self.run_dir, name))
            shutil.rmtree(self.snap_dir, ignore_errors=True)
        os.makedirs(self.snap_dir, exist_ok=True)

        init = self._blob["init"] if self._blob is not None else a.init
        if not init:
            raise SystemExit("--init <NetV3 clone checkpoint> is required for a fresh run")
        self.init_path = init if os.path.isabs(init) else os.path.join(REPO, init)
        self.init_blob = torch.load(self.init_path, map_location="cpu", weights_only=False)
        if not N3.is_v3(self.init_blob):
            raise SystemExit(f"{self.init_path}: not a NetV3 checkpoint")
        self.net_cfg = dict(self.init_blob["cfg"])
        dt = a.decision_ticks or (self._blob or {}).get("decision_ticks") \
            or self.init_blob.get("decision_ticks") or 5
        if self._blob is not None and int(self._blob["decision_ticks"]) != int(dt):
            raise SystemExit(f"state.pth was trained at {self._blob['decision_ticks']} ticks per decision, "
                             f"not {dt}")
        self.dt = int(dt)
        self.conv = convert_constants(self.dt, a.gamma, a.gae_lambda, a.rollout_seconds, a.chunk_seconds,
                                      a.rollout_steps, a.chunk_steps)
        self.n_updates = int(self._blob["n_updates"]) if self._blob is not None else 0
        # The potential shaping is policy-invariant only at the PPO discount.
        self.spec = R.RewardSpec(**dict(G.DEFAULT_SPEC, gamma=self.conv["gamma"]))
        R.install_reward(self.spec)
        cfg = bt.PPOConfig(
            num_envs=int(a.num_envs), gamma=self.conv["gamma"], gae_lambda=self.conv["gae_lambda"],
            eps_clip=a.eps_clip, lr=a.lr, update_timestep=self.conv["update_timestep"],
            bptt_chunk=self.conv["bptt_chunk"], ppo_epochs=a.epochs, num_minibatches=a.minibatches,
            aux_card_coef=0.0, aux_warmup_episodes=0, cycle_id_coef=0.0,
            # a resumed run must not replay the episodes its first start drew
            seed=int(a.seed) + 7919 * self.n_updates)
        super().__init__(cfg)
        self.device = torch.device(a.device)
        self.act_device = torch.device(a.act_device)
        self.actor = None
        self.kl_w = [float(v) for v in a.kl_head_weights.split(",")]
        self.guard_ppm = tuple(float(v) for v in a.guard_ppm.split(","))
        self.kl_boost = 1.0
        self.priv = None             # priv_critic.PrivCritic with --priv-critic
        self._clean = None           # CleanRows with --sim2real-anchor-clean
        self.snapshots = []          # [{"path", "episodes"}], oldest first
        self._last_snapshot_ep = 0
        self.kstats = {k: _kind_row() for k in PE.KIND_NAMES}
        self.started = {k: 0 for k in PE.KIND_NAMES}
        self.by_ckpt = {}
        self.baseline = {"plays": 0.0, "steps": 0.0}     # the clone's own play rate (warm-up rollouts)
        self.guard = {"streak": 0, "tripped": False, "tripped_at": None}
        self._stop = ""
        self._coefs = (0.0, 0.0)
        self._warming = False
        self._ref = None
        self._bc_iter = None
        self._mb = {}
        self._rec = None
        self.bc = None
        self.progress = None

    # -- envs and pool ---------------------------------------------------------
    def _meta(self, **extra):
        b = self.init_blob
        meta = {"names": b["names"], "decision_ticks": self.dt, "lead": b["lead"],
                "dataset": b["dataset"], "dataset_fp": b["dataset_fp"], "train": b.get("train")}
        meta.update(extra)
        return meta

    def _older(self):
        self.snapshots = [s for s in self.snapshots if os.path.exists(s["path"])]
        return [{"path": s["path"], "name": f"snap_ep{s['episodes']}"} for s in self.snapshots]

    def _self_ckpts(self):
        a = self.args
        return self_entries(self.latest_path, self._older(), a.self_latest_share, a.self_max_older,
                            seeds=self._self_seeds)

    def build_envs(self):
        envs = self._build_envs()
        a = self.args
        return self._keep_reset_infos(envs) if (a.priv_critic or a.sim2real_anchor_clean) else envs

    def _build_envs(self):
        a = self.args
        if self._blob is not None:
            self.snapshots = list(self._blob.get("snapshots") or [])
        if not os.path.exists(self.latest_path):
            # The pool checks that every checkpoint exists before the net does:
            # the first "latest" is the init checkpoint under this run's interval.
            _atomic_save(dict(self.init_blob, decision_ticks=self.dt), self.latest_path)
        over = json.loads(a.pool_override) if a.pool_override else None
        pool, self.pool_note = build_pool(a.pool, over, a.clone_ckpt, a.clone_fallback)
        self._self_seeds, dropped = [], []
        for e in pool["self"].get("ckpts") or ():
            e = {"path": e} if isinstance(e, str) else dict(e)
            path = e["path"] if os.path.isabs(e["path"]) else os.path.join(REPO, e["path"])
            (self._self_seeds if os.path.exists(path) else dropped).append(dict(e, path=path))
        if dropped:
            self.pool_note += f"; self seeds not found, dropped: {[d['path'] for d in dropped]}"
        pool["self"]["ckpts"] = self._self_ckpts()
        self.pool = pool
        if self._envs_override is not None:
            return self._envs_override
        from training import obs_noise
        extra = {}
        if a.board_hp_live and a.no_board_noise:
            raise SystemExit("--board-hp-live needs the board noise (drop --no-board-noise)")
        board = None if a.no_board_noise else obs_noise.cli_spec("default", a.obs_noise_scale,
                                                                 hp_live=bool(a.board_hp_live))
        if not a.no_tail_noise and self.net_cfg.get("use_tail", True):
            extra["obs_v2_noise"] = tail_noise_spec(a, self.init_blob.get("names") or "spells")
            if "evo_no_tile" in extra["obs_v2_noise"]:
                obs_noise.load_tail_config(extra["obs_v2_noise"])       # refuse a bad spec here, not in a worker
                if not getattr(self, "_evo_logged", False):
                    self._evo_logged = True
                    fl = extra["obs_v2_noise"].get("flight", {}).get("Goblin Barrel", {})
                    print(f">>> tail evo_no_tile {extra['obs_v2_noise']['evo_no_tile']} (--tail-evo-no-tile): "
                          f"evolved casts get no landing estimate; flight Goblin Barrel fail "
                          f"{fl.get('fail', 'unchanged')} (preset {a.tail_preset} 0.07); engine {C.ENGINE} "
                          f"has get_evolution_state", flush=True)
        elif getattr(a, "tail_evo_no_tile", ""):
            raise SystemExit("--tail-evo-no-tile needs the tail noise and a net that uses the tail")
        s2r = self._sim2real_spec()
        if s2r is not None:
            extra["sim2real"] = s2r
        spec = {k: v for k, v in self.spec.as_dict().items() if k != "c_leak"}
        seeds = bt.worker_seeds(self.cfg.seed, self.cfg.num_envs)
        thunks = [PE.make_pool_env(s, pool=pool, spec=spec, act_delay=parse_delay(a.act_delay),
                                   obs_noise=board, decision_ticks=self.dt, **extra) for s in seeds]
        if a.vec == "sync":
            return gym.vector.SyncVectorEnv(thunks)
        return gym.vector.AsyncVectorEnv(thunks)

    # -- sim-to-real randomization (opt-in) -----------------------------------------
    def _sim2real_spec(self):
        """The envs' `sim2real` value at the current episode count, or None."""
        a = self.args
        est = bool(getattr(a, "sim2real_elixir_est", False))
        if not (a.sim2real or a.sim2real_scale or a.sim2real_only or a.sim2real_off or est):
            if a.sim2real_widen:
                raise SystemExit("--sim2real-widen needs --sim2real")
            if a.sim2real_anchor_clean:
                raise SystemExit("--sim2real-anchor-clean needs --sim2real")
            return None
        from training import sim2real
        spec = sim2real.cli_spec(a.sim2real, a.sim2real_scale, a.sim2real_only, a.sim2real_off,
                                 **({"elixir_est": True} if est else {}))
        if est and float(a.sim2real_elixir_est_share) < 1.0:
            spec = dict(spec, elixir_est_share=float(a.sim2real_elixir_est_share))
        if a.sim2real_anchor_clean:
            spec = dict(spec, publish_clean=True)
        self._s2r_range = spec.get("scale_range")
        self._s2r_now = None
        if a.sim2real_widen > 0:
            if self._s2r_range is None:
                raise SystemExit("--sim2real-widen needs --sim2real-scale lo:hi")
            done = int((self._blob or {}).get("episodes", 0))
            self._s2r_now = sim2real.widened(self._s2r_range, done / a.sim2real_widen)
            spec = dict(spec, scale_range=self._s2r_now)
        return spec

    def _widen_sim2real(self):
        a = self.args
        if not getattr(a, "sim2real_widen", 0) or getattr(self, "_s2r_now", None) is None:
            return
        from training import sim2real
        new = sim2real.widened(self._s2r_range, self.episodes_completed / a.sim2real_widen)
        if abs(new[1] - self._s2r_now[1]) >= 0.01 or (new[1] >= self._s2r_range[1] > self._s2r_now[1]):
            self._s2r_now = new
            self.envs.call("set_sim2real", {"scale_range": new})      # from each env's next episode

    def _push_self_pool(self):
        entries = self._self_ckpts()
        self.pool["self"]["ckpts"] = entries
        self.envs.call("set_pool", {"self": {"ckpts": entries}})
        return entries

    # -- checkpoints -------------------------------------------------------------
    def load_checkpoint(self):
        b = self._blob
        if b is None:
            self.net.load_state_dict(self.init_blob["model"], strict=True)
            print(f">>> init: {self.init_path}", flush=True)
            return False
        self.net.load_state_dict(b["model"], strict=True)
        self.optimizer.load_state_dict(b["optimizer"])
        self.episodes_completed = int(b["episodes"])
        self.kl_boost = float(b.get("kl_boost", 1.0))
        self._last_snapshot_ep = int(b.get("last_snapshot_ep", 0))
        for name in ("kstats", "started", "by_ckpt", "baseline", "guard"):
            if b.get(name) is not None:
                setattr(self, name, b[name])
        print(f">>> resume: {self.weight_path} (update {self.n_updates}, episode {self.episodes_completed})",
              flush=True)
        return True

    def save_latest(self):
        N3.save_v3(self.net, self.latest_path, **self._meta(
            ppo={"run": self.args.run, "init": self.init_path, "episodes": int(self.episodes_completed),
                 "updates": int(self.n_updates), "constants": self.conv}))

    def save_checkpoint(self):
        self.save_latest()
        _atomic_save({
            "model": {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()},
            "optimizer": self.optimizer.state_dict(), "cfg": self.net_cfg,
            "episodes": int(self.episodes_completed), "n_updates": int(self.n_updates),
            "decision_ticks": self.dt, "init": self.init_path, "args": vars(self.args),
            "kl_boost": self.kl_boost, "snapshots": self.snapshots,
            "last_snapshot_ep": self._last_snapshot_ep, "kstats": self.kstats, "started": self.started,
            "by_ckpt": self.by_ckpt, "baseline": self.baseline, "guard": self.guard,
            "rng": {"numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                    "python": random.getstate()},
            **({"priv_critic": self.priv.state()} if self.priv is not None else {}),
        }, self.weight_path)

    def take_snapshot(self):
        """Freeze the current policy as an older `self` opponent."""
        ep = int(self.episodes_completed)
        path = os.path.join(self.snap_dir, f"ep{ep:07d}.pth")
        N3.save_v3(self.net, path, **self._meta(
            ppo={"run": self.args.run, "init": self.init_path, "episodes": ep,
                 "updates": int(self.n_updates), "constants": self.conv}))
        self.snapshots.append({"path": path, "episodes": ep})
        self._last_snapshot_ep = ep
        entries = self._push_self_pool()
        print(f">>> snapshot {path}; self pool: "
              + ", ".join(f"{e['name']} {e['weight']:.3f}" for e in entries), flush=True)
        return path

    # -- setup -------------------------------------------------------------------
    def setup(self):
        a = self.args
        if a.torch_threads > 0:
            torch.set_num_threads(a.torch_threads)
        keep = (bt.MicroRoyaleNet, bt.run_path, bt.HISTORICAL_CHECKPOINT_DIR)
        # The engine trainer builds its net through this module global, and
        # makes two directories under engine/: both stay inside the run.
        bt.MicroRoyaleNet = lambda num_ability_slots=0, **_: TrainNetV3(**self.net_cfg)
        bt.run_path = lambda *p: os.path.join(self.run_dir, *p)
        bt.HISTORICAL_CHECKPOINT_DIR = self.snap_dir
        try:
            super().setup()
        finally:
            bt.MicroRoyaleNet, bt.run_path, bt.HISTORICAL_CHECKPOINT_DIR = keep
        try:
            os.rmdir(os.path.join(self.run_dir, "replays"))
        except OSError:
            pass
        n = self.cfg.num_envs
        if self.act_device != self.device:
            # A few envs per step is a batch too small for a GPU to pay off:
            # the rollout runs on a copy of the net on the act device.
            self.actor = TrainNetV3(**self.net_cfg).to(self.act_device)
            for p in self.actor.parameters():
                p.requires_grad_(False)
            self._hx, self._cx = self._hx.to(self.act_device), self._cx.to(self.act_device)
        self.updater = PB.AnchoredUpdater(self.net, self.optimizer, self.cfg, self)
        self.ref_net, _ = N3.load_v3(self._anchor_blob(), self.device)
        for p in self.ref_net.parameters():
            p.requires_grad_(False)
        self.deck = [int(c) for c in C.deck_ids()]
        self._setup_bc()
        self._ep_open = [False] * n
        self._ep_kind = [0] * n
        self._ep_name = [""] * n
        self._ep_mid = [0.0] * n
        self._ep_start = [np.ones(6, dtype=np.float64) for _ in range(n)]
        self._ep_steps = [0] * n
        self._ep_leak = [0.0] * n
        self._reset_window()
        self._wrap_env_step()
        if a.sim2real_anchor_clean:
            from training import sim2real
            self._clean = sim2real.CleanRows()
            print(">>> sim2real: the reference policy reads the true values of the layer's span", flush=True)
        if a.priv_critic:
            from training import priv_critic
            self.priv = priv_critic.PrivCritic(n, lr=a.priv_critic_lr, device=self.device)
            if self.priv.load((self._blob or {}).get("priv_critic")):
                print(">>> privileged critic: resumed", flush=True)
            else:
                print(">>> privileged critic: new (zero residual)", flush=True)
        self.progress = open(self.progress_path, "a")
        self._t_last = time.time()
        if self.full_resume and self._blob.get("rng"):
            rng = self._blob["rng"]
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            random.setstate(rng["python"])
        self._blob = None
        c = self.conv
        print(f">>> ppo_pool: run {self.run_dir}\n"
              f"    net cfg {self.net_cfg}; update on {self.device}, rollout on {self.act_device}; {n} envs ({a.vec}), "
              f"{self.dt} ticks per decision, act delay {parse_delay(a.act_delay)}, "
              f"board noise {'off' if a.no_board_noise else ('on, units at full health as live' if a.board_hp_live else 'on')}, "
              f"tail noise {'off' if a.no_tail_noise else 'on'}"
              f"{' (' + a.tail_preset + ')' if getattr(a, 'tail_preset', '') and not a.no_tail_noise else ''}\n"
              f"    per decision: gamma {c['gamma']:.6f}, GAE lambda {c['gae_lambda']:.5f} "
              f"(horizon {c['gae_horizon_seconds']:.1f} s), rollout {c['update_timestep']} decisions "
              f"({c['rollout_seconds']:.0f} s), segment {c['bptt_chunk']} ({c['chunk_seconds']:.0f} s)\n"
              f"    lr {a.lr} (warm-up {a.warmup_lr} x {a.critic_warmup} updates), KL {a.kl_start}->{a.kl_end} "
              f"mix {a.kl_mix}, BC {a.bc_start}->{a.bc_end} over {a.anchor_decay:.0f} episodes, "
              f"entropy when/what/where {a.ent_when}/{a.ent_what}/{a.ent_where}\n"
              f"    pool weights {self.pool['weights']}, p_midmatch {self.pool['p_midmatch']}"
              + (f"\n    POOL NOTE: {self.pool_note}" if self.pool_note else ""), flush=True)

    def _anchor_blob(self):
        """The frozen KL reference's checkpoint: the init, or --anchor-ckpt (2026-11-27, default off)."""
        path = getattr(self.args, "anchor_ckpt", None)
        if not path:
            return self.init_blob
        path = path if os.path.isabs(path) else os.path.join(REPO, path)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if not N3.is_v3(blob):
            raise SystemExit(f"--anchor-ckpt {path}: not a NetV3 checkpoint")
        for k in ("cfg", "names", "lead", "dataset_fp", "decision_ticks", "base_obs_dim"):
            if blob.get(k) != self.init_blob.get(k):
                raise SystemExit(f"--anchor-ckpt {path}: {k} {blob.get(k)!r} differs from the init's "
                                 f"{self.init_blob.get(k)!r} ({self.init_path})")
        print(f">>> KL anchor (ref_net): {path} (--anchor-ckpt); the init stays {self.init_path}", flush=True)
        return blob

    def _setup_bc(self):
        a = self.args
        if a.bc_start <= 0 and a.bc_end <= 0:
            print(">>> cloning loss off (--bc-start and --bc-end are 0)", flush=True)
            return
        root = a.bc_data or self.init_blob.get("dataset")
        root = root if os.path.isabs(root) else os.path.join(REPO, root)
        ready = os.path.exists(os.path.join(root, "READY"))
        if not ready and os.path.basename(os.path.normpath(root)) != "pilot" and not a.bc_allow_unready:
            raise SystemExit(f"{root} has no READY file: the dataset is not finished "
                             "(use data/bc2/pilot for a smoke run)")
        tr = self.init_blob.get("train") or {}
        ds = D2.BCDataset(root, expect_obs_dim=self.net.obs_dim)
        groups = list(tr.get("groups") or BT.HOG_GROUPS)
        flt = D2.RowFilter(split="train", mirrored=None, groups=groups)
        tg = BT.Targets(ds, int(self.init_blob["lead"]), tr.get("interval"))
        cache = None
        if tr.get("noise", True):
            nroot = getattr(a, "bc_noise_root", None)
            nkw = {}
            if nroot:
                nkw["root"] = nroot if os.path.isabs(nroot) else os.path.join(REPO, nroot)
            cache = BT.NoiseCache(ds, seed=int(tr.get("noise_seed", 0)),
                                  names=self.init_blob.get("names") or "spells", p=float(tr.get("noise_p", 1.0)),
                                  **nkw)
            if nroot:
                need = list(ds.select("train", None, flt.groups))
                miss = [ds.metas[int(i)]["name"] for i in need if not os.path.exists(cache.path(i))]
                if miss:
                    raise SystemExit(f"--bc-noise-root: {len(miss)} of {len(need)} shards missing in {cache.dir} "
                                     f"(e.g. {miss[:3]}); not building them with the default reader")
                print(f">>> cloning noise cache: {cache.dir} (--bc-noise-root, all {len(need)} shards present)",
                      flush=True)
            else:
                cache.ensure(ds.select("train", None, flt.groups), procs=a.bc_procs, log=print)
        grid = BT.dataset_grid(ds)
        self.bc = {"ds": ds, "flt": flt, "tg": tg, "cache": cache, "root": root, "grid": grid,
                   "what_h": tr.get("what_h", 10), "what_on_when": tr.get("what_on_when", True),
                   # the critic is fitted to this run's returns, not to human outcomes
                   "weights": dict(tr.get("loss_weights") or {}, value=0.0)}
        _bw = [float(v) for v in a.bc_head_weights.split(",")]
        if _bw != [1.0, 1.0, 1.0]:
            _W = dict(BT.LOSS_W, **self.bc["weights"])
            for _h, _v in zip(("when", "what", "where"), _bw):
                self.bc["weights"][_h] = _W[_h] * _v
            print(f">>> cloning head weights (--bc-head-weights {a.bc_head_weights}): {self.bc['weights']}", flush=True)
        print(f">>> cloning rows: {root} (train split, groups {groups}, grid {grid} ticks, "
              f"lead {self.init_blob['lead']}, noise {'on' if cache is not None else 'off'})", flush=True)
        if grid != self.dt:
            print(f">>> WARNING: the cloning rows are on a {grid}-tick grid and this run decides every "
                  f"{self.dt} ticks: the WHEN target's rate per decision does not match", flush=True)

    def _bc_stream(self):
        b = self.bc
        ld = D2.BCLoader(b["ds"], batch_size=self.args.bc_batch, device=self.device, flt=b["flt"],
                         shuffle=True, seed=int(self.args.seed) * 100003 + self.n_updates,
                         block=self.args.bc_block, epochs=None, drop_last=True, noise=b["cache"],
                         label_transform=b["tg"], prefetch=4)
        return iter(ld)

    def _keep_reset_infos(self, envs):
        """--priv-critic: the privileged state of the first observation comes
        with the reset the engine trainer makes (it drops the infos). An extra
        env.privileged() call instead would not be free of side effects."""
        inner = envs.reset
        self._reset_infos = None

        def reset(*args, **kw):
            out = inner(*args, **kw)
            self._reset_infos = out[1]
            return out
        envs.reset = reset
        return envs

    def _wrap_env_step(self):
        """Time spent waiting on the workers, for the throughput split."""
        self._env_s = 0.0
        inner = self.envs.step

        def step(action):
            t0 = time.perf_counter()
            # The engine trainer sends target_x / target_y as (n, 1); each env
            # then gets a 1-element array, which reward.RewardEnv._own_spell
            # (board noise on, own spell played) passes to float(): a TypeError
            # under numpy >= 2.4. Flat arrays reach the envs as scalars.
            action = dict(action)
            for key in ("target_x", "target_y"):
                action[key] = np.asarray(action[key]).reshape(-1)
            out = inner(action)
            self._env_s += time.perf_counter() - t0
            return out
        self.envs.step = step

    # -- rollout bookkeeping ---------------------------------------------------------
    def _reset_window(self):
        self._win = {"kstats": {k: _kind_row() for k in PE.KIND_NAMES}, "episodes": 0, "leak": 0.0,
                     "dropped": 0.0, "steps": 0}

    def collect_rollout(self):
        self._env_s = 0.0
        t0 = time.time()
        if self._clean is not None and self._clean.cur is None:
            self._clean.start(self._reset_infos)
        if self.priv is not None and self.priv.cur is None:
            self.priv.start(self._reset_infos)
        self._reset_infos = None
        if self.actor is None:
            super().collect_rollout()
        else:
            self.actor.load_state_dict(self.net.state_dict())
            net, dev = self.net, self.device
            self.net, self.device = self.actor, self.act_device
            try:
                super().collect_rollout()
            finally:
                self.net, self.device = net, dev
        self._rollout_s = time.time() - t0

    def _draw_coverage(self, obs_tensor, card_mask):
        # The coverage term is off and TrainNetV3 skips its placement pass.
        return {"slot": torch.full((obs_tensor.shape[0],), self.net.hand_size, dtype=torch.long,
                                   device=obs_tensor.device)}

    def on_step(self, ctx):
        n = self.cfg.num_envs
        infos = ctx.infos
        kind = _col(infos, "opp_kind", n, -1, np.int64)
        mid = _col(infos, "midmatch", n, 0.0)
        names = infos.get("opp_ckpt")
        priv = infos.get("priv")
        if self.priv is not None:
            self.priv.record(infos)
        if self._clean is not None:
            self._clean.record(infos)
        leak = np.asarray(ctx.stats.get("leak_elixir", np.zeros(n)), dtype=np.float64).reshape(-1)
        dropped = _col(infos, "act_dropped", n, 0.0)
        for i in range(n):
            if (ctx.prev_dones[i] or not self._ep_open[i]) and kind[i] >= 0:
                # A phantom step returns the next episode's reset info; the very
                # first episodes are opened by their first real step.
                self._ep_open[i] = True
                self._ep_kind[i] = int(kind[i])
                self._ep_name[i] = str(names[i]) if names is not None and names[i] is not None else ""
                self._ep_mid[i] = float(mid[i])
                self._ep_start[i] = (np.asarray(priv[i], dtype=np.float64)[PE.PRIV["towers"]].copy()
                                     if priv is not None else np.ones(6))
                self._ep_steps[i], self._ep_leak[i] = 0, 0.0
                self.started[PE.KIND_NAMES[int(kind[i])]] += 1
            if not ctx.prev_dones[i]:
                self._ep_steps[i] += 1
                self._ep_leak[i] += float(leak[i])
                self._win["leak"] += float(leak[i])
                self._win["dropped"] += float(dropped[i])
                self._win["steps"] += 1

    def on_episode_end(self, i, ctx):
        priv = ctx.infos.get("priv")
        end = (np.asarray(priv[i], dtype=np.float64)[PE.PRIV["towers"]] if priv is not None
               else np.ones(6))
        start = self._ep_start[i]
        outcome = self.metrics.finish_episode(i, float(ctx.raw_rewards[i]), float(end[:3].sum()),
                                              float(end[3:].sum()))
        kind = PE.KIND_NAMES[self._ep_kind[i]]
        add = {"episodes": 1, "wins": int(outcome > 0), "draws": int(outcome == 0),
               "losses": int(outcome < 0),
               # sum over the three towers of the health fraction lost in this episode
               "dmg_dealt": float(np.clip(start[3:] - end[3:], 0.0, None).sum()),
               "dmg_taken": float(np.clip(start[:3] - end[:3], 0.0, None).sum()),
               "crowns_for": 3 if end[3] <= 0 else int((end[4:] <= 0).sum()),
               "crowns_against": 3 if end[0] <= 0 else int((end[1:3] <= 0).sum()),
               "midmatch": self._ep_mid[i], "steps": self._ep_steps[i], "leak": self._ep_leak[i]}
        for rows in (self.kstats[kind], self._win["kstats"][kind]):
            for k, v in add.items():
                rows[k] += v
        if self._ep_name[i]:
            r = self.by_ckpt.setdefault(f"{kind}:{self._ep_name[i]}", {"episodes": 0, "wins": 0})
            r["episodes"] += 1
            r["wins"] += int(outcome > 0)
        self._win["episodes"] += 1
        self._ep_open[i] = False

    # -- schedules ------------------------------------------------------------------
    def warming(self):
        return self.n_updates < self.args.critic_warmup

    def coefs(self):
        a = self.args
        f = self.episodes_completed / max(1.0, a.anchor_decay)
        kl = log_interp(a.kl_start, a.kl_end, f)
        if getattr(a, "kl_final", 0.0) > 0 and a.kl_final_at > a.kl_final_from:
            f2 = (self.episodes_completed - a.kl_final_from) / (a.kl_final_at - a.kl_final_from)
            if f2 > 0:
                kl = min(kl, log_interp(a.kl_end, a.kl_final, f2))
        kl *= self.kl_boost
        bc = log_interp(a.bc_start, a.bc_end, f) if self.bc is not None else 0.0
        return kl, bc

    def when_ref(self, obs):
        """1.0 where a decision keeps its WHEN reference, 0.0 where
        --anchor-free-elixir exempts it (obs: (..., obs_dim), the learner's rows)."""
        e, b = self.args.anchor_free_elixir, self.args.anchor_free_behind
        if e is None and not b:
            return None
        return BT.when_kept(obs, e, b).float()

    # -- the anchor terms of one minibatch ---------------------------------------------
    def anchor_terms(self, loc):
        """KL to the clone by head (both directions) and the head entropies of
        one PPO minibatch, from the update's own tensors."""
        tt, ee = loc["tt"], loc["ee"]
        dec, placed = loc["mb_decision"], loc["mb_placed"]
        n_dec = dec.sum().clamp(min=1.0)
        cl = loc["cl_seq"]
        lq = self._ref["card"][tt, ee]
        kw_r, kh_r, pp = kl_heads(cl, lq)            # KL(pi || clone)
        kw_f, kh_f, pq = kl_heads(lq, cl)            # KL(clone || pi)
        lpp = torch.log_softmax(loc["pl_seq"], dim=-1)
        lqp = self._ref["place"][tt, ee]
        wref = self.when_ref(loc["mb_obs_seq"])
        dec_w = dec if wref is None else dec * wref
        t = {"when_r": (kw_r * dec_w).sum() / n_dec, "when_f": (kw_f * dec_w).sum() / n_dec,
             # the slot choice weighs in by the probability of playing at all
             "what_r": (pp * kh_r * dec).sum() / n_dec, "what_f": (pq * kh_f * dec).sum() / n_dec,
             # WHERE of the card that was taken: a one-sample estimate of the
             # expectation over cards
             "where_r": (kl_cat(lpp, lqp) * placed).sum() / n_dec,
             "where_f": (kl_cat(lqp, lpp) * placed).sum() / n_dec}
        zero = torch.zeros((), device=cl.device)
        t["ability_r"], t["ability_f"] = zero, zero
        fwd = loc["fwd"]
        if self._ref["ability"] is not None and len(fwd) > 6 and fwd[6]:
            ready = loc["ability_ready_seq"][tt, ee].float()
            for k, lg in enumerate(fwd[6]):
                la = torch.log_softmax(lg, dim=-1)
                lb = torch.log_softmax(self._ref["ability"][tt, ee, k], dim=-1)
                t["ability_r"] = t["ability_r"] + (kl_cat(la, lb) * ready[..., k]).sum() / n_dec
                t["ability_f"] = t["ability_f"] + (kl_cat(lb, la) * ready[..., k]).sum() / n_dec
        m = self.args.kl_mix
        t["kl"] = sum(w * ((1.0 - m) * t[h + "_r"] + m * t[h + "_f"])
                      for w, h in zip(self.kl_w, ("when", "what", "where", "ability")))
        hw, hh, _ = head_entropies(cl)
        t["h_when"] = (hw * dec).sum() / n_dec
        w = pp.detach() * dec
        t["h_what"] = (hh * w).sum() / w.sum().clamp(min=1e-6)
        if wref is not None:
            t["when_free"] = (dec * (1.0 - wref)).sum() / n_dec      # share of decisions exempt
        return t

    def extra_loss(self, loss, loc):
        a = self.args
        with torch.set_grad_enabled(not self._warming):
            t = self.anchor_terms(loc)
        for k, v in t.items():
            self._mb.setdefault(k, []).append(float(v.detach()))
        if self.priv is not None:
            self.priv.train_step(loc)
        if self._warming:
            return 0.5 * loc["critic_loss"]
        kl_c, bc_c = self._coefs
        total = loss + kl_c * t["kl"] - a.ent_when * t["h_when"] - a.ent_what * t["h_what"]
        if bc_c > 0 and self._bc_iter is not None:
            b = self.bc
            lbc, parts = BT.forward_losses(self.net, next(self._bc_iter), what_h=b["what_h"],
                                           what_on_when=b["what_on_when"], weights=b["weights"],
                                           when_free_elixir=a.anchor_free_elixir,
                                           when_free_behind=a.anchor_free_behind)
            total = total + bc_c * lbc
            self._mb.setdefault("bc", []).append(float(lbc.detach()))
            for k, v in parts.items():
                self._mb.setdefault("bc_" + k, []).append(float(v.detach() if torch.is_tensor(v) else v))
        return total

    # -- diagnostics on the rollout: learner and clone on the same states ----------------
    @torch.no_grad()
    def diagnose(self, batch, ref=None, cur=None):
        net = self.net
        ref = ref or policy_tables(self.ref_net, batch)
        cur = cur or policy_tables(net, batch)
        dec = batch["decision"] > 0
        valid = batch["valid"] > 0
        cards = batch["card_actions"]
        placed = dec & (cards != net.hand_size)
        n_dec = float(dec.sum().clamp(min=1))
        n_valid = float(valid.sum().clamp(min=1))
        d = dec.float()
        out = {"decision_rows": int(dec.sum()), "valid_rows": int(valid.sum())}

        def _mean(x, w=d, n=n_dec):
            return float((x * w).sum()) / n

        kl = {}
        for tag, (p, q) in (("r", ("cur", "ref")), ("f", ("ref", "cur"))):
            tab = {"cur": cur, "ref": ref}
            kw, kh, pp = kl_heads(tab[p]["card"], tab[q]["card"])
            wref = self.when_ref(batch["obs"])
            kl["when_" + tag] = _mean(kw if wref is None else kw * wref)
            if wref is not None and tag == "r":
                # what the exempt decisions would have added, for the log only
                kl["when_free_r"] = _mean(kw * (1.0 - wref))
                out["when_free_share"] = _mean(1.0 - wref)
            kl["what_" + tag] = _mean(pp * kh)
            kl["where_" + tag] = _mean(kl_cat(tab[p]["place"], tab[q]["place"]), placed.float())
        m = self.args.kl_mix
        kl["total"] = sum(w * ((1.0 - m) * kl[h + "_r"] + m * kl[h + "_f"])
                          for w, h in zip(self.kl_w, ("when", "what", "where")))
        out["kl_pre"] = kl

        hw, hh, pp = head_entropies(cur["card"])
        hw_q, hh_q, pq = head_entropies(ref["card"])
        pl = placed.float()
        n_pl = float(pl.sum().clamp(min=1))

        def _h_where(tab):
            lp = tab["place"]
            return float(((-(lp.exp() * torch.where(torch.isfinite(lp), lp, torch.zeros_like(lp))).sum(-1))
                          * pl).sum()) / n_pl
        out["entropy"] = {"when": _mean(hw), "what": float((hh * pp * d).sum() / (pp * d).sum().clamp(min=1e-6)),
                          "where": _h_where(cur)}
        out["entropy_clone"] = {"when": _mean(hw_q),
                                "what": float((hh_q * pq * d).sum() / (pq * d).sum().clamp(min=1e-6)),
                                "where": _h_where(ref)}
        out["p_play"] = _mean(pp)
        out["p_play_clone"] = _mean(pq)
        out["play_ratio"] = out["p_play"] / max(1e-9, out["p_play_clone"])
        # plays per game minute: sampled, and expected under each policy on these states
        plays = valid & (cards != net.hand_size)
        out["plays_per_min"] = per_minute(float(plays.sum()), n_valid, self.dt)
        out["plays_per_min_expected"] = per_minute(float((pp * d).sum()), n_valid, self.dt)
        out["plays_per_min_clone_expected"] = per_minute(float((pq * d).sum()), n_valid, self.dt)
        out["sampled_plays"] = int(plays.sum())

        T, N = cards.shape
        ids = net.hand_card_ids(batch["obs"].reshape(T * N, -1))                  # (TN, hand)
        ids = ids.clamp(min=0)
        n_ids = int(net.num_card_ids)

        def _by_card(weights):                                                    # (TN, hand)
            acc = torch.zeros(n_ids, dtype=torch.float32, device=ids.device)
            return acc.index_add_(0, ids.reshape(-1), weights.reshape(-1).float())
        slot_p = cur["card"][..., :-1].exp() * d.unsqueeze(-1)
        slot_q = ref["card"][..., :-1].exp() * d.unsqueeze(-1)
        e_p, e_q = _by_card(slot_p.reshape(T * N, -1)), _by_card(slot_q.reshape(T * N, -1))
        taken = F.one_hot(cards.clamp(max=net.hand_size - 1), net.hand_size) * plays.unsqueeze(-1)
        s_n = _by_card(taken.reshape(T * N, -1))
        tot_p, tot_q, tot_n = float(e_p.sum()), float(e_q.sum()), float(s_n.sum())
        cards_out, ratios = {}, {}
        for cid in sorted(set(self.deck) | {int(c) for c in (s_n > 0).nonzero().reshape(-1).tolist()}):
            name = card_name(cid)
            sp, sq = float(e_p[cid]) / max(1e-9, tot_p), float(e_q[cid]) / max(1e-9, tot_q)
            cards_out[name] = {"plays": int(s_n[cid]), "share": round(float(s_n[cid]) / max(1.0, tot_n), 4),
                               "per_min": round(per_minute(float(s_n[cid]), n_valid, self.dt), 3),
                               "share_expected": round(sp, 4), "share_clone": round(sq, 4)}
            if sq >= 0.02:
                ratios[name] = sp / sq
        out["cards"] = cards_out
        out["share_ratio"] = {k: round(v, 3) for k, v in ratios.items()}
        return out, ratios

    # -- update -----------------------------------------------------------------------
    def run_update(self):
        cfg, net, a = self.cfg, self.net, self.args
        batch = self.buffer.drain()
        t0 = time.time()
        if a.update_threads > 0:
            torch.set_num_threads(a.update_threads)
        if self.actor is not None:
            batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            next_obs_t = torch.tensor(self._obs, dtype=torch.float32).to(self.device)
            next_value = net.heads(net.extract_features_hires(next_obs_t)[0])["value"].squeeze(-1)
        if self.priv is None:
            advantages = gae_mod.compute_gae(batch["rewards"], batch["values"], batch["masks"], next_value,
                                             cfg.gamma, cfg.gae_lambda)
            returns = advantages + batch["values"]
        else:
            # V = V_net(obs) + g(true state) for the advantages and the return
            # target; batch["values"] (the net's own, for its value clipping) stays.
            res, res_next = self.priv.residuals(batch["values"].shape[0])
            v_priv = batch["values"] + res.to(batch["values"].dtype)
            advantages = gae_mod.compute_gae(batch["rewards"], v_priv, batch["masks"],
                                             next_value + res_next.to(next_value.dtype),
                                             cfg.gamma, cfg.gae_lambda)
            returns = advantages + v_priv
        adv_norm = gae_mod.normalize(advantages, mask=batch["decision"])
        with torch.no_grad():
            keep = batch["valid"] > 0.5
            r, v = returns[keep], batch["values"][keep]
            ev = float(gae_mod.explained_variance(r, v))
            self._ev_priv = (float(gae_mod.explained_variance(r, v_priv[keep]))
                             if self.priv is not None else None)
            vf_clip_range = max(cfg.vf_clip_std_frac * gae_mod.safe_std(r), cfg.eps_clip)
            reward_mean = float(batch["rewards"][keep].mean())
        if self._clean is None:
            self._ref = policy_tables(self.ref_net, batch)
        else:
            # the clone's answer from the true values (the learner's rows keep
            # the readings: the span is put back before anything else runs)
            with self._clean.swapped(batch["obs"]):
                self._ref = policy_tables(self.ref_net, batch)
        diag, ratios = self.diagnose(batch, ref=self._ref)
        t_diag = time.time() - t0

        warming = self._warming = self.warming()
        self._coefs = kl_c, bc_c = self.coefs()
        self._mb = {}
        for g in self.optimizer.param_groups:
            g["lr"] = a.warmup_lr if warming else a.lr
        if warming:
            for name, p in net.named_parameters():
                p.requires_grad_(name.startswith("value_head"))
        elif bc_c > 0 and self.bc is not None:
            self._bc_iter = self._bc_stream()
        try:
            stats = self.updater.update(
                batch, adv_norm, returns, vf_clip_range,
                # the engine's card entropy is over the composed (slots, wait)
                # distribution, i.e. a bonus for playing more: off; the heads
                # have their own terms in extra_loss
                ent_coef_card=0.0, ent_coef_placement=0.0 if warming else a.ent_where,
                coverage_coef=0.0, collect_per_card=False, deck_coverage_coef=0.0, aux_scale=0.0)
        finally:
            if warming:
                for p in net.parameters():
                    p.requires_grad_(True)
            if self._bc_iter is not None:
                self._bc_iter.close()
                self._bc_iter = None
            self._ref = None
            if self.priv is not None:
                self.priv.done()
            if a.update_threads > 0 and a.torch_threads > 0:
                torch.set_num_threads(a.torch_threads)
        self.n_updates += 1
        self._explained_variance, self._vf_clip_range = ev, vf_clip_range
        self._record(stats, diag, ratios, warming, kl_c, bc_c, ev, reward_mean, time.time() - t0, t_diag)
        return stats

    def _record(self, stats, diag, ratios, warming, kl_c, bc_c, ev, reward_mean, update_s, diag_s):
        a, cfg = self.args, self.cfg

        def mb(key, first=False):
            v = self._mb.get(key) or []
            if not v:
                return None
            return float(v[0]) if first else float(np.mean(v))
        # the clone's own play rate: rollouts collected while the policy was still the clone
        if warming:
            self.baseline["plays"] += diag["sampled_plays"]
            self.baseline["steps"] += diag["valid_rows"]
        base_ppm = (per_minute(self.baseline["plays"], self.baseline["steps"], self.dt)
                    if self.baseline["steps"] > 0 else None)
        ppm_ratio = None if (warming or not base_ppm) else diag["plays_per_min"] / base_ppm
        reasons = [] if warming else guard_reasons(
            diag["play_ratio"], ratios, ppm_ratio, a.guard_play_lo, a.guard_play_hi, a.guard_share_lo,
            self.guard_ppm)
        g = self.guard
        g["streak"] = g["streak"] + 1 if reasons else 0
        if g["streak"] >= a.guard_patience and not g["tripped"]:
            g["tripped"], g["tripped_at"] = True, int(self.n_updates)
            print(f">>> GUARD TRIPPED at update {self.n_updates}: {'; '.join(reasons)}"
                  + (" -- stopping (--guard-stop)" if a.guard_stop else " -- run continues (no --guard-stop)"),
                  flush=True)
            if a.guard_stop:
                self._stop = "guard"
        if a.kl_max > 0 and not warming:
            if diag["kl_pre"]["total"] > a.kl_max:
                self.kl_boost = min(30.0, self.kl_boost * 1.5)
            elif diag["kl_pre"]["total"] < a.kl_max / 1.5:
                self.kl_boost = max(1.0, self.kl_boost / 1.5)

        now = time.time()
        wall = max(1e-6, now - self._t_last)
        self._t_last = now
        win = self._win
        steps = cfg.update_timestep * cfg.num_envs
        total_started = max(1, sum(self.started.values()))
        rec = {
            "t": now, "update": self.n_updates, "episodes": int(self.episodes_completed),
            "warmup": bool(warming), "decision_ticks": self.dt,
            "actor_loss": stats.actor_loss, "critic_loss": stats.critic_loss, "ev": ev,
            **({"ev_priv": self._ev_priv, "priv_loss": float(np.mean(self.priv.losses or [0.0]))}
               if self.priv is not None else {}),
            **({"sim2real_scale_range": self._s2r_now} if getattr(self, "_s2r_now", None) else {}),
            "clip_frac": stats.clip_frac, "ratio_dev_first": stats.ratio_dev_first,
            "nonfinite_skips": stats.nonfinite_skips, "reward_per_decision": reward_mean,
            "lr": a.warmup_lr if warming else a.lr,
            "kl_coef": kl_c, "kl_boost": self.kl_boost, "bc_coef": 0.0 if warming else bc_c,
            # KL to the clone before this update, over the whole rollout
            "kl_pre": {k: round(v, 7) for k, v in diag["kl_pre"].items()},
            # share of decisions --anchor-free-elixir left without a WHEN reference
            "when_free_share": diag.get("when_free_share"),
            # and through the update's minibatches (first = before any step of this update)
            "kl_upd": {h: mb(h) for h in ("when_r", "what_r", "where_r", "ability_r",
                                           "when_f", "what_f", "where_f", "ability_f", "kl")},
            "kl_first": mb("kl", first=True),
            "bc_loss": mb("bc"),
            "bc_parts": {k[3:]: round(float(np.mean(v)), 5) for k, v in self._mb.items()
                         if k.startswith("bc_")},
            "entropy": {k: round(v, 5) for k, v in diag["entropy"].items()},
            "entropy_clone": {k: round(v, 5) for k, v in diag["entropy_clone"].items()},
            "entropy_where_frac": stats.ent_placement,
            "p_play": diag["p_play"], "p_play_clone": diag["p_play_clone"],
            "plays_per_min": diag["plays_per_min"],
            "plays_per_min_expected": diag["plays_per_min_expected"],
            "plays_per_min_clone_expected": diag["plays_per_min_clone_expected"],
            "plays_per_min_clone_baseline": base_ppm,
            "cards": diag["cards"],
            "elixir_leaked_per_min": per_minute(win["leak"], max(1, win["steps"]), self.dt),
            "plays_dropped_by_delay": win["dropped"],
            "by_kind_recent": kind_summary(win["kstats"]),
            "by_kind": kind_summary(self.kstats),
            "by_ckpt": {k: {"episodes": v["episodes"], "win_rate": round(v["wins"] / v["episodes"], 4)}
                        for k, v in sorted(self.by_ckpt.items()) if v["episodes"]},
            "mix": {k: round(v / total_started, 4) for k, v in self.started.items()},
            "mix_target": {k: round(float(v) / max(1e-9, sum(self.pool["weights"].values())), 4)
                           for k, v in self.pool["weights"].items()},
            "episodes_started": dict(self.started),
            "guard": {"play_ratio": round(diag["play_ratio"], 4), "share_ratio": diag["share_ratio"],
                      "ppm_ratio": None if ppm_ratio is None else round(ppm_ratio, 4),
                      "breach": reasons, "streak": g["streak"], "tripped": g["tripped"]},
            "episodes_this_update": win["episodes"], "episodes_per_s": win["episodes"] / wall,
            "decisions_per_s": steps / max(1e-6, self._rollout_s),
            "rollout_s": self._rollout_s, "env_wait_s": self._env_s,
            "update_s": update_s, "diag_s": diag_s, "wall_s": wall,
            "pool_note": self.pool_note,
        }
        outs = list(self.metrics.outcomes)[-200:]
        if outs:
            rec["win_rate_200"] = float((np.asarray(outs, dtype=np.float64) > 0.5).mean())
        self._rec = rec
        self._reset_window()

    def log_update(self, stats):
        rec = self._rec
        self.progress.write(json.dumps(rec) + "\n")
        self.progress.flush()
        w, ep = self.writer, self.episodes_completed
        for key in ("actor_loss", "critic_loss", "ev", "clip_frac", "kl_coef", "bc_coef", "bc_loss",
                    "p_play", "plays_per_min", "elixir_leaked_per_min", "episodes_per_s"):
            if rec.get(key) is not None:
                w.add_scalar("Pool/" + key, rec[key], ep)
        for k, v in rec["kl_pre"].items():
            w.add_scalar("KL/" + k, v, ep)
        for k, v in rec["by_kind"].items():
            w.add_scalar("WinRate/" + k, v["win_rate"], ep)
        kp = rec["kl_pre"]
        wr = " ".join(f"{k} {v['win_rate']:.2f}/{v['episodes']}" for k, v in rec["by_kind"].items())
        bc = "-" if rec["bc_loss"] is None else f"{rec['bc_loss']:.3f}"
        print(f">>> [pool] upd {rec['update']} ep {rec['episodes']} "
              f"{'[critic warm-up] ' if rec['warmup'] else ''}"
              f"actor {rec['actor_loss']:.4f} critic {rec['critic_loss']:.4f} EV {rec['ev']:.2f} "
              f"KL when/what/where {kp['when_r']:.4f}/{kp['what_r']:.4f}/{kp['where_r']:.4f} "
              f"(x{rec['kl_coef']:.2f}) BC {bc} (x{rec['bc_coef']:.2f}) "
              f"plays/min {rec['plays_per_min']:.1f} (clone {rec['plays_per_min_clone_expected']:.1f}) "
              f"wins {wr} | {rec['episodes_per_s']:.2f} ep/s, rollout {rec['rollout_s']:.0f}s "
              f"(env {rec['env_wait_s']:.0f}s) update {rec['update_s']:.0f}s"
              + (f" | GUARD {rec['guard']['breach']}" if rec["guard"]["breach"] else "")
              + (f" | {self.pool_note}" if self.pool_note else ""), flush=True)

    def periodic(self, stats):
        if self.episodes_completed - self._last_snapshot_ep >= self.args.snapshot_every:
            self.take_snapshot()
        self._widen_sim2real()
        self.save_checkpoint()

    def should_stop(self):
        a = self.args
        if self._stop:
            return True
        if a.max_updates and self.n_updates >= a.max_updates:
            return True
        return self.episodes_completed >= a.max_episodes

    def request_stop(self, why="signal"):
        self._stop = self._stop or why

    def on_finish(self):
        self.save_checkpoint()
        print(f">>> finished ({self._stop or 'limit reached'}): update {self.n_updates}, "
              f"episode {self.episodes_completed}; {self.latest_path}", flush=True)

    def close(self):
        if self.progress is not None:
            self.progress.close()
            self.progress = None
        if self.envs is not None and self._envs_override is None:
            try:
                self.envs.close()
            except Exception as e:  # noqa: BLE001
                print(f">>> closing the envs: {e}", flush=True)
            self.envs = None


def main(args=None):
    a = args or ARGS
    print("argv:", " ".join(sys.argv), flush=True)
    tr = PoolTrainer(a)

    def _stop(signum, _frame):
        print(f">>> signal {signum}: stopping after this update", flush=True)
        tr.request_stop("signal")
    signal.signal(signal.SIGTERM, _stop)
    try:
        tr.run()
    finally:
        tr.close()
    return tr


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
