"""PPO from the BC policy, minimal reward, real-deck opponents.

    .venv/bin/python -m training.ppo_bc --run ppo_bc26 --init checkpoints/bc_hog26.pth \
        --num-envs 20 [--start-rung 0] [--max-episodes 400000]

The engine's own Phase1Trainer (engine/python_ai/trainers/train.py) runs the
rollouts, GAE, the PPO update and the teacher curriculum; this subclass changes,
from outside engine/:

  reward     training/reward.py: outcome +-1, PBRS tower term with Phi(terminal)=0,
             elixir-leak penalty (worst case 0.2 of a win). Engine shaping,
             flawless-defense bonus and draw penalty are switched off.
  opponents  every episode an opponent deck from the REAL sample
             (data/decks/train_pool.json, training/opponents.py), piloted by the
             UtilityTeacher at the current curriculum rung.
  regularizers off: advisor coverage, placement coverage, deck coverage,
             defensive scenarios. Entropy: FIXED small coefficients instead of
             the adaptive controller (it would push placement entropy back up
             and dissolve the human placements).
  BC anchor  * critic warm-up: the first --critic-warmup updates train only the
               value head (actor frozen), so the first policy gradients use a
               critic fitted to THIS reward and opponent mix;
             * KL(pi || pi_BC) on rollout states (card + placement-of-the-taken-
               card + ability), coefficient log-decaying --kl-start -> --kl-end
               over --anchor-decay episodes and held at --kl-end after;
             * BC cross-entropy on human dataset chunks every minibatch, weight
               --bc-start -> --bc-end on the same schedule.
             The two floors never reach zero: the policy may move away from the
             data where it pays, but keeps paying to forget it.
  logging    TensorBoard runs/<run>/ (engine series + BC/*, Opp/*), and
             runs/<run>/progress.jsonl one line per update.

Resumable: re-running the same command resumes from checkpoints/<run>.pth.
"""
import argparse
import inspect
import json
import math
import os
import shutil
import sys
import textwrap
import time


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="ppo_bc26")
    p.add_argument("--init", default=None, help="BC weights (bare state_dict) for a fresh start")
    p.add_argument("--num-envs", type=int, default=int(os.environ.get("CLASH_NUM_ENVS", 8)))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-episodes", type=int, default=1_000_000)
    p.add_argument("--start-rung", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--k", type=float, default=0.0, help="convex tower transform")
    p.add_argument("--leak-worst", type=float, default=0.2)
    p.add_argument("--draw", type=float, default=0.0)
    p.add_argument("--p-freq", type=float, default=0.5,
                   help="share of episodes with an opponent deck drawn by real frequency (rest PFSP)")
    p.add_argument("--ent-card", type=float, default=0.01)
    p.add_argument("--ent-place", type=float, default=0.005)
    p.add_argument("--critic-warmup", type=int, default=6, help="updates")
    p.add_argument("--kl-start", type=float, default=0.1)
    p.add_argument("--kl-end", type=float, default=0.01)
    p.add_argument("--bc-start", type=float, default=0.1)
    p.add_argument("--bc-end", type=float, default=0.01)
    p.add_argument("--anchor-decay", type=float, default=30000, help="episodes")
    p.add_argument("--bc-batch", type=int, default=16, help="human matches per BC chunk")
    p.add_argument("--val-every", type=int, default=10, help="updates between held-out BC evals")
    p.add_argument("--save-every", type=int, default=200, help="episodes")
    # FirstLight-Zero ports; every default is the old behaviour.
    p.add_argument("--arch", default="v1", choices=("v1", "net_v2"),
                   help="net_v2: training/net_v2.NetV2 (cfg from --init's blob)")
    p.add_argument("--act-delay", default="",
                   help="lo,hi,jitter ticks of actuation delay (training/reward.ActuationDelay)")
    p.add_argument("--w-unilateral", type=float, default=0.0,
                   help="asymmetric elixir-overflow penalty coefficient (FLZ: 0.075)")
    p.add_argument("--king-chip", type=float, default=1.0,
                   help="King damage weight in Phi while both Princesses stand (FLZ: 0.2)")
    # Perception noise on the actor's observation (training/obs_noise.py). Off by default.
    p.add_argument("--obs-noise", default="",
                   help="'default' (training/obs_noise_default.json) or a config JSON path; "
                        "'' = off")
    p.add_argument("--obs-noise-scale", default="",
                   help="override the config's scale: '0.5', or 'lo:hi' redrawn every episode")
    p.add_argument("--obs-noise-hp-live", action="store_true",
                   help="write unit HP as the live driver does (full HP, never measured)")
    return p.parse_args()


def parse_delay(s):
    return tuple(int(v) for v in s.split(",")) if s else None


def obs_noise_spec(args):
    """env_config["obs_noise"] for these flags, or None when --obs-noise is not given."""
    if not getattr(args, "obs_noise", ""):
        return None
    from training import obs_noise
    return obs_noise.cli_spec(args.obs_noise, getattr(args, "obs_noise_scale", ""),
                              getattr(args, "obs_noise_hp_live", False))


ARGS = _parse() if __name__ == "__main__" else None

# --- environment switches, BEFORE any engine import reads them -------------
if ARGS is not None:
    os.environ["CLASH_NUM_ENVS"] = str(ARGS.num_envs)
    os.environ["CLASH_GAMMA"] = str(ARGS.gamma)
    os.environ["CLASH_SAVE_EVERY"] = str(ARGS.save_every)
    if ARGS.seed is not None:
        os.environ["CLASH_SEED"] = str(ARGS.seed)
for k, v in {"CLASH_ADVISOR_COVERAGE_COEF": "0", "CLASH_PLACEMENT_COVERAGE_COEF": "0",
             "CLASH_DECK_COVERAGE_COEF": "0", "CLASH_PHASE1_SCENARIO_PROB": "0",
             "CLASH_OFFENSIVE_SCENARIO_PROB": "0", "CLASH_PHASE1_DECK_POOL": "0"}.items():
    os.environ.setdefault(k, v)

from training import common as C  # noqa: E402

REPO = C.REPO
if ARGS is not None:
    os.environ["CLASH_WEIGHTS"] = os.path.join(REPO, "checkpoints", f"{ARGS.run}.pth")
    os.environ["CLASH_LOGDIR"] = os.path.join(REPO, "runs", ARGS.run)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import gymnasium as gym  # noqa: E402

from python_ai.rl import base_trainer as bt  # noqa: E402
from python_ai.rl import ppo as ppo_mod  # noqa: E402
from python_ai.rl import checkpointing as ckpt  # noqa: E402
from python_ai.rl import gae as gae_mod  # noqa: E402
from python_ai.rl import abilities as ability_mod  # noqa: E402
from python_ai.rl.config import aux_warmup_scale  # noqa: E402
from python_ai.trainers import train as train_mod  # noqa: E402

from training import reward as R  # noqa: E402
from training import bc as BC  # noqa: E402
from training import bcdata as D  # noqa: E402


def log_interp(a, b, frac):
    frac = min(1.0, max(0.0, frac))
    return float(a * (b / a) ** frac) if a > 0 and b > 0 else float(a + (b - a) * frac)


# ---------------------------------------------------------------------------
# PPOUpdater with one extra hook in the loss. The engine's update() source is
# reused verbatim with a single injected line before `optimizer.zero_grad()`, so
# every engine fix to update() is picked up; the anchor is asserted.
# ---------------------------------------------------------------------------
_ANCHOR = "                self.optimizer.zero_grad()\n                loss.backward()\n"


def _build_hooked_update():
    src = textwrap.dedent(inspect.getsource(ppo_mod.PPOUpdater.update))
    # The method body is dedented by 4, so the in-loop anchor sits at 12 spaces.
    anchor = "".join(l[4:] for l in _ANCHOR.splitlines(keepends=True))
    assert src.count(anchor) == 1, "PPOUpdater.update changed: hook anchor not found"
    hook = anchor.replace("self.optimizer.zero_grad()",
                          "loss = self._extra_loss(loss, locals())\n"
                          "            self.optimizer.zero_grad()", 1)
    src = src.replace(anchor, hook)
    ns = dict(vars(ppo_mod))
    exec(compile(src, ppo_mod.__file__ + ":hooked_update", "exec"), ns)
    return ns["update"]


class AnchoredUpdater(ppo_mod.PPOUpdater):
    update = _build_hooked_update()

    def __init__(self, net, optimizer, cfg, owner):
        super().__init__(net, optimizer, cfg)
        self.owner = owner

    def _extra_loss(self, loss, loc):
        return self.owner.extra_loss(loss, loc)


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------
def make_reward_env(seed, spec, pool, p_freq, act_delay=None, obs_noise=None):
    def _init():
        from training import common  # noqa: F401  (worker: sys.path, CLASH_DECK)
        from training.reward import RewardEnv
        cfg = {"opponent": "teacher", "teacher_stage": 0, "scenario_seed": seed,
               "deck_pool": False, "defensive_scenario_prob": 0.0,
               "reward_spec": spec, "real_pool": pool, "p_freq": p_freq}
        if act_delay is not None:
            cfg["act_delay"] = act_delay
        if obs_noise is not None:
            cfg["obs_noise"] = obs_noise
        return RewardEnv(cfg)
    return _init


def _install_arch(args):
    """--arch net_v2: the engine trainer builds `bt.MicroRoyaleNet(num_ability_slots=...)`
    through the module global, so rebinding it swaps the network everywhere the
    trainer constructs one. Returns the NetV2 cfg (None for v1)."""
    if args.arch != "net_v2":
        return None
    from training import net_v2 as N2
    cfg = dict(N2.DEFAULTS)
    for src in (args.init, os.environ.get("CLASH_WEIGHTS")):
        if src and os.path.exists(src):
            ck = torch.load(src, map_location="cpu", weights_only=False)
            if isinstance(ck, dict) and ck.get("cfg"):
                cfg.update(ck["cfg"])
                break

    def _factory(num_ability_slots=0, **_):
        return N2.NetV2(num_ability_slots=num_ability_slots, **cfg)
    bt.MicroRoyaleNet = _factory
    return cfg


class BCPPOTrainer(train_mod.Phase1Trainer):
    pipeline_name = "pipeline1"
    replay_prefix = "bcppo"

    def __init__(self, args):
        self.args = args
        self.spec = R.RewardSpec(gamma=args.gamma, k=args.k, leak_worst=args.leak_worst,
                                 draw=args.draw, king_chip=args.king_chip,
                                 w_unilateral=args.w_unilateral)
        R.install_reward(self.spec)
        self.arch_cfg = _install_arch(args)
        super().__init__()
        self.pool_path = C.data_dir("decks", "train_pool.json")
        self.n_updates = 0
        self.run_dir = self.log_dir
        self._bc_h = None
        self._ref = None
        self._stream = None
        self._stream_h = None
        self._kl_log, self._bc_log = [], []
        # Snapshots stay in this repo, not engine/.
        hist = os.path.join(REPO, "checkpoints", f"{args.run}_snapshots")
        stage = os.path.join(REPO, "checkpoints", f"{args.run}_stages")
        orig_hist = ckpt.save_historical_snapshot

        def _hist(net, ep, pipeline, directory=None):
            return orig_hist(net, ep, pipeline, directory=hist)
        bt.save_historical_snapshot = _hist
        train_mod.STAGE_CHECKPOINT_DIR = stage
        bt.HISTORICAL_CHECKPOINT_DIR = hist

    # -- envs ---------------------------------------------------------------
    def build_envs(self):
        spec = {k: v for k, v in self.spec.as_dict().items() if k != "c_leak"}
        seeds = bt.worker_seeds(self.cfg.seed, self.cfg.num_envs)
        return gym.vector.AsyncVectorEnv(
            [make_reward_env(s, spec, self.pool_path, self.args.p_freq,
                             parse_delay(self.args.act_delay), obs_noise_spec(self.args))
             for s in seeds])

    def build_replay_env(self):
        return None

    def should_stop(self):
        return self.episodes_completed >= self.args.max_episodes

    def on_finish(self):
        self.save_checkpoint()

    # -- resume -------------------------------------------------------------
    def checkpoint_payload(self):
        d = super().checkpoint_payload()
        d["bcppo"] = {"n_updates": self.n_updates, "args": vars(self.args),
                      "reward": self.spec.as_dict()}
        if self.arch_cfg is not None:
            from training import net_v2 as N2
            d["arch"], d["cfg"] = N2.ARCH, self.arch_cfg
        return d

    def load_checkpoint(self):
        if not os.path.exists(self.weight_path) and self.args.init:
            # Fresh start from BC: the bare state_dict loads as "legacy weights".
            ck = torch.load(self.args.init, map_location="cpu", weights_only=False)
            if isinstance(ck, dict) and "model" in ck and "optimizer" not in ck:
                torch.save(ck["model"], self.weight_path)   # a net_v2 BC blob
            else:
                shutil.copyfile(self.args.init, self.weight_path)
            print(f">>> BC init: {self.args.init} -> {self.weight_path}")
        full = super().load_checkpoint()
        if full:
            ck = torch.load(self.weight_path, map_location="cpu", weights_only=False)
            self.n_updates = int((ck.get("bcppo") or {}).get("n_updates", 0))
        return full

    # -- setup --------------------------------------------------------------
    def setup(self):
        super().setup()
        a = self.args
        for g in self.optimizer.param_groups:
            g["lr"] = a.lr
        self.updater = AnchoredUpdater(self.net, self.optimizer, self.cfg, self)
        if not self.full_resume and a.start_rung:
            self.curriculum.stage = int(a.start_rung)
            self.envs.call("set_teacher_stage", self.curriculum.teacher_stage)
        # Frozen BC reference.
        self.bc_net = None
        if a.init:
            from training.net_v2 import load_any
            self.bc_net = load_any(a.init, self.device)
            for p in self.bc_net.parameters():
                p.requires_grad_(False)
        # Human data stream (train split) and a fixed held-out slice.
        self.bcdata = None
        try:
            self.bcdata = D.BCData(verbose=True)
            D.set_obs_dim(self.bcdata.obs_dim)
            self._rng = np.random.default_rng(12345)
            self.val_eps = self.bcdata.val[:120]
        except Exception as e:  # noqa: BLE001
            self.bcdata = None   # BCData may have loaded before a later line raised
            print(f">>> no BC data for the auxiliary loss ({e}); BC term off")
        self._bc_h = BC.zero_hidden(self.net, self.cfg.num_envs, self.device)
        self.progress = open(os.path.join(self.log_dir, "progress.jsonl"), "a")
        print(f">>> BC-PPO: run {a.run} lr {a.lr} ent {a.ent_card}/{a.ent_place} "
              f"KL {a.kl_start}->{a.kl_end} BC {a.bc_start}->{a.bc_end} over {a.anchor_decay:.0f} eps, "
              f"critic warm-up {a.critic_warmup} updates, reward {self.spec.as_dict()}, "
              f"start rung {self.curriculum.stage}, updates so far {self.n_updates}, arch {a.arch}, "
              f"act_delay {parse_delay(a.act_delay)}", flush=True)

    # -- schedules ------------------------------------------------------------
    def coefs(self):
        a = self.args
        f = self.episodes_completed / max(1.0, a.anchor_decay)
        kl = log_interp(a.kl_start, a.kl_end, f) if self.bc_net is not None else 0.0
        bcw = log_interp(a.bc_start, a.bc_end, f) if self.bcdata is not None else 0.0
        return kl, bcw

    def warming(self):
        return self.n_updates < self.args.critic_warmup

    # -- BC reference over the rollout ------------------------------------------
    @torch.no_grad()
    def _bc_reference(self, batch):
        if self.bc_net is None:
            return None
        net = self.bc_net
        obs_all = batch["obs"]
        T, N = obs_all.shape[:2]
        L = self.cfg.bptt_chunk
        cards = batch["card_actions"]
        dec = batch["decision"]
        out_card = torch.zeros(T, N, net.hand_size + 1, device=obs_all.device)
        out_place = torch.zeros(T, N, net.placement_cells, device=obs_all.device)
        out_ab = None
        h = self._bc_h
        for t0 in range(0, T, L):
            sl = slice(t0, min(T, t0 + L))
            obs = obs_all[sl]
            l = obs.shape[0]
            flat = obs.reshape(l * N, -1)
            feats, emb, sp, hi = net.extract_features_hires(flat)
            cm = net.affordability_mask(flat)
            ci = cards[sl]
            act = ((dec[sl] > 0) & (ci != net.hand_size)).reshape(-1).nonzero(as_tuple=True)[0]
            fwd = net.forward_sequence(
                feats.view(l, N, -1), emb.view(l, N, net.hand_size + 1, -1),
                sp.view(l, N, *sp.shape[1:]), obs, cm.view(l, N, -1), ci, batch["masks"][sl], h,
                hires_seq=hi.view(l, N, *hi.shape[1:]), active_rows=act,
                with_ability=bool(net.num_ability_slots))
            h = fwd[4]
            out_card[sl] = torch.log_softmax(fwd[0].masked_fill(~torch.isfinite(fwd[0]), -1e9), -1)
            out_place[sl] = torch.log_softmax(fwd[1].masked_fill(~torch.isfinite(fwd[1]), -1e9), -1)
            if len(fwd) > 6 and fwd[6]:
                if out_ab is None:
                    out_ab = torch.zeros(T, N, len(fwd[6]), 2, device=obs_all.device)
                out_ab[sl] = torch.stack(fwd[6], dim=2)
        self._bc_h = (h[0].detach(), h[1].detach())
        return {"card": out_card, "place": out_place, "ability": out_ab}

    def _next_bc_chunk(self):
        if self.bcdata is None:
            return None, None
        if self._stream is None:
            self._stream = self._bc_gen()
        b, first = next(self._stream)
        if first or self._stream_h is None or self._stream_h[0].shape[0] != b["obs"].shape[1]:
            self._stream_h = BC.zero_hidden(self.net, b["obs"].shape[1], self.device)
        return b, first

    def _bc_gen(self):
        while True:
            for item in BC.chunk_batches(self.bcdata.train, self.args.bc_batch, self.cfg.bptt_chunk,
                                         self._rng, self.device, self.bcdata.obs_dim):
                yield item

    # -- the injected loss term --------------------------------------------------
    def extra_loss(self, loss, loc):
        if self.warming():
            return 0.5 * loc["critic_loss"]
        kl_c, bc_c = self.coefs()
        net = self.net
        extra = torch.zeros((), device=loss.device)
        if self._ref is not None and kl_c > 0:
            tt, ee = loc["tt"], loc["ee"]
            dec, placed = loc["mb_decision"], loc["mb_placed"]
            cl = loc["cl_seq"]
            lp = torch.log_softmax(cl.masked_fill(~torch.isfinite(cl), -1e9), -1)
            lq = self._ref["card"][tt, ee]
            legal = torch.isfinite(cl)
            kl_card = (lp.exp() * torch.where(legal, lp - lq, torch.zeros_like(lp))).sum(-1)
            pl = loc["pl_seq"]
            legal_p = torch.isfinite(pl) & (placed > 0).unsqueeze(-1)
            lpp = torch.log_softmax(pl.masked_fill(~torch.isfinite(pl), -1e9), -1)
            lqp = self._ref["place"][tt, ee]
            kl_place = (lpp.exp() * torch.where(legal_p, lpp - lqp, torch.zeros_like(lpp))).sum(-1)
            n_dec = dec.sum().clamp(min=1.0)
            kl = (kl_card * dec).sum() / n_dec + (kl_place * placed).sum() / n_dec
            fwd = loc["fwd"]
            if self._ref["ability"] is not None and len(fwd) > 6 and fwd[6]:
                ready = loc["ability_ready_seq"][tt, ee].bool()
                for k, lg in enumerate(fwd[6]):
                    r = ready[..., k]
                    if r.any():
                        lpa = torch.log_softmax(lg, -1)
                        lqa = torch.log_softmax(self._ref["ability"][tt, ee, k], -1)
                        kla = (lpa.exp() * (lpa - lqa)).sum(-1)
                        kl = kl + (kla * r.float()).sum() / n_dec
            extra = extra + kl_c * kl
            self._kl_log.append(float(kl.detach()))
        if bc_c > 0 and self.bcdata is not None:
            b, _ = self._next_bc_chunk()
            out, h = BC.forward_chunk(net, b, self._stream_h)
            self._stream_h = (h[0].detach(), h[1].detach())
            if getattr(net, "gate", False):   # net_v2: gate / card / cell, as it was cloned
                from training.bc_v2 import v2_losses
                lbc, st = v2_losses(net, b, out, smooth=0.05, cell_smooth=0.1, w_noop=1.0,
                                    value_coef=0.0)
            else:
                lbc, st = BC.bc_losses(net, b, out, smooth=0.05, cell_smooth=0.1, w_noop=1.0,
                                       value_coef=0.0)
            extra = extra + bc_c * lbc
            self._bc_log.append(float(lbc.detach()))
        return loss + extra

    # -- update ----------------------------------------------------------------
    def run_update(self):
        cfg, net = self.cfg, self.net
        batch = self.buffer.drain()
        with torch.no_grad():
            next_obs_t = torch.tensor(self._obs, dtype=torch.float32).to(self.device)
            feats, _, _ = net.extract_features(next_obs_t)
            _, _, _, next_value, _ = net.step_lstm_and_card(feats, (self._hx, self._cx))
            next_value = next_value.squeeze(-1)
        advantages = gae_mod.compute_gae(
            batch["rewards"], batch["values"], batch["masks"], next_value,
            cfg.gamma, cfg.gae_lambda,
            boot_nonterminal=batch.get("boot_nonterminal"),
            trunc_flag=batch.get("trunc_flag"), trunc_boot=batch.get("trunc_boot"))
        returns = advantages + batch["values"]
        adv_norm = gae_mod.normalize(advantages, mask=batch["decision"])
        self._note_placements(batch)
        with torch.no_grad():
            keep = batch["valid"] > 0.5
            r, v = returns[keep], batch["values"][keep]
            self._explained_variance = float(gae_mod.explained_variance(r, v))
            vf_clip_range = max(cfg.vf_clip_std_frac * gae_mod.safe_std(r), cfg.eps_clip)
        self._vf_clip_range = vf_clip_range
        self._ref = self._bc_reference(batch)
        self._kl_log, self._bc_log = [], []
        warming = self.warming()
        if warming:
            for n_, p in net.named_parameters():
                p.requires_grad_(n_.startswith("value_head"))
        t0 = time.time()
        try:
            stats = self.updater.update(
                batch, adv_norm, returns, vf_clip_range,
                ent_coef_card=0.0 if warming else self.args.ent_card,
                ent_coef_placement=0.0 if warming else self.args.ent_place,
                coverage_coef=0.0,
                deck_coverage_coef=0.0,
                aux_scale=aux_warmup_scale(self.episodes_completed, cfg.aux_warmup_episodes))
        finally:
            if warming:
                for p in net.parameters():
                    p.requires_grad_(True)
        self._ref = None
        self.n_updates += 1
        self._post_update(stats, time.time() - t0, warming, batch)
        return stats

    def _post_update(self, stats, dt, warming, batch):
        w, ep = self.writer, self.episodes_completed
        kl_c, bc_c = self.coefs()
        kl = float(np.mean(self._kl_log)) if self._kl_log else float("nan")
        bcl = float(np.mean(self._bc_log)) if self._bc_log else float("nan")
        w.add_scalar("BC/kl_to_bc", kl, ep)
        w.add_scalar("BC/kl_coef", kl_c, ep)
        w.add_scalar("BC/aux_bc_loss", bcl, ep)
        w.add_scalar("BC/bc_coef", bc_c, ep)
        w.add_scalar("BC/critic_warmup", float(warming), ep)
        w.add_scalar("Reward/explained_variance", self._explained_variance, ep)
        rec = {"t": time.time(), "update": self.n_updates, "episodes": ep,
               "stage": int(self.curriculum.stage), "phase": self.curriculum.phase,
               "warmup": warming, "kl_to_bc": kl, "kl_coef": kl_c, "bc_loss": bcl, "bc_coef": bc_c,
               "actor_loss": stats.actor_loss, "critic_loss": stats.critic_loss,
               "ent_card": stats.ent_card, "ent_place": stats.ent_placement,
               "clip_frac": stats.clip_frac, "ev": self._explained_variance,
               "update_s": dt}
        outs = list(self.metrics.outcomes)[-200:]
        if outs:
            o = np.asarray(outs, dtype=np.float64)
            rec["win_rate_200"] = float((o > 0.5).mean())
            rec["n_recent"] = len(outs)
        # Per-opponent-bucket win rates (count-weighted across workers).
        try:
            st = self.envs.call("get_deck_pool_stats")
            cn = self.envs.call("get_deck_pool_counts")
            agg, n = {}, {}
            for s_, c_ in zip(st, cn):
                for kk, vv in s_.items():
                    m = c_.get(kk, 0)
                    agg[kk] = agg.get(kk, 0.0) + vv * m
                    n[kk] = n.get(kk, 0) + m
            opp = {kk: agg[kk] / n[kk] for kk in agg if n[kk] > 0}
            for kk, vv in opp.items():
                w.add_scalar(f"Opp/{kk}", vv, ep)
            rec["opp_wr"] = {kk: round(vv, 3) for kk, vv in opp.items()}
            rec["opp_n"] = n
        except Exception as e:  # noqa: BLE001
            rec["opp_err"] = str(e)[:100]
        # Held-out human-data metrics of the CURRENT policy.
        if self.bcdata is not None and self.n_updates % self.args.val_every == 1:
            m = BC.evaluate(self.net, self.val_eps, self.device, self.bcdata.obs_dim, B=40,
                            loss_kw=dict(smooth=0.0, cell_smooth=0.0, w_noop=1.0,
                                         lost_weight=1.0, value_coef=0.0))
            for kk in ("card_match", "card_match_given_play", "cell_match", "cell_within1",
                       "bc_mean_p_play", "human_play_rate", "val_loss_card", "val_loss_place"):
                if m.get(kk) is not None:
                    w.add_scalar(f"BC/val_{kk}", m[kk], ep)
                    rec[f"val_{kk}"] = m[kk]
        self.progress.write(json.dumps(rec) + "\n")
        self.progress.flush()
        print(f">>> [BCPPO] upd {self.n_updates} ep {ep} rung {self.curriculum.stage} "
              f"wr200 {rec.get('win_rate_200', float('nan')):.3f} KL {kl:.4f} (x{kl_c:.3f}) "
              f"BC {bcl:.3f} (x{bc_c:.3f}) EV {self._explained_variance:.2f} "
              f"{'[critic warm-up] ' if warming else ''}upd {dt:.1f}s", flush=True)


def main():
    a = ARGS
    os.makedirs(os.path.join(REPO, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(REPO, "runs"), exist_ok=True)
    print("argv:", " ".join(sys.argv), flush=True)
    BCPPOTrainer(a).run()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
