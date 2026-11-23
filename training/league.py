"""ppo_league26: BC-PPO (training/ppo_bc.py) plus a FirstLight-Zero-style self-play league.

    .venv/bin/python -m training.league --run ppo_league26 --fork-from checkpoints/ppo_bc26.pth \
        --init checkpoints/bc_hog26.pth --num-envs 12 --lr 5e-5 --seed 26 --save-every 200

Everything ppo_bc26 does is inherited unchanged (reward, real-deck teacher,
curriculum, critic warm-up, KL-to-BC and BC floors at 0.01, logging), except:

  start     the weights, Adam moments, update count, curriculum rung and PFSP
            state of a COPY of --fork-from (checkpoints/<run>.pth is created
            once from it; re-running resumes from <run>.pth).
  opponents per episode (training/league_env.py), by --p-mirror / --p-snapshot,
            the rest teacher:
              mirror    current policy vs current policy, BOTH SEATS TRAIN: the
                        seat-1 rows (its own mirrored observation, its own LSTM
                        state, its own reward) are extra columns of the PPO batch;
              snapshot  a frozen snapshot of our own lineage, PFSP-weighted
                        (1-p)^2 with a --snap-uniform share, from the ppo_bc26
                        snapshots up to the fork plus this run's own, excluding
                        anything younger than --snap-min-age episodes;
              teacher   the real-deck-pool teacher; with probability --rung-bias-p
                        at a rung uniform in [--rung-bias-lo, --rung-bias-hi].
            Neural opponents run on the trainer's device, no grad, one LSTM state
            per env, batched per network.
  curriculum only teacher episodes at the curriculum's own rung feed the win-rate
            window, the gate and the plateau valve (exactly the population
            ppo_bc26's curriculum sees).
  reward    + the enemy-King-activation term (--w-king, 0.1 of a win; reward.py).
  level     opponent card+tower level curriculum (gen-2): every episode team 1
            plays at the cap with probability --opp-level-p-cap, else uniform in
            [--opp-level-lo, cap]. The cap starts at --opp-level-lo and rises by
            one, up to --opp-level-hi, once --opp-level-every episodes have passed
            since the last rise AND the teacher win rate at the cap over the last
            --opp-level-window such episodes is >= --opp-level-gate (0 = schedule
            only). The cap is checkpointed. --opp-level-hi equal to lo = off.
  logging   runs/<run>/league.jsonl, one line per update: the opponent mix,
            results per kind, neural plays, King wakes by cause, the seat-sum
            check. TensorBoard League/*.
"""
import argparse
import dataclasses
import glob
import json
import os
import random
import re
import shutil
import sys
import time
from collections import OrderedDict

RUN_DEFAULT = "ppo_league26"


def _pre_parse():
    """League flags, plus the flags the env-var block needs before engine import.
    Everything else is ppo_bc's own parser."""
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--run", default=RUN_DEFAULT)
    pre.add_argument("--num-envs", type=int, default=int(os.environ.get("CLASH_NUM_ENVS", 8)))
    pre.add_argument("--gamma", type=float, default=0.999)
    pre.add_argument("--save-every", type=int, default=200)
    pre.add_argument("--seed", type=int, default=None)
    env_args, _ = pre.parse_known_args()

    lg = argparse.ArgumentParser(allow_abbrev=False)
    lg.add_argument("--fork-from", default="checkpoints/ppo_bc26.pth",
                    help="full checkpoint copied to checkpoints/<run>.pth on the first start")
    lg.add_argument("--p-mirror", type=float, default=0.25)
    lg.add_argument("--p-snapshot", type=float, default=0.25)
    lg.add_argument("--mirror-both-seats", type=int, default=1)
    lg.add_argument("--snap-dirs", default="checkpoints/ppo_bc26_snapshots",
                    help="comma list of other lineages' snapshot dirs (capped at the fork episode)")
    lg.add_argument("--snap-min-age", type=int, default=6000, help="episodes")
    lg.add_argument("--snap-uniform", type=float, default=0.10)
    lg.add_argument("--snap-cache", type=int, default=16, help="frozen nets kept on the device")
    lg.add_argument("--rung-bias-p", type=float, default=0.5)
    lg.add_argument("--rung-bias-lo", type=int, default=4)
    lg.add_argument("--rung-bias-hi", type=int, default=10)
    lg.add_argument("--w-king", type=float, default=0.1)
    lg.add_argument("--torch-threads", type=int, default=4)
    lg.add_argument("--opp-level-lo", type=int, default=11)
    lg.add_argument("--opp-level-hi", type=int, default=11, help="15 = the gen-2 curriculum")
    lg.add_argument("--opp-level-p-cap", type=float, default=0.5)
    lg.add_argument("--opp-level-every", type=int, default=15000, help="episodes between rises")
    lg.add_argument("--opp-level-gate", type=float, default=0.0)
    lg.add_argument("--opp-level-window", type=int, default=400)
    lg.add_argument("--pfsp-window", type=int, default=0,
                    help="0 = engine EMA; N = Laplace-smoothed (n*s+1)/(n+2) win rate over each "
                         "snapshot's last N games (FirstLight-Zero: 300)")
    largs, rest = lg.parse_known_args()
    # ppo_bc's --run defaults to ppo_bc26: always pass ours explicitly.
    if "--run" not in rest:
        rest = ["--run", env_args.run] + rest
    return env_args, largs, rest


if __name__ == "__main__":
    ENV_ARGS, LARGS, REST = _pre_parse()
    if ENV_ARGS.run == "ppo_bc26":
        sys.exit("refusing --run ppo_bc26: that is the running A arm")
    os.environ["CLASH_NUM_ENVS"] = str(ENV_ARGS.num_envs)
    os.environ["CLASH_GAMMA"] = str(ENV_ARGS.gamma)
    os.environ["CLASH_SAVE_EVERY"] = str(ENV_ARGS.save_every)
    if ENV_ARGS.seed is not None:
        os.environ["CLASH_SEED"] = str(ENV_ARGS.seed)
    for k, v in {"CLASH_ADVISOR_COVERAGE_COEF": "0", "CLASH_PLACEMENT_COVERAGE_COEF": "0",
                 "CLASH_DECK_COVERAGE_COEF": "0", "CLASH_PHASE1_SCENARIO_PROB": "0",
                 "CLASH_OFFENSIVE_SCENARIO_PROB": "0", "CLASH_PHASE1_DECK_POOL": "0"}.items():
        os.environ.setdefault(k, v)
    from training import common as _C
    os.environ["CLASH_WEIGHTS"] = os.path.join(_C.REPO, "checkpoints", f"{ENV_ARGS.run}.pth")
    os.environ["CLASH_LOGDIR"] = os.path.join(_C.REPO, "runs", ENV_ARGS.run)

from training import common as C  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

from training import ppo_bc as P  # noqa: E402
from training import league_env as LE  # noqa: E402
from python_ai.rl import base_trainer as bt  # noqa: E402
from python_ai.rl import gae as gae_mod  # noqa: E402
from python_ai.rl.buffer import RolloutBuffer  # noqa: E402
from python_ai.rl.config import aux_warmup_scale  # noqa: E402
from python_ai.envs.selfplay_env import PFSP_EXPONENT, PFSP_EMA_ALPHA  # noqa: E402

REPO = C.REPO
_EP_RE = re.compile(r"_ep(\d+)\.pth$")


# ---------------------------------------------------------------------------
# PPO update that skips segments with no valid row (seat-1 columns are mostly
# empty outside mirror episodes).
# ---------------------------------------------------------------------------
class LeagueUpdater(P.AnchoredUpdater):
    valid_cols = None

    def _segments(self, device):
        segments, offsets = super()._segments(device)
        v = self.valid_cols
        if v is not None:
            L = self.cfg.bptt_chunk
            T, N = v.shape[:2]
            per = (v.reshape(T // L, L, N).sum(1) > 0).cpu().numpy()
            keep = per[segments[:, 0] // L, segments[:, 1]]
            segments = segments[keep]
        return segments, offsets


# ---------------------------------------------------------------------------
# Frozen snapshots of our own lineage, PFSP-weighted.
# ---------------------------------------------------------------------------
class SnapshotPool:
    def __init__(self, sources, min_age, uniform, cache_size, device, rng, window=0):
        #: [(directory, max_episode or None)]
        self.sources = sources
        self.min_age, self.uniform = int(min_age), float(uniform)
        self.cache_size, self.device, self.rng = int(cache_size), device, rng
        self.stats, self.n = {}, {}
        #: window > 0: per-snapshot deque of the last `window` results, and the
        #: PFSP win rate is (wins + 1) / (games + 2) over it.
        self.window, self.hist = int(window), {}
        self.entries = {}          # name -> (path, ep)
        self._nets = OrderedDict()
        self._scan_ep = None

    def refresh(self, ep_now):
        found = {}
        for d, cap in self.sources:
            for p in glob.glob(os.path.join(d, "*.pth")):
                m = _EP_RE.search(p)
                if not m:
                    continue
                ep = int(m.group(1))
                if cap is not None and ep > cap:
                    continue
                if ep > ep_now - self.min_age:
                    continue
                found[os.path.basename(p)] = (p, ep)
        self.entries = found
        self._scan_ep = ep_now

    def weights(self):
        names = sorted(self.entries)
        if not names:
            return names, np.zeros(0)
        pf = np.asarray([(1.0 - self.stats.get(k, 0.5)) ** PFSP_EXPONENT for k in names])
        pf = pf / pf.sum() if pf.sum() > 0 else np.full(len(names), 1.0 / len(names))
        w = (1.0 - self.uniform) * pf + self.uniform / len(names)
        return names, w / w.sum()

    def sample(self):
        names, w = self.weights()
        if not names:
            return None
        return names[int(np.searchsorted(np.cumsum(w), self.rng.random(), side="right").clip(0, len(names) - 1))]

    def record(self, name, won):
        n = self.n[name] = self.n.get(name, 0) + 1
        if self.window:
            from collections import deque
            h = self.hist.setdefault(name, deque(maxlen=self.window))
            h.append(1.0 if won else 0.0)
            self.stats[name] = (sum(h) + 1.0) / (len(h) + 2.0)
            return
        a = max(PFSP_EMA_ALPHA, 1.0 / (n + 1))
        self.stats[name] = (1 - a) * self.stats.get(name, 0.5) + a * (1.0 if won else 0.0)

    def net(self, name):
        if name in self._nets:
            self._nets.move_to_end(name)
            return self._nets[name]
        from training.net_v2 import load_any   # v1 or net_v2 snapshots alike
        net = load_any(self.entries[name][0], self.device)
        for p in net.parameters():
            p.requires_grad_(False)
        self._nets[name] = net
        while len(self._nets) > self.cache_size:
            self._nets.popitem(last=False)
        return net

    def state_dict(self):
        return {"stats": dict(self.stats), "n": dict(self.n),
                "hist": {k: list(v) for k, v in self.hist.items()}}

    def load_state_dict(self, d):
        from collections import deque
        self.stats.update((d or {}).get("stats", {}))
        self.n.update((d or {}).get("n", {}))
        if self.window:
            for k, v in ((d or {}).get("hist") or {}).items():
                self.hist[k] = deque(v, maxlen=self.window)


class LeagueVec:
    """The AsyncVectorEnv, with team 1's neural actions computed before every step."""

    def __init__(self, envs, trainer):
        self._envs, self._t = envs, trainer

    def __getattr__(self, name):
        return getattr(self._envs, name)

    def reset(self, **kw):
        obs, infos = self._envs.reset(**kw)
        self._t._league_infos(infos, np.zeros(self._envs.num_envs, dtype=bool), first=True)
        return obs, infos

    def step(self, action):
        action = dict(action)
        action["opp"] = self._t._league_act()
        out = self._envs.step(action)
        self._t._league_after_step(out)
        return out


def _col(infos, key, n, default, dtype=np.float64):
    v = infos.get(key)
    if v is None:
        return np.full(n, default, dtype=dtype)
    v = np.asarray(v)
    m = infos.get("_" + key)
    out = v.astype(dtype)
    if m is not None:
        out = np.where(np.asarray(m, dtype=bool), out, default).astype(dtype)
    return out


class LeagueTrainer(P.BCPPOTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.spec = P.R.RewardSpec(gamma=args.gamma, k=args.k, leak_worst=args.leak_worst,
                                   draw=args.draw, w_king=args.w_king, king_chip=args.king_chip,
                                   w_unilateral=args.w_unilateral)
        P.R.install_reward(self.spec)
        self.league_rng = random.Random((args.seed or 0) + 4242)
        self.fork_ep = None
        self._league_restored = None
        self.opp_level_cap = int(args.opp_level_lo)
        self._lvl_raise_ep = 0
        self._lvl_window = []

    # -- envs -------------------------------------------------------------------
    def build_envs(self):
        a = self.args
        spec = {k: v for k, v in self.spec.as_dict().items() if k != "c_leak"}
        seeds = bt.worker_seeds(self.cfg.seed, self.cfg.num_envs)
        p_t = max(0.0, 1.0 - a.p_mirror - a.p_snapshot)
        mix = (p_t, a.p_mirror, a.p_snapshot)
        rb = (a.rung_bias_p, a.rung_bias_lo, a.rung_bias_hi) if a.rung_bias_p > 0 else None
        lvl = ((self.opp_level_cap, a.opp_level_p_cap, a.opp_level_lo)
               if a.opp_level_hi > a.opp_level_lo else None)
        envs = gym.vector.AsyncVectorEnv(
            [LE.make_league_env(s, spec, self.pool_path, a.p_freq, mix, rb,
                                act_delay=P.parse_delay(a.act_delay), opp_level=lvl,
                                obs_noise=P.obs_noise_spec(a))
             for s in seeds])
        return LeagueVec(envs, self)

    # -- resume -----------------------------------------------------------------
    def load_checkpoint(self):
        a = self.args
        if not os.path.exists(self.weight_path) and a.fork_from:
            src = a.fork_from if os.path.isabs(a.fork_from) else os.path.join(REPO, a.fork_from)
            tmp = self.weight_path + ".forking"
            shutil.copyfile(src, tmp)
            os.replace(tmp, self.weight_path)
            print(f">>> League fork: {src} -> {self.weight_path}", flush=True)
        full = super().load_checkpoint()
        ck = torch.load(self.weight_path, map_location="cpu", weights_only=False)
        lg = ck.get("league") if isinstance(ck, dict) else None
        self.fork_ep = int(lg["fork_ep"]) if lg else int(self.episodes_completed)
        self._league_restored = lg
        ol = (lg or {}).get("opp_level")
        if ol:
            self.opp_level_cap = int(ol["cap"])
            self._lvl_raise_ep = int(ol["raise_ep"])
        elif self.args.opp_level_hi > self.args.opp_level_lo:
            self._lvl_raise_ep = int(self.episodes_completed)
        return full

    def checkpoint_payload(self):
        d = super().checkpoint_payload()
        d["league"] = {"fork_ep": self.fork_ep, "snapshots": self.snap.state_dict(),
                       "args": vars(self.args),
                       "opp_level": {"cap": self.opp_level_cap, "raise_ep": self._lvl_raise_ep}}
        return d

    # -- setup ------------------------------------------------------------------
    def setup(self):
        a = self.args
        n = self.cfg.num_envs
        # Everything the league proxy reads during the first envs.reset() inside
        # super().setup() must exist first.
        self._kind = np.zeros(n, dtype=np.int64)
        self._ep_id = np.zeros(n, dtype=np.int64)
        self._snap_name = [None] * n
        self._obs1 = np.zeros((n, 1), dtype=np.float32)
        self._ready1 = None
        self._h1 = self._c1 = None
        self._pending = None
        self._lstat = self._new_lstat()
        self.snap = None
        self._pending_infos = None
        super().setup()
        self.updater = LeagueUpdater(self.net, self.optimizer, self.cfg, self)
        dev = self.device
        H = self._hx.shape[1]
        self._h1 = torch.zeros(n, H, device=dev)
        self._c1 = torch.zeros(n, H, device=dev)
        self._ready1 = torch.zeros((n, len(self._ability_slots)), dtype=torch.bool)
        self.side = RolloutBuffer(self.buffer.fields)
        self._bc_h1 = P.BC.zero_hidden(self.net, n, dev)
        sources = []
        for d in [s for s in a.snap_dirs.split(",") if s]:
            sources.append((d if os.path.isabs(d) else os.path.join(REPO, d), self.fork_ep))
        sources.append((os.path.join(REPO, "checkpoints", f"{a.run}_snapshots"), None))
        self.snap = SnapshotPool(sources, a.snap_min_age, a.snap_uniform, a.snap_cache, dev,
                                 self.league_rng, window=a.pfsp_window)
        self.snap.load_state_dict((self._league_restored or {}).get("snapshots"))
        self.snap.refresh(self.episodes_completed)
        if self._pending_infos is not None:   # the reset inside super().setup()
            self._league_infos(*self._pending_infos)
            self._pending_infos = None
        self.league_log = open(os.path.join(self.log_dir, "league.jsonl"), "a")
        if a.opp_level_hi > a.opp_level_lo:
            self.envs.call("set_opp_level_cap", self.opp_level_cap)
            print(f">>> LEVEL curriculum: cap {self.opp_level_cap} in [{a.opp_level_lo},"
                  f"{a.opp_level_hi}], p_cap {a.opp_level_p_cap}, every {a.opp_level_every} eps, "
                  f"gate {a.opp_level_gate} over {a.opp_level_window}, last rise at ep "
                  f"{self._lvl_raise_ep}", flush=True)
        print(f">>> LEAGUE: run {a.run} fork_ep {self.fork_ep} mix teacher "
              f"{max(0.0, 1 - a.p_mirror - a.p_snapshot):.2f} / mirror {a.p_mirror:.2f} / snapshot "
              f"{a.p_snapshot:.2f}, both seats train in mirror: {bool(a.mirror_both_seats)}, "
              f"snapshots eligible {len(self.snap.entries)} (min age {a.snap_min_age}, "
              f"uniform {a.snap_uniform}), rung bias p={a.rung_bias_p} in "
              f"[{a.rung_bias_lo},{a.rung_bias_hi}], w_king {a.w_king}, "
              f"reward {self.spec.as_dict()}, device {dev}", flush=True)

    # -- per-step league bookkeeping ---------------------------------------------
    @staticmethod
    def _new_lstat():
        return {"episodes": [0, 0, 0], "wins": [0, 0, 0], "draws": [0, 0, 0],
                "teacher_on_curriculum": 0, "teacher_off_rungs": {},
                "neural_steps": 0, "neural_plays": [0, 0, 0], "neural_ability": 0,
                "king_enemy_damage": 0, "king_enemy_tower": 0,
                "king_own_damage": 0, "king_own_tower": 0,
                "king_r_on_damage": 0.0, "king_r_on_tower": 0.0,
                "zs_resid_max": 0.0, "zs_n": 0, "seat1_rows": 0, "seat1_decisions": 0,
                "snap_eps": {}, "snap_wins": {}, "level_eps": {}, "level_wins": {}}

    def _league_infos(self, infos, dones, first=False):
        """Track each env's opponent kind; pick a snapshot when a new episode starts."""
        if self.snap is None:            # called from inside super().setup()
            self._pending_infos = (infos, dones, first)
            return
        n = self.cfg.num_envs
        kind = _col(infos, "opp_kind", n, -1, np.int64)
        ep = _col(infos, "league_episode", n, -1, np.int64)
        obs1 = infos.get("obs1")
        if obs1 is not None and self._obs1.shape[1] != np.asarray(obs1).shape[1]:
            self._obs1 = np.zeros((n, np.asarray(obs1).shape[1]), dtype=np.float32)
        for i in range(n):
            if kind[i] < 0:
                continue
            if ep[i] != self._ep_id[i]:       # a reset happened: new episode in env i
                self._ep_id[i] = ep[i]
                self._kind[i] = kind[i]
                self._snap_name[i] = None
                if kind[i] == LE.SNAPSHOT:
                    name = self.snap.sample()
                    if name is None:          # nothing old enough yet: play the mirror
                        self._kind[i] = LE.MIRROR
                    self._snap_name[i] = name
                self._h1[i].zero_()
                self._c1[i].zero_()
        if obs1 is not None:
            m = np.asarray(infos.get("_obs1", np.ones(n, dtype=bool)), dtype=bool)
            self._obs1[m] = np.asarray(obs1, dtype=np.float32)[m]
        cols = []
        for slot in self._ability_slots:
            cols.append(_col(infos, f"opp_ability_slot{slot}_ready", n, False, bool))
        if cols:
            self._ready1 = torch.as_tensor(np.stack(cols, axis=1))

    @torch.no_grad()
    def _league_act(self):
        n = self.cfg.num_envs
        act = np.zeros((n, LE.OPP_DIM), dtype=np.float32)
        act[:, 0] = C.NOOP
        live = (self._kind != LE.TEACHER) & ~np.asarray(self._prev_dones, dtype=bool)
        self._pending = None
        if not live.any():
            return act
        dev = self.device
        obs1 = torch.as_tensor(self._obs1, dtype=torch.float32, device=dev)
        groups = {}
        for i in np.nonzero(live)[0]:
            key = "__mirror__" if self._kind[i] == LE.MIRROR else self._snap_name[i]
            groups.setdefault(key, []).append(int(i))
        mirror = None
        h_in, c_in = self._h1.clone(), self._c1.clone()
        for key, idx in groups.items():
            net = self.net if key == "__mirror__" else self.snap.net(key)
            it = torch.as_tensor(idx, device=dev)
            out = LE.policy_step(net, obs1[it], (self._h1[it], self._c1[it]), self._ready1[idx])
            self._h1[it], self._c1[it] = out["h"], out["c"]
            act[idx] = LE.opp_rows(out, self._ability_slots)
            if key == "__mirror__":
                mirror = (idx, out)
        self._pending = {"obs1": obs1, "h_in": h_in, "c_in": c_in, "mirror": mirror,
                         "live": live.copy(), "kind": self._kind.copy(),
                         "snap": list(self._snap_name), "act": act.copy(),
                         "ready1": self._ready1.clone()}
        return act

    def _league_after_step(self, out):
        next_obs, raw, term, trunc, infos = out
        n = self.cfg.num_envs
        dones = np.asarray(term) | np.asarray(trunc)
        dev = self.device
        pend = self._pending
        st = self._lstat
        prev_d = np.asarray(self._prev_dones, dtype=bool)
        # -- seat-1 rows for mirror envs ------------------------------------------
        if self.args.mirror_both_seats and self._ability_slots is not None:
            self._add_seat1_row(pend, infos, term, dones, prev_d)
        # -- stats on real (non-phantom) steps --------------------------------------
        real = ~prev_d
        kind_now = self._kind.copy()
        neural = real & (kind_now != LE.TEACHER)
        if pend is not None and neural.any():
            st["neural_steps"] += int(neural.sum())
            played = pend["act"][:, 0] != C.NOOP
            for k in (LE.MIRROR, LE.SNAPSHOT):
                st["neural_plays"][k] += int((played & neural & (kind_now == k)).sum())
            st["neural_ability"] += int((pend["act"][:, 3:5].sum(1) > 0)[neural].sum())
            r0 = np.asarray(raw, np.float64) + _col(infos, "r_shape", n, 0.0)
            r1 = _col(infos, "r1", n, 0.0)
            leak = (_col(infos, "leak_elixir", n, 0.0) + _col(infos, "leak1_elixir", n, 0.0))
            # Outcome, tower PBRS and King term cancel; each seat's own leak and a
            # draw's `draw` (both seats, 0 by default) are not zero-sum by design.
            drew = np.asarray(term, bool) & (np.abs(np.asarray(raw, np.float64)) < 0.5)
            uni = _col(infos, "r_unilateral", n, 0.0) + _col(infos, "r1_unilateral", n, 0.0)
            resid = np.abs(r0 + r1 + self.spec.c_leak * leak - uni
                           - 2.0 * self.spec.draw * drew)[neural]
            st["zs_resid_max"] = max(st["zs_resid_max"], float(resid.max()))
            st["zs_n"] += int(neural.sum())
        ke = _col(infos, "king_wake_enemy", n, 0, np.int64)
        ko = _col(infos, "king_wake_own", n, 0, np.int64)
        rk = _col(infos, "r_king", n, 0.0)
        st["king_enemy_damage"] += int(((ke == 1) & real).sum())
        st["king_enemy_tower"] += int(((ke == 2) & real).sum())
        st["king_own_damage"] += int(((ko == 1) & real).sum())
        st["king_own_tower"] += int(((ko == 2) & real).sum())
        on_damage = ((ke == 1) | (ko == 1)) & real
        st["king_r_on_damage"] += float(np.abs(rk[on_damage]).sum())
        st["king_r_on_tower"] += float(np.abs(rk[~on_damage & real]).sum())
        # -- episode ends -------------------------------------------------------------
        oncur = _col(infos, "on_curriculum", n, 0.0)
        rung = _col(infos, "opp_rung", n, -1, np.int64)
        olvl = _col(infos, "opp_level", n, 11, np.int64)
        for i in np.nonzero(dones & real)[0]:
            k = int(kind_now[i])
            won, draw = bool(raw[i] > 0.5), bool(abs(raw[i]) < 0.5)
            L = str(int(olvl[i]))
            st["level_eps"][L] = st["level_eps"].get(L, 0) + 1
            st["level_wins"][L] = st["level_wins"].get(L, 0) + int(won)
            if k == LE.TEACHER and int(olvl[i]) == self.opp_level_cap:
                self._lvl_window.append(int(won))
                del self._lvl_window[:-max(1, int(self.args.opp_level_window))]
            st["episodes"][k] += 1
            st["wins"][k] += int(won)
            st["draws"][k] += int(draw)
            if k == LE.TEACHER:
                if oncur[i] > 0.5:
                    st["teacher_on_curriculum"] += 1
                else:
                    r = str(int(rung[i]))
                    st["teacher_off_rungs"][r] = st["teacher_off_rungs"].get(r, 0) + 1
            elif k == LE.SNAPSHOT and self._snap_name[i] is not None:
                nm = self._snap_name[i]
                self.snap.record(nm, won)
                st["snap_eps"][nm] = st["snap_eps"].get(nm, 0) + 1
                st["snap_wins"][nm] = st["snap_wins"].get(nm, 0) + int(won)
        # LSTM state of seat 1 resets at an episode end.
        keep = torch.as_tensor(1.0 - dones.astype(np.float32), device=dev).unsqueeze(1)
        self._h1 *= keep
        self._c1 *= keep
        self._league_infos(infos, dones)
        self._pending = None

    def _add_seat1_row(self, pend, infos, term, dones, prev_d):
        n = self.cfg.num_envs
        dev = self.device
        valid_np = np.zeros(n, dtype=np.float32)
        card = torch.full((n,), C.NOOP, dtype=torch.long, device=dev)
        cell = torch.zeros(n, dtype=torch.long, device=dev)
        lp = torch.zeros(n, device=dev)
        val = torch.zeros(n, device=dev)
        k = len(self._ability_slots)
        ab = torch.zeros((n, k), dtype=torch.long, device=dev)
        ready = torch.zeros((n, k), dtype=torch.bool, device=dev)
        if pend is not None:
            obs1 = pend["obs1"]
            h_in, c_in = pend["h_in"], pend["c_in"]
            if pend["mirror"] is not None:
                idx, out = pend["mirror"]
                it = torch.as_tensor(idx, device=dev)
                valid_np[idx] = 1.0
                card[it], cell[it] = out["card"], out["cell"]
                lp[it], val[it] = out["logprob"], out["value"]
                if k:
                    ab[it] = out["ab"].to(dev)
                    ready[it] = out["ready"].to(dev)
        else:
            # Before any env has had a neural opponent, _obs1 is still a (n, 1)
            # placeholder; an all-invalid row still needs full-width obs.
            o1 = self._obs1 if self._obs1.shape[1] > 1 else np.zeros_like(np.asarray(self._obs, np.float32))
            obs1 = torch.as_tensor(o1, dtype=torch.float32, device=dev)
            h_in, c_in = self._h1.clone(), self._c1.clone()
        valid = torch.as_tensor(valid_np, device=dev)
        card_mask = self.net.affordability_mask(obs1)
        r1 = torch.as_tensor(_col(infos, "r1", n, 0.0, np.float32), device=dev) * valid
        mask = torch.as_tensor(1.0 - dones.astype(np.float32), device=dev) * valid
        cov = self._draw_coverage(obs1, card_mask)
        aux = _col(infos, "me_played_card", n, -1, np.int64)
        row = {"obs": obs1, "card_actions": card, "placement_actions": cell,
               "decision": ((card_mask.sum(dim=1) > 1) | ready.any(dim=1)).float() * valid,
               "hx_in": h_in, "cx_in": c_in, "logprobs": lp, "values": val,
               "rewards": r1, "masks": mask, "valid": valid,
               "aux_opp_played": torch.as_tensor(aux, device=dev),
               "coverage_slot": cov["slot"]}
        f = self.side.fields
        if "boot_nonterminal" in f:
            row["boot_nonterminal"] = torch.as_tensor(
                1.0 - np.asarray(term, np.float32), device=dev) * valid
            row["trunc_flag"] = torch.zeros(n, device=dev)
            row["trunc_boot"] = torch.zeros(n, device=dev)
        if "ability_actions" in f:
            row["ability_actions"] = ab
            row["ability_ready"] = ready
        if "coverage_target" in f:
            row["coverage_target"] = cov["target"]
            row["coverage_has"] = cov["has"]
        self.side.add(**row)
        self._lstat["seat1_rows"] += int(valid_np.sum())
        self._lstat["seat1_decisions"] += int(row["decision"].sum().item())

    # -- curriculum: only the population ppo_bc26's curriculum sees ----------------
    def on_episode_end(self, i, ctx):
        k = int(_col(ctx.infos, "opp_kind", self.cfg.num_envs, 0, np.int64)[i])
        oncur = _col(ctx.infos, "on_curriculum", self.cfg.num_envs, 1.0)[i] > 0.5
        if k != LE.TEACHER or not oncur:
            self.metrics.reset_env(i)
            return
        super().on_episode_end(i, ctx)

    def periodic(self, stats):
        super().periodic(stats)
        self._maybe_raise_level()
        if self.snap is not None and self.episodes_completed - (self.snap._scan_ep or 0) >= 500:
            self.snap.refresh(self.episodes_completed)

    def _maybe_raise_level(self):
        a = self.args
        if self.opp_level_cap >= a.opp_level_hi:
            return
        if self.episodes_completed - self._lvl_raise_ep < a.opp_level_every:
            return
        w = self._lvl_window
        wr = float(np.mean(w)) if w else 0.0
        if a.opp_level_gate > 0 and (len(w) < a.opp_level_window // 2 or wr < a.opp_level_gate):
            return
        self.opp_level_cap += 1
        self._lvl_raise_ep = int(self.episodes_completed)
        self._lvl_window = []
        self.envs.call("set_opp_level_cap", self.opp_level_cap)
        print(f">>> LEVEL cap -> {self.opp_level_cap} at ep {self.episodes_completed} "
              f"(teacher win at the old cap {wr:.3f} over {len(w)})", flush=True)

    # -- update: seat-1 columns appended -------------------------------------------
    def run_update(self):
        cfg, net = self.cfg, self.net
        n = cfg.num_envs
        b0 = self.buffer.drain()
        b1 = self.side.drain() if len(self.side) else None
        cols = (torch.nonzero(b1["valid"].sum(0) > 0).squeeze(1) if b1 is not None
                else torch.zeros(0, dtype=torch.long))
        m = int(cols.numel())
        with torch.no_grad():
            next_obs_t = torch.tensor(self._obs, dtype=torch.float32).to(self.device)
            feats, _, _ = net.extract_features(next_obs_t)
            _, _, _, nv0, _ = net.step_lstm_and_card(feats, (self._hx, self._cx))
            next_value = nv0.squeeze(-1)
            if m:
                o1 = torch.as_tensor(self._obs1, dtype=torch.float32, device=self.device)[cols]
                f1, _, _ = net.extract_features(o1)
                _, _, _, nv1, _ = net.step_lstm_and_card(f1, (self._h1[cols], self._c1[cols]))
                next_value = torch.cat([next_value, nv1.squeeze(-1)])
        if m:
            batch = {k: torch.cat([b0[k], b1[k][:, cols]], dim=1) for k in b0}
            del b1
        else:
            batch = b0
        ucfg = dataclasses.replace(cfg, num_envs=n + m)
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
        # BC reference over all columns; seat-1 BC hidden state kept per env.
        h0 = self._bc_h
        if m:
            self._bc_h = (torch.cat([h0[0], self._bc_h1[0][cols]]),
                          torch.cat([h0[1], self._bc_h1[1][cols]]))
        self._ref = self._bc_reference(batch)
        if m and self._bc_h is not None and self._ref is not None:
            hh, cc = self._bc_h
            self._bc_h1[0][cols], self._bc_h1[1][cols] = hh[n:], cc[n:]
            self._bc_h = (hh[:n], cc[:n])
        self._kl_log, self._bc_log = [], []
        warming = self.warming()
        if warming:
            for n_, p in net.named_parameters():
                p.requires_grad_(n_.startswith("value_head"))
        self.updater.cfg = ucfg
        self.updater.valid_cols = batch["valid"]
        t0 = time.time()
        try:
            stats = self.updater.update(
                batch, adv_norm, returns, vf_clip_range,
                ent_coef_card=0.0 if warming else self.args.ent_card,
                ent_coef_placement=0.0 if warming else self.args.ent_place,
                coverage_coef=0.0, deck_coverage_coef=0.0,
                aux_scale=aux_warmup_scale(self.episodes_completed, cfg.aux_warmup_episodes))
        finally:
            if warming:
                for p in net.parameters():
                    p.requires_grad_(True)
            self.updater.cfg = cfg
            self.updater.valid_cols = None
        self._ref = None
        self.n_updates += 1
        self._seat1_cols = m
        self._post_update(stats, time.time() - t0, warming, batch)
        self._log_league(stats)
        return stats

    def _log_league(self, stats):
        st, ep, w = self._lstat, self.episodes_completed, self.writer
        tot = max(1, sum(st["episodes"]))
        rec = {"t": time.time(), "update": self.n_updates, "episodes": ep,
               "seat1_columns": self._seat1_cols, **{k: v for k, v in st.items()}}
        rec["mix_share"] = {LE.KIND_NAMES[k]: round(st["episodes"][k] / tot, 3) for k in range(3)}
        rec["win_rate"] = {LE.KIND_NAMES[k]: (round(st["wins"][k] / st["episodes"][k], 3)
                                              if st["episodes"][k] else None) for k in range(3)}
        rec["snapshots_eligible"] = len(self.snap.entries)
        rec["opp_level_cap"] = self.opp_level_cap
        rec["level_win"] = {L: round(st["level_wins"].get(L, 0) / v, 3)
                            for L, v in sorted(st["level_eps"].items()) if v}
        w.add_scalar("League/opp_level_cap", self.opp_level_cap, ep)
        for L, v in rec["level_win"].items():
            w.add_scalar(f"League/win_level{L}", v, ep)
        rec["finite"] = bool(np.isfinite([stats.actor_loss, stats.critic_loss]).all())
        for k in range(3):
            w.add_scalar(f"League/share_{LE.KIND_NAMES[k]}", st["episodes"][k] / tot, ep)
            if st["episodes"][k]:
                w.add_scalar(f"League/win_{LE.KIND_NAMES[k]}", st["wins"][k] / st["episodes"][k], ep)
        w.add_scalar("League/king_enemy_damage_wakes", st["king_enemy_damage"], ep)
        w.add_scalar("League/king_own_damage_wakes", st["king_own_damage"], ep)
        w.add_scalar("League/seat_sum_resid_max", st["zs_resid_max"], ep)
        w.add_scalar("League/seat1_decisions", st["seat1_decisions"], ep)
        self.league_log.write(json.dumps(rec) + "\n")
        self.league_log.flush()
        wr = rec["win_rate"]
        print(f">>> [LEAGUE] upd {self.n_updates} ep {ep} mix "
              f"T{st['episodes'][0]}/M{st['episodes'][1]}/S{st['episodes'][2]} "
              f"wr T{wr['teacher']} M{wr['mirror']} S{wr['snapshot']} "
              f"plays M{st['neural_plays'][1]} S{st['neural_plays'][2]} "
              f"seat1 rows {st['seat1_rows']} (dec {st['seat1_decisions']}, cols {self._seat1_cols}) "
              f"king enemy dmg/twr {st['king_enemy_damage']}/{st['king_enemy_tower']} "
              f"own {st['king_own_damage']}/{st['king_own_tower']} "
              f"zs {st['zs_resid_max']:.2e} snaps {len(self.snap.entries)} "
              f"lvl cap {self.opp_level_cap} win {rec['level_win']}", flush=True)
        self._lstat = self._new_lstat()


def main():
    P.ARGS = None
    sys.argv = [sys.argv[0]] + REST
    base = P._parse()
    for k, v in vars(LARGS).items():
        setattr(base, k, v)
    assert base.run == ENV_ARGS.run != "ppo_bc26"
    torch.set_num_threads(max(1, int(base.torch_threads)))
    os.makedirs(os.path.join(REPO, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(REPO, "runs"), exist_ok=True)
    print("argv:", " ".join([sys.argv[0]] + REST), "league:", vars(LARGS), flush=True)
    LeagueTrainer(base).run()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
