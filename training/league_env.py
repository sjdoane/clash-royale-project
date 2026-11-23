"""The env side of the ppo_league26 self-play league (training/league.py).

`LeagueEnv` is `RewardEnv` (same reward, same real-deck teacher sampler) plus a
per-episode OPPONENT KIND, drawn at every reset from its own seeded rng:

  teacher   the existing real-deck-pool UtilityTeacher, exactly as in ppo_bc26,
            optionally at a rung drawn from [lo, hi] instead of the curriculum's
  mirror    team 1 is the CURRENT policy (the trainer runs it, both seats train)
  snapshot  team 1 is a frozen snapshot of our own policy (the trainer picks it)

For the two neural kinds the env cannot run a network itself (it lives in an
AsyncVectorEnv worker, the nets live on the trainer's GPU), so the trainer
computes team 1's action and passes it in the extra action key "opp" =
[card slot, x, y, ability slot 1, ability slot 2], in TEAM 1's OWN FRAME, as the
teacher's does: `step_self_play` mirrors it back. The engine's teacher path is
reused verbatim by swapping `self.teacher` for a stub that returns that action.
Team 1 plays OUR deck (CLASH_DECK) in neural episodes: the networks were trained
on it and the observation's hand is their own.

What the trainer needs for team 1 is published in `info` (reset and step):
obs1 (team 1's mirrored observation), r1 (team 1's reward), readiness of team
1's Champion/Hero abilities, and team 0's played card (the next-card aux label
seen from seat 1).

Seat-1 reward, same definition seen from the other side:
    r1 = -outcome + F_tower(obs1) - c_leak * leak(obs1) + r_king_seat1
Outcome, tower PBRS (k = 0, so Phi1 = -Phi0 exactly) and the King term are
zero-sum; the elixir-leak penalty is each seat's own cost, as in ppo_bc26.
"""
import random

import numpy as np
from gymnasium import spaces

from training import common as C
from training.reward import RewardEnv, KingWakeLatch, SKIP_TICKS, FULL_ELIXIR

TEACHER, MIRROR, SNAPSHOT = 0, 1, 2
KIND_NAMES = ("teacher", "mirror", "snapshot")
#: "opp" action layout.
OPP_DIM = 5


class _StubTeacher:
    """Plays whatever the trainer put in `action`; the teacher interface the
    engine's MicroRoyaleEnv calls (set_deck/set_stage/reset/act/ability_flags).
    `delayable`: with act_delay on, RewardEnv routes its card plays through the
    same ActuationDelay as ours (a network seat has the same touch latency)."""

    delayable = True

    def __init__(self):
        self.action = (C.NOOP, 0.0, 0.0, False, False)

    def set_deck(self, deck):
        pass

    def set_stage(self, stage):
        pass

    def reset(self):
        self.action = (C.NOOP, 0.0, 0.0, False, False)

    def act(self, game, obs1):
        c, x, y, _, _ = self.action
        return int(c), float(x), float(y)

    def ability_flags(self, game, obs1):
        return bool(self.action[3]), bool(self.action[4])


class LeagueEnv(RewardEnv):
    def __init__(self, env_config=None):
        cfg = dict(env_config or {})
        mix = [float(v) for v in cfg.pop("league_mix", (1.0, 0.0, 0.0))]
        self._mix_cum = np.cumsum(np.asarray(mix) / sum(mix))
        #: (p, lo, hi): a teacher episode plays a rung uniform in [lo, hi] with
        #: probability p, else the curriculum's rung.
        self._rung_bias = cfg.pop("rung_bias", None)
        seed = cfg.get("scenario_seed")
        super().__init__(cfg)
        assert self.teacher is not None, "LeagueEnv needs opponent='teacher'"
        self._kind_rng = random.Random(None if seed is None else int(seed) + 31337)
        self._real_teacher = self.teacher
        self._stub = _StubTeacher()
        self._neural_deck = list(self.ai_deck)
        self._kind = TEACHER
        self._episode = 0
        self._curr_rung = 0
        self._ep_rung = 0
        self._on_curriculum = True
        self._obs1 = None
        self._me_last_tick = {}
        # Seat 1's King term needs the wake causes even when w_king = 0 (they
        # are logged); RewardEnv only builds its latch when w_king != 0.
        self._latch_own = None if self._king is not None else KingWakeLatch()
        from python_ai.rl.abilities import ability_engine_slots
        self._slots = ability_engine_slots(self._neural_deck)
        sp = dict(self.action_space.spaces)
        sp["opp"] = spaces.Box(low=0.0, high=64.0, shape=(OPP_DIM,), dtype=np.float32)
        self.action_space = spaces.Dict(sp)

    # -- curriculum hook ------------------------------------------------------
    def set_teacher_stage(self, stage):
        self._curr_rung = int(stage)
        if self._kind == TEACHER and self._on_curriculum:
            self._real_teacher.set_stage(int(stage))
            self._ep_rung = int(stage)

    # -- episode ----------------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._kind_rng.seed(int(seed) + 31337)
        kind = int(np.searchsorted(self._mix_cum, self._kind_rng.random(), side="right"))
        kind = min(kind, SNAPSHOT)
        self._kind = kind
        self._episode += 1
        self._me_last_tick = {}
        if kind == TEACHER:
            self.teacher = self._real_teacher
            rung, self._on_curriculum = self._curr_rung, True
            if self._rung_bias and self._kind_rng.random() < float(self._rung_bias[0]):
                rung = self._kind_rng.randint(int(self._rung_bias[1]), int(self._rung_bias[2]))
                self._on_curriculum = rung == self._curr_rung
            self._real_teacher.set_stage(rung)
            self._ep_rung = rung
            obs, info = super().reset(seed=seed, options=options)
            self._obs1 = None
        else:
            self.teacher = self._stub
            sampler, fixed = self._sampler, self._fixed_deck
            self._sampler, self._fixed_deck = None, self._neural_deck
            try:
                obs, info = super().reset(seed=seed, options=options)
            finally:
                self._sampler, self._fixed_deck = sampler, fixed
            self._cur_bucket = None      # no real-deck PFSP update for this episode
            self.current_deck_name = "league_" + KIND_NAMES[kind]
            self._ep_rung, self._on_curriculum = -1, False
            self._obs1 = np.asarray(self.game.get_observation_for_team(1), dtype=np.float32)
        if self._latch_own is not None:
            self._latch_own.reset()
        info = dict(info)
        info.update(self._league_info())
        if self._obs1 is not None:
            info["obs1"] = self._obs1
        return obs, info

    def _league_info(self):
        out = {"opp_kind": self._kind, "league_episode": self._episode,
               "on_curriculum": float(self._on_curriculum), "opp_rung": self._ep_rung}
        if self._kind != TEACHER:
            for s in (1, 2):
                out[f"opp_ability_slot{s}_ready"] = bool(self.game.is_champion_ability_ready(1, s))
        return out

    def _poll_my_play(self):
        """Team 0's play this step (card id, -1 for none): seat 1's next-card label."""
        best_tick, best_card = None, -1
        for card in set(self._neural_deck):
            tick = self.game.get_last_played_tick(0, card)
            prev = self._me_last_tick.get(card)
            self._me_last_tick[card] = tick
            if prev is not None and tick > prev:
                if best_tick is None or tick < best_tick:
                    best_tick, best_card = tick, card
        return best_card

    def step(self, action, skip_frames=SKIP_TICKS):
        neural = self._kind != TEACHER
        if neural:
            a = np.asarray(action["opp"], dtype=np.float32).reshape(-1)
            self._stub.action = (int(round(float(a[0]))), float(a[1]), float(a[2]),
                                 bool(a[3] > 0.5), bool(a[4] > 0.5))
        obs, reward, terminated, truncated, info = super().step(action, skip_frames)
        if self._latch_own is not None:      # wake causes for logging when w_king = 0
            c0, c1 = self._latch_own.update(self.game)
            info["king_wake_own"] = {None: 0, "damage": 1, "tower": 2}[c0]
            info["king_wake_enemy"] = {None: 0, "damage": 1, "tower": 2}[c1]
            info.setdefault("r_king", 0.0)
            info.setdefault("r_king_seat1", 0.0)
        info.update(self._league_info())
        if neural:
            obs1 = np.asarray(self.game.get_observation_for_team(1), dtype=np.float32)
            F1, leak1 = self._spec.shaping(self._obs1, obs1, [float(terminated)])
            r1 = (-float(reward) + float(F1[0]) - self._spec.c_leak * float(leak1[0])
                  + float(info.get("r_king_seat1", 0.0)))
            if terminated and abs(reward) < 0.5:
                r1 += self._spec.draw
            if self._uni is not None:
                e0, e1 = self.game.get_elixir_for_team(0), self.game.get_elixir_for_team(1)
                pen1 = self._uni[1].step(leak1[0], e1 >= FULL_ELIXIR, e0 >= FULL_ELIXIR)
                r1 -= pen1
                info["r1_unilateral"] = -pen1
            info["obs1"] = self.inflight_obs(obs1, 1)
            info["r1"] = r1
            info["r1_tower"] = float(F1[0])
            info["leak1_elixir"] = float(leak1[0])
            info["me_played_card"] = self._poll_my_play()
            self._obs1 = obs1
        return obs, reward, terminated, truncated, info


def make_league_env(seed, spec, pool, p_freq, mix, rung_bias, act_delay=None, opp_level=None,
                    obs_noise=None):
    def _init():
        from training import common  # noqa: F401  (worker: sys.path, CLASH_DECK)
        from training.league_env import LeagueEnv
        cfg = {"opponent": "teacher", "teacher_stage": 0, "scenario_seed": seed,
               "deck_pool": False, "defensive_scenario_prob": 0.0,
               "reward_spec": spec, "real_pool": pool, "p_freq": p_freq,
               "league_mix": mix, "rung_bias": rung_bias}
        if act_delay is not None:
            cfg["act_delay"] = act_delay
        if opp_level is not None:
            cfg["opp_level"] = opp_level
        if obs_noise is not None:
            cfg["obs_noise"] = obs_noise      # seat 0's observation only; info["obs1"] stays clean
        return LeagueEnv(cfg)
    return _init


# ---------------------------------------------------------------------------
# One batched policy step for a seat-1 network (trainer and h2h share it).
# ---------------------------------------------------------------------------
def policy_step(net, obs_t, hc, ready, temp=1.0):
    """Sample one joint action, exactly as base_trainer.collect_rollout does.

    temp != 1 (eval / h2h only) samples card and cell at that temperature
    (0 = argmax); logprob stays the T=1 one.

    obs_t (B, D) on the net's device, hc = (h, c) (B, H), ready (B, k) bool.
    Returns a dict of tensors: card, cell, x, y, ab (B, k), logprob, value,
    card_mask, h, c. Call under torch.no_grad().
    """
    import torch
    from torch.distributions import Categorical
    from python_ai.rl import abilities as A
    card_mask = net.affordability_mask(obs_t)
    feats, emb, sp, hi = net.extract_features_hires(obs_t)
    cl, a1, a2, value, (h, c) = net.step_lstm_and_card(feats, hc, card_mask)
    from training.net_v2 import sample_logits
    cd = Categorical(logits=cl)
    card = cd.sample() if temp == 1.0 else sample_logits(cl, temp)
    pl = net.placement_given_card(h, emb, card, obs_t, sp, hires_map=hi)
    pd = Categorical(logits=pl)
    cell = pd.sample() if temp == 1.0 else sample_logits(pl, temp)
    lp = cd.log_prob(card) + pd.log_prob(cell)
    ab_logits = [l for l in (a1, a2) if l is not None]
    ab, ab_lp, _ = A.sample(ab_logits, ready.to(obs_t.device))
    if ab_logits:
        lp = lp + ab_lp
    x, y = net.cell_to_xy(cell)
    return {"card": card, "cell": cell, "x": x.reshape(-1), "y": y.reshape(-1),
            "ab": ab, "logprob": lp, "value": value.squeeze(-1), "card_mask": card_mask,
            "h": h, "c": c, "ready": ready.to(obs_t.device)}


def opp_rows(out, slots):
    """(B, OPP_DIM) float32 "opp" actions from a policy_step result."""
    card = out["card"].detach().cpu().numpy()
    a = np.zeros((len(card), OPP_DIM), dtype=np.float32)
    a[:, 0] = card
    a[:, 1] = out["x"].detach().cpu().numpy()
    a[:, 2] = out["y"].detach().cpu().numpy()
    ab = out["ab"].detach().cpu().numpy()
    for k, slot in enumerate(slots):
        a[:, 2 + slot] = ab[:, k]
    return a
