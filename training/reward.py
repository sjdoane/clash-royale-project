"""The minimal reward (TRAINING_PLAN.md section 5), computed outside the engine.

    r_t = outcome_t                                   +1 win / -1 loss / `draw` on a draw, terminal only
        + gamma * Phi(s_{t+1}) * (1 - done) - Phi(s_t) potential-based tower shaping (Ng et al. 1999)
        - c * leak_t                                   elixir lost while the bar sits at 10

Phi(s) = w_tower * ( mean_i g(d_enemy_i) - mean_i g(d_own_i) ), d = damage fraction of
a tower (1 - hp/hp0, King/left/right), g(d) = d for k=0 else (e^{kd}-1)/(e^k-1)
(convex: finishing a hurt tower is worth more than chipping a full one).
Phi(terminal) := 0, so sum_t gamma^t F_t = -Phi(s_0) = 0 on every episode and the
term is exactly policy-invariant (the engine's own tower term is not: it leaves
gamma^T Phi(s_T) as a hidden terminal bonus -- engine CLAUDE.md, "Reward").

leak_t = elixir generated during the step that could not be banked:
    gen  = ELIXIR_REGEN_RATE * multiplier * ticks
    kept = e_{t+1} - e_t + spent_{t+1} - spent_t
    leak = clip(gen - kept, 0, gen)   only when e_{t+1} is at the cap
c is sized so that leaking for an ENTIRE max-length match (3600 ticks incl.
overtime, ~273 elixir) costs `leak_worst` (default 0.2) of a win.

Everything is a function of the observation (tower HP scalars, own elixir,
own cumulative spend, elixir phase), so the BC stage computes the SAME returns
from the human dataset to pre-train the critic.

Optional (off by default, ppo_league26 only): the enemy-KING-ACTIVATION term,
`w_king` of a win, ZERO-SUM between the seats. When the team-1 King first
wakes from DAMAGE we pay -w_king (the opponent +w_king); when ours first wakes
from damage we get +w_king (they -w_king). A wake caused by a Princess Tower
falling is not penalised (`KingWakeLatch`). w_king = 0 leaves every number
above bit-identical.

Optional FirstLight-Zero ports (docs/firstlight_zero_study.md R4/R6/R7), all off
by default so every number above stays bit-identical:

  king_chip      weight on a King's damage fraction in Phi while BOTH of its
                 Princesses stand (FLZ `king_shaping_mode='hp_point_weighted'`,
                 0.2). Applied to both teams, so Phi1 = -Phi0 still holds.
  w_unilateral   the unilateral overflow term (FLZ reward.py, `0.075 * w^2`):
                 w accumulates the elixir we leak while the opponent is NOT also
                 at the cap and resets when both are full; the step pays the
                 increase of c*w^2, capped at 0.1 per transition. Per seat, like
                 the leak term (not zero-sum).
  act_delay      (env config, `ActuationDelay`): a card chosen at decision t is
                 executed d ticks later, d = an episode delay in [lo, hi] plus a
                 per-play jitter in [0, jitter]. While it is in flight the
                 returned observation already shows it spent (its cost taken off
                 the elixir scalar, its hand slot's cost zeroed, so the
                 affordability mask blocks it): the "in-flight card" feature,
                 built from the existing observation layout, no engine change.

How it reaches the PPO loop without editing engine/: `RewardEnv` (a subclass of
the engine's MicroRoyaleEnv) publishes `r_shape` (tower F - c*leak) in `info`
and keeps the env's own reward = the pure engine outcome. `install_reward()`
then rebinds three names in `python_ai.rl.base_trainer`'s namespace:
compute_shaping -> reads info["r_shape"], flawless_defense_bonus -> 0,
DRAW_PENALTY -> -draw. Everything else in the engine's loop is unchanged.
"""
import math
import random

import numpy as np

from training import common as C

#: Engine regen per tick at 1x (GameManager::ELIXIR_REGEN_RATE, mirrored in advisors/tactics).
from python_ai.advisors.tactics import ELIXIR_REGEN_RATE  # noqa: E402

DEFAULT_W_TOWER = 1.0
DEFAULT_K = 0.0
DEFAULT_LEAK_WORST = 0.2
DEFAULT_DRAW = 0.0
DEFAULT_W_KING = 0.0
DEFAULT_KING_CHIP = 1.0
DEFAULT_W_UNILATERAL = 0.0
#: FLZ's per-transition cap on the overflow penalty (reward.py, step_penalty_cap).
UNILATERAL_STEP_CAP = 0.1
SKIP_TICKS = 10
# The engine's hard match end (patch 0004: 3000, overtime expires at 5:00);
# 3600 on an engine without it, the training max_ticks.
MAX_TICKS = int(getattr(C.E.ClashRoyaleEnv, "MATCH_END_TICK", 3600))


def max_match_elixir(max_ticks=MAX_TICKS):
    """Elixir generated over a full match under the engine's phase schedule."""
    return float(sum(ELIXIR_REGEN_RATE * C.E.elixir_multiplier_at_tick(t)
                     for t in range(max_ticks)))


def leak_coef(leak_worst=DEFAULT_LEAK_WORST, max_ticks=MAX_TICKS):
    return float(leak_worst) / max_match_elixir(max_ticks)


def _g(d, k):
    d = np.clip(d, 0.0, 1.0)
    if abs(k) < 1e-9:
        return d
    return (np.exp(k * d) - 1.0) / (math.exp(k) - 1.0)


def tower_damage_fracs(obs):
    """(own (N,3), enemy (N,3)) damage fractions from the observation.

    Each is 1 - hp / that tower's own max hp, so a tower of any level runs 0 -> 1.
    On a gen-2 engine (C.TOWER_OBS_OWN_MAX) the scalar already is that fraction.
    On an older one it is hp / MAX_BUILDING_HP, rescaled by the level-11 hp0,
    which is right only at level 11: against a level-15 tower it read the first
    31% of damage as none (engine UPSTREAM item 44).
    """
    obs = np.atleast_2d(obs)
    own = obs[:, C.EXTRA + 3:C.EXTRA + 6].astype(np.float64)
    enemy = obs[:, C.EXTRA + 6:C.EXTRA + 9].astype(np.float64)
    if not C.TOWER_OBS_OWN_MAX:
        hp0 = np.asarray(C.TOWER_HP0, dtype=np.float64)
        own = own * C.MAX_BUILDING_HP / hp0
        enemy = enemy * C.MAX_BUILDING_HP / hp0
    return 1.0 - np.clip(own, 0, 1), 1.0 - np.clip(enemy, 0, 1)


def _king_weighted(g, d, king_chip):
    """g (N,3) with the King column scaled by king_chip while both Princesses stand."""
    if king_chip == 1.0:
        return g
    g = np.array(g, dtype=np.float64, copy=True)
    both = (d[:, 1] < 1.0 - 1e-9) & (d[:, 2] < 1.0 - 1e-9)
    g[:, 0] *= np.where(both, king_chip, 1.0)
    return g


def potential(obs, w_tower=DEFAULT_W_TOWER, k=DEFAULT_K, king_chip=DEFAULT_KING_CHIP):
    """Phi(s) for a batch of observations, (N,)."""
    d_own, d_enemy = tower_damage_fracs(obs)
    ge = _king_weighted(_g(d_enemy, k), d_enemy, king_chip)
    go = _king_weighted(_g(d_own, k), d_own, king_chip)
    return (w_tower * (ge.mean(1) - go.mean(1))).astype(np.float64)


def elixir_leak(obs0, obs1, ticks=SKIP_TICKS):
    """(N,) elixir lost at the cap between two consecutive observations."""
    obs0, obs1 = np.atleast_2d(obs0), np.atleast_2d(obs1)
    e0 = obs0[:, C.SPATIAL_SIZE].astype(np.float64) * 10.0
    e1 = obs1[:, C.SPATIAL_SIZE].astype(np.float64) * 10.0
    s0 = obs0[:, C.EXTRA + 1].astype(np.float64) * C.MAX_MATCH_ELIXIR
    s1 = obs1[:, C.EXTRA + 1].astype(np.float64) * C.MAX_MATCH_ELIXIR
    mult = obs0[:, C.EXTRA + 9].astype(np.float64) * C.MAX_ELIXIR_MULT
    gen = ELIXIR_REGEN_RATE * mult * ticks
    kept = e1 - e0 + np.maximum(0.0, s1 - s0)
    at_cap = e1 >= 9.99
    return np.where(at_cap, np.clip(gen - kept, 0.0, gen), 0.0)


class KingWakeLatch:
    """Per-episode, per-team "has this King been activated yet", from tower HP.

    The engine's `Tower::awake` latch is not bound to Python, so it is
    reconstructed from what is: `get_tower_hp` / `get_tower_max_hp` (slot 0 =
    King, 1/2 = Princesses, board coordinates, destroyed reads <= 0). The
    engine wakes a King on ANY damage (hp < maxHp) or on a Princess of its team
    being destroyed. Called once after every env step:

      * a Princess of that team is down (this step or earlier) -> latch, cause
        "tower" (not penalised). A step where the King is hit AND a Princess
        falls counts as "tower": the conservative reading.
      * otherwise King hp < max -> latch, cause "damage" (penalised). Damage to
        a team's King can only come from the other team, so it is attributable.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.woken = [False, False]

    def update(self, game):
        """-> (cause0, cause1): None / "damage" / "tower" for a King that woke
        on this step (team 0 = ours, team 1 = the opponent's)."""
        out = [None, None]
        for team in (0, 1):
            if self.woken[team]:
                continue
            princess_down = (game.get_tower_hp(team, 1) <= 0
                             or game.get_tower_hp(team, 2) <= 0)
            if princess_down:
                self.woken[team] = True
                out[team] = "tower"
            elif game.get_tower_hp(team, 0) < game.get_tower_max_hp(team, 0):
                self.woken[team] = True
                out[team] = "damage"
        return tuple(out)


class UnilateralOverflow:
    """FLZ's unilateral overflow term for one seat (firstlight_zero/reward.py).

    step(leak, self_full, opp_full) -> penalty >= 0. Only a step that ends with
    this seat at the cap counts; leaking while the opponent is also full resets
    the account (no penalty), otherwise w grows by the leak and the step pays
    c*w^2 minus what was already paid, capped at `cap`.
    """

    def __init__(self, coef, cap=UNILATERAL_STEP_CAP):
        self.coef, self.cap = float(coef), float(cap)
        self.reset()

    def reset(self):
        self.w = 0.0
        self.paid = 0.0

    def step(self, leak, self_full, opp_full):
        if not self_full:
            return 0.0
        if opp_full:
            self.w, self.paid = 0.0, 0.0
            return 0.0
        self.w += max(0.0, float(leak))
        pen = min(self.cap, max(0.0, self.coef * self.w * self.w - self.paid))
        self.paid += pen
        return pen


def king_terms(w_king, cause0, cause1):
    """(r_seat0, r_seat1) from one step's wake causes; zero-sum by construction."""
    r = 0.0
    if cause1 == "damage":      # we woke THEIR King
        r -= w_king
    if cause0 == "damage":      # they woke OURS
        r += w_king
    return r, -r


class RewardSpec:
    def __init__(self, gamma, w_tower=DEFAULT_W_TOWER, k=DEFAULT_K,
                 leak_worst=DEFAULT_LEAK_WORST, draw=DEFAULT_DRAW, w_king=DEFAULT_W_KING,
                 king_chip=DEFAULT_KING_CHIP, w_unilateral=DEFAULT_W_UNILATERAL):
        self.gamma, self.w_tower, self.k = float(gamma), float(w_tower), float(k)
        self.leak_worst, self.draw = float(leak_worst), float(draw)
        self.w_king = float(w_king)
        self.king_chip = float(king_chip)
        self.w_unilateral = float(w_unilateral)
        self.c_leak = leak_coef(leak_worst)

    def as_dict(self):
        d = {"gamma": self.gamma, "w_tower": self.w_tower, "k": self.k,
             "leak_worst": self.leak_worst, "c_leak": self.c_leak, "draw": self.draw}
        if self.w_king:
            # Only when on, so ppo_bc26's stamped reward dict is unchanged.
            d["w_king"] = self.w_king
        if self.king_chip != DEFAULT_KING_CHIP:
            d["king_chip"] = self.king_chip
        if self.w_unilateral:
            d["w_unilateral"] = self.w_unilateral
        return d

    def shaping(self, obs0, obs1, done, ticks=SKIP_TICKS):
        """(F_tower, leak_elixir) for one transition batch; Phi(terminal) = 0.
        `ticks`: engine ticks between obs0 and obs1 (the decision interval)."""
        phi0 = potential(obs0, self.w_tower, self.k, self.king_chip)
        phi1 = potential(obs1, self.w_tower, self.k, self.king_chip) * (1.0 - np.asarray(done, np.float64))
        return self.gamma * phi1 - phi0, elixir_leak(obs0, obs1, ticks)

    def episode_rewards(self, obs, outcome, ticks=SKIP_TICKS):
        """Per-step rewards for one recorded episode (obs (T, D) in time order, the
        last row is the final decision state). Used by BC for critic targets.
        `ticks`: engine ticks between consecutive rows (the decision interval).

        The transition out of the last row is terminal (we have no s_T row), so
        its shaping is -Phi(s_{T-1}) and the outcome lands on it."""
        T = len(obs)
        r = np.zeros(T, dtype=np.float64)
        if T > 1:
            F, leak = self.shaping(obs[:-1], obs[1:], np.zeros(T - 1), ticks)
            r[:-1] = F - self.c_leak * leak
        r[-1] = -potential(obs[-1:], self.w_tower, self.k, self.king_chip)[0] + (
            outcome if outcome != 0 else self.draw)
        return r

    def returns(self, rewards):
        out = np.zeros_like(rewards)
        acc = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            acc = rewards[t] + self.gamma * acc
            out[t] = acc
        return out


# ---------------------------------------------------------------------------
# The env side
# ---------------------------------------------------------------------------
from python_ai.envs import gym_wrapper  # noqa: E402

from training import obs_v2  # noqa: E402

obs_v2.install_step_tap(gym_wrapper.MicroRoyaleEnv)   # notes each play's tile for the tail


#: An elixir bar reads "full" from here (elixir_leak's at-cap test).
FULL_ELIXIR = 9.99


class _FixedTeacher:
    """Team 1 for one engine sub-step: a given (slot, x, y) and ability flags,
    in team 1's own frame (step_self_play mirrors it)."""

    def __init__(self, slot=C.NOOP, x=0.0, y=0.0, ab=(False, False)):
        self.slot, self.x, self.y, self.ab = int(slot), float(x), float(y), tuple(ab)

    def act(self, game, obs1):
        return self.slot, self.x, self.y

    def ability_flags(self, game, obs1):
        return self.ab


class ActuationDelay:
    """Touch -> execution latency for network-controlled seats (FLZ R4).

    A card chosen at tick t is queued as (t + d, card id, x, y) and played when
    its tick comes, from whatever hand slot holds that card id then (dropped if
    it left the hand). d = the episode's delay, uniform in [lo, hi], plus a
    per-play jitter uniform in [0, jitter]. A card already in flight cannot be
    queued again.
    """

    def __init__(self, lo, hi, jitter, rng):
        self.lo, self.hi, self.jitter = int(lo), int(hi), int(jitter)
        self.rng = rng
        self.d = self.lo
        self.pending = {0: [], 1: []}
        self.dropped = 0

    def new_episode(self):
        self.d = self.rng.randint(self.lo, self.hi)
        self.pending = {0: [], 1: []}
        self.dropped = 0

    def push(self, team, tick, card_id, x, y):
        if any(e[1] == card_id for e in self.pending[team]):
            self.dropped += 1
            return False
        j = self.rng.randint(0, self.jitter) if self.jitter > 0 else 0
        self.pending[team].append([int(tick) + self.d + j, int(card_id), float(x), float(y)])
        self.pending[team].sort(key=lambda e: e[0])
        return True

    def pop_due(self, team, tick):
        q = self.pending[team]
        if q and q[0][0] <= tick:
            return q.pop(0)
        return None

    def next_event(self, tick):
        """The earliest tick > `tick` at which something is due, or None; tick + 1
        if a second play is already due at `tick` (one card per engine step)."""
        ts = [e[0] for q in self.pending.values() for e in q]
        if any(t <= tick for t in ts):
            return tick + 1
        later = [t for t in ts if t > tick]
        return min(later) if later else None


_SPELL_RADIUS = {}


def _spell_radius(card_id):
    """A spell card's effect radius in tiles (engine CardStats), 0 for anything else."""
    r = _SPELL_RADIUS.get(card_id)
    if r is None:
        try:
            if C.E.get_card_info(card_id)["is_spell"]:
                base = C.E.get_card_stats(card_id)["base"]
                r = float(max(base.get("spell_radius", 0.0), base.get("roll_width", 0.0) / 2.0))
            else:
                r = 0.0
        except Exception:
            r = 0.0
        _SPELL_RADIUS[card_id] = r
    return r


class RewardEnv(gym_wrapper.MicroRoyaleEnv):
    """MicroRoyaleEnv + the minimal reward's shaping in info['r_shape'], and the
    real-deck opponent sampler (training/opponents.py) when `real_pool` is set.

    The returned reward stays the engine's pure outcome (+1/-1/0), so the engine
    trainer's win/draw bookkeeping (|r| > 0.5) keeps working unchanged.

    `act_delay` = (lo, hi, jitter) ticks turns on ActuationDelay for our seat
    (and, in LeagueEnv, for a network-controlled seat 1). The UtilityTeacher is
    never delayed and still decides once per decision step.

    `opp_level` sets team 1's card AND tower level (engine set_team_level). An
    int fixes it (eval: 11, 13, 15). (cap, p_cap, lo) samples it per episode:
    the cap with probability p_cap, else uniform in [lo, cap], so the levels
    already learned stay in the mix; `set_opp_level_cap` moves the cap (the
    trainer's 11 -> 15 curriculum). Our own seat stays at the tournament level.
    """

    def __init__(self, env_config=None):
        env_config = dict(env_config or {})
        self._spec = RewardSpec(**env_config.pop("reward_spec"))
        pool_path = env_config.pop("real_pool", None)
        p_freq = env_config.pop("p_freq", 0.5)
        self._fixed_deck = env_config.pop("fixed_opp_deck", None)
        act_delay = env_config.pop("act_delay", None)
        opp_level = env_config.pop("opp_level", None)
        # Engine ticks per decision step when step() is called without
        # skip_frames (10 = the 1 Hz every gen-1 policy decided at; gen-2: 5).
        # The elixir-leak term uses the ticks a step really advanced.
        self.decision_ticks = int(env_config.pop("decision_ticks", SKIP_TICKS))
        if self.decision_ticks < 1:
            raise ValueError("decision_ticks must be >= 1")
        # Perception noise on the ACTOR's observation only (training/obs_noise.py).
        # None / False = off. True, a JSON path, or a dict of overrides = on.
        noise_spec = env_config.pop("obs_noise", None)
        # Observation v2 tail (training/obs_v2.py): recent plays of both sides,
        # spells in the air, our next card, appended after the engine's floats.
        # obs_v2=False gives the engine observation alone (old checkpoints run
        # either way: they ignore the tail). obs_v2_noise (True / dict / JSON
        # path) degrades the opponent part like the live reader (obs_noise.TailNoise).
        v2_on = env_config.pop("obs_v2", True)
        v2_noise = env_config.pop("obs_v2_noise", None)
        self._v2 = None
        if pool_path:
            env_config["deck_pool"] = False     # engine's 16-deck pool off; we sample
        super().__init__(env_config)
        self._sampler = None
        seed = env_config.get("scenario_seed")
        if pool_path:
            from training import opponents
            self._sampler = opponents.RealDeckSampler(
                opponents.load_pool(pool_path), p_freq=p_freq,
                rng=random.Random(seed if seed is not None else None))
        self._cur_bucket = None
        self._last_obs = None
        self._king = KingWakeLatch() if self._spec.w_king else None
        self._delay = None
        if act_delay is not None and int(act_delay[1]) > 0:
            self._delay = ActuationDelay(*act_delay[:3], rng=random.Random(
                None if seed is None else int(seed) + 4099))
        self._opp_level = None
        self._level_rng = random.Random(None if seed is None else int(seed) + 6151)
        if opp_level is not None:
            lv = (int(opp_level), 1.0, int(opp_level)) if np.isscalar(opp_level) else opp_level
            self._opp_level = [int(lv[0]), float(lv[1]), int(lv[2])]
        self.ep_opp_level = self._team1_level()
        self._noise = None
        if noise_spec is not None and noise_spec is not False:
            from training import obs_noise
            self._noise = obs_noise.ObsNoise(obs_noise.load_config(noise_spec),
                                             seed=None if seed is None else int(seed) + 9973)
        self._uni = None
        if self._spec.w_unilateral:
            self._uni = (UnilateralOverflow(self._spec.w_unilateral),
                         UnilateralOverflow(self._spec.w_unilateral))
        if v2_on:
            self._v2 = obs_v2.EnvTail(self, v2_noise, seed, C.E)

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        obs_v2.wrap_outputs(cls)        # a subclass's own reset / step get the tail too

    def reset(self, seed=None, options=None):
        if self._sampler is not None:
            if seed is not None:
                self._sampler.rng.seed(int(seed) + 7919)
            d = self._sampler.sample()
            self._cur_bucket = d["bucket"]
            self.current_deck_name = d["bucket"]
            self.set_opponent_deck(d["card_ids"])
        elif self._fixed_deck is not None:
            self.set_opponent_deck(self._fixed_deck)
        if self._opp_level is not None:
            if seed is not None:
                self._level_rng.seed(int(seed) + 6151)
            cap, p_cap, lo = self._opp_level
            lv = cap if (lo >= cap or self._level_rng.random() < p_cap) else \
                self._level_rng.randint(min(lo, cap), cap)
            assert self.game.set_team_level(1, int(lv)), lv
        self.ep_opp_level = self._team1_level()
        obs, info = super().reset(seed=seed, options=options)
        info = dict(info)
        info["opp_level"] = self.ep_opp_level
        self._last_obs = obs
        if self._king is not None:
            self._king.reset()
        if self._delay is not None:
            if seed is not None:
                self._delay.rng.seed(int(seed) + 4099)
            self._delay.new_episode()
            info = dict(info)
            info["act_delay_ticks"] = self._delay.d
        if self._uni is not None:
            for u in self._uni:
                u.reset()
        if self._noise is not None:
            # _last_obs (reward shaping) keeps the clean observation.
            info = dict(info)
            if self._noise.cfg.publish_clean:
                info["obs_clean"] = obs
            obs = self._noise.reset(obs, seed=None if seed is None else int(seed) + 9973,
                                    prefer=self._decks_in_play())
            info["obs_noise_scale"] = self._noise.scale
        if self._v2 is not None:
            return self._v2.reset_out(self, obs, info, seed)
        return obs, info

    def _team1_level(self):
        """Team 1's card level; 11 on an engine without set_team_level (pre-0008)."""
        get = getattr(self.game, "get_team_card_level", None)
        return int(get(1)) if get is not None else 11

    def set_opp_level_cap(self, cap):
        """The curriculum hook: the next episodes sample team 1's level up to `cap`."""
        if self._opp_level is not None:
            self._opp_level[0] = int(cap)

    # -- actuation delay -----------------------------------------------------------
    def _team1_delayed(self):
        """True when team 1 is a network seat whose card plays go through the delay."""
        return self._delay is not None and getattr(self.teacher, "delayable", False)

    def _hand(self, team):
        return list(self.game.get_hand() if team == 0 else self.game.get_hand_for_team(1))

    def inflight_obs(self, obs, team=0):
        """`obs` as the live screen shows it while cards are in flight: each one's
        cost already off the elixir scalar and its slot's cost zeroed (so the
        affordability mask blocks it)."""
        if self._delay is None or not self._delay.pending[team]:
            return obs
        o = np.array(obs, dtype=np.float32, copy=True)
        hand = self._hand(team)
        s = C.SPATIAL_SIZE
        for _, cid, _, _ in self._delay.pending[team]:
            if cid in hand:
                o[s + 1 + hand.index(cid)] = 0.0
            o[s] = max(0.0, float(o[s]) - float(C.E.get_card_info(cid)["cost"]) / 10.0)
        return o

    def _delayed_step(self, action, k):
        dl = self._delay
        t0 = int(self.game.get_current_tick())
        slot = int(gym_wrapper._to_scalar(action["card_index"]))
        if slot != C.NOOP:
            hand = self._hand(0)
            if 0 <= slot < len(hand) and hand[slot] >= 0:
                dl.push(0, t0, hand[slot], float(gym_wrapper._to_scalar(action["target_x"])),
                        float(gym_wrapper._to_scalar(action["target_y"])))
        ab0 = {key: action.get(key, 0) for key in ("activate_ability_slot1", "activate_ability_slot2")}
        real_teacher = self.teacher
        t1_neural = self._team1_delayed()
        ab1 = (False, False)
        if t1_neural:
            c1, x1, y1 = real_teacher.act(self.game, None)
            ab1 = real_teacher.ability_flags(self.game, None)
            if int(c1) != C.NOOP:
                hand1 = self._hand(1)
                if 0 <= int(c1) < len(hand1) and hand1[int(c1)] >= 0:
                    dl.push(1, t0, hand1[int(c1)], x1, y1)
        t, end = t0, t0 + int(k)
        first = True
        reward_sum, opp_played = 0.0, -1
        out = None
        try:
            while t < end:
                a0 = {"card_index": C.NOOP, "target_x": 0.0, "target_y": 0.0,
                      "activate_ability_slot1": ab0["activate_ability_slot1"] if first else 0,
                      "activate_ability_slot2": ab0["activate_ability_slot2"] if first else 0}
                e = dl.pop_due(0, t)
                if e is not None:
                    hand = self._hand(0)
                    if e[1] in hand:
                        a0.update(card_index=hand.index(e[1]), target_x=e[2], target_y=e[3])
                    else:
                        dl.dropped += 1
                if t1_neural or not first:
                    tt = _FixedTeacher(ab=ab1 if first else (False, False))
                    e1 = dl.pop_due(1, t) if t1_neural else None
                    if e1 is not None:
                        hand1 = self._hand(1)
                        if e1[1] in hand1:
                            tt.slot, tt.x, tt.y = hand1.index(e1[1]), e1[2], e1[3]
                        else:
                            dl.dropped += 1
                    self.teacher = tt
                else:
                    self.teacher = real_teacher     # the teacher decides once per decision
                nxt = dl.next_event(t)
                seg = (min(end, nxt) if nxt is not None else end) - t
                out = gym_wrapper.MicroRoyaleEnv.step(self, a0, seg)
                first = False
                t += seg
                reward_sum += float(out[1])
                if opp_played < 0:
                    opp_played = out[4].get("opp_played_card", -1)
                if out[2] or out[3]:
                    break
        finally:
            self.teacher = real_teacher
        obs, _, term, trunc, info = out
        info["opp_played_card"] = opp_played
        info["act_delay_ticks"] = dl.d
        info["act_inflight"] = len(dl.pending[0])
        info["act_dropped"] = dl.dropped
        return obs, reward_sum, term, trunc, info

    def _decks_in_play(self):
        deck = getattr(self, "current_opp_deck", None) or getattr(self, "opp_deck", None) or []
        return list(C.deck_ids()) + [int(c) for c in deck]

    def _own_spell(self, action):
        """(card id, x, y, radius, last-played tick) if the action asks for a
        spell from the hand, else None. The noise model hides units under it."""
        try:
            slot = int(action["card_index"])
        except (KeyError, TypeError, IndexError):
            return None
        hand = list(self.game.get_hand())
        if not 0 <= slot < len(hand):
            return None
        cid = int(hand[slot])
        rad = _spell_radius(cid)
        if rad <= 0.0:
            return None
        return (cid, float(action["target_x"]) + 0.5, float(action["target_y"]) + 0.5, rad,
                self.game.get_last_played_tick(0, cid))

    def _noised(self, clean, obs, action_spell, info, skip_frames):
        spells = []
        if action_spell is not None:
            cid, x, y, rad, before = action_spell
            if self.game.get_last_played_tick(0, cid) != before:      # it was really played
                spells.append((x, y, rad))
        opp = int(info.get("opp_played_card", -1))
        opp_rad = _spell_radius(opp) if opp >= 0 else 0.0
        if self._noise.cfg.publish_clean:
            info["obs_clean"] = clean
        info["obs_noise_scale"] = self._noise.scale
        return self._noise.apply(obs, skip_frames, spells=spells, opp_spell_radius=opp_rad or None)

    def step(self, action, skip_frames=None):
        if skip_frames is None:
            skip_frames = self.decision_ticks
        obs0 = self._last_obs
        spell = self._own_spell(action) if self._noise is not None else None
        if self._delay is None:
            obs, reward, terminated, truncated, info = super().step(action, skip_frames)
        else:
            obs, reward, terminated, truncated, info = self._delayed_step(action, skip_frames)
        F, leak = self._spec.shaping(obs0, obs, [float(terminated)], skip_frames)
        info["r_tower"] = float(F[0])
        info["leak_elixir"] = float(leak[0])
        info["r_shape"] = float(F[0] - self._spec.c_leak * leak[0])
        if self._uni is not None:
            e0, e1 = self.game.get_elixir_for_team(0), self.game.get_elixir_for_team(1)
            pen = self._uni[0].step(leak[0], e0 >= FULL_ELIXIR, e1 >= FULL_ELIXIR)
            info["r_shape"] -= pen
            info["r_unilateral"] = -pen
        if self._king is not None:
            c0, c1 = self._king.update(self.game)
            rk0, rk1 = king_terms(self._spec.w_king, c0, c1)
            info["r_shape"] += rk0
            info["r_king"] = rk0
            info["r_king_seat1"] = rk1
            # 1 = woke from damage (penalised), 2 = woke from a Princess falling.
            info["king_wake_own"] = {None: 0, "damage": 1, "tower": 2}[c0]
            info["king_wake_enemy"] = {None: 0, "damage": 1, "tower": 2}[c1]
        if terminated and self._sampler is not None:
            self._sampler.record(self._cur_bucket, reward > 0.5)
        info["opp_level"] = self.ep_opp_level
        self._last_obs = obs
        out = self.inflight_obs(obs, 0)
        if self._noise is not None:
            out = self._noised(obs, out, spell, info, skip_frames)
        if self._v2 is not None:
            return self._v2.step_out(self, (out, reward, terminated, truncated, info))
        return out, reward, terminated, truncated, info

    # The trainer's PFSP read-out / checkpoint hooks, at bucket granularity.
    def get_deck_pool_stats(self):
        return dict(self._sampler.stats) if self._sampler else super().get_deck_pool_stats()

    def get_deck_pool_counts(self):
        return dict(self._sampler.n) if self._sampler else super().get_deck_pool_counts()

    def set_deck_pool_stats(self, stats, counts=None):
        if self._sampler is None:
            return super().set_deck_pool_stats(stats, counts)
        for k, v in (stats or {}).items():
            if k in self._sampler.stats:
                self._sampler.stats[k] = float(v)
        for k, v in (counts or {}).items():
            if k in self._sampler.n:
                self._sampler.n[k] = int(v)


def install_reward(spec):
    """Rebind the engine loop's shaping names to the minimal reward (see module doc)."""
    from python_ai.rl import base_trainer as bt
    from python_ai.rl import engine_stats as es

    orig_extract = es.extract_engine_stats

    def extract_with_shape(infos, num_envs):
        stats = orig_extract(infos, num_envs)
        stats["r_shape"] = np.asarray(infos.get("r_shape", np.zeros(num_envs)), np.float32)
        stats["r_tower"] = np.asarray(infos.get("r_tower", np.zeros(num_envs)), np.float32)
        stats["leak_elixir"] = np.asarray(infos.get("leak_elixir", np.zeros(num_envs)), np.float32)
        return stats

    def minimal_shaping(stats, prev_stats, gamma=None, w_spell=None, **_):
        return np.asarray(stats["r_shape"], dtype=np.float32)

    def no_flawless(dones, step_rewards, stats, prev_stats, w=None):
        return np.zeros(len(dones), dtype=np.float32)

    bt.extract_engine_stats = extract_with_shape
    bt.compute_shaping = minimal_shaping
    bt.flawless_defense_bonus = no_flawless
    # base_trainer subtracts DRAW_PENALTY * is_draw; a draw is worth `spec.draw`.
    bt.DRAW_PENALTY = -float(spec.draw)
