"""Sim-to-real randomization for gen-2 training and evaluation (opt-in).

What the live driver gives the policy differs from what the simulator gives it
in ways the board noise (training/obs_noise.py), the tail noise and the
actuation delay (training/reward.py) do not cover. This module adds them, each
behind its own switch, each with a range taken from the live recordings (the
sources are in training/SIM2REAL_REPORT.md and next to every default below).

  name      what it models                                             drawn
  elixir    the HUD digit: an integer (floor), its tick-over phase,     match + step
            a wrong digit for a moment after a play, rare misreads
  refill    the hand slot stays empty ~1.4 s after a card leaves it     match + play
  hand      a slot read as empty / as another card for a few frames     match + step
  clock     the match clock in whole seconds, off by up to 1 s          match
  towers    a tower numeral held at its last value while unreadable     match + step
  opptrack  opponent cycle block and spend built from DETECTED plays    match
            (the simulated detector that already feeds the opponent
            state estimate), not from the truth
  oppdet    that detector's miss / unknown / false / late rates         match
  tapfail   a tap that does not register: nothing is played, the card   match + play
            looks in flight for ~1.6 s, failures come in bursts
  place     a placement that lands one tile off                         match + play
  timing    decisions every ~0.55 s instead of 0.5 s                    match + step
  hold      a decision skipped because the board frame was stale        match + step
  levels    opponent one level above / below what the policy is told    match
            (the engine's only handle on unit-stat error)

Two time scales: every episode draws a strength `s` (scale x U(scale_range))
and its own systematic parameters (this device is slow, this reader is off
today); inside the episode errors are events with a duration in engine ticks,
so a slot stays unreadable and a tap burst stays a burst, as live.

s = 0 is exactly the clean environment: the action, the step length and the
observation object are passed through untouched. Structural differences
(integer elixir, whole-second clock, detected opponent plays) are on with
probability min(1, s) per episode; rates and durations scale linearly with s.

Only the ACTOR's observation and its action are touched. The reward, the
shaping terms, info["priv"] (the privileged-critic input) and the opponent see
the true state. The layer reads from the engine only what the live driver has:
ENGINE_READS below (a test holds it to that list).

Use: env config key `sim2real` = "v1" | a JSON path | a dict
{"preset": "v1", "scale": 1.0, "scale_range": [0.0, 1.5], "only": [...],
"off": [...], <any Sim2RealConfig field>: value}; `cli_spec` builds it from the
command-line flags of ppo_pool / eval_ghost. Without the key nothing here runs.
"""
import contextlib
import dataclasses
import json
import math
import os

import numpy as np

from training import common as C

NAMES = ("elixir", "refill", "hand", "clock", "towers", "opptrack", "oppdet",
         "tapfail", "place", "timing", "hold", "levels")

#: Everything the layer may call on the engine match. Own-seat state the live
#: driver also has, the simulated detector's reports, and placement legality.
ENGINE_READS = frozenset({"get_current_tick", "get_hand", "get_last_played_tick",
                          "get_opp_reports", "is_valid_placement"})

TICKS_PER_S = 10
CYCLE_TAU = 200.0                 # engine: recency = exp(-(now - play tick) / 200)
TIME_DENOM = 3600.0               # engine: obs time = tick / 3600
DOUBLE_TICK, TRIPLE_TICK = 1200, 2400
BOARD_W, BOARD_H = C.BOARD_W, C.BOARD_H
HOLD_MAX = 6                      # decisions one stale spell may swallow


def _layout():
    CE = C.E.ClashRoyaleEnv
    s = C.SPATIAL_SIZE
    n = int(CE.NUM_CARD_IDS)
    return {"S": s, "COST": s + 1, "ONEHOT": s + 1 + C.HAND_SIZE, "N": n, "X": int(C.EXTRA),
            "CYCLE": int(CE.CYCLE_START), "LEVEL": int(CE.LEVEL_START),
            "MAX_SPEND": float(C.MAX_MATCH_ELIXIR)}


def clean_span():
    """(a, b): the one contiguous stretch of the observation the layer may
    change (elixir, slot costs, hand one-hots, the time / spend / tower
    scalars, the cycle block, the level entries)."""
    L = _layout()
    return int(L["S"]), int(L["LEVEL"]) + 4


@dataclasses.dataclass
class Sim2RealConfig:
    # -- strength ----------------------------------------------------------------
    scale: float = 1.0
    scale_range: tuple = None                 # (lo, hi): per-episode U(lo, hi) x scale
    #: also put the true values of every entry the layer may change into
    #: info["s2r_clean"] (clean_span()): for the trainer's frozen reference
    #: policy, never for the actor. Not a randomization.
    publish_clean: bool = False
    # -- switches ----------------------------------------------------------------
    elixir: bool = True
    refill: bool = True
    hand: bool = True
    clock: bool = True
    towers: bool = True
    opptrack: bool = True
    oppdet: bool = True
    tapfail: bool = True
    place: bool = True
    timing: bool = True
    hold: bool = True
    levels: bool = True
    # -- elixir (live feeds the integer digit; runs/live/20261120-185026 and
    #    20261122-174340: 476/476 decisions integer) -------------------------------
    #: elixir added before the floor: when the digit ticks over relative to the
    #: true bar. Measured read - truth: mean -0.56 against -0.50 for a pure floor,
    #: and the digit one below the floor on up to 15 % of the frames away from our
    #: plays (1,927 frames, one match; the truth is reconstructed and drifts, so
    #: 15 % is an upper bound: the range runs from just past it to a slightly
    #: early digit, the pure floor inside).
    elixir_offset: tuple = (-0.25, 0.05)
    #: P(the digit is off by one for a moment after one of our plays). Bar drop
    #: seen 0.13-0.44 s after tap end (n = 32); mismatch within 1 s of a play
    #: 41 % of frames against 15 % elsewhere (upper bounds).
    elixir_transient_p: tuple = (0.0, 0.3)
    elixir_transient_ticks: tuple = (3, 10)   # mismatch runs: median 2, mean 4.4, max 15 frames at 11 fps
    elixir_minus_share: float = 0.8           # -1 on 14.6 % of frames, +1 on 3.6 %
    #: P(per decision) of a misread away from plays: labelled digit accuracy
    #: 99.95 % [99.92, 99.98] (runs/perception/STATE_ACCURACY_REPORT.md).
    elixir_misread_p: float = 0.0005
    # -- refill: slot empty after the card leaves. Median 1.37 s, p5-p95 1.25-1.55
    #    (n = 32, 20261120-185026); median 1.34 s (n = 41, 20261120-093139). ---------
    refill_ticks: tuple = (12.0, 16.0)
    # -- hand misreads: slot accuracy 99.13 % [98.96, 99.29] on human play, about
    #    half of the errors from drags the bot never makes -> 0.4 % per slot-frame.
    hand_misread_p: float = 0.004
    hand_misread_steps: float = 2.0           # mean duration, decisions
    hand_blank_share: float = 0.5             # read as empty; else as another card of our deck
    # -- clock: within 1 s on 99.68 % of labelled frames; fed in whole seconds. ----
    clock_offset_ticks: tuple = (-10, 10)
    # -- towers: numerals correct on 99.3-99.8 % of frames, own right Princess
    #    97.76 % (level badge over the numeral); the reader holds the last value.
    tower_stale_p: float = 0.005
    tower_stale_steps: float = 3.0
    tower_own_right_mult: float = 4.0
    # -- oppdet: the engine detector's rates, per episode between its defaults
    #    (miss 0.03-0.04, unknown 0.02, false 0.02) and twice-the-measured-minus-
    #    default, so the range is centred on what the driver's fused opponent
    #    plays scored on the test videos (runs/perception/STATE_ACCURACY_REPORT.md,
    #    89 gold plays): miss 0.146 [0.09, 0.23], card identity wrong 0.213
    #    [0.12, 0.35] (37/47 right), false plays 0.073 [0.03, 0.15].
    oppdet_miss_hi: float = 0.25
    oppdet_unknown_hi: float = 0.40
    oppdet_false_hi: float = 0.13
    oppdet_late_hi: float = 0.15              # not measured; the engine default is 0.05
    # -- tapfail: 0 of 32 with the current driver (95 % bound 9 %), 9 of 45 with the
    #    old one, four of them in a row. Per-episode share of taps that fail; bursts
    #    by a two-state chain.
    tapfail_p: tuple = (0.0, 0.06)
    tapfail_burst: float = 0.5                # P(fail | the previous attempt failed)
    tapfail_ticks: int = 16                   # play_watch: judged 1.2 s after tap end (0.38 s)
    # -- place: stationary troops landed on the commanded tile 10 of 12 times
    #    (two doubtful); no tile error shown. Conservative.
    place_p: tuple = (0.0, 0.05)
    # -- timing: 2 Hz driver, 313 intervals: median 0.550 s, p90 0.58, max 0.60.
    timing_extra_ticks: tuple = (0.3, 0.8)
    # -- hold: 0 stale-frame holds in 645 recorded decisions (95 % bound 0.46 %).
    hold_p: tuple = (0.0, 0.004)
    hold_continue: float = 0.5                # P(the next decision is skipped too)
    # -- levels: not measurable from recordings (friendly play is level 11 both
    #    sides; card levels differ per card). P(opponent is one level off).
    levels_p: float = 0.3
    # -- elixir estimate (opt-in, not a randomization; SIM2REAL_REPORT.md section 9):
    #    the digit stream above goes through the estimator the live driver runs
    #    (live_ppo/elixir_estimate.py: dead reckoning at the regeneration rate,
    #    corrected at each tick-over) and the observation carries its fractional
    #    estimate instead of the digit. The layer feeds it once per decision, and
    #    tells it of our plays when they are chosen and when they are seen to land.
    elixir_est: bool = False
    elixir_est_kw: dict = None                # ElixirEstimator keyword arguments
    #: with `elixir_est`, the digit's timing is this range instead of
    #: `elixir_offset`. Measured against the bar's fill (section 9): the digit
    #: turns over 0.024 +- 0.025 elixir before the bar reaches the whole number
    #: (118 tick-overs, three recordings), and reads one below the bar's floor on
    #: 0.3-0.7 % of frames away from our plays, not the 15 % `elixir_offset`
    #: allows for. The estimator's own tick-over window (`lead`) is the same range.
    elixir_est_offset: tuple = (0.0, 0.05)
    #: a card chosen that the bar could not pay for (the estimate was high) is a
    #: tap that does nothing: the driver shows it in flight for `tapfail_ticks`.
    elixir_est_fail: bool = True
    #: with `elixir_est`: the share of integer-digit episodes that go through the
    #: estimator; the rest show the integer as `v1` does (`elixir_offset`), so a
    #: policy trained here can also be run with the estimator off. Drawn from its
    #: own generator: at 1.0 (the default) nothing else about an episode changes.
    elixir_est_share: float = 1.0

    def enabled(self):
        return [n for n in NAMES if getattr(self, n)]


PRESETS = {"v1": {}}        # v1 = the defaults above, everything on


def load_config(spec):
    """None / False / "" -> None (off). "v1", a JSON path, or a dict -> config."""
    if spec is None or spec is False or spec == "":
        return None
    if isinstance(spec, Sim2RealConfig):
        return spec
    if isinstance(spec, str):
        if spec in PRESETS:
            spec = {"preset": spec}
        else:
            with open(spec if os.path.isabs(spec) else os.path.join(C.REPO, spec)) as fh:
                spec = json.load(fh)
    spec = {k: v for k, v in dict(spec).items() if not k.startswith("_")}
    preset = spec.pop("preset", "v1")
    if preset not in PRESETS:
        raise ValueError(f"sim2real preset {preset!r}: one of {sorted(PRESETS)}")
    only, off = spec.pop("only", None), spec.pop("off", None)
    vals = dict(PRESETS[preset])
    vals.update(spec)
    fields = {f.name for f in dataclasses.fields(Sim2RealConfig)}
    bad = sorted(set(vals) - fields)
    if bad:
        raise ValueError(f"unknown sim2real keys: {bad}")
    for k, v in list(vals.items()):
        if isinstance(v, list):
            vals[k] = tuple(v)
    cfg = Sim2RealConfig(**vals)
    for group, value in ((only, True), (off, False)):
        if group is None:
            continue
        group = [g for g in (group.split(",") if isinstance(group, str) else group) if g]
        unknown = sorted(set(group) - set(NAMES))
        if unknown:
            raise ValueError(f"unknown sim2real randomization(s) {unknown}: {NAMES}")
        if value:
            for n in NAMES:
                setattr(cfg, n, n in group)
        else:
            for n in group:
                setattr(cfg, n, False)
    if cfg.scale_range is not None:
        lo, hi = float(cfg.scale_range[0]), float(cfg.scale_range[1])
        if lo < 0 or hi < lo:
            raise ValueError(f"sim2real scale_range {cfg.scale_range}")
        cfg.scale_range = (lo, hi)
    if cfg.scale < 0:
        raise ValueError("sim2real scale must be >= 0")
    return cfg


def cli_spec(name, scale="", only="", off="", elixir_est=False):
    """Command-line flags -> the env's `sim2real` value (None = off).
    scale: "0.5" (fixed) or "0:1.5" (per-episode range)."""
    if not name:
        if elixir_est:
            raise SystemExit("--sim2real-elixir-est needs --sim2real <preset | JSON path>")
        if scale or only or off:
            raise SystemExit("--sim2real-scale / -only / -off need --sim2real <preset | JSON path>")
        return None
    if name in PRESETS:
        spec = {"preset": name}
    else:
        with open(name if os.path.isabs(name) else os.path.join(C.REPO, name)) as fh:
            spec = json.load(fh)
    if scale:
        if ":" in str(scale):
            lo, hi = str(scale).split(":")
            spec["scale_range"] = [float(lo), float(hi)]
            spec.setdefault("scale", 1.0)
        else:
            spec["scale"] = float(scale)
    if only:
        spec["only"] = [s for s in only.split(",") if s]
    if off:
        spec["off"] = [s for s in off.split(",") if s]
    if elixir_est:
        spec["elixir_est"] = True
    load_config(spec)                # refuse a bad spec here, not in a worker
    return spec


def widened(scale_range, frac):
    """The scale range at training progress `frac` in [0, 1]: the upper end
    grows linearly from the lower end to its full value."""
    lo, hi = float(scale_range[0]), float(scale_range[1])
    return [lo, lo + (hi - lo) * min(1.0, max(0.0, float(frac)))]


_COST = {}


def _cost(cid):
    c = _COST.get(cid)
    if c is None:
        c = _COST[cid] = float(C.E.get_card_info(int(cid))["cost"])
    return c


def _slot_of(action):
    v = action["card_index"]
    return int(np.asarray(v).reshape(-1)[0])


def _xy(action):
    return (float(np.asarray(action["target_x"]).reshape(-1)[0]),
            float(np.asarray(action["target_y"]).reshape(-1)[0]))


def _like(v, new):
    """`new` in the type / shape / dtype the action field came in."""
    if isinstance(v, np.ndarray):
        return np.full(v.shape, new, dtype=v.dtype)
    if isinstance(v, np.generic):
        return type(v)(new)
    return float(new)


def _regime(tick):
    return 1 if tick < DOUBLE_TICK else 2 if tick < TRIPLE_TICK else 3


_EST = []


def _estimator_class():
    """live_ppo/elixir_estimate.ElixirEstimator: the class the live driver runs."""
    if not _EST:
        import importlib.util  # noqa: PLC0415
        path = os.path.join(C.REPO, "live_ppo", "elixir_estimate.py")
        spec = importlib.util.spec_from_file_location("cr_elixir_estimate", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _EST.append(mod.ElixirEstimator)
    return _EST[0]


class Sim2Real:
    """The per-env state of the layer. The env mixin below calls, per episode,
    next_seed -> begin -> (super().reset) -> view, and per step
    pre_step -> (super().step) -> view."""

    def __init__(self, cfg, seed=None):
        self.cfg = cfg
        self.L = _layout()
        self.span = clean_span()
        self._stream = np.random.default_rng(None if seed is None else [int(seed), 77011])
        self._pending = None
        self._default_det = None
        self._det_dirty = False
        self._det_game = None
        self.rng = np.random.default_rng(0)
        self.scale = 0.0
        self.on = {n: False for n in NAMES}
        self.d = {}
        self.stats = {}
        self.last_ticks = 0
        self.est = None                        # ElixirEstimator, with cfg.elixir_est
        self._clear()

    # -- configuration ---------------------------------------------------------
    def set(self, over):
        """Merge {field: value} into the config from the next episode on (the
        trainer's widening schedule: {"scale_range": [lo, hi]})."""
        vals = dataclasses.asdict(self.cfg)
        vals.update({k: (tuple(v) if isinstance(v, list) else v) for k, v in dict(over).items()})
        self._pending = load_config(dict(vals, preset="v1"))
        return True

    @property
    def active(self):
        return self.scale > 0 and any(self.on.values())

    # -- episode -----------------------------------------------------------------
    def next_seed(self, seed):
        """The episode's seed: the given one (which restarts this env's stream),
        or the next of the stream. Passed down, so every sub-generator of the
        env restarts from it and the episode replays from it."""
        if seed is None:
            return int(self._stream.integers(1, 2 ** 31 - 1))
        self._stream = np.random.default_rng([int(seed), 77011])
        return int(seed)

    def _clear(self):
        self.blank_until = [0] * C.HAND_SIZE
        self.phantom = []                      # [until tick, card id, cost]
        self.last_fail = False
        self.el_tr = (0, 0)                    # (until tick, sign)
        self.mis = [None] * C.HAND_SIZE        # (until tick, card id or -1 = blank)
        self.tower_hold = [0] * 6
        self.tower_shown = None
        self.acc = 0.0
        self.hand_before = None
        self.lv_delta = 0
        self.est_picks = []                    # (tick, card id, cost, true elixir then, last-played tick then)
        self.e_clean = None
        self.stats = {"attempts": 0, "tap_fail": 0, "held": 0, "blocked_pick": 0, "moved": 0,
                      "plays": 0, "steps": 0, "ticks": 0}

    def begin(self, env, seed):
        """Draw the episode's strength and systematic parameters; set what must
        be set before the engine resets (opponent level, detector rates)."""
        if self._pending is not None:
            self.cfg, self._pending = self._pending, None
        cfg = self.cfg
        r = self.rng = np.random.default_rng([int(seed), 5501])
        lo, hi = cfg.scale_range or (1.0, 1.0)
        s = self.scale = float(cfg.scale) * float(r.uniform(lo, hi))
        gate = min(1.0, s)
        # Every draw is made whatever the switches say, in a fixed order: turning
        # one randomization off leaves the others' draws as they were.
        u = r.random(len(NAMES))
        ln3 = math.log(3.0)
        d = self.d = {}
        est_ep = bool(cfg.elixir_est)
        if est_ep and float(cfg.elixir_est_share) < 1.0:
            est_ep = float(np.random.default_rng([int(seed), 5503]).random()) < float(cfg.elixir_est_share)
        d["el_off"] = s * float(r.uniform(*(cfg.elixir_est_offset if est_ep else cfg.elixir_offset)))
        d["el_tr_p"] = min(1.0, s * float(r.uniform(*cfg.elixir_transient_p)))
        d["el_mis_p"] = min(0.5, s * cfg.elixir_misread_p)
        d["refill"] = s * float(r.uniform(*cfg.refill_ticks))
        d["hand_p"] = min(0.5, s * cfg.hand_misread_p * math.exp(float(r.uniform(-ln3, ln3))))
        d["clock_off"] = int(round(s * float(r.uniform(*cfg.clock_offset_ticks))))
        d["tower_p"] = min(0.5, s * cfg.tower_stale_p * math.exp(float(r.uniform(-ln3, ln3))))
        det = r.random(4)
        d["fail_p"] = min(1.0, s * float(r.uniform(*cfg.tapfail_p)))     # share of taps that fail
        # ... as a two-state chain: after a failure the next tap fails with
        # `tapfail_burst`, so the entry probability that gives that share is
        d["fail_enter"] = (1.0 if d["fail_p"] >= 1.0 else
                           min(1.0, d["fail_p"] * max(0.0, 1.0 - cfg.tapfail_burst) / (1.0 - d["fail_p"])))
        d["fail_ticks"] = int(cfg.tapfail_ticks) + int(r.integers(-2, 3))
        d["place_p"] = min(0.5, s * float(r.uniform(*cfg.place_p)))
        d["extra"] = s * float(r.uniform(*cfg.timing_extra_ticks))
        d["hold_p"] = min(0.5, s * float(r.uniform(*cfg.hold_p)))
        lv_u, lv_sign = float(r.random()), (1 if r.random() < 0.5 else -1)
        on = self.on = {n: bool(getattr(cfg, n)) and s > 0 for n in NAMES}
        d["el_floor"] = on["elixir"] and u[0] < gate
        d["clock_q"] = on["clock"] and u[3] < gate
        d["opptrack"] = on["opptrack"] and u[5] < gate
        self._clear()
        self.acc = float(r.random())
        self.est = None
        if est_ep and d["el_floor"]:
            # The engine refills at once after a play from a full bar (the game
            # waits 1 s: the estimator's cap_hold_s), so no wait here.
            self.est = _estimator_class()(**dict({"cap_hold_s": 0.0}, **(cfg.elixir_est_kw or {})))
            self.stats.update(unafford=0, est_n=0, est_sum=0.0, est_sq=0.0, est_above=0, est_below=0)
        # -- opponent level: one above / below the level the observation shows.
        if getattr(env, "_s2r_levels_ok", False):
            self.lv_delta = lv_sign if (on["levels"] and lv_u < min(1.0, s * cfg.levels_p)) else 0
            env._opp_level = [11 + self.lv_delta, 1.0, 11 + self.lv_delta] if self.lv_delta else None
        # -- the simulated detector's rates for this episode (engine binding).
        game = env.game
        if (on["oppdet"] or self._det_dirty) and hasattr(game, "set_opp_detection_noise"):
            if self._default_det is None:
                cur = game.get_opp_detection_noise()
                self._default_det = {k: getattr(cur, k) for k in dir(cur) if not k.startswith("_")}
            n = C.E.OppDetectionNoise()
            for k, v in self._default_det.items():
                setattr(n, k, v)
            if on["oppdet"]:
                for key, top, f in (("miss_troop", cfg.oppdet_miss_hi, det[0]),
                                    ("miss_spell", cfg.oppdet_miss_hi, det[0]),
                                    ("unknown_card", cfg.oppdet_unknown_hi, det[1]),
                                    ("false_play", cfg.oppdet_false_hi, det[2]),
                                    ("late_troop", cfg.oppdet_late_hi, det[3])):
                    base = float(self._default_det[key])
                    setattr(n, key, float(min(0.6, max(0.0, base + s * f * (top - base)))))
                d["det"] = {k: getattr(n, k) for k in ("miss_troop", "unknown_card", "false_play", "late_troop")}
            game.set_opp_detection_noise(n)
            self._det_dirty = on["oppdet"]
            self._det_game = game

    def info(self):
        st = self.stats
        return {"s2r_scale": float(self.scale), "s2r_tap_fail": int(st["tap_fail"]),
                "s2r_attempts": int(st["attempts"]), "s2r_held": int(st["held"]),
                "s2r_opp_level_delta": int(self.lv_delta)}

    # -- action side ---------------------------------------------------------------
    def _blocked(self, slot, cid, t):
        if self.blank_until[slot] > t:
            return True
        if any(p[1] == cid for p in self.phantom):
            return True
        m = self.mis[slot]
        return m is not None and m[0] > t and m[1] < 0

    def pre_step(self, env, action, skip):
        """-> (action, engine ticks). The same objects when nothing applies."""
        base = int(env.decision_ticks if skip is None else skip)
        if not self.active:
            return action, (skip if skip is not None else base)
        cfg, d, r, on = self.cfg, self.d, self.rng, self.on
        game = env.game
        t = int(game.get_current_tick())
        k = base
        if on["timing"] and d["extra"] > 0:
            self.acc += d["extra"] * float(r.uniform(0.5, 1.5))
            e = int(self.acc)
            self.acc -= e
            k = base + e
        if on["hold"] and d["hold_p"] > 0 and r.random() < d["hold_p"]:
            # The live driver makes no decision while its board frame is stale:
            # the policy is simply not asked, so this step lasts that much longer.
            n = min(HOLD_MAX, int(r.geometric(1.0 - cfg.hold_continue)))
            self.stats["held"] += n
            k += base * n
        hand = self.hand_before = [int(c) for c in game.get_hand()]
        slot = _slot_of(action)
        act = action
        if slot != C.NOOP and 0 <= slot < len(hand) and hand[slot] >= 0:
            cid = hand[slot]
            if self._blocked(slot, cid, t):
                # The live mask forbids this slot; the cost scalar is zero in our
                # observation, so a policy using the mask never gets here.
                self.stats["blocked_pick"] += 1
                act = dict(action, card_index=C.NOOP)
            else:
                self.stats["attempts"] += 1
                fail = False
                if on["tapfail"] and d["fail_p"] > 0:
                    fail = bool(r.random() < (cfg.tapfail_burst if self.last_fail else d["fail_enter"]))
                    self.last_fail = fail
                if self.est is not None:
                    self.est.own_play(t / TICKS_PER_S)
                    if not fail:
                        self.est_picks.append((t, cid, _cost(cid), self.e_clean,
                                               int(game.get_last_played_tick(0, cid))))
                if fail:
                    self.stats["tap_fail"] += 1
                    self.phantom.append([t + d["fail_ticks"], cid, _cost(cid)])
                    act = dict(action, card_index=C.NOOP)
                elif on["place"] and d["place_p"] > 0 and r.random() < d["place_p"]:
                    x, y = _xy(action)
                    dx, dy = ((int(r.choice((-1, 1))), 0) if r.random() < 0.5 else (0, int(r.choice((-1, 1)))))
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < BOARD_W and 0 <= ny < BOARD_H and \
                            game.is_valid_placement(cid, float(nx), float(ny), 0):
                        act = dict(action, target_x=_like(action["target_x"], nx),
                                   target_y=_like(action["target_y"], ny))
                        self.stats["moved"] += 1
        self.stats["steps"] += 1
        self.stats["ticks"] += k
        self.last_ticks = k
        return act, k

    # -- observation side ------------------------------------------------------------
    def view(self, env, obs, reset=False):
        """The actor's observation as the live readers would report it. Returns
        `obs` itself when nothing changes."""
        if not self.active:
            return obs
        cfg, d, r, on, L = self.cfg, self.d, self.rng, self.on, self.L
        game = env.game
        t = int(game.get_current_tick())
        S, COST, X = L["S"], L["COST"], L["X"]
        hand = [int(c) for c in game.get_hand()]
        out = [obs]

        def W():
            if out[0] is obs:
                out[0] = np.array(obs, dtype=np.float32, copy=True)
            return out[0]

        if self.est is not None:
            self.e_clean = float(obs[S]) * 10.0
            self._est_resolve(env, game, t)
        # our plays that executed in this step
        if not reset and self.hand_before is not None:
            for i, (a, b) in enumerate(zip(self.hand_before, hand)):
                if a != b and a >= 0:
                    te = int(game.get_last_played_tick(0, a))
                    if te <= 0 or te > t:
                        te = t
                    self.stats["plays"] += 1
                    if self.est is not None:
                        self.est.own_play(t / TICKS_PER_S)
                    if on["refill"] and d["refill"] > 0:
                        self.blank_until[i] = te + int(round(d["refill"] + float(r.uniform(-1.0, 1.0))))
                    if on["elixir"] and d["el_tr_p"] > 0 and r.random() < d["el_tr_p"]:
                        dur = int(r.integers(cfg.elixir_transient_ticks[0], cfg.elixir_transient_ticks[1] + 1))
                        self.el_tr = (t + dur, -1 if r.random() < cfg.elixir_minus_share else 1)
        self.hand_before = None
        if on["elixir"] and d["el_mis_p"] > 0 and self.el_tr[0] <= t and r.random() < d["el_mis_p"]:
            self.el_tr = (t + 5 * int(r.integers(1, 3)), -1 if r.random() < 0.5 else 1)

        # refill lag: the slot is empty on screen; the live mask forbids it
        for i in range(C.HAND_SIZE):
            if self.blank_until[i] > t and obs[COST + i] != 0.0:
                W()[COST + i] = 0.0
        # a tap that did not register: the driver shows the card in flight
        # (cost off the bar, slot blocked) until it gives up
        self.phantom = [p for p in self.phantom if p[0] > t]
        for _, cid, cost in self.phantom:
            o = W()
            if cid in hand:
                o[COST + hand.index(cid)] = 0.0
            o[S] = max(0.0, float(o[S]) - cost / 10.0)
        # hand misreads
        if on["hand"] and d["hand_p"] > 0:
            for i in range(C.HAND_SIZE):
                m = self.mis[i]
                if m is not None and m[0] <= t:
                    m = self.mis[i] = None
                if m is None and not reset and r.random() < d["hand_p"]:
                    dur = 5 * int(r.geometric(1.0 / max(1.0, cfg.hand_misread_steps)))
                    wrong = -1
                    if r.random() >= cfg.hand_blank_share:
                        others = [int(c) for c in env.ai_deck if int(c) not in hand]
                        if others:
                            wrong = others[int(r.integers(len(others)))]
                    m = self.mis[i] = (t + dur, wrong)
                if m is not None:
                    o = W()
                    a = L["ONEHOT"] + i * L["N"]
                    o[a:a + L["N"]] = 0.0
                    if m[1] >= 0:
                        o[a + m[1]] = 1.0
                        if o[COST + i] != 0.0:          # an unavailable slot stays unavailable
                            o[COST + i] = _cost(m[1]) / 10.0
                    else:
                        o[COST + i] = 0.0
        # elixir digit
        if on["elixir"]:
            cur = out[0]
            e = float(cur[S]) * 10.0
            e2 = e
            if d["el_floor"]:
                whole = abs(e - round(e)) < 1e-4       # a full bar (or a debit of one) is read exactly
                e2 = float(round(e)) if whole else float(math.floor(e + d["el_off"] + 1e-4))
            if self.el_tr[0] > t:
                e2 += self.el_tr[1]
            e2 = min(10.0, max(0.0, e2))
            if self.est is not None:
                e2 = self._est_value(env, e2, t, reset)
            if abs(e2 - e) > 1e-6:
                W()[S] = e2 / 10.0
        # clock in whole seconds, with this match's offset
        if d["clock_q"]:
            cur = out[0]
            tk = int(round(float(cur[X]) * TIME_DENOM))
            disp = max(0, tk + d["clock_off"])
            disp -= disp % TICKS_PER_S
            if disp != tk:
                W()[X] = disp / TIME_DENOM
                if _regime(disp) != _regime(tk):
                    W()[X + 9] = _regime(disp) / float(C.MAX_ELIXIR_MULT)
        # tower numerals held while unreadable
        if on["towers"] and d["tower_p"] > 0:
            v = np.asarray(obs[X + 3:X + 9], dtype=np.float32)
            if self.tower_shown is None:
                self.tower_shown = v.copy()
            for j in range(6):
                if self.tower_hold[j] <= t and not reset:
                    p = d["tower_p"] * (cfg.tower_own_right_mult if j == 2 else 1.0)
                    if r.random() < min(0.5, p):
                        self.tower_hold[j] = t + 5 * int(r.geometric(1.0 / max(1.0, cfg.tower_stale_steps)))
                if self.tower_hold[j] > t:
                    if self.tower_shown[j] != v[j]:
                        W()[X + 3 + j] = self.tower_shown[j]
                else:
                    self.tower_shown[j] = v[j]
        # opponent cycle block and spend from the detector's reports
        if d["opptrack"] and hasattr(game, "get_opp_reports"):
            n = L["N"]
            block = np.zeros(2 * n, dtype=np.float32)
            spend = 0.0
            for rep in game.get_opp_reports(0):
                rp = rep.report
                spend += float(rp.cost)
                cid = int(rp.card_id)
                if 0 <= cid < n:
                    block[cid] = 1.0
                    rec = math.exp(-max(0.0, t - float(rp.tick)) / CYCLE_TAU)
                    if rec > block[n + cid]:
                        block[n + cid] = rec
            a = L["CYCLE"]
            if not np.array_equal(block, obs[a:a + 2 * n]):
                W()[a:a + 2 * n] = block
            sp = min(1.0, spend / L["MAX_SPEND"])
            if abs(sp - float(obs[X + 2])) > 1e-7:
                W()[X + 2] = sp
        # the opponent's level as the driver states it (the nominal one)
        if self.lv_delta:
            a = L["LEVEL"]
            if obs[a + 2] != 0.0 or obs[a + 3] != 0.0:
                W()[a + 2:a + 4] = 0.0
        return out[0]

    # -- elixir estimate (cfg.elixir_est) ------------------------------------------
    @staticmethod
    def _in_flight(env):
        """Card ids of our plays chosen and not yet executed (the env's actuation
        delay): what RewardEnv.inflight_obs has taken off the elixir scalar."""
        dl = getattr(env, "_delay", None)
        return [int(e[1]) for e in dl.pending[0]] if dl is not None else []

    def _est_resolve(self, env, game, t):
        """Our picks since the last view: played, still on their way, or refused.
        One refused because the bar could not pay for it becomes a failed tap."""
        if not self.est_picks:
            return
        flying, keep = self._in_flight(env), []
        for p in self.est_picks:
            pt, cid, cost, e0, lpt = p
            if int(game.get_last_played_tick(0, cid)) != lpt:
                continue
            if cid in flying:
                keep.append(p)
                continue
            until = pt + self.d["fail_ticks"]
            if self.cfg.elixir_est_fail and e0 is not None and cost > e0 + 1e-6 and until > t:
                self.phantom.append([until, cid, cost])
                self.stats["unafford"] += 1
        self.est_picks = keep

    def _est_value(self, env, digit_net, t, reset):
        """The digit the layer would write (net of our cards in flight) -> the
        estimator's fraction, net of the same cards. The estimator sees the
        digit as the screen shows it: before those cards are taken off."""
        E, d, st = self.est, self.d, self.stats
        if reset:
            E.reset()
        flying = sum(_cost(c) for c in self._in_flight(env))
        owed = flying + sum(p[2] for p in self.phantom)
        # at zero with cards owed the scalar was clamped: the digit is unknown
        digit = None if (digit_net <= 0.0 and owed > 0) else min(10.0, digit_net + owed)
        tk = max(0, t + d["clock_off"]) if d["clock_q"] else t
        x = E.update(t / TICKS_PER_S, digit=digit, mult=_regime(tk))
        if x is None:
            return digit_net
        if self.e_clean > 0.0 or flying <= 0:
            true = min(10.0, self.e_clean + flying)
            err = x - true
            st["est_n"] += 1
            st["est_sum"] += err
            st["est_sq"] += err * err
            fx, ft = math.floor(x + 1e-9), math.floor(true + 1e-6)
            st["est_above"] += int(fx > ft)
            st["est_below"] += int(fx < ft)
        return min(10.0, max(0.0, x - owed))


# ------------------------------------------------------------------ env glue ---
class _Mixin:
    """reset / step around a RewardEnv subclass. Without the `sim2real` config
    key it adds nothing (and the plain classes are what everything else uses)."""

    s2r = None
    _s2r_depth = 0
    _s2r_levels_ok = False

    def __init__(self, env_config=None):
        cfg = dict(env_config or {})
        spec = cfg.pop("sim2real", None)
        super().__init__(cfg)
        c = load_config(spec)
        if c is not None:
            self.s2r = Sim2Real(c, cfg.get("scenario_seed"))
            # The level jitter owns RewardEnv's per-episode level: only when the
            # env config left it alone and ghosts play at the nominal level.
            self._s2r_levels_ok = (self._opp_level is None and not getattr(self, "_levels", False)
                                   and hasattr(self.game, "set_team_level"))

    def set_sim2real(self, over):
        return self.s2r.set(over) if self.s2r is not None else False

    def reset(self, seed=None, options=None):
        m = self.s2r
        if m is None:
            return super().reset(seed=seed, options=options)
        if self._s2r_depth:
            # GhostEnv / PoolEnv fall back to a start from tick 0 by calling
            # self.reset again with the same seed: the episode is already drawn.
            return super().reset(seed=seed, options=options)
        seed = m.next_seed(seed)
        m.begin(self, seed)
        self._s2r_depth += 1
        try:
            obs, info = super().reset(seed=seed, options=options)
        finally:
            self._s2r_depth -= 1
        if m.d.get("det") is not None and self.game is not m._det_game:
            m.d["det"] = None          # the match was rebuilt (tower troops): engine default rates
        info = dict(info)
        if m.cfg.publish_clean:
            info["s2r_clean"] = self._s2r_clean(obs)
        obs = m.view(self, obs, reset=True)
        info.update(m.info())
        return obs, info

    def _s2r_clean(self, obs):
        a, b = self.s2r.span
        return np.array(obs[a:b], dtype=np.float32, copy=True)

    def step(self, action, skip_frames=None):
        m = self.s2r
        if m is None:
            return super().step(action, skip_frames)
        action, k = m.pre_step(self, action, skip_frames)
        obs, reward, terminated, truncated, info = super().step(action, k)
        if m.cfg.publish_clean:
            info["s2r_clean"] = self._s2r_clean(obs)
        obs = m.view(self, obs)
        info.update(m.info())
        return obs, reward, terminated, truncated, info


class CleanRows:
    """Trainer side of `publish_clean`: for every env step, the true values of
    the layer's span in the observation the action was chosen in; `swapped`
    puts them into a rollout's observations for the length of a `with` (the
    frozen reference policy is evaluated inside it) and takes them out again."""

    def __init__(self):
        self.span = clean_span()
        self.cur = None            # (n, span): of the observation the trainer holds
        self.rows = []

    @staticmethod
    def _stack(infos):
        v = None if infos is None else infos.get("s2r_clean")
        if v is None:
            raise RuntimeError("the envs do not publish info['s2r_clean'] (sim2real: publish_clean)")
        return np.stack([np.asarray(x, dtype=np.float32) for x in v])

    def start(self, reset_infos):
        self.cur = self._stack(reset_infos)

    def record(self, infos):
        """Once per env step, after it (info is post-step: the next row's)."""
        self.rows.append(self.cur)
        self.cur = self._stack(infos)

    @contextlib.contextmanager
    def swapped(self, obs):
        """obs: the rollout's (T, n, obs) tensor, edited in place and restored."""
        import torch  # noqa: PLC0415
        a, b = self.span
        if len(self.rows) != int(obs.shape[0]):
            raise RuntimeError(f"clean rows {len(self.rows)} != rollout steps {int(obs.shape[0])}")
        clean = torch.as_tensor(np.stack(self.rows), dtype=obs.dtype, device=obs.device)
        self.rows = []
        keep = obs[..., a:b].clone()
        obs[..., a:b] = clean
        try:
            yield
        finally:
            obs[..., a:b] = keep


_CLASSES = {}


def ghost_env_class():
    if "ghost" not in _CLASSES:
        from training import ghost as G  # noqa: PLC0415

        class GhostEnvS2R(_Mixin, G.GhostEnv):
            pass
        _CLASSES["ghost"] = GhostEnvS2R
    return _CLASSES["ghost"]


def pool_env_class():
    if "pool" not in _CLASSES:
        from training import pool_env as PE  # noqa: PLC0415

        class PoolEnvS2R(_Mixin, PE.PoolEnv):
            pass
        _CLASSES["pool"] = PoolEnvS2R
    return _CLASSES["pool"]
