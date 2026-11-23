"""Run a gen-1 checkpoint on the gen-2 engine: observation adapter + loader.

    net, info = obs_compat.load_policy("remote_results/bc_hog26.pth")
    net.set_opp_deck(env.current_opp_deck)      # optional, see (4)
    mask = net.affordability_mask(obs_gen2)     # the MicroRoyaleNet inference API

The gen-1 checkpoints (bc_hog26, ppo_league26_*) were trained on engine/'s
13,977-value observation. build/gen2/engine emits 14,459 values, and
`training.common.load_weights` / `net_v2.load_any` there build a gen-2 shaped
net and load "flexibly", which DROPS every layer whose shape changed (hand
one-hot embedding, cycle embedding, the extra-scalar layer, ...). This module
instead builds the network with the gen-1 layout, loads the checkpoint
strictly, and translates each gen-2 observation into the gen-1 layout.

`adapt(obs2)` -> gen-1 layout, exactly:

 (1) spatial planes (21 x 34 x 18): copied, except the two building channels
     (3 ours, 7 theirs):
       * Crown Tower cells: gen 2 writes hp / that tower's own max (1.0 = full);
         gen 1 wrote hp / 4008, so a full King read 1.0 and a full Princess
         2534/4008 = 0.632. Adapted value = fraction * (gen-1 max / 4008): the
         old net sees the same FRACTION of tower health it was trained on, not
         the same hit points (gen-2 level-11 towers have 4824 / 3052 hp).
       * deployed buildings: gen 2 writes hp / 4824, gen 1 hp / 4008; adapted =
         value * 4824 / 4008, capped at 1 (same hit points -> same value).
     and the Crown Tower cells of the two damage-per-tick planes (15 ours, 16
     theirs): a default-level gen-2 tower reads 0.1817 (King) / 0.2271
     (Princess), gen 1 read 0.15 / 0.1875. A cell holding exactly the gen-2
     tower value is replaced by the gen-1 value; anything else (a tower at
     another level, a stronger flyer over the tower) is left as written.
 (2) own elixir, 4 hand costs: copied. 4 hand one-hots: the first 185 of 195
     columns (ids 0-175 mean the same card in both registries; ids 185-194 do
     not exist in gen 1 and cannot be in our hand).
 (3) 10 extra scalars: time, both elixir-spent, elixir phase copied; the six
     tower scalars rescaled as in (1) (King x 1.0, Princess x 0.632).
     Time is tick / env max_ticks in both (the gym wrapper's 3600).
 (4) opponent cycle blocks seen[] / recency[] (2 x 195 -> 2 x 185): gen 2 folds
     an Evolution or Hero play onto its face card id; gen 1 lit the played deck
     id itself. With `set_opp_deck(deck)` the adapter moves each Evolution /
     Hero deck card's entry back to its own id (when that id exists in gen 1,
     i.e. < 185), which reproduces gen 1; without it the fold is left in place
     (the old net then sees e.g. "Skeletons" where it used to see "evo:Skeletons").
 (5) dropped: the 4 level scalars and the 418 opponent-state values
     (opponent elixir estimate, own Evolution charge, ability state, the
     opponent's Evolution blocks). The gen-1 net has no input for them.

What the old net still sees differently from its training engine (mechanics,
not layout; nothing here can change it): towers have about 20% more hit
points per unit of the fraction it reads; triple elixir starts at tick 2400
(gen 1: 1800); the match ends at tick 3000 at the latest (gen 1: 3600) while
the time scalar still runs on the 3600 scale; cards 176-194 (Minion Giant and
the newer Evolutions/Heroes) are invisible in the opponent cycle block (their
units are still on the board planes); elixir regenerates at 1 per 2.8 s
(gen 1: 0.035 per tick, 1 per 2.857 s); every card stat the gen-2 patches
changed. Measured check (training/tests/test_obs_compat.py): for a mirror match
with the same seed, adapt(gen-2 obs) equals the gen-1 engine's observation
value for value at reset and one tick after both sides deploy a card; 40 ticks
later the only difference is the elixir value (the regeneration rate).
"""
import contextlib

from training import common as C

import numpy as np
import torch

E, CE = C.E, C.CE

GEN1 = {"obs": 13977, "num_card_ids": 185, "cycle_block": 370,
        "max_building_hp": 4008.0, "tower_hp": (4008.0, 2534.0, 2534.0),
        # value of a Crown Tower's cell in the damage-per-tick planes (King, Princess),
        # read off the gen-1 engine's fresh board
        "tower_dps_cell": (0.15, 0.1875)}
N_CH = CE.NUM_CHANNELS
SPATIAL = C.SPATIAL_SIZE
CELLS = C.BOARD_H * C.BOARD_W
HAND = C.HAND_SIZE
NUM_EXTRA = CE.NUM_EXTRA_SCALARS
BUILDING_CH = (3, 7)                     # own, enemy
DPS_CH = int(getattr(CE, "CH_DPS", 15))  # own; enemy = +1
IS_GEN2 = hasattr(CE, "LEVEL_START")


def checkpoint_layout(sd):
    """(num_card_ids, has_opp_state, is_v2) read off a state dict."""
    n = int(sd["scalar_mlp.hand_slot.weight"].shape[1])
    return n, any(k.startswith("scalar_mlp.opp_state") for k in sd), \
        any(k.startswith("board.") for k in sd)


def state_dict_of(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return (ck["model"] if isinstance(ck, dict) and "model" in ck else ck), ck


class _Gen1EnvClass:
    """clash_royale_env.ClashRoyaleEnv as python_ai.models.net sees it while a
    gen-1 shaped net is constructed: gen-1 layout constants, no level / opponent
    state sections; everything else is the live class."""

    NUM_CARD_IDS = GEN1["num_card_ids"]
    CYCLE_BLOCK_SIZE = GEN1["cycle_block"]
    _HIDDEN = ("NUM_LEVEL_SCALARS", "LEVEL_START", "OPP_STATE_SIZE", "OPP_STATE_START",
               "OPP_EVO_OFFSET")

    def __getattr__(self, name):
        if name in self._HIDDEN:
            raise AttributeError(name)
        return getattr(CE, name)

    def __call__(self, *a, **k):
        return CE(*a, **k)


class _Gen1Module:
    ClashRoyaleEnv = _Gen1EnvClass()

    def __getattr__(self, name):
        return getattr(E, name)


@contextlib.contextmanager
def _gen1_layout():
    from python_ai.models import net as netmod
    saved = netmod.clash_royale_env, netmod.NUM_CARD_IDS_LIVE
    netmod.clash_royale_env, netmod.NUM_CARD_IDS_LIVE = _Gen1Module(), GEN1["num_card_ids"]
    try:
        yield netmod
    finally:
        netmod.clash_royale_env, netmod.NUM_CARD_IDS_LIVE = saved


def new_gen1_net(deck=None):
    """A MicroRoyaleNet with the gen-1 observation layout, on the live engine."""
    from python_ai.rl.abilities import ability_engine_slots
    with _gen1_layout() as netmod:
        net = netmod.MicroRoyaleNet(num_card_ids=GEN1["num_card_ids"],
                                    num_ability_slots=len(ability_engine_slots(deck or C.deck_ids())))
    assert net.spatial_size + net.scalar_size == GEN1["obs"], (net.spatial_size, net.scalar_size)
    return net


# ------------------------------------------------------------------ adapter ---
def _fresh_obs():
    g = CE(list(C.deck_ids()), list(C.deck_ids()), 100)
    g.seed(0)
    return g.get_observation_for_team(0)


def _tower_cells():
    """[(side, which, flat cell)] of the six Crown Towers in a team-0 observation,
    side 0 = ours; which 0 King, 1 left, 2 right. Read off a fresh board."""
    obs = np.asarray(_fresh_obs(), np.float32)[:SPATIAL].reshape(N_CH, C.BOARD_H, C.BOARD_W)
    out = []
    for side, ch in enumerate(BUILDING_CH):
        ys, xs = np.nonzero(obs[ch])
        assert len(ys) == 3, f"expected 3 tower cells in channel {ch}, found {len(ys)}"
        cells = sorted(zip(xs.tolist(), ys.tolist()))            # by column: left, King, right
        (xl, yl), (xk, yk), (xr, yr) = cells
        assert xl < xk < xr and yl == yr
        for which, (x, y) in ((0, (xk, yk)), (1, (xl, yl)), (2, (xr, yr))):
            out.append((side, which, y * C.BOARD_W + x))
    return out


from training.obs_v2 import TAIL_SIZE as _TAIL  # noqa: E402  (numpy only)


class Adapter:
    """gen-2 observation(s) -> gen-1 layout. Works on a (B, 14459) or (14459,)
    torch tensor or numpy array; returns the same kind."""

    def __init__(self, opp_deck=None, time_ratio=1.0):
        if not IS_GEN2:
            raise RuntimeError("obs_compat.Adapter needs the gen-2 engine (CR_ENGINE)")
        n2, n1 = CE.NUM_CARD_IDS, GEN1["num_card_ids"]
        self.n_in = int(CE(list(C.deck_ids()), list(C.deck_ids()), 100).observation_size())
        r_build = float(C.MAX_BUILDING_HP) / GEN1["max_building_hp"]
        idx, scale, cap = [], [], []

        def put(src, s=1.0, c=np.inf):
            idx.append(src)
            scale.append(s)
            cap.append(c)

        for i in range(SPATIAL):                                   # (1)
            ch = i // CELLS
            if ch in BUILDING_CH:
                put(i, r_build, 1.0)
            else:
                put(i)
        s2 = SPATIAL
        for i in range(1 + HAND):                                  # (2) elixir, costs
            put(s2 + i)
        oh2 = s2 + 1 + HAND
        for slot in range(HAND):
            for c in range(n1):
                put(oh2 + slot * n2 + c)
        ex2 = oh2 + HAND * n2
        assert ex2 == CE.EXTRA_SCALARS_START
        tower = [h / GEN1["max_building_hp"] for h in GEN1["tower_hp"]]
        for i in range(NUM_EXTRA):                                 # (3)
            if i == 0:
                put(ex2 + i, float(time_ratio))
            elif 3 <= i <= 8:
                put(ex2 + i, tower[(i - 3) % 3])
            else:
                put(ex2 + i)
        cy2 = CE.CYCLE_START
        assert cy2 == ex2 + NUM_EXTRA
        src = list(range(n1))                                      # (4)
        zero = set()
        face = getattr(E, "face_card_id_of", None)
        for d in (opp_deck or ()):
            d = int(d)
            f = int(face(d)) if face is not None else d
            if f != d and d < n1 and f < n1:
                src[d] = f
                zero.add(f)
        for block in range(2):
            for c in range(n1):
                put(cy2 + block * n2 + src[c], 0.0 if c in zero else 1.0)
        assert len(idx) == GEN1["obs"], len(idx)
        self.idx = torch.tensor(idx, dtype=torch.long)
        self.scale = torch.tensor(scale, dtype=torch.float32)
        self.cap = torch.tensor(cap, dtype=torch.float32)
        # Tower cells: fraction * (gen-1 max / 4008) while that tower stands.
        self.towers = [(ch * CELLS + cell, ex2 + 3 + 3 * side + which, tower[which])
                       for side, which, cell in _tower_cells()
                       for ch in (BUILDING_CH[side],)]
        # Damage-per-tick planes: a standing tower's own value -> the gen-1 value.
        fresh = np.asarray(_fresh_obs(), np.float32)
        self.tower_dps = [((DPS_CH + side) * CELLS + cell,
                           float(fresh[(DPS_CH + side) * CELLS + cell]),
                           GEN1["tower_dps_cell"][min(which, 1)])
                          for side, which, cell in _tower_cells()]

    def __call__(self, obs):
        is_np = isinstance(obs, np.ndarray)
        o = torch.as_tensor(obs, dtype=torch.float32)
        single = o.dim() == 1
        if single:
            o = o.unsqueeze(0)
        if o.shape[1] == self.n_in + _TAIL:
            o = o[:, :self.n_in]            # the observation v2 tail: a gen-1 network never saw it
        if o.shape[1] != self.n_in:
            raise ValueError(f"expected a gen-2 observation of {self.n_in} values, got {o.shape[1]}")
        out = torch.minimum(o[:, self.idx] * self.scale, self.cap)
        for cell, scalar, k in self.towers:
            alive = o[:, scalar] > 0
            out[:, cell] = torch.where(alive, o[:, cell] * k, out[:, cell])
        for cell, v2, v1 in self.tower_dps:
            mine = (o[:, cell] - v2).abs() < 1e-5
            out[:, cell] = torch.where(mine, torch.full_like(out[:, cell], v1), out[:, cell])
        if single:
            out = out[0]
        return out.numpy() if is_np else out


class Gen1Policy:
    """A gen-1 net behind the MicroRoyaleNet inference API, fed gen-2
    observations. Only the methods that read the observation are wrapped;
    everything else is the net's."""

    def __init__(self, net, opp_deck=None):
        self.net = net
        self._adapters = {}
        self._last = (None, None)
        self.set_opp_deck(opp_deck)

    def set_opp_deck(self, deck):
        key = tuple(int(c) for c in deck) if deck else ()
        if key not in self._adapters:
            self._adapters[key] = Adapter(key or None)
        self.adapter = self._adapters[key]
        self._last = (None, None)

    def adapt(self, obs):
        if obs is None:
            return None
        if self._last[0] is obs:
            return self._last[1]
        out = self.adapter(obs)
        self._last = (obs, out)
        return out

    def affordability_mask(self, obs):
        return self.net.affordability_mask(self.adapt(obs))

    def extract_features_hires(self, obs):
        return self.net.extract_features_hires(self.adapt(obs))

    def extract_features(self, obs):
        return self.net.extract_features(self.adapt(obs))

    def placement_given_card(self, hx, card_embeds, card_idx, obs=None, spatial_map=None, **kw):
        return self.net.placement_given_card(hx, card_embeds, card_idx, self.adapt(obs), spatial_map, **kw)

    def placement_mask(self, obs, *a, **k):
        return self.net.placement_mask(self.adapt(obs), *a, **k)

    def eval(self):
        self.net.eval()
        return self

    def __getattr__(self, name):
        return getattr(self.net, name)


def load_policy_v3(path, ck=None, when_internal=True):
    """A NetV3 checkpoint (training/net_v3.py) -> (net, info), built from the
    checkpoint's own metadata. Raises on ANY mismatch with this engine and this
    code: observation length, obs_v2, card ids, the hash of net_v3.py, a missing
    metadata field, a state dict that does not fit.

    info carries what a caller must honour: `decision_ticks` (engine ticks per
    decision the net was trained for), `lead`, `names` (tail names policy of the
    training noise), `obs_v2` = True, the dataset and its fingerprint.

    `when_internal=True` (default): the net draws its WHEN coin itself, so a
    caller that tempers the 5-way card logits only sharpens WHAT / WHERE. A
    caller that uses `net.set_sampling(...)` (eval_ghost) switches to the
    composed, normalised logits instead."""
    from training import net_v3  # noqa: PLC0415
    if ck is None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    if not IS_GEN2:
        raise ValueError(f"{path}: a v3 checkpoint needs the gen-2 engine (CR_ENGINE)")
    try:
        net, meta = net_v3.load_v3(ck)
    except (ValueError, RuntimeError, KeyError) as ex:
        raise ValueError(f"{path}: {ex}") from ex
    from python_ai.rl import abilities as A  # noqa: PLC0415
    net.set_ability_slots(A.ability_engine_slots(C.deck_ids()))
    net.set_sampling(when_internal=when_internal)
    info = {"path": path, "arch": "v3", "checkpoint_card_ids": int(meta["num_card_ids"]),
            "engine_card_ids": int(CE.NUM_CARD_IDS), "adapter": False, "obs_v2": True,
            "obs_dim": int(meta["obs_dim"]), "decision_ticks": int(meta["decision_ticks"]),
            "lead": int(meta["lead"]), "names": meta["names"], "dataset": meta["dataset"],
            "dataset_fp": meta["dataset_fp"], "code_hash": meta["code_hash"], "cfg": meta["cfg"],
            "train": meta.get("train")}
    return net, info


def load_policy(path, opp_deck=None):
    """Any checkpoint -> (policy for gen-2 observations, info).

    A checkpoint whose layout matches the live engine goes through
    net_v2.load_any unchanged. A gen-1 v1 checkpoint on the gen-2 engine is
    loaded STRICTLY into a gen-1 shaped net and wrapped in Gen1Policy."""
    sd, ck = state_dict_of(path)
    if isinstance(ck, dict) and ck.get("arch") == "v3":
        return load_policy_v3(path, ck)
    n_ids, has_opp, is_v2 = checkpoint_layout(sd)
    info = {"path": path, "checkpoint_card_ids": n_ids, "engine_card_ids": int(CE.NUM_CARD_IDS),
            "adapter": False}
    if n_ids == int(CE.NUM_CARD_IDS):
        from training.net_v2 import load_any
        return load_any(path), info
    if not (IS_GEN2 and n_ids == GEN1["num_card_ids"] and not has_opp and not is_v2):
        raise ValueError(f"{path}: layout with {n_ids} card ids matches neither the live engine "
                         f"({CE.NUM_CARD_IDS}) nor a gen-1 v1 checkpoint")
    deck = ck.get("deck") if isinstance(ck, dict) else None
    net = new_gen1_net(deck)
    params = {k for k, _ in net.named_parameters()}
    # Buffers describe the ENGINE (which ids are spells), not what was learned:
    # keep the live engine's. Measured: the only saved buffer is spell_flags and
    # it differs from gen 1 at id 180 alone.
    live = {k: v.clone() for k, v in net.state_dict().items() if k not in params}
    res = net.load_state_dict(sd, strict=False)
    net.load_state_dict(live, strict=False)
    # A missing or unexpected PARAMETER is an error.
    missing = [k for k in res.missing_keys if k in params]
    if missing or res.unexpected_keys:
        raise ValueError(f"{path}: not a gen-1 MicroRoyaleNet: missing parameters {missing}, "
                         f"unexpected keys {list(res.unexpected_keys)}")
    info.update(adapter=True, missing_buffers=list(res.missing_keys), tensors=len(sd))
    return Gen1Policy(net.eval(), opp_deck), info
