"""NetV3: the gen-2 policy network (factorised WHEN / WHAT / WHERE, reads the tail).

Why (measured on gen 1): the clone kept the humans' spell RATES but not their
triggers, and at T = 0.5 it almost stopped playing, because one 5-way softmax
held "wait" (5 : 1 in the data, jittered in time) next to the four cards, so
sharpening the card choice also sharpened "wait". Here the three decisions are
separate heads:

  WHEN   one logit w: the probability that a play should start in this
         decision interval. Sampled as it is (its own knobs: `when_temp`,
         `when_bias`); the card temperature never touches it.
  WHAT   a score per hand slot from one scorer shared by the four slots (it
         reads the state and that slot's card embedding), softmax over the
         playable slots. This is "which card, given that a play starts".
  WHERE  the engine net's placement decoder (coarse map + full-resolution
         residual), conditioned on the chosen slot's card embedding.

The 5-way card distribution every caller expects is composed from them:
    log p(wait)   = logsigmoid(-w)
    log p(slot s) = logsigmoid(w) + log_softmax_playable(what)[s]
(no playable card: wait has log-probability 0 and the slots -inf). The values
returned as "card logits" are these normalised log-probabilities, so
Categorical(logits=...) and the PPO ratio work unchanged.

Inputs. The full gen-2 observation: the engine's 14,459 floats and the 148-float
tail of training/obs_v2.py. The tail's card codes go through one shared
`nn.Embedding(256, d, padding_idx=0)`; play entries and spell entries are
encoded per entry and pooled per side (TailEncoder). A row without a tail (or
`use_tail=False`, the ablation) gets zero tail features. The engine's opponent
cycle branch feeds the actor WITH gradient here (the engine net detaches it).

Feature vector (what `extract_features*` returns first):
    [ cnn 1440 | scalar branches without cycle 56 | hand card embeddings 4 x 16
      | cycle 24 | tail T ]
The last `skip_dim` = 64 + 24 + T values are also a skip into every head.

No recurrence. The cloning data (training/bcdata2.py) is served as shuffled
rows, so the net is trained with no carried state; what a recurrent state would
have held (who played what, when, where) is in the tail, the cycle block and
the opponent-state block. `core` is a feed-forward stack; the (hx, cx) pair is
kept for the interface: hx = the core's output for THIS observation (the
placement head needs it), cx = zeros, and an incoming hidden state is ignored.

Deck-agnostic. Nothing is sized by CLASH_DECK. The two ability heads are
"ability of engine slot 1" and "ability of engine slot 2" (the only two deck
positions that can hold a Champion / Hero); `set_ability_slots` says which of
them the current deck uses, and `step_lstm_and_card` returns logits for those,
in slot order, as callers expect (`[x for x in (a1, a2) if x is not None]`).

Sampling knobs (`set_sampling`): what_temp, where_temp divide the WHAT and
WHERE logits inside the net; when_temp / when_bias act on w alone. A caller that
uses them samples the returned logits at temperature 1. For callers that still
temper the composed 5-way logits themselves (eval_panel, league_env, the live
driver), `when_internal=True` makes the net draw the WHEN coin itself and return
logits that are already decided between wait and play, so the caller's
temperature can only reach WHAT and WHERE.
"""
from __future__ import annotations

import ast
import hashlib
import os

from training import common as C  # noqa: F401  (engine path + CLASH_DECK first)

import torch
import torch.nn as nn
import torch.nn.functional as F

from python_ai.models.net import (MicroRoyaleNet, CARD_EMBED_DIM, CYCLE_BRANCH_DIM,
                                  _conv_batch_invariant)
from python_ai.rl import abilities as A
from training import obs_v2 as OV

ARCH = "v3"
DEFAULTS = dict(
    tail_dim=48,        # T: tail features
    tail_embed=16,      # d of the shared card embedding
    tail_hidden=32,     # per-entry encoder width
    core_hidden=512,    # first core layer
    core_layers=1,      # hidden layers of `core_hidden` before the 256 output
    what_hidden=64,
    use_tail=True,      # False: the tail is not read (ablation); same shapes
)
#: dt_next buckets (ticks): [0,5) [5,10) [10,20) [20,40) [40,80) [80,inf) and "no next play"
DT_EDGES = (5, 10, 20, 40, 80)
N_DT = len(DT_EDGES) + 2


def engine_obs_dim():
    """Length of the engine observation (without the tail), from the engine."""
    global _ENGINE_OBS_DIM
    if _ENGINE_OBS_DIM is None:
        deck = list(C.deck_ids())
        g = C.CE(deck, deck, 3600)
        _ENGINE_OBS_DIM = len(g.get_observation_for_team(0))
    return _ENGINE_OBS_DIM


_ENGINE_OBS_DIM = None


def code_hash(path=None):
    """Hash of this file's syntax tree (comments and layout do not count).
    Stored in checkpoints; obs_compat.load_policy refuses a checkpoint whose
    network code is not this one."""
    with open(path or __file__, "r") as f:
        tree = ast.parse(f.read())
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()[:16]


def dt_bucket(dt):
    """dt_next in ticks (long tensor, bcdata2.NO_NEXT = none) -> bucket index."""
    edges = torch.tensor(DT_EDGES, device=dt.device)
    b = torch.bucketize(dt.clamp(max=60000), edges, right=True)
    return torch.where(dt >= 65535, torch.full_like(b, N_DT - 1), b)


class TailEncoder(nn.Module):
    """The 148-float tail -> T features.

    Each play entry (6 opponent + 6 own) becomes [its 10 floats | card
    embedding | side flag] -> MLP; per side the valid entries are pooled by sum
    and max, and the newest entry is kept as it is (the play a response is
    owed to). Spell entries likewise (sum per side). The next card's embedding
    and the tail scalars join, and one layer mixes them down to T.
    """

    def __init__(self, embed=16, hidden=32, out=48):
        super().__init__()
        self.embed = nn.Embedding(OV.NUM_CARD_INDEX, embed, padding_idx=0)
        self.play = nn.Sequential(nn.Linear(OV.ENTRY + embed + 1, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.spell = nn.Sequential(nn.Linear(OV.SPELL_ENTRY + embed + 1, hidden // 2), nn.ReLU())
        mixed = 2 * 3 * hidden + 2 * (hidden // 2) + embed + 2 + 4 + 1
        self.out = nn.Sequential(nn.Linear(mixed, out), nn.ReLU())
        self.out_dim = out

    def _index(self, code):
        return (code * OV.CODE_DIV).round().long().clamp_(0, OV.NUM_CARD_INDEX - 1)

    def forward(self, tail):
        B = tail.shape[0]
        parts = []
        for sl, k, side in ((OV.OPP_PLAYS, OV.K_OPP, 0.0), (OV.OWN_PLAYS, OV.K_OWN, 1.0)):
            e = tail[:, sl].reshape(B, k, OV.ENTRY)
            valid = (e[..., OV.E_VALID] > 0.5).unsqueeze(-1).to(e.dtype)
            x = torch.cat((e, self.embed(self._index(e[..., OV.E_CODE])),
                           torch.full((B, k, 1), side, dtype=e.dtype, device=e.device)), dim=-1)
            h = self.play(x) * valid
            parts += [h.sum(1), h.max(1).values, h[:, 0]]
        for sl, side in ((OV.OPP_SPELLS, 0.0), (OV.OWN_SPELLS, 1.0)):
            e = tail[:, sl].reshape(B, OV.N_SPELL, OV.SPELL_ENTRY)
            valid = (e[..., OV.S_VALID] > 0.5).unsqueeze(-1).to(e.dtype)
            x = torch.cat((e, self.embed(self._index(e[..., OV.S_CODE])),
                           torch.full((B, OV.N_SPELL, 1), side, dtype=e.dtype, device=e.device)), dim=-1)
            parts.append((self.spell(x) * valid).sum(1))
        nc = tail[:, OV.NEXT_CARD]
        parts += [self.embed(self._index(nc[:, 1])) * nc[:, 0:1], nc[:, (0, 2)],
                  tail[:, OV.SCALARS], tail[:, OV.PRESENT:OV.PRESENT + 1]]
        return self.out(torch.cat(parts, dim=1))


class NetV3(MicroRoyaleNet):
    ARCH = ARCH

    def __init__(self, num_ability_slots=0, ability_slots=None, **cfg):  # noqa: ARG002
        # num_ability_slots is accepted (the PPO trainer passes it) and ignored:
        # both generic ability heads always exist.
        super().__init__(num_ability_slots=0)
        unknown = set(cfg) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"NetV3: unknown config keys {sorted(unknown)}")
        self.cfg = dict(DEFAULTS, **cfg)
        c = self.cfg
        H = self.LSTM_HIDDEN
        for name in ("lstm", "card_head", "value_head", "place_ctx", "place_ctx_hi"):
            delattr(self, name)
        self.base_obs_dim = engine_obs_dim()
        self.obs_dim = self.base_obs_dim + OV.TAIL_SIZE
        self.use_tail = bool(c["use_tail"])
        self.tail_enc = TailEncoder(c["tail_embed"], c["tail_hidden"], c["tail_dim"])
        self.tail_dim = int(c["tail_dim"])
        self.hand_embed_dim = self.hand_size * CARD_EMBED_DIM
        #: the skip into every head: hand card embeddings, cycle branch, tail
        self.skip_dim = self.hand_embed_dim + CYCLE_BRANCH_DIM + self.tail_dim
        self.scalar_nocycle_dim = self.scalar_mlp.out_dim - CYCLE_BRANCH_DIM
        self.lstm_input_dim = self.cnn_out_dim + self.scalar_nocycle_dim + self.skip_dim
        # The engine net calls this width "cycle_feature_dim" wherever it
        # splits the skip off the features; here the skip is wider.
        self.cycle_feature_dim = self.skip_dim
        self.head_dim = H + self.skip_dim

        layers, d = [], self.lstm_input_dim
        for _ in range(int(c["core_layers"])):
            layers += [nn.Linear(d, int(c["core_hidden"])), nn.ReLU()]
            d = int(c["core_hidden"])
        layers += [nn.Linear(d, H), nn.ReLU()]
        self.core = nn.Sequential(*layers)

        self.when_head = nn.Linear(self.head_dim, 1)
        nn.init.constant_(self.when_head.bias, -2.0)     # plays are rare
        wh = int(c["what_hidden"])
        self.what_head = nn.Sequential(nn.Linear(self.head_dim + CARD_EMBED_DIM, wh), nn.ReLU(),
                                       nn.Linear(wh, 1))
        self.value_head = nn.Linear(self.head_dim, 1)
        self.place_ctx = nn.Linear(self.head_dim + CARD_EMBED_DIM, 32)
        self.place_ctx_hi = nn.Linear(self.head_dim + CARD_EMBED_DIM, self.place_hires[0].in_channels - 16)
        # ability of engine slot 1 / engine slot 2 (deck positions 1 and 2)
        self.ability_heads = nn.ModuleList([nn.Linear(H, 2), nn.Linear(H, 2)])
        # auxiliary: time to the next own play (bucketed) and which slot it is
        self.dt_head = nn.Linear(self.head_dim, N_DT)
        self.next_head = nn.Sequential(nn.Linear(self.head_dim + CARD_EMBED_DIM, wh), nn.ReLU(),
                                       nn.Linear(wh, 1))

        self.set_ability_slots(A.ability_engine_slots(C.deck_ids()) if ability_slots is None
                               else ability_slots)
        self.set_sampling()
        self.when_internal = False

    # ------------------------------------------------------------- knobs ---
    def set_ability_slots(self, slots):
        """Engine slots (1 and / or 2) of the current deck's Champions / Heroes.
        Routing only: the weights do not depend on it."""
        slots = [int(s) for s in slots]
        if any(s not in (1, 2) for s in slots) or sorted(set(slots)) != slots:
            raise ValueError(f"ability slots must be a sorted subset of (1, 2), got {slots}")
        self.ability_slots = slots
        self.num_ability_slots = len(slots)
        return self

    def set_sampling(self, what_temp=1.0, where_temp=1.0, when_temp=1.0, when_bias=0.0,
                     when_internal=False):
        """Temperatures applied inside the net (see the module docstring); a
        caller that sets them samples the returned logits at temperature 1."""
        for k, v in (("what_temp", what_temp), ("where_temp", where_temp), ("when_temp", when_temp)):
            if not v > 0:
                raise ValueError(f"{k} must be > 0 (got {v}); use a small value for near-argmax")
        self.what_temp, self.where_temp = float(what_temp), float(where_temp)
        self.when_temp, self.when_bias = float(when_temp), float(when_bias)
        self.when_internal = bool(when_internal)
        return self

    # the engine net's two Champion heads, as the heads the current deck uses
    @property
    def ability_slot1_head(self):
        return self.ability_heads[self.ability_slots[0] - 1] if len(self.ability_slots) >= 1 else None

    @ability_slot1_head.setter
    def ability_slot1_head(self, value):   # the parent assigns None in __init__
        if value is not None:
            raise AttributeError("NetV3 owns its ability heads")

    @property
    def ability_slot2_head(self):
        return self.ability_heads[self.ability_slots[1] - 1] if len(self.ability_slots) >= 2 else None

    @ability_slot2_head.setter
    def ability_slot2_head(self, value):
        if value is not None:
            raise AttributeError("NetV3 owns its ability heads")

    # ---------------------------------------------------------- features ---
    def tail_features(self, obs):
        """(B, T) tail features; zeros without a tail or with use_tail=False."""
        if not self.use_tail or obs.shape[1] < self.obs_dim:
            return obs.new_zeros(obs.shape[0], self.tail_dim)
        return self.tail_enc(obs[:, self.base_obs_dim:self.obs_dim])

    def _hand_embeds(self, scalar_obs):
        onehot_start = 1 + self.hand_size
        onehots = scalar_obs[:, onehot_start:onehot_start + self.hand_size * self.num_card_ids]
        emb = self.card_id_embed(onehots.reshape(-1, self.hand_size, self.num_card_ids))
        noop = self.noop_embed.view(1, 1, -1).expand(emb.shape[0], 1, -1)
        return torch.cat([emb, noop], dim=1)                       # (B, hand+1, 16)

    def _skip(self, obs, card_embeds=None):
        scalar_obs = obs[:, self.spatial_size:self.base_obs_dim]
        if card_embeds is None:
            card_embeds = self._hand_embeds(scalar_obs)
        cycle = self.scalar_mlp.cycle_branch(
            scalar_obs[:, self.cycle_start:self.cycle_start + self.cycle_block_size])
        return torch.cat((card_embeds[:, :self.hand_size].flatten(1), cycle,
                          self.tail_features(obs)), dim=1)

    def extract_features_hires(self, obs):
        # contiguous copies: a column slice of the observation is a strided
        # view, and MPS convolution backward on one is about five times slower
        spatial_obs = obs[:, :self.spatial_size].contiguous().view(
            -1, self.channels, self.board_height, self.board_width)
        scalar_obs = obs[:, self.spatial_size:self.base_obs_dim].contiguous()
        hires_map = self.cnn_trunk[:2](spatial_obs)
        spatial_map = self.cnn_trunk[2:](hires_map)
        cnn = self.cnn_flatten(spatial_map)
        # the engine encoder puts its (detached) cycle branch last: dropped
        # here, the skip carries the branch with its gradient
        scalar = self.scalar_mlp(scalar_obs)[:, :self.scalar_nocycle_dim]
        card_embeds = self._hand_embeds(scalar_obs)
        combined = torch.cat((cnn, scalar, self._skip(obs, card_embeds)), dim=1)
        return combined, card_embeds, spatial_map, hires_map

    def _split_cycle(self, features):
        """The skip (hand embeddings, cycle, tail) at the end of `features`, with
        its gradient: in this net the cycle branch feeds the actor."""
        return features[..., -self.skip_dim:]

    def cycle_features(self, obs, detached=False):
        """The head skip rebuilt from a raw observation (the engine net's name)."""
        out = self._skip(obs)
        return out.detach() if detached else out

    def predict_cycle_card(self, cycle_feat):
        a = self.hand_embed_dim
        return self.cycle_id_head(cycle_feat[..., a:a + CYCLE_BRANCH_DIM])

    # -------------------------------------------------------------- heads ---
    def _slot_scores(self, scorer, head_in):
        a = self.hand_embed_dim
        emb = head_in[..., -self.skip_dim:-self.skip_dim + a].reshape(
            head_in.shape[0], self.hand_size, CARD_EMBED_DIM)
        x = torch.cat((head_in.unsqueeze(1).expand(-1, self.hand_size, -1), emb), dim=-1)
        return scorer(x).squeeze(-1)                                 # (B, hand)

    def heads(self, features, card_mask=None):
        """Every head that does not depend on the chosen card, raw (no sampling
        knobs). Dict: hx, head_in, when (B,), what (B, hand; -inf where not
        playable), has_legal (B,), value (B, 1), ability [(B, 2), (B, 2)] by
        engine slot 1 / 2, dt (B, N_DT), next (B, hand)."""
        hx = self.core(features)
        head_in = torch.cat((hx, self._split_cycle(features)), dim=-1)
        what = self._slot_scores(self.what_head, head_in)
        if card_mask is None:
            has_legal = torch.ones(hx.shape[0], dtype=torch.bool, device=hx.device)
        else:
            legal = card_mask[:, :self.hand_size]
            has_legal = legal.any(dim=1)
            # a row with nothing playable keeps finite scores (no nan in
            # log_softmax); compose_card_logits sends its slots to -inf
            what = what.masked_fill(~(legal | ~has_legal.unsqueeze(1)), float("-inf"))
        return {"hx": hx, "head_in": head_in, "when": self.when_head(head_in).squeeze(-1),
                "what": what, "has_legal": has_legal, "value": self.value_head(head_in),
                "ability": [h(hx) for h in self.ability_heads],
                "dt": self.dt_head(head_in), "next": self._slot_scores(self.next_head, head_in)}

    @staticmethod
    def compose_card_logits(when, what, has_legal):
        """Normalised log-probabilities over (slot 0..hand-1, wait)."""
        log_what = F.log_softmax(what, dim=-1)
        slots = F.logsigmoid(when).unsqueeze(-1) + log_what
        wait = F.logsigmoid(-when)
        neg = torch.full_like(slots, float("-inf"))
        slots = torch.where(has_legal.unsqueeze(-1), slots, neg)
        wait = torch.where(has_legal, wait, torch.zeros_like(wait))
        return torch.cat((slots, wait.unsqueeze(-1)), dim=-1)

    def _active_ability(self, logits):
        out = [logits[s - 1] for s in self.ability_slots]
        return (out + [None, None])[:2]

    def step_lstm_and_card(self, features, hidden_state, card_mask=None):
        """(card_logits (B, hand+1), ability logits of the deck's first and second
        Champion slot or None, value (B, 1), (hx, cx)). The incoming hidden
        state is ignored (no recurrence); the sampling knobs apply."""
        o = self.heads(features, card_mask)
        when = (o["when"] + self.when_bias) / self.when_temp
        what = o["what"] / self.what_temp if self.what_temp != 1.0 else o["what"]
        if self.when_internal:
            play = (torch.rand_like(when) < torch.sigmoid(when)) & o["has_legal"]
            neg = torch.full_like(when, float("-inf"))
            zero = torch.zeros_like(when)
            slots = torch.where(play.unsqueeze(-1), F.log_softmax(what, dim=-1),
                                neg.unsqueeze(-1).expand_as(what))
            card_logits = torch.cat((slots, torch.where(play, neg, zero).unsqueeze(-1)), dim=-1)
        else:
            card_logits = self.compose_card_logits(when, what, o["has_legal"])
        a1, a2 = self._active_ability(o["ability"])
        cx = torch.zeros_like(o["hx"])
        return card_logits, a1, a2, o["value"], (o["hx"], cx)

    def _placement_raw(self, hx, card_embeds, card_idx, obs=None, spatial_map=None,
                       hires_map=None, ctx=None, ctx_hi=None, cycle_feat=None):
        return MicroRoyaleNet.placement_given_card(
            self, hx, card_embeds, card_idx, obs, spatial_map, hires_map=hires_map,
            ctx=ctx, ctx_hi=ctx_hi, cycle_feat=cycle_feat)

    def placement_given_card(self, hx, card_embeds, card_idx, obs=None, spatial_map=None,
                             hires_map=None, ctx=None, ctx_hi=None, cycle_feat=None):
        """WHERE: placement logits (B, cells) for the chosen slot; `where_temp`
        applies. Without `cycle_feat` the skip is rebuilt from `obs`."""
        logits = self._placement_raw(hx, card_embeds, card_idx, obs, spatial_map, hires_map,
                                     ctx, ctx_hi, cycle_feat)
        return logits / self.where_temp if self.where_temp != 1.0 else logits

    def forward_sequence(self, feats_seq, card_embeds_seq, spatial_seq, obs_seq, card_mask_seq,
                         card_idx_seq, reset_seq, hidden_state, extra_card_idx_seq=None,
                         hires_seq=None, active_rows=None, with_ability=False):
        """The engine net's sequence interface (PPO update). No recurrence, so
        every row is computed independently; sampling knobs are NOT applied
        (these are the training-time log-probabilities)."""
        L, B = feats_seq.shape[0], feats_seq.shape[1]
        flat_feats = feats_seq.reshape(L * B, -1)
        mask_flat = card_mask_seq.reshape(L * B, -1)
        o = self.heads(flat_feats, mask_flat)
        flat_hx, flat_head = o["hx"], o["head_in"]
        flat_skip = self._split_cycle(flat_feats)
        card_logits = self.compose_card_logits(o["when"], o["what"], o["has_legal"])
        values = o["value"].squeeze(-1)
        aux = self.aux_card_head(flat_hx)

        flat_obs = obs_seq.reshape(L * B, -1)
        flat_hires = (self.hires_features(flat_obs) if hires_seq is None
                      else hires_seq.reshape(L * B, *hires_seq.shape[2:]))
        flat_embeds = card_embeds_seq.reshape(L * B, *card_embeds_seq.shape[2:])
        flat_spatial = spatial_seq.reshape(L * B, *spatial_seq.shape[2:])

        def _placement(idx_seq):
            flat_idx = idx_seq.reshape(L * B)
            if active_rows is None:
                return self._placement_raw(flat_hx, flat_embeds, flat_idx, flat_obs, flat_spatial,
                                           hires_map=flat_hires, cycle_feat=flat_skip)
            out = flat_hx.new_zeros(L * B, self.placement_cells)
            if not _conv_batch_invariant(flat_hx.device):
                full = self._placement_raw(flat_hx, flat_embeds, flat_idx, flat_obs, flat_spatial,
                                           hires_map=flat_hires, cycle_feat=flat_skip)
                return out.index_copy(0, active_rows, full[active_rows])
            if active_rows.numel() == 0:
                return out
            rows = torch.arange(L * B, device=flat_hx.device)
            joint = torch.cat((flat_head, flat_embeds[rows, flat_idx]), dim=-1)
            sub = self._placement_raw(
                flat_hx[active_rows], flat_embeds[active_rows], flat_idx[active_rows],
                flat_obs[active_rows], flat_spatial[active_rows],
                hires_map=flat_hires[active_rows], ctx=self.place_ctx(joint)[active_rows],
                ctx_hi=self.place_ctx_hi(joint)[active_rows], cycle_feat=flat_skip[active_rows])
            return out.index_copy(0, active_rows, sub)

        place_logits = _placement(card_idx_seq)
        extra_logits = None
        if extra_card_idx_seq is not None:
            extra_logits = _placement(extra_card_idx_seq).view(L, B, -1)
        last = reset_seq[L - 1].unsqueeze(1)
        hx = flat_hx.view(L, B, -1)[L - 1] * last
        out = (card_logits.view(L, B, -1), place_logits.view(L, B, -1), values.view(L, B),
               aux.view(L, B, -1), (hx, torch.zeros_like(hx)), extra_logits)
        if not with_ability:
            return out
        return out + ([a.view(L, B, 2) for a in self._active_ability(o["ability"]) if a is not None],)

    def forward_from_features(self, features, card_embeds, hidden_state, card_idx, card_mask=None,
                              obs=None, spatial_map=None):
        card_logits, a1, a2, value, (hx, cx) = self.step_lstm_and_card(features, hidden_state, card_mask)
        place = self.placement_given_card(hx, card_embeds, card_idx, obs, spatial_map,
                                          cycle_feat=self._split_cycle(features))
        return card_logits, place, value, a1, a2, (hx, cx)


# --------------------------------------------------------------- helpers ---
def new_net_v3(device="cpu", **cfg):
    return NetV3(**cfg).to(device)


def param_table(net):
    """Parameter count by top-level module, and the total."""
    rows = {}
    for name, p in net.named_parameters():
        rows[name.split(".")[0]] = rows.get(name.split(".")[0], 0) + p.numel()
    rows["total"] = sum(p.numel() for p in net.parameters())
    return rows


def metadata(net, **extra):
    """What every v3 checkpoint carries next to its weights."""
    meta = {"arch": ARCH, "cfg": dict(net.cfg), "obs_dim": int(net.obs_dim),
            "base_obs_dim": int(net.base_obs_dim), "obs_v2": True,
            "num_card_ids": int(net.num_card_ids), "code_hash": code_hash()}
    meta.update(extra)
    return meta


REQUIRED_META = ("arch", "cfg", "obs_dim", "base_obs_dim", "obs_v2", "num_card_ids", "code_hash",
                 "names", "decision_ticks", "lead", "dataset", "dataset_fp")


def save_v3(net, path, **extra):
    """Atomic save of {"model": state_dict, **metadata(net), **extra}. `extra`
    must supply names (tail names policy), decision_ticks, lead, dataset,
    dataset_fp; anything else (training settings, metrics) rides along."""
    blob = metadata(net, **extra)
    missing = [k for k in REQUIRED_META if k not in blob]
    if missing:
        raise ValueError(f"save_v3: missing metadata {missing}")
    blob["model"] = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, path)
    return blob


def is_v3(ck):
    return isinstance(ck, dict) and ck.get("arch") == ARCH


def load_v3(path_or_blob, device="cpu", allow_code_mismatch=False):
    """A v3 checkpoint -> (net in eval mode, metadata dict without the weights).
    Raises on anything that does not match this code and this engine."""
    ck = path_or_blob
    if not isinstance(ck, dict):
        ck = torch.load(ck, map_location="cpu", weights_only=False)
    if not is_v3(ck):
        raise ValueError("not a v3 checkpoint (arch != 'v3')")
    missing = [k for k in REQUIRED_META + ("model",) if k not in ck]
    if missing:
        raise ValueError(f"v3 checkpoint lacks metadata {missing}")
    meta = {k: v for k, v in ck.items() if k != "model"}
    problems = []
    if int(ck["base_obs_dim"]) != engine_obs_dim():
        problems.append(f"engine observation is {engine_obs_dim()} floats, checkpoint was trained "
                        f"on {ck['base_obs_dim']}")
    if int(ck["obs_dim"]) != engine_obs_dim() + OV.TAIL_SIZE:
        problems.append(f"observation length {ck['obs_dim']} != {engine_obs_dim() + OV.TAIL_SIZE}")
    if ck["obs_v2"] is not True:
        problems.append("obs_v2 is not True")
    if int(ck["num_card_ids"]) != int(C.CE.NUM_CARD_IDS):
        problems.append(f"card ids {ck['num_card_ids']} != engine {int(C.CE.NUM_CARD_IDS)}")
    if ck["code_hash"] != code_hash() and not allow_code_mismatch:
        problems.append(f"net_v3.py code hash {code_hash()} != checkpoint {ck['code_hash']}")
    if problems:
        raise ValueError("v3 checkpoint does not match: " + "; ".join(problems))
    net = NetV3(**ck["cfg"])
    net.load_state_dict(ck["model"], strict=True)
    return net.to(device).eval(), meta
