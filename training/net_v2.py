"""net_v2: a wider board encoder, a small LSTM input, and a FiLM placement head.

A subclass of the engine's MicroRoyaleNet with the SAME observation and the SAME
action interface, so the PPO trainer, eval_panel and the live loop can swap it
in (`bt.MicroRoyaleNet = NetV2`, the name-rebinding pattern ppo_bc.py already
uses):

  * observation: 13,977 floats, read through the parent's offsets
    (spatial_size, extra_start, cycle_start, ...); nothing restated here;
  * card head: hand_size slots + no-op, masked by the parent's
    `affordability_mask`; placement: placement_cells (612) logits masked by the
    parent's `placement_mask`; Hero ability heads off hx; (hx, cx) of
    LSTM_HIDDEN = 256, so the trainer's hidden-state buffers are unchanged;
  * every method the trainer calls keeps its signature: extract_features_hires,
    extract_features, hires_features, step_lstm_and_card, placement_given_card,
    forward_sequence, predict_opp_next_card, cycle_features, ...

What changes (engine CLAUDE.md "Network": the v1 trunk is 14k parameters and
the LSTM's 1504-wide input makes it 96% of the net):

  board      coord channels + 3x3 stem -> 4 pre-activation residual blocks at the
             FULL 34x18 resolution, width 64, dilations (1, 2, 4, 8) (radius 31
             rows) plus a global mean/max-pool bias in blocks 2 and 4, so every
             cell's feature sees the whole board. The last conv of each block
             is zero-initialised (the parent's DilatedContextBlock device).
  lstm input two stride-2 convs to 16 x 9 x 5, + global mean/max -> Linear -> 256,
             concatenated with the parent's 64-dim ScalarEncoder (cycle branch
             last, so the parent's `_split_cycle` skip still works):
             LSTMCell(320, 256) ~0.59M, against LSTMCell(1504, 256) 1.80M.
  card head optional wait/act gate (default): column NOOP of `card_head` is the
             wait logit w, columns 0..3 the card logits; the returned 5 logits
             are normalised log-probs
                 log P(noop) = logsigmoid(w),
                 log P(slot i) = logsigmoid(-w) + log_softmax_legal(c)_i,
             so Categorical(logits=...) in the trainer is unchanged and the
             BC prior-shift correction is still `card_head.bias[NOOP] += s`.
  placement  FiLM: (head_in, chosen card embedding) -> per-channel scale/shift
             of the full-resolution board features -> two 3x3 convs -> 1x1 ->
             612 logits, plus a per-card prior map Linear(card embed, 612).
             The location head reads the board directly at full resolution.
"""
import math

from training import common as C  # noqa: F401  (engine path + CLASH_DECK first)

import torch
import torch.nn as nn
import torch.nn.functional as F

from python_ai.models.net import MicroRoyaleNet, CARD_EMBED_DIM

ARCH = "net_v2"

DEFAULTS = dict(width=64, dilations=(1, 2, 4, 8), global_blocks=(1, 3), proj_channels=16,
                board_embed=256, gate=True, film_hidden=128, place_hidden=32)


class ResBlock(nn.Module):
    """x + conv2(relu([+gpool bias] conv1(relu(x)))), both 3x3 at `dilation`.

    conv2 is zero-initialised: the block is the identity at init. With
    `global_pool`, a bias from the mean and max over the board of the first
    conv's activation is added before the second conv (KataGo-style), which
    gives every cell a whole-board receptive field for 2*width*width params.
    """

    def __init__(self, width, dilation, global_pool=False):
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
        self.gp = nn.Linear(2 * width, width) if global_pool else None
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        h = self.conv1(F.relu(x))
        if self.gp is not None:
            g = torch.cat((h.mean((2, 3)), h.amax((2, 3))), dim=1)
            h = h + self.gp(g)[:, :, None, None]
        return x + self.conv2(F.relu(h))


class BoardEncoder(nn.Module):
    """(B, channels, H, W) -> (B, width, H, W) at full resolution."""

    def __init__(self, channels, board_h, board_w, width, dilations, global_blocks):
        super().__init__()
        rows = torch.linspace(-1.0, 1.0, board_h).view(1, 1, board_h, 1).expand(1, 1, board_h, board_w)
        cols = torch.linspace(-1.0, 1.0, board_w).view(1, 1, 1, board_w).expand(1, 1, board_h, board_w)
        # Absolute position: the convs are translation-equivariant, placement is not.
        self.register_buffer("coords", torch.cat((rows, cols), 1).contiguous(), persistent=False)
        self.stem = nn.Conv2d(channels + 2, width, 3, padding=1)
        self.blocks = nn.ModuleList(ResBlock(width, d, i in global_blocks)
                                    for i, d in enumerate(dilations))

    def forward(self, x):
        x = torch.cat((x, self.coords.expand(x.shape[0], -1, -1, -1)), 1)
        h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        return F.relu(h)


class NetV2(MicroRoyaleNet):
    ARCH = ARCH

    def __init__(self, num_ability_slots=0, **cfg):
        super().__init__(num_ability_slots=num_ability_slots)
        cfg = {**DEFAULTS, **cfg}
        cfg["dilations"] = tuple(cfg["dilations"])
        cfg["global_blocks"] = tuple(cfg["global_blocks"])
        self.cfg = cfg
        # Drop the v1 modules this replaces (the parent builds them).
        for name in ("cnn_trunk", "cnn_flatten", "lstm", "card_head", "place_ctx", "place_up",
                     "place_ctx_hi", "place_hires"):
            delattr(self, name)
        W = cfg["width"]
        self.board_feat_dim = W
        self.board = BoardEncoder(self.channels, self.board_height, self.board_width, W,
                                  cfg["dilations"], cfg["global_blocks"])
        P = cfg["proj_channels"]
        self.board_down = nn.Sequential(
            nn.Conv2d(W, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, P, 3, stride=2, padding=1), nn.ReLU())
        dh = math.ceil(math.ceil(self.board_height / 2) / 2)
        dw = math.ceil(math.ceil(self.board_width / 2) / 2)
        self.board_proj = nn.Linear(P * dh * dw + 2 * W, cfg["board_embed"])
        self.cnn_out_dim = cfg["board_embed"]
        self.lstm_input_dim = self.cnn_out_dim + self.scalar_feature_dim
        self.lstm = nn.LSTMCell(self.lstm_input_dim, self.LSTM_HIDDEN)
        head_dim = self.LSTM_HIDDEN + self.cycle_feature_dim
        self.head_dim = head_dim
        self.gate = bool(cfg["gate"])
        self.card_head = nn.Linear(head_dim, self.hand_size + 1)
        # Placement: FiLM on the full-resolution board features.
        self.place_film = nn.Sequential(
            nn.Linear(head_dim + CARD_EMBED_DIM, cfg["film_hidden"]), nn.ReLU(),
            nn.Linear(cfg["film_hidden"], 2 * W))
        nn.init.zeros_(self.place_film[-1].weight)
        nn.init.zeros_(self.place_film[-1].bias)
        Hp = cfg["place_hidden"]
        self.place_conv = nn.Sequential(
            nn.Conv2d(W, Hp, 3, padding=1), nn.ReLU(),
            nn.Conv2d(Hp, Hp, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(Hp, 1, 1))
        self.place_prior = nn.Linear(CARD_EMBED_DIM, self.placement_cells)
        nn.init.zeros_(self.place_prior.weight)
        nn.init.zeros_(self.place_prior.bias)

    # ------------------------------------------------------------------ trunk
    def _board_features(self, obs):
        spatial = obs[:, :self.spatial_size].view(-1, self.channels, self.board_height, self.board_width)
        return self.board(spatial)

    def _board_vector(self, fmap):
        d = self.board_down(fmap).flatten(1)
        g = torch.cat((fmap.mean((2, 3)), fmap.amax((2, 3))), 1)
        return F.relu(self.board_proj(torch.cat((d, g), 1)))

    def extract_features_hires(self, obs):
        """(combined, card_embeds, board_map, board_map).

        board_map (B, width, 34, 18) is both the 'spatial_map' and the
        'hires_map' of the parent's contract: v2 has one full-resolution map.
        """
        fmap = self._board_features(obs)
        scalar_features = self.scalar_mlp(obs[:, self.spatial_size:])
        combined = torch.cat((self._board_vector(fmap), scalar_features), dim=1)
        scalar_obs = obs[:, self.spatial_size:]
        onehot_start = 1 + self.hand_size
        oh = scalar_obs[:, onehot_start:onehot_start + self.hand_size * self.num_card_ids]
        emb = self.card_id_embed(oh.view(-1, self.hand_size, self.num_card_ids))
        noop = self.noop_embed.view(1, 1, -1).expand(emb.shape[0], 1, -1)
        return combined, torch.cat([emb, noop], dim=1), fmap, fmap

    def hires_features(self, obs):
        return self._board_features(obs)

    # ------------------------------------------------------------------ heads
    def card_log_probs(self, head_in, card_mask=None):
        raw = self.card_head(head_in)
        if not self.gate:
            return raw if card_mask is None else raw.masked_fill(~card_mask, float("-inf"))
        n = self.hand_size
        w, c = raw[:, n], raw[:, :n]
        if card_mask is None:
            any_legal = torch.ones_like(w, dtype=torch.bool)
        else:
            legal = card_mask[:, :n]
            any_legal = legal.any(-1)
            c = c.masked_fill(~legal, float("-inf"))
        c = torch.where(any_legal[:, None], c, torch.zeros_like(c))
        lsm = torch.log_softmax(c, -1) + F.logsigmoid(-w)[:, None]
        lsm = torch.where(any_legal[:, None] & torch.isfinite(c), lsm, torch.full_like(lsm, float("-inf")))
        lp_noop = torch.where(any_legal, F.logsigmoid(w), torch.zeros_like(w))
        return torch.cat((lsm, lp_noop[:, None]), dim=1)

    def step_lstm_and_card(self, features, hidden_state, card_mask=None):
        hx, cx = self.lstm(features, hidden_state)
        head_in = self._head_input(hx, self._split_cycle(features))
        card_logits = self.card_log_probs(head_in, card_mask)
        state_value = self.value_head(head_in)
        a1 = self.ability_slot1_head(hx) if self.ability_slot1_head is not None else None
        a2 = self.ability_slot2_head(hx) if self.ability_slot2_head is not None else None
        return card_logits, a1, a2, state_value, (hx, cx)

    def film_context(self, head_in, chosen_embed):
        return self.place_film(torch.cat((head_in, chosen_embed), dim=-1))

    def placement_given_card(self, hx, card_embeds, card_idx, obs=None, spatial_map=None,
                             hires_map=None, ctx=None, ctx_hi=None, cycle_feat=None):
        fmap = spatial_map if spatial_map is not None else hires_map
        if fmap is None:
            if obs is None:
                raise ValueError("placement_given_card needs the board map or obs to rebuild it")
            fmap = self._board_features(obs)
        b = torch.arange(card_embeds.shape[0], device=card_embeds.device)
        chosen = card_embeds[b, card_idx]
        if ctx is None:
            if cycle_feat is None and self.cycle_feature_dim:
                if obs is None:
                    raise ValueError("placement_given_card needs cycle_feat or obs")
                cycle_feat = self.cycle_features(obs, detached=True)
            ctx = self.film_context(self._head_input(hx, cycle_feat), chosen)
        W = self.board_feat_dim
        gamma, beta = ctx[:, :W, None, None], ctx[:, W:, None, None]
        h = F.relu(fmap * (1.0 + gamma) + beta)
        logits = self.place_conv(h)[:, 0].reshape(-1, self.placement_cells)
        logits = logits + self.place_prior(chosen)
        if obs is not None:
            logits = logits.masked_fill(~self.placement_mask(obs, card_idx), float("-inf"))
        return logits

    def forward_sequence(self, feats_seq, card_embeds_seq, spatial_seq, obs_seq, card_mask_seq,
                         card_idx_seq, reset_seq, hidden_state, extra_card_idx_seq=None,
                         hires_seq=None, active_rows=None, with_ability=False):
        L, B = feats_seq.shape[0], feats_seq.shape[1]
        hx, cx = hidden_state
        hx_steps = []
        for l in range(L):
            hx, cx = self.lstm(feats_seq[l], (hx, cx))
            hx_steps.append(hx)
            r = reset_seq[l].unsqueeze(1)
            hx, cx = hx * r, cx * r
        flat_hx = torch.stack(hx_steps).reshape(L * B, -1)
        flat_cycle = self._split_cycle(feats_seq.reshape(L * B, -1))
        flat_head = self._head_input(flat_hx, flat_cycle)
        card_logits = self.card_log_probs(flat_head, card_mask_seq.reshape(L * B, -1))
        values = self.value_head(flat_head).squeeze(-1)
        aux = self.aux_card_head(flat_hx)
        flat_obs = obs_seq.reshape(L * B, -1)
        fm = spatial_seq if spatial_seq is not None else hires_seq
        flat_map = fm.reshape(L * B, *fm.shape[2:])
        flat_embeds = card_embeds_seq.reshape(L * B, *card_embeds_seq.shape[2:])

        def _placement(idx_seq):
            flat_idx = idx_seq.reshape(L * B)
            if active_rows is None:
                return self.placement_given_card(flat_hx, flat_embeds, flat_idx, flat_obs, flat_map,
                                                 cycle_feat=flat_cycle)
            out = flat_hx.new_zeros(L * B, self.placement_cells)
            if active_rows.numel() == 0:
                return out
            rows = active_rows
            chosen = flat_embeds[rows, flat_idx[rows]]
            ctx = self.film_context(flat_head[rows], chosen)
            sub = self.placement_given_card(flat_hx[rows], flat_embeds[rows], flat_idx[rows],
                                            flat_obs[rows], flat_map[rows], ctx=ctx)
            return out.index_copy(0, rows, sub.to(out.dtype))

        place_logits = _placement(card_idx_seq)
        extra = None
        if extra_card_idx_seq is not None:
            extra = _placement(extra_card_idx_seq).view(L, B, -1)
        out = (card_logits.view(L, B, -1), place_logits.view(L, B, -1), values.view(L, B),
               aux.view(L, B, -1), (hx, cx), extra)
        if not with_ability:
            return out
        ability = [h(flat_hx).view(L, B, 2) for h in (self.ability_slot1_head, self.ability_slot2_head)
                   if h is not None]
        return out + (ability,)

    def forward_from_features(self, features, card_embeds, hidden_state, card_idx, card_mask=None,
                              obs=None, spatial_map=None):
        cl, a1, a2, v, (hx, cx) = self.step_lstm_and_card(features, hidden_state, card_mask)
        pl = self.placement_given_card(hx, card_embeds, card_idx, obs, spatial_map)
        return cl, pl, v, a1, a2, (hx, cx)


# ---------------------------------------------------------------------------
def new_net_v2(device="cpu", **cfg):
    from python_ai.rl.abilities import ability_engine_slots
    return NetV2(num_ability_slots=len(ability_engine_slots(C.deck_ids())), **cfg).to(device)


def param_table(net):
    """{block: params} for the report."""
    groups = {
        "board encoder (stem + 4 res blocks)": ["board."],
        "board -> LSTM projection": ["board_down.", "board_proj."],
        "scalar encoder (engine ScalarEncoder)": ["scalar_mlp."],
        "card embeddings": ["card_id_embed.", "noop_embed"],
        "lstm": ["lstm."],
        "card head (gate + cards)": ["card_head."],
        "placement head (FiLM + convs + prior)": ["place_film.", "place_conv.", "place_prior."],
        "value head": ["value_head."],
        "ability head": ["ability_slot"],
        "aux heads (next card, cycle id)": ["aux_card_head.", "cycle_id_head."],
    }
    out = {k: 0 for k in groups}
    other = 0
    for n, p in net.named_parameters():
        for k, prefixes in groups.items():
            if any(n.startswith(pf) for pf in prefixes):
                out[k] += p.numel()
                break
        else:
            other += p.numel()
    out["other"] = other
    out["total"] = sum(p.numel() for p in net.parameters())
    return out


def save_v2(net, path, extra=None):
    sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    blob = {"arch": ARCH, "cfg": net.cfg, "model": sd}
    if extra:
        blob.update(extra)
    tmp = path + ".tmp"
    torch.save(blob, tmp)
    import os
    os.replace(tmp, path)


def is_v2_state(sd):
    """A net_v2 state_dict (bare weights, e.g. a PFSP snapshot) by its keys."""
    return isinstance(sd, dict) and "board.stem.weight" in sd


def load_any(path, device="cpu"):
    """A v1 (bare state_dict / {'model': ...}) or v2 checkpoint -> net, in eval mode.

    v2 is recognised by `arch` or, for blobs saved without it (trainer
    snapshots), by its keys; `cfg` then defaults to DEFAULTS."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    if (isinstance(ck, dict) and ck.get("arch") == ARCH) or is_v2_state(sd):
        net = new_net_v2(device, **(ck.get("cfg") or {}))
        net.load_state_dict(sd)
    else:
        net = C.load_weights(C.new_net(device), path)
    return net.to(device).eval()


def sample_logits(logits, temp=1.0):
    """One draw per row at temperature `temp` (0 = argmax). Masked -inf logits
    stay -inf, so legality is unchanged."""
    if temp <= 0:
        return logits.argmax(-1)
    from torch.distributions import Categorical
    return Categorical(logits=logits / temp if temp != 1.0 else logits).sample()
