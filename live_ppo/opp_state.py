"""Live feed of the observation's opponent-state section (engine item 46, patch 0012).

The estimator is the engine's own C++ `OpponentStateEstimator`, bound as
`clash_royale_env.OpponentStateEstimator`: the same class ClashEnv runs inside
training, fed there by the engine's play events through a simulated detector
whose rates are this pipeline's measured ones. Here it is fed by
`opp_plays.OppModel`'s plays (births back-dated, emitted ~1.5 s later), so the
policy sees the same kind of estimate in play as in training.

Section layout, offsets from `ClashRoyaleEnv.OPP_STATE_START` (all bound; see
runs/sim_fidelity/OPP_STATE_LAYOUT.md):

  OPP_ELIXIR_OFFSET   est/10, lower/10, upper/10, confidence
  OWN_EVO_OFFSET      per own hand slot: is Evolution, evo charge
  OWN_ABILITY_OFFSET  per own Champion/Hero slot 1, 2: has one, deployed, ready, cooldown frac
  OPP_ABILITY_OFFSET  per opponent Champion/Hero: seen, cooldown frac, cost/10, recency
  OPP_EVO_OFFSET      known[c] then charge[c] per Evolution id

Identity: YOLO `-ev1` on most of a play's detections -> the Evolution id,
reported evolved; `-hero` -> the Hero card id. Otherwise the base id, reported
plain (the engine never reports a plain play under the Evolution's id either).

Not detected live (so training's detector must not detect them either; the
ClashEnv default ability_detected is 0.0): opponent ability activations. YOLO
has no class for any ability effect. `ability(card_id, t)` is the hook for a
future detector.

Own side: our hand's Evolution charge is counted from our own plays
(`own_play`, from the hand reader's cycle) by the same PlayerState rule; our
Hero's "deployed" is a YOLO side-0 `<name>-hero` sighting in the last
DEPLOYED_HOLD_S, "ready" is deployed with the activation unused and the bar
covering its cost.

When the driver reads the Hero ability button (ability_button.py) it sets
`own_button = {card id: (present, ready)}` before each encode, and that wins:
"deployed" is the sighting OR the button being on screen (the button exists
exactly while the Hero is on the field), and "ready" is the driver's
`ability_tap.readiness` value: the button read as ready on two boards, the bar
covering the cost, no press of ours open or already counted. That same value
masks the policy's ability head, as `is_champion_ability_ready` does both jobs
in the simulator.
"""
from __future__ import annotations

import numpy as np

import clash_royale_env as E

C = E.ClashRoyaleEnv
OSE = getattr(E, "OpponentStateEstimator", None)
HAS_SECTION = OSE is not None and hasattr(C, "OPP_STATE_SIZE")

#: A play is reported evolved when at least this share of its class votes carry
#: `-ev1` (YOLO flips a body between the two classes from frame to frame).
EVO_VOTE_FRAC = 0.34
#: How long one own-Hero sighting keeps it "deployed".
DEPLOYED_HOLD_S = 1.5
#: The prior on unreported plays: troops 0/9 and spells 1/51 missed on the
#: labelled recordings (runs/live/PERCEPTION_FILL_2026-11-18.md) -> ~3%, the
#: mean of ClashEnv's default missTroop / missSpell.
ASSUMED_MISS_RATE = 0.035
TICKS_PER_S = 10


def _require():
    if not HAS_SECTION:
        raise RuntimeError("--opp-state needs an engine with patch 0012 (OpponentStateEstimator, "
                           "OPP_STATE_SIZE); this one has neither")


def _names():
    by = {}
    for c in range(C.NUM_CARD_IDS):
        try:
            by[c] = E.get_card_info(c)
        except Exception:  # noqa: BLE001 -- unregistered id
            continue
    return by


class OppState:
    def __init__(self, deck):
        _require()
        info = _names()
        self.info = info
        self.hero_of = {}           # base id -> Hero id
        by_name = {v["name"]: k for k, v in info.items()}
        for k, v in info.items():
            if v.get("is_hero") and v["name"].startswith("Hero "):
                base = by_name.get(v["name"][5:])
                if base is not None and base != k:
                    self.hero_of[base] = k
        cfg = E.OpponentStateConfig()
        cfg.assumed_miss_rate = ASSUMED_MISS_RATE
        self.est = OSE(cfg)
        # Our own counter: the engine's rule applied to our plays, our deck's
        # Evolutions known from the start.
        own_cfg = E.OpponentStateConfig()
        own_cfg.assumed_miss_rate = 0.0
        self.own = OSE(own_cfg)
        self.deck = list(deck)
        for c in self.deck:
            if E.get_card_stats(c)["is_evolution"]:
                self.own.declare_evolution(c)
        self.own_slots = {}         # deck slot 1/2 -> card id (Champion/Hero only)
        for slot in (1, 2):
            if slot < len(self.deck):
                v = info.get(self.deck[slot], {})
                if v.get("is_champion") or v.get("is_hero"):
                    self.own_slots[slot] = self.deck[slot]
        self._own_seen_t = {}       # card id -> last side-0 sighting (match s)
        self._own_used = {}         # card id -> activation used since this deployment
        self.own_button = None      # card id -> (present, ready) from the button reader, or None
        self.log: list[dict] = []

    # ---- identity
    def identify(self, p):
        """(card id, evolved flag) the estimator is told for an OppModel Play."""
        if p.card is None:
            return -1, -1
        form, frac = getattr(p, "form", ""), getattr(p, "form_frac", 0.0)
        if form == "-hero" and frac >= EVO_VOTE_FRAC and p.card in self.hero_of:
            return self.hero_of[p.card], 0
        evo = OSE.evolution_id_of(p.card)
        if evo >= 0 and form == "-ev1" and frac >= EVO_VOTE_FRAC:
            return evo, 1
        return p.card, 0 if evo >= 0 else -1

    # ---- feed
    def play(self, p, emit_s: float):
        cid, evolved = self.identify(p)
        self.est.advance_with_schedule(int(round(max(0.0, emit_s) * TICKS_PER_S)))
        self.est.report_play(int(round(max(0.0, p.t) * TICKS_PER_S)), int(cid), -1.0, int(evolved))
        self.log.append({"t": round(p.t, 2), "emit": round(emit_s, 2), "card": int(cid),
                         "evolved": int(evolved), "form": getattr(p, "form", ""),
                         "form_frac": round(getattr(p, "form_frac", 0.0), 2),
                         "est": round(self.est.estimate(), 2), "conf": round(self.est.confidence(), 2),
                         "contradictions": self.est.contradictions()})

    def ability(self, card_id: int, t_s: float, emit_s: float | None = None):
        self.est.advance_with_schedule(int(round(max(0.0, emit_s if emit_s is not None else t_s) * TICKS_PER_S)))
        self.est.report_ability(int(round(max(0.0, t_s) * TICKS_PER_S)), int(card_id))

    def own_play(self, card_id: int, t_s: float):
        tick = int(round(max(0.0, t_s) * TICKS_PER_S))
        self.own.advance_with_schedule(tick)
        evo = OSE.evolution_id_of(card_id) if not E.get_card_stats(card_id)["is_evolution"] else card_id
        if evo >= 0 and evo in self.deck:
            # the counter decides whether this one was evolved
            self.own.report_play(tick, OSE.base_id_of(evo), -1.0, -1)
        else:
            self.own.report_play(tick, int(card_id), -1.0, 0)
        if card_id in self.own_slots.values():
            self._own_used[card_id] = False

    def own_ability_used(self, card_id: int):
        self._own_used[card_id] = True

    def see_own(self, dets, t_s: float):
        """Side-0 YOLO dets of this frame: our Hero/Champion on the board."""
        for d in dets or ():
            if d.get("side") == 0 and d["cls"].endswith("-hero"):
                base = d["cls"][: -len("-hero")]
                for cid in self.own_slots.values():
                    if _slug(self.info[cid]["name"]).endswith(base):
                        self._own_seen_t[cid] = t_s

    def advance(self, t_s: float):
        tick = int(round(max(0.0, t_s) * TICKS_PER_S))
        self.est.advance_with_schedule(tick)
        self.own.advance_with_schedule(tick)

    # ---- read-out
    def section(self, hand, my_elixir: float, t_s: float) -> np.ndarray:
        """The OPP_STATE_SIZE-long section, as ClashEnv::writeOpponentState
        writes it. `hand`: our four hand ids in TRAIN ids (Evolutions as such)."""
        self.advance(t_s)
        o = np.zeros(C.OPP_STATE_SIZE, np.float32)
        o[C.OPP_ELIXIR_OFFSET:C.OPP_ELIXIR_OFFSET + 4] = self.est.elixir_features()
        for i, c in enumerate(list(hand)[:C.HAND_SIZE]):
            if c is None or c < 0 or not E.get_card_stats(int(c))["is_evolution"]:
                continue
            ch = self.own.evo_charge_for(int(c))
            o[C.OWN_EVO_OFFSET + 2 * i] = 1.0
            o[C.OWN_EVO_OFFSET + 2 * i + 1] = max(0.0, ch)
        for k, slot in enumerate((1, 2)):
            cid = self.own_slots.get(slot)
            if cid is None:
                continue
            f = C.OWN_ABILITY_OFFSET + k * C.ABILITY_FEATURES
            st = E.get_card_stats(cid)["base"]
            deployed = t_s - self._own_seen_t.get(cid, -1e9) <= DEPLOYED_HOLD_S
            button = (self.own_button or {}).get(cid)
            o[f] = 1.0
            if button is not None:
                present, ready = button
                o[f + 1] = float(deployed or present)
                o[f + 2] = float(bool(ready) and my_elixir >= st["ability_cost"])
            else:
                o[f + 1] = float(deployed)
                o[f + 2] = float(deployed and not self._own_used.get(cid, False)
                                 and my_elixir >= st["ability_cost"])
            o[f + 3] = 0.0   # cooldown left: no live reading; 0 for every Hero (one use)
        o[C.OPP_ABILITY_OFFSET:C.OPP_ABILITY_OFFSET + C.NUM_OPP_ABILITY_CARDS * C.ABILITY_FEATURES] = \
            self.est.ability_features()
        o[C.OPP_EVO_OFFSET:] = self.est.evo_blocks()
        return o


def _slug(name: str) -> str:
    return name.lower().replace(" ", "-")
