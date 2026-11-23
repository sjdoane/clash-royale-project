"""Generation-2 (NetV3, training/net_v3.py) checkpoints in the live driver: the
rules that need no engine, so they are tested without one
(eval/test_policy_v3.py). driver.py does the loading and the wiring.

What a v3 checkpoint is, to the driver
    arch 'v3'; no `deck` key (the training deck is fixed in training/common.py:
    HOG26_DECK, which must equal the driver's TRAIN_DECK); feed-forward; the
    observation is the gen-2 engine's plus the observation v2 tail; the card
    choice is three heads: WHEN (play now, a coin), WHAT (which slot), WHERE.
    Its metadata says how it must be run: `names` (tail names policy of the
    training noise), `decision_ticks` (engine ticks per decision), `obs_dim`.

Defaults follow the checkpoint (`settings`): the tail is on with the
checkpoint's `names`; the three live-only crutches added for generation 1 are
off, each with a flag to turn it back on:
    tactical officer   off (--tactical-officer). It replaces the WHERE head for
                       Cannon, Fireball and Giant because gen-1's placement of
                       them was worse than random. The clone learned WHERE from
                       human play, and the deck has no Giant.
    solvency gate      off (--solvency-gate), as for every checkpoint: training
                       never used it.
    leak floor         off (--leak-floor F). It forced a card at a full bar
                       because gen-1 sat on 10 elixir. The clone's WHEN head
                       plays on its own; forcing it would be a play the data
                       never showed. (For v3 the floor cannot work through the
                       no-op mask, which NetV3 does not read: with the flag it
                       forces the WHEN coin instead.)

Sampling (`sampling`): --temp is the WHAT and WHERE temperature, --when-temp and
--when-bias go to WHEN, all through net.set_sampling, which is how the simulator
evaluation (training/eval_ghost.py) applies them. --greedy takes the most likely
card and cell but leaves WHEN a coin at the trained probability: the argmax of a
coin that says "play" with p < 0.5 on most decisions is "never play", and the
argmax of the composed five-way distribution is the same thing.

Plays in flight (`blocked_slots`, `inflight_obs`): the simulator's actuation
delay (training/reward.py ActuationDelay / inflight_obs) refuses a card that is
already on its way and shows the policy that card's slot with cost 0 and the
elixir already paid. Live, the card stays in its slot for about half a second
after the placement is issued, which at 2 decisions a second is long enough to
ask for it again; the same treatment is applied from play_watch's open plays.
"""
from __future__ import annotations

import math

V1, NET_V2, V3 = "v1", "net_v2", "v3"
#: WHEN bias that makes sigmoid() exactly 1.0 in float32 (the --leak-floor force)
FORCE_PLAY_BIAS = 1.0e4


class V3Error(ValueError):
    """A v3 checkpoint cannot be run the way this process was started."""


def arch(blob) -> str:
    """'v3' | 'net_v2' | 'v1' for a checkpoint blob: by `arch`, else by the
    state dict's keys (net_v2 has a `board.stem`), else the engine's own net."""
    if not isinstance(blob, dict):
        return V1
    a = blob.get("arch")
    if a == V3:
        return V3
    state = blob.get("model", blob)
    if a == NET_V2 or (isinstance(state, dict) and "board.stem.weight" in state):
        return NET_V2
    return V1


def parse_deck(text) -> list[int]:
    """training/common.HOG26_DECK ('125,168,...') -> [125, 168, ...]."""
    if isinstance(text, (list, tuple)):
        return [int(v) for v in text]
    return [int(v) for v in str(text).replace(" ", "").split(",") if v]


def check(blob, *, path="checkpoint", engine_gen2, engine="", obs_v2="auto", names=None,
          driver_deck=None, train_deck=None) -> None:
    """Refuse (V3Error, with what to change) to run a v3 checkpoint when
    the gen-2 engine is not selected, the tail is forced off, the tail names
    policy asked for is not the one it was trained with, or the deck the
    driver plays is not the one the training code fixes.

    `names`: the --obs-v2-names given on the command line, None when it was not.
    `train_deck`: training/common.HOG26_DECK, the stand-in for the `deck` key a
    v3 checkpoint does not carry (a checkpoint that does carry one wins)."""
    if arch(blob) != V3:
        raise V3Error(f"{path}: not a v3 checkpoint")
    if not engine_gen2:
        raise V3Error(
            f"{path}: a v3 (generation-2) checkpoint needs the gen-2 engine, and this process "
            f"runs {engine or 'the pinned engine'}. Start it with "
            f"CR_ENGINE=<repo>/build/gen2/engine (and run from $CR_ENGINE/perception)")
    if obs_v2 == "off":
        raise V3Error(
            f"{path}: this checkpoint was trained on the observation v2 tail "
            f"({blob.get('obs_dim')} floats = {blob.get('base_obs_dim')} + tail); --obs-v2 off "
            f"would feed it a shorter observation. Drop --obs-v2 off")
    trained = blob.get("names")
    if names is not None and trained is not None and names != trained:
        raise V3Error(
            f"{path}: --obs-v2-names {names} but the checkpoint was trained with opponent "
            f"names '{trained}'. Drop the flag (the checkpoint's value is used) or pass "
            f"--obs-v2-names {trained}")
    deck = blob.get("deck") or train_deck
    if driver_deck is not None and deck is not None and parse_deck(deck) != list(driver_deck):
        raise V3Error(
            f"{path}: trained on deck {parse_deck(deck)} "
            f"({'checkpoint' if blob.get('deck') else 'training/common.HOG26_DECK'}) but the "
            f"driver plays {list(driver_deck)}")


def settings(kind, blob=None, *, names=None, leak_floor=None, tactical_loop=True,
             tactical_officer=False, pending_mask="auto", play_watch=True) -> dict:
    """What the driver runs with, from the checkpoint kind and the flags that
    were GIVEN (None = not given). v1 / net_v2: exactly what the driver did
    before v3 existed. v3: see the module docstring.

    names           opponent names in the observation v2 tail
    leak_floor      0 = off
    tactical        the tactical officer (and with it the base class's gate)
    pending_mask    plays in flight shown as in the simulator
    """
    v3 = kind == V3
    out = {
        "names": names if names is not None else ((blob or {}).get("names") if v3 else None) or "spells",
        "leak_floor": float(leak_floor) if leak_floor is not None else (0.0 if v3 else 10.0),
        "tactical": bool(tactical_officer) if v3 else bool(tactical_loop),
    }
    if pending_mask not in ("auto", "on", "off"):
        raise ValueError(f"pending_mask {pending_mask!r}")
    want = pending_mask == "on" or (pending_mask == "auto" and v3)
    if pending_mask == "on" and not play_watch:
        raise V3Error("--pending-mask on needs the play watch (drop --no-play-watch): it is what "
                      "knows which placements are still in flight")
    out["pending_mask"] = bool(want and play_watch)
    out["pending_mask_lost"] = bool(want and not play_watch)    # auto, but nothing to read it from
    return out


def sampling(kind, *, temp=1.0, greedy=False, when_temp=None, when_bias=None) -> dict:
    """The command line's sampling flags -> how the driver samples.

    Returns {"patch": None | "argmax" | "tempered", "patch_temp": T,
             "set_sampling": None | kwargs for NetV3.set_sampling, "note": str}.

    v1 / net_v2 (unchanged): the five-way card logits and the cell logits are
    sampled through torch's Categorical, which the driver patches: --greedy ->
    argmax, else --temp != 1 -> logits / T. --when-temp / --when-bias are
    refused: those networks have no WHEN head.

    v3: the temperatures are set inside the net, so the Categorical is sampled
    at 1 and never patched for temperature (patching it too would apply --temp
    twice). --greedy patches the argmax in for WHAT and WHERE only; WHEN stays
    the net's own coin (when_internal) at the trained probability, shaped by
    --when-temp / --when-bias."""
    wt = 1.0 if when_temp is None else float(when_temp)
    wb = 0.0 if when_bias is None else float(when_bias)
    if kind != V3:
        if when_temp is not None or when_bias is not None:
            raise V3Error("--when-temp / --when-bias shape the WHEN head, which only a v3 "
                          "checkpoint has")
        if greedy:
            return {"patch": "argmax", "patch_temp": 1.0, "set_sampling": None,
                    "note": "argmax of the card and cell logits"}
        if temp != 1.0:
            return {"patch": "tempered", "patch_temp": float(temp), "set_sampling": None,
                    "note": f"card and cell logits / {temp:g}"}
        return {"patch": None, "patch_temp": 1.0, "set_sampling": None, "note": "sampled at 1"}
    if not greedy and not temp > 0:
        raise V3Error(f"--temp {temp:g}: a v3 checkpoint needs a temperature > 0 "
                      f"(--greedy is the argmax of WHAT and WHERE)")
    if not wt > 0:
        raise V3Error(f"--when-temp {wt:g}: must be > 0 (there is no greedy WHEN: it would "
                      f"never play)")
    if not math.isfinite(wb):
        raise V3Error(f"--when-bias {wb}: must be finite")
    t = 1.0 if greedy else float(temp)
    return {"patch": "argmax" if greedy else None, "patch_temp": 1.0,
            "set_sampling": {"what_temp": t, "where_temp": t, "when_temp": wt, "when_bias": wb,
                             "when_internal": True},
            "note": (f"WHAT / WHERE {'argmax' if greedy else f'temperature {t:g}'}; WHEN a coin at "
                     f"sigmoid((w + {wb:g}) / {wt:g})")}


# ----------------------------------------------------------------- plays in flight
def blocked_slots(in_flight, hand_names=None) -> dict:
    """{slot: debit} for the hand slots that hold a card already on its way.

    `in_flight`: (slot, card name or None, cost or None) of each placement that
    was issued and is neither seen landed nor given up (PlayWatch.in_flight()).
    `hand_names`: the hand being decided on; a slot that no longer shows the
    card that was played from it is not blocked (the play has landed and the
    slot holds the next card).

    debit: True when the play's cost was not booked anywhere (a dry run: the
    loop only records the cost of a tap it sent), so the observation must pay
    it; False when the driver's elixir already has it off (PlayWatch.owed())."""
    out = {}
    for slot, card, cost in in_flight:
        slot = int(slot)
        if not 0 <= slot < 4:
            continue
        if hand_names is not None and card not in (None, "?") and slot < len(hand_names) \
                and hand_names[slot] != card:
            continue
        out[slot] = out.get(slot, False) or cost is None
    return out


def inflight_obs(obs, spatial_size, blocked) -> list:
    """training/reward.py inflight_obs on a live observation, in place: each
    blocked slot's cost scalar -> 0 (net.affordability_mask then refuses it) and,
    where the cost is not paid yet (`blocked[slot]`), off the elixir scalar.
    Elixir and costs are /10 in the observation. Returns [(slot, cost)]."""
    s = int(spatial_size)
    done = []
    for slot in sorted(blocked):
        cost = float(obs[s + 1 + slot])
        if blocked[slot]:
            obs[s] = max(0.0, float(obs[s]) - cost)
        obs[s + 1 + slot] = 0.0
        done.append((slot, round(cost * 10.0, 2)))
    return done


# ----------------------------------------------------------------- the decision log
def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _softmax(xs, temp=1.0):
    fin = [x / temp for x in xs if x != float("-inf")]
    if not fin:
        return [0.0] * len(xs)
    m = max(fin)
    es = [0.0 if x == float("-inf") else math.exp(x / temp - m) for x in xs]
    z = sum(es)
    return [e / z for e in es]


def card_probs(when, what, has_legal, knobs=None) -> dict:
    """The distribution a v3 decision is drawn from, for the log, from the raw
    heads (NetV3.heads) and the sampling knobs in force:
        p_play  sigmoid((when + when_bias) / when_temp), 0 with nothing playable
        what    softmax(what / what_temp) over the hand (0 where not playable)
        p       [p_play * what..., 1 - p_play]: the five-way row the driver's
                log and overlay already read (last = hold)"""
    k = knobs or {}
    if not has_legal:
        return {"p_play": 0.0, "what": [0.0] * len(what), "p": [0.0] * len(what) + [1.0]}
    p_play = _sigmoid((float(when) + float(k.get("when_bias", 0.0))) / float(k.get("when_temp", 1.0)))
    w = _softmax([float(x) for x in what], float(k.get("what_temp", 1.0)))
    return {"p_play": p_play, "what": w, "p": [p_play * x for x in w] + [1.0 - p_play]}


def ability_prob(logits2) -> float:
    """P(activate) of one ability head's (wait, activate) logits."""
    return _softmax([float(logits2[0]), float(logits2[1])])[1]


def decision_record(probs, hand_names, *, ability=None, pending=(), knobs=None, obs_len=None,
                    forced=False) -> dict:
    """The `v3` block of a perception.jsonl decision row."""
    order = sorted(range(len(probs["what"])), key=lambda i: -probs["what"][i])
    return {
        "p_play": round(probs["p_play"], 4),
        "what": [[hand_names[i] if i < len(hand_names) else "?", round(probs["what"][i], 3)]
                 for i in order if probs["what"][i] > 0.0],
        "ability": ability,
        "pending": [list(p) for p in pending],
        "forced_play": bool(forced),
        "sampling": dict(knobs or {}),
        "obs_len": obs_len,
    }
