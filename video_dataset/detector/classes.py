"""Detector class taxonomy for the CURRENT card pool.

Classes are on-board *unit* identities (what a box looks like), named after
the card (RoyaleAPI keys) with the form as a suffix: `-ev1` for an Evolution,
`-hero` for a Hero. A card can put several units on the board (Goblin Gang ->
goblin + spear-goblin; Golem -> golem + golemite) and a unit can come from
several cards (skeleton <- skeletons, skeleton-army, graveyard, tombstone,
witch ...), so card -> units is a table (CARD_UNITS) and elixir cost belongs
to the card, not the box.

KataCR unit names are kept for units that already existed in 2024 (singular:
skeleton, goblin, barbarian ...); KataCR's `<unit>-evolution` becomes
`<unit>-ev1`.

The current pool = every card_key seen in data/il_replay/hog26/plays.parquet
(2026 ladder, includes -ev1 / -hero forms) plus card art keys that are real
ladder cards; `super-*`, event and tower-troop art is excluded (EVENT_ONLY).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "data" / "card_art" / "cards-150"
PLAYS = REPO / "data" / "il_replay" / "hog26" / "plays.parquet"

# Non-unit classes the detector also predicts (towers / HP bars / deploy clock).
UI_CLASSES = ["king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower",
              "tower-bar", "king-tower-bar", "bar", "bar-level", "clock"]

EVENT_ONLY = {"super-archers", "super-hog-rider", "super-ice-golem", "super-knight",
              "super-lava-hound", "super-magic-archer", "super-mini-pekka", "super-witch",
              "santa-hog-rider", "terry", "warmth", "party-hut", "party-rocket",
              # tower troops (board towers, covered by the tower classes)
              "tower-princess", "cannoneer", "dagger-duchess", "royal-chef"}

# KataCR name -> our name (only renames; everything else keeps its name)
KATACR_RENAME = {k: k.replace("-evolution", "-ev1") for k in [
    "skeleton-evolution", "bat-evolution", "barbarian-evolution", "knight-evolution",
    "firecracker-evolution", "royal-recruit-evolution", "mortar-evolution",
    "royal-giant-evolution", "archer-evolution", "ice-spirit-evolution",
    "valkyrie-evolution", "bomber-evolution", "wall-breaker-evolution",
    "tesla-evolution", "battle-ram-evolution", "zap-evolution"]}
# KataCR classes that are not detector targets (UI / effects) or no longer in the game
KATACR_DROP = {"emote", "text", "elixir", "selected", "skeleton-king-bar", "evolution-symbol",
               "ice-spirit-evolution-symbol", "skeleton-king-skill", "tesla-evolution-shock",
               "dirt", "dagger-duchess-tower-bar", "big-text", "small-text", "background-items"}

# card form -> on-board unit classes. Cards not listed map to [card_key] for
# base/hero forms; `-ev1` forms of listed cards map to their main unit + "-ev1".
_BASE = {
    "skeletons": ["skeleton"], "skeleton-army": ["skeleton"], "graveyard": ["graveyard", "skeleton"],
    "goblins": ["goblin"], "goblin-gang": ["goblin", "spear-goblin"], "spear-goblins": ["spear-goblin"],
    "goblin-hut": ["goblin-hut", "spear-goblin"], "goblin-barrel": ["goblin-barrel", "goblin"],
    "goblin-drill": ["goblin-drill", "goblin"], "goblin-cage": ["goblin-cage", "goblin-brawler"],
    "bats": ["bat"], "minions": ["minion"], "minion-horde": ["minion"], "barbarians": ["barbarian"],
    "elite-barbarians": ["elite-barbarian"], "barbarian-hut": ["barbarian-hut", "barbarian"],
    "wall-breakers": ["wall-breaker"], "archers": ["archer"], "royal-recruits": ["royal-recruit"],
    "guards": ["guard"], "rascals": ["rascal-boy", "rascal-girl"], "zappies": ["zappy"],
    "royal-hogs": ["royal-hog"], "elixir-golem": ["elixir-golem-big", "elixir-golem-mid", "elixir-golem-small"],
    "golem": ["golem", "golemite"], "lava-hound": ["lava-hound", "lava-pup"],
    "phoenix": ["phoenix-big", "phoenix-egg", "phoenix-small"], "skeleton-dragons": ["skeleton-dragon"],
    "skeleton-barrel": ["skeleton-barrel", "skeleton"], "furnace": ["furnace", "fire-spirit"],
    "fire-spirits": ["fire-spirit"], "tombstone": ["tombstone", "skeleton"], "witch": ["witch", "skeleton"],
    "night-witch": ["night-witch", "bat"], "three-musketeers": ["musketeer"],
    "battle-ram": ["battle-ram", "barbarian"], "mother-witch": ["mother-witch", "hog"],
    "bomb-tower": ["bomb-tower", "bomb"], "giant-skeleton": ["giant-skeleton", "bomb"],
}


def card_units(card: str) -> list[str]:
    """On-board unit classes a played card form produces (first = main unit)."""
    if card in _BASE:
        return list(_BASE[card])
    if card.endswith("-ev1"):
        base = card[:-4]
        if base.endswith("-hero"):            # hero art variant, board form is the hero
            return card_units(base)
        units = card_units(base)
        return [units[0] + "-ev1"] + units[1:]
    return [card]


@lru_cache(maxsize=1)
def current_pool() -> dict[str, int]:
    """card form -> plays count (0 = only in card art)."""
    import pandas as pd
    counts = pd.read_parquet(PLAYS, columns=["card_key"]).card_key.value_counts().to_dict()
    for p in ART.glob("*.png"):
        k = p.stem
        if k in EVENT_ONLY or k.endswith("-hero-ev1"):
            continue
        counts.setdefault(k, 0)
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


@lru_cache(maxsize=1)
def unit_classes() -> list[str]:
    """Detector unit classes, most-played first (by summed card plays)."""
    score: dict[str, int] = {}
    for card, n in current_pool().items():
        for u in card_units(card):
            score[u] = score.get(u, 0) + n
    return [u for u, _ in sorted(score.items(), key=lambda kv: -kv[1])]


def all_classes() -> list[str]:
    return UI_CLASSES + unit_classes()


def katacr_to_ours(name: str) -> str | None:
    if name in KATACR_DROP:
        return None
    n = KATACR_RENAME.get(name, name)
    return n if n in set(all_classes()) else None


if __name__ == "__main__":
    pool = current_pool()
    cls = unit_classes()
    print(len(pool), "card forms;", len(cls), "unit classes;", len(UI_CLASSES), "UI classes")
    print(cls)
