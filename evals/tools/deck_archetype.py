"""Archetype of a Clash Royale deck from its cards (engine card ids or names, any form).

    from deck_archetype import archetype, flags, ARCHETYPES
    archetype([15, 25, ...]) -> "Hog"; flags(deck) -> {"building": bool, "big_spell": bool}

One label per deck, by its main win condition. A deck with several win conditions takes the first
matching rule in ORDER (siege before air before beatdown ...), e.g. Giant + Graveyard = beatdown,
Lava Hound + Miner = air, Hog Rider + Goblin Barrel = Hog, Miner + Goblin Barrel = bait.
Minion Giant (new 2026-09, flying building-targeter, 8 % of the ladder pool of 2026-11-18) gets its own
row, because training had none of it until 2026-11-23 and it is answered unlike Balloon or Lava.

Flags: a defensive building (shoots or spawns: Cannon, Tesla, Inferno Tower, Bomb Tower, Goblin Cage,
Tombstone, Goblin Hut, Barbarian Hut, Furnace; not X-Bow / Mortar / Drill / Collector), and a big spell
(Fireball, Poison, Rocket, Lightning, Earthquake, Void).
"""
import sys

sys.path.insert(0, ".")

ORDER = [
    ("siege", {"X-Bow", "Mortar"}),
    ("Lava / Balloon air", {"Lava Hound", "Balloon"}),
    ("Minion Giant", {"Minion Giant"}),
    ("Golem / Giant beatdown", {"Golem", "Giant", "Electro Giant", "Goblin Giant", "Elixir Golem", "Rune Giant",
                                "Giant Skeleton", "Goblinstein"}),
    ("Royal Giant", {"Royal Giant"}),
    ("Hog", {"Hog Rider"}),
    ("Goblin Drill", {"Goblin Drill"}),
    ("Goblin Barrel bait", {"Goblin Barrel", "Skeleton Barrel"}),
    ("bridge spam", {"Ram Rider", "Battle Ram", "Bandit", "Royal Hogs", "Wall Breakers", "Boss Bandit"}),
    ("Miner / Graveyard control", {"Miner", "Graveyard", "Mighty Miner"}),
]
ARCHETYPES = [a for a, _ in ORDER] + ["other"]
DEF_BUILDINGS = {"Cannon", "Tesla", "Inferno Tower", "Bomb Tower", "Goblin Cage", "Tombstone", "Goblin Hut",
                 "Barbarian Hut", "Furnace"}
BIG_SPELLS = {"Fireball", "Poison", "Rocket", "Lightning", "Earthquake", "Void"}
SHORT = {"siege": "siege", "Lava / Balloon air": "air", "Minion Giant": "MinionGiant",
         "Golem / Giant beatdown": "beatdown", "Royal Giant": "RG", "Hog": "Hog", "Goblin Drill": "drill",
         "Goblin Barrel bait": "bait", "bridge spam": "bridge", "Miner / Graveyard control": "control",
         "other": "other"}


def base_name(c):
    if not isinstance(c, str):
        from training import ghost as G
        c = G.card_name(int(c))
    for p in ("evo:", "hero:", "Hero "):
        if c.startswith(p):
            c = c[len(p):]
    return c


def names(deck):
    return {base_name(c) for c in deck}


def archetype(deck):
    n = names(deck)
    for a, s in ORDER:
        if n & s:
            return a
    return "other"


def flags(deck):
    n = names(deck)
    return {"building": bool(n & DEF_BUILDINGS), "big_spell": bool(n & BIG_SPELLS)}


if __name__ == "__main__":
    import json
    for d in json.load(open(sys.argv[1]))["decks"][:int(sys.argv[2]) if len(sys.argv) > 2 else 20]:
        print(archetype(d["card_ids"]), flags(d["card_ids"]), d["cards"])
