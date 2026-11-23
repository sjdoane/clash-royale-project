"""Inventory of the decoded mirsella/clash-royale assets per detector class (read-only).

    .venv-det/bin/python video_dataset/detector/mirsella_inventory.py
    -> data/ext/mirsella_inventory.json

Only lists decoded files (sprites/sc/decoded/{chr,building}_*/exports/<anim>/frame_*.png,
models/decoded/characters__*.glb); never runs anything from the clone.
Sprite export dirs are counted as ally / enemy (name contains "enemy") / fx (vmc_*,
fx_*: effects, not the unit body). Name matching: strip _dl/_troop/_rework, _evolution
-> -ev1, _hero -> -hero, then ALIAS; GLBs: characters__<compact name>__ and
characters__hero__hero_<name>__.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
M = REPO / "data" / "ext" / "mirsella"
OUT = REPO / "data" / "ext" / "mirsella_inventory.json"
CLASSES = REPO / "data" / "det" / "classes_191.json"

ALIAS = {   # mirsella workspace stem -> our class
    "axe-man": "executioner", "baby-goblin-dagger": "baby-goblins", "bats": "bat", "bats-ev1": "bat-ev1",
    "battle-healer": "battle-healer", "boss-bandit": "boss-bandit", "fire-firespirit": "fire-spirit",
    "flying-machine": "flying-machine", "furnace-rework": "furnace", "goblin-archer": "dart-goblin",
    "goblin-blowdart": "dart-goblin", "goblin-blowdart-ev1": "dart-goblin-ev1", "goblin-bush": "suspicious-bush",
    "goblin-cage-ev1": "goblin-cage-ev1", "goblin-thug": "goblin-brawler", "goblinstein-doctor": "goblinstein",
    "goblinstein-monster": "goblinstein", "guard02": "guard", "heal-spirit": "heal-spirit",
    "ice-spirits": "ice-spirit", "ice-golem-hero": "ice-golem-hero", "knives-thrower": "spear-goblin",
    "lava-pups": "lava-pup", "lumberjack-ev1": "lumberjack-ev1", "minionhorde-ev1": "minion-ev1",
    "minipekka-hero": "mini-pekka-hero", "movingcannon": "cannon-cart", "rage-barbarian": "elite-barbarian",
    "rascals": "rascal-boy", "royalgiant-ev1": "royal-giant-ev1", "skeleton-balloon": "skeleton-barrel",
    "skeleton-balloon-ev1": "skeleton-barrel-ev1", "skeleton-warrior": "guard", "snowman": "giant-snowball",
    "wall-breakers-ev1": "wall-breaker-ev1", "zappies": "zappy", "zap-machine": "sparky",
    "mother-witch-prestige": "mother-witch", "goblin-referee": None, "goblin-tower-mask": None,
    "giant-buffing": "rune-giant", "merge-maiden": None, "merge-maiden-dragon": None, "rockera": None,
    "barbarian-barrel": "barbarian-barrel", "fisherman": "fisherman", "golden-knight": "golden-knight",
    "goblin-queen": None, "king": None, "royal-chef": None, "cannoneer": None, "dark-witch": "night-witch",
    "electro-giant": "electro-giant", "phoenix": "phoenix-big", "electro-dragon": "electro-dragon",
    "goblin-giant": "goblin-giant", "three-musketeers": "musketeer", "ram-rider": "ram-rider",
    "berserker": "berserker", "ghost": "royal-ghost", "elixir-golem": "elixir-golem-big", "mighty-monk": "monk",
    # buildings
    "basic-cannon": "cannon", "cannon-ev1": "cannon-ev1", "xbow": "x-bow", "barbarian-mortar": "mortar",
    "firespirit-hut": None, "mega-bomb": None, "mighty-miner-bomb": None, "pancakes": None,
    "present-santa": None, "rage-bottle": "rage", "vines-bottle": "vines", "heist-safe": None,
    "throne-goblin-queen": None, "tower": None, "clanwars-towers": None, "prestige": None,
}


def norm(ws: str) -> str:
    s = re.sub(r"^(chr|building)_", "", ws).lower()
    s = re.sub(r"_(dl|troop|rework)$", "", s)
    s = re.sub(r"_(dl|troop)(?=_)", "", s)
    s = re.sub(r"_evolution$", "_ev1", s)
    return s.replace("_", "-")


def to_ours(ws: str, ours: set[str]):
    n = norm(ws)
    if re.match(r"(prestige\d*|towerskin)", n):
        return None, n
    if n in ALIAS:
        return ALIAS[n], n
    if n in ours:
        return n, n
    return None, n


def sprite_inventory(ours):
    per = defaultdict(lambda: {"workspaces": [], "ally_exports": 0, "enemy_exports": 0, "fx_exports": 0, "pngs": 0})
    unmatched = {}
    for d in sorted((M / "sprites" / "sc" / "decoded").iterdir()):
        if not d.is_dir() or not re.match(r"(chr|building)_", d.name) or d.name.startswith("building_towerskin"):
            continue
        cls, n = to_ours(d.name, ours)
        ex = d / "exports"
        c = Counter()
        pngs = 0
        if ex.exists():
            for a in ex.iterdir():
                if not a.is_dir():
                    continue
                k = "fx" if re.match(r"(vmc|fx)_", a.name) else "enemy" if "enemy" in a.name.lower() else "ally"
                c[k] += 1
                pngs += sum(1 for _ in a.glob("*.png"))
        if cls is None:
            unmatched[d.name] = {"norm": n, "exports": dict(c)}
            continue
        e = per[cls]
        e["workspaces"].append(d.name)
        e["ally_exports"] += c["ally"]
        e["enemy_exports"] += c["enemy"]
        e["fx_exports"] += c["fx"]
        e["pngs"] += pngs
    return per, unmatched


def glb_inventory(ours):
    compact = {c.replace("-", ""): c for c in ours if not c.endswith(("-ev1", "-hero"))}
    compact.update({"daggergoblin": None, "dummy": None, "rabbit": None, "monkspirit": "monk", "archerqueen": "archer-queen"})
    per = defaultdict(lambda: {"models": 0, "animations": 0})
    other = Counter()
    for p in (M / "models" / "decoded").glob("characters__*.glb"):
        parts = p.name.split("__")
        if parts[1] == "hero":
            cls = parts[2].removeprefix("hero_").replace("_", "-") + "-hero"
            cls = cls if cls in ours else None
        else:
            cls = compact.get(parts[1])
        if cls is None:
            other[parts[1] if parts[1] != "hero" else parts[2]] += 1
            continue
        per[cls]["models" if "model" in p.name.split("__")[-2] else "animations"] += 1
    return per, other


def main():
    ours_l = json.loads(CLASSES.read_text())
    ours = set(ours_l)
    spr, unmatched = sprite_inventory(ours)
    glb, glb_other = glb_inventory(ours)
    per_class = {}
    for c in ours_l:
        s, g = spr.get(c), glb.get(c)
        per_class[c] = {"sprites": dict(s) if s else None, "glb": dict(g) if g else None}

    def has_body(c):
        s = per_class[c]["sprites"]
        return bool(s and s["ally_exports"] + s["enemy_exports"])
    kinds = {"base": [c for c in ours_l if not c.endswith(("-ev1", "-hero"))],
             "evo": [c for c in ours_l if c.endswith("-ev1")], "hero": [c for c in ours_l if c.endswith("-hero")]}
    summary = {}
    for k, cs in kinds.items():
        summary[k] = {"classes": len(cs),
                      "with_unit_sprites": sum(has_body(c) for c in cs),
                      "with_enemy_sprites": sum(bool(per_class[c]["sprites"] and per_class[c]["sprites"]["enemy_exports"]) for c in cs),
                      "with_glb": sum(per_class[c]["glb"] is not None for c in cs),
                      "with_any": sum(has_body(c) or per_class[c]["glb"] is not None for c in cs),
                      "missing": [c for c in cs if not has_body(c) and per_class[c]["glb"] is None]}
    res = {"source": "github.com/mirsella/clash-royale (decoded assets, read only; private use, do not re-host)",
           "summary": summary,
           "sprite_workspaces_total": sum(len(v["workspaces"]) for v in spr.values()) + len(unmatched),
           "unmatched_workspaces": unmatched, "glb_unmatched_dirs": dict(glb_other), "per_class": per_class}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(summary, indent=1))
    print("unmatched workspaces:", sorted(unmatched))


if __name__ == "__main__":
    main()
