"""SudhishNeel/clash-royale-detection class names -> our detector classes.

    .venv-det/bin/python video_dataset/detector/ext_classmap.py      # writes data/ext/classmap_report.json

Their taxonomy (README / LABELLING.md of the HF dataset, 208 classes + `drop`)
names a class after the on-field unit, like ours, with `<card>-evolution` for
Evolutions and `<card>-hero` for Heroes. Ours (classes.py) uses `-ev1` for
Evolutions and the same `-hero` suffix, with the card key's own spelling
(`goblins-hero` for the hero of the Goblins card).

The map is name rules + a few explicit renames; nothing is fuzzy-matched.
Their classes that are not detector targets for us (towers, tower troops,
UI / effects, `drop`) are listed separately from the real gaps (units we have
no class for), which are the proposals to ADD to classes.py.

Their pending splits are labelled as the base card in their data (README,
"Pending splits"): `bowler` may be Hero Bowler, `barbarian-barrel` may be the
hero, `minion` may be Evo Minion. Hero units labelled before 2026-10-08 may
also still carry the base class (LABELLING.md, data note). They are mapped to
our base class and flagged `noisy_base` here so a consumer can choose.
"""
from __future__ import annotations

import json
from pathlib import Path

from classes import UI_CLASSES

REPO = Path(__file__).resolve().parents[2]
DET = REPO / "data" / "det"
# classes.unit_classes() frozen at build time: the training pod has neither the
# plays parquet nor the card art that unit_classes() derives the list from.
FROZEN = DET / "classes_191.json"
EXT = REPO / "data" / "ext"
SN = EXT / "sudhishneel"
REPORT = EXT / "classmap_report.json"

# Their name -> our name, where the name rules below do not already give it.
RENAME = {
    "goblin-hero": "goblins-hero",     # hero of the Goblins card; our card key is plural
    "baby-goblin": "baby-goblins",     # our class comes from the card key (classes.card_units)
}
# Their classes we deliberately do not train on (not detector unit targets for us).
NOT_TARGET = {
    # towers + tower troops: board towers, our UI_CLASSES, not in the 191 unit classes
    "king-tower": "tower", "queen-tower": "tower", "cannoneer-tower": "tower",
    "dagger-duchess-tower": "tower", "royal-chef": "tower-troop", "royal-guardian": "tower-troop",
    # UI / effects: KataCR_DROP in classes.py treats these as background
    "clock": "ui", "elixir": "ui", "dirt": "effect", "skeleton-king-skill": "effect",
    "drop": "not-a-unit",
    # event-only cards (classes.EVENT_ONLY)
    "party-hut": "event", "party-rocket": "event", "terry": "event",
}
# Mapped, but their label can also be a different form of ours (see module doc).
NOISY_BASE = {
    "bowler": "may include bowler-hero (pending split)",
    "barbarian-barrel": "may include barbarian-barrel-hero (pending split)",
    "minion": "may include minion-ev1 (pending split)",
}


def our_classes() -> list[str]:
    try:
        from classes import unit_classes  # noqa: PLC0415
        cls = unit_classes()
    except (FileNotFoundError, OSError, ImportError):
        return json.loads(FROZEN.read_text())
    if not FROZEN.exists() or json.loads(FROZEN.read_text()) != cls:
        FROZEN.parent.mkdir(parents=True, exist_ok=True)
        FROZEN.write_text(json.dumps(cls, indent=0))
    return cls


def their_classes() -> list[str]:
    """208 names + drop, in their permanent id order. classes.txt has only the
    first 187 (v1 release); the full list is the `crops` ClassLabel in README.md."""
    import yaml  # noqa: PLC0415
    readme = (SN / "dataset" / "README.md").read_text()
    front = yaml.safe_load(readme.split("---")[1])
    crops = next(c for c in front["dataset_info"] if c["config_name"] == "crops")
    feat = next(f for f in crops["features"] if f["name"] == "label")
    names = feat["dtype"]["class_label"]["names"]
    return [names[str(i)] if str(i) in names else names[i] for i in range(len(names))]


def to_ours(name: str, ours: set[str]) -> str | None:
    if name in NOT_TARGET:
        return None
    n = RENAME.get(name, name)
    if n.endswith("-evolution"):
        n = n[: -len("-evolution")] + "-ev1"
    return n if n in ours else None


def build() -> dict:
    ours_list = our_classes()
    ours = set(ours_list)
    theirs = their_classes()
    mapped, unmapped, not_target = {}, [], {}
    for t in theirs:
        o = to_ours(t, ours)
        if o is not None:
            mapped[t] = o
        elif t in NOT_TARGET:
            not_target[t] = NOT_TARGET[t]
        else:
            unmapped.append(t)
    hit = set(mapped.values())
    no_source = [c for c in ours_list if c not in hit]
    # proposals: their unit classes we lack, named in our convention
    def proposed(t):
        if t == "skeleton-army-evolution":
            return "skeleton-army-ev1"   # General Gerry; NOT our skeleton-ev1 (LABELLING.md section 1)
        return t[: -len("-evolution")] + "-ev1" if t.endswith("-evolution") else t
    add = [{"theirs": t, "propose": proposed(t),
            "kind": "hero" if t.endswith("-hero") else "evo" if t.endswith("-evolution") else "unit"}
           for t in unmapped]
    return {
        "source": "SudhishNeel/clash-royale-detection README crops ClassLabel (208 + drop)",
        "n_theirs": len(theirs), "n_ours": len(ours_list),
        "n_mapped": len(mapped), "n_unmapped_units": len(unmapped), "n_not_target": len(not_target),
        "n_ours_no_source": len(no_source),
        "mapped": mapped,
        "renamed": {k: v for k, v in mapped.items() if k != v},
        "noisy_base": {k: v for k, v in NOISY_BASE.items() if k in mapped},
        "theirs_unmapped_propose_add": add,
        "theirs_not_target": not_target,
        "ours_with_no_source": no_source,
        "ui_classes_ours_not_in_detector": UI_CLASSES,
        "notes": [
            "Hero units labelled before 2026-10-08 may carry the base class in their data (LABELLING.md).",
            "Their README lists no examples yet for terry, zap, zap-evolution, rocket, vines, heal-spirit.",
            "classes.py not changed; adding a proposed class means appending it to unit_classes() "
            "(class ids shift, so retrain from scratch or remap heads).",
        ],
    }


def load_map(extend: bool = False) -> dict[str, str]:
    """their name -> our name. extend=True also maps their unmapped unit classes to
    the proposed names appended after ours (see target_classes)."""
    rep = build()
    m = dict(rep["mapped"])
    if extend:
        m.update({a["theirs"]: a["propose"] for a in rep["theirs_unmapped_propose_add"]})
    return m


def target_classes(extend: bool = False) -> list[str]:
    """Training class list: our 191 (ids unchanged), plus with extend=True the proposed
    additions appended in their id order (ids 191..). Names, not ids, are what
    detector.py / verify.py consume, so appended classes need no downstream change."""
    cls = our_classes()
    if extend:
        cls = cls + [a["propose"] for a in build()["theirs_unmapped_propose_add"]]
    return cls


def main():
    rep = build()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=1))
    print({k: v for k, v in rep.items() if k.startswith("n_")})
    print("unmapped:", [a["theirs"] for a in rep["theirs_unmapped_propose_add"]])
    print("ours no source:", rep["ours_with_no_source"])
    print("->", REPORT)


if __name__ == "__main__":
    main()
