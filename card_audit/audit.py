"""Compare engine card stats (clash_royale_env.get_card_stats) with the
level-11 wiki numbers in data/wiki/level11.json, and write
card_audit/REPORT.md plus card_audit/audit_results.json.

  .venv/bin/python card_audit/audit.py

Run card_audit/fetch_wiki.py first. A field is a mismatch when the two values
differ by more than TOL (relative), or at all for booleans and counts.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# CR_ENGINE_PY points at another engine build's python_ai (a scratch tree).
ENGINE_PY = os.environ.get("CR_ENGINE_PY", os.path.join(ROOT, "engine", "python_ai"))
sys.path.insert(0, ENGINE_PY)
import clash_royale_env as cre  # noqa: E402

WIKI = os.path.join(ROOT, "data", "wiki", "level11.json")
META = os.path.join(ROOT, "engine", "python_ai", "opponents", "decks", "meta_decks.json")
OUT_JSON = os.path.join(HERE, "audit_results.json")
OUT_MD = os.path.join(HERE, "REPORT.md")
TOL = 0.02

GROUP_A = [125, 168, 129, 15, 40, 72, 33, 7,   # the deck
           24, 6, 25,                          # base versions of its Evo/Hero cards
           140, 144, 175]                      # alternatives
# Derived from the engine, never copied: a new Evolution or Hero id is picked
# up without an edit here.
EVOLUTION_IDS = tuple(cre.get_evolution_card_ids())
HERO_IDS = tuple(i for i in sorted(cre.get_all_card_ids()) if cre.get_card_info(i)["is_hero"])


# ---------------------------------------------------------------- parsing --
def num(text):
    """First number in a wiki cell; a parenthesised number wins
    ('Medium (60)' -> 60, 'Melee: Short (0.5)' -> 0.5)."""
    if text is None:
        return None
    t = text.replace(",", "")
    m = re.search(r"\(([-+]?\d+(?:\.\d+)?)\)", t)
    if m and not re.match(r"^\s*\d", t.split("(")[0] or ""):
        return float(m.group(1))
    m = re.search(r"[-+]?\d+(?:\.\d+)?", t)
    return float(m.group(0)) if m else None


def seconds(text):
    """'1 min 50 sec' -> 110; '30 sec' -> 30; '10 seconds' -> 10."""
    if text is None:
        return None
    m = re.search(r"(\d+)\s*min(?:ute)?s?\s*(?:(\d+(?:\.\d+)?)\s*s)?", text)
    if m:
        return 60 * float(m.group(1)) + float(m.group(2) or 0)
    return num(text)


def range_pair(text):
    """'3.5-11.5' -> (3.5, 11.5); '6' -> (None, 6)."""
    if text is None:
        return None, None
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*$", text)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, num(text)


def pick(d, *keys):
    for k in keys:
        if k in d and num(d[k]) is not None:
            return num(d[k])
    return None


# (card id, field) pairs whose engine and wiki numbers mean different things.
NOT_COMPARABLE = {
    (36, "hit_speed_s"): "engine cooldown spans the whole axe throw (0.9 s hit + 1.5 s axe time)",
    (157, "hit_speed_s"): "engine cooldown spans the whole axe throw (0.9 s hit + 1.5 s axe time)",
    (109, "radius"): "wiki radius is the Goblins' landing spread; engine radius is a hitbox",
    (132, "radius"): "wiki radius is the Goblins' landing spread; engine radius is a hitbox",
    # Stale tables: History records a later change and the engine follows History.
    (70, "hit_speed_s"): "table stale: History sets 1.7 s on 12/1/2026; engine follows History",
    (70, "range"): "table stale: History sets 5.5 on 1/6/2026; engine follows History",
    (138, "hit_speed_s"): "table stale: History sets 1.7 s on 12/1/2026; engine follows History",
    (138, "range"): "table stale: History sets 5.5 on 1/6/2026; engine follows History",
    (93, "hit_speed_s"): "table stale: History sets 4.7 s on 4/8/2026; engine follows History",
    (152, "hit_speed_s"): "table stale: History sets 4.7 s on 4/8/2026; engine follows History",
    (123, "damage"): "Evolution page row disagrees with the base Wall Breakers page (302)",
    (127, "damage"): "Evolution page row disagrees with the base Bomber page (212)",
    (32, "crown_tower_damage"): "official June 2026 notes: 21 per pulse (23% of 92); wiki 22 rounds up",
    # Evolution/Hero pages whose level row disagrees with the base card's page
    # or describes another unit. The engine keeps the base card's value.
    (138, "speed_tier"): "Evolution page says Slow; base Furnace page says Medium",
    (127, "hp"): "Evolution page row disagrees with the base Bomber page",
    (143, "hit_speed_s"): "Evolution page says 1.7 s; base Royal Giant page says 1.8 s",
    (159, "hp"): "Evolution page row disagrees with the base Goblin Giant page",
    (151, "hp"): "Evolution page row does not match the base Electro Dragon page; not trusted",
    (151, "damage"): "Evolution page row does not match the base Electro Dragon page; not trusted",
    (161, "hp"): "Evolution page row is the Barbarians' stats, not the Ram's",
    (161, "damage"): "Evolution page row is the Barbarians' stats, not the Ram's",
    (139, "damage"): "wiki figure is the evolved cage's trapped damage, not an attack",
    (153, "hp"): "wiki 202 is a spawned Goblin's hp",
    (133, "count"): "wiki counts 15 Skeletons; engine adds the General as a 16th body",
    (164, "cost"): "Mirror costs the mirrored card + 1; no fixed cost",
    (180, "cost"): "Mirror costs the mirrored card + 1; no fixed cost",
    # Patches 0014-0019: the engine follows newer official notes than the table.
    (84, "damage"): "table stale: official 26 Aug 2026 notes set 184; engine follows the notes",
    (176, "damage"): "table stale: official 19 Oct 2026 notes set 153 (from 168); engine follows the notes",
    (174, "damage"): "OPEN: Hero page 232 vs base Barbarian Barrel page 215; engine keeps the base value",
    (174, "crown_tower_damage"): "OPEN: Hero page 116 vs base 108; engine keeps the base value",
    (178, "damage"): "Evolution page's Damage cell repeats its Spear Damage (220); base page swing is 384",
    (166, "shield_hp"): "512 is the Triumphant Taunt ability shield (patch 0018), not a base shield",
    (165, "hp"): "two forms; wiki row is the other form (TODO)",
    (165, "damage"): "two forms; wiki row is the other form (TODO)",
    (165, "range"): "two forms; wiki row is the other form (TODO)",
    (121, "hit_speed_s"): "wiki attributes are the Doctor; engine models the Monster (TODO)",
    (121, "range"): "wiki attributes are the Doctor; engine models the Monster (TODO)",
    (121, "speed_tier"): "wiki attributes are the Doctor; engine models the Monster (TODO)",
    (121, "targets_air"): "wiki attributes are the Doctor; engine models the Monster (TODO)",
}


def merged(entry, key):
    """First-table-wins merge of the attribute tables or level-11 rows."""
    out = {}
    for t in entry.get(key, []):
        for k, v in t["values"].items():
            out.setdefault(k, v)
    return out


def primary_attrs(entry):
    a = entry.get("attributes", [])
    return a[0]["values"] if a else {}


def history_changes(history):
    """[(date, sentence)] for dated History lines."""
    out = []
    for line in history.split("\n"):
        m = re.match(r"\s*On (\d+/\d+/\d{4})", line)
        if m:
            out.append((m.group(1), line.strip()))
    return out


def pct_changes(history, stat_words, since_year=2025):
    """Percent changes to a stat in History since `since_year`."""
    res = []
    for date, s in history_changes(history):
        if int(date.split("/")[-1]) < since_year:
            continue
        low = s.lower()
        if not any(w in low for w in stat_words):
            continue
        for m in re.finditer(r"(increased|decreased)[^.]*?by (\d+(?:\.\d+)?)%", low):
            p = float(m.group(2)) / 100.0
            res.append((date, (1 + p) if m.group(1) == "increased" else (1 - p), s))
    return res


# ------------------------------------------------------------- comparing --
def rel_diff(e, w):
    if w == 0:
        return 0.0 if e == 0 else 1.0
    return abs(e - w) / abs(w)


def compare_stats(card_id, name, stats, entry, is_evo_entry=False):
    """Yield (field, engine, wiki, note) mismatches for one stats dict."""
    attrs = primary_attrs(entry)
    allattrs = merged(entry, "attributes")
    l11 = merged(entry, "level11")
    hist = entry.get("history", "")
    arche = stats["archetype"]
    is_spell = arche == "Spell"
    is_building = arche == "DefensiveBuilding"
    rows = []
    base = re.sub(r"^Hero ", "", name)

    def check(field, eng, wiki_val, stat_words=(), tol=TOL, note=""):
        if eng is None or wiki_val is None:
            return
        if (card_id, field) in NOT_COMPARABLE:
            rows.append(dict(field=field, engine=eng, wiki=wiki_val, status="not comparable",
                             note=NOT_COMPARABLE[(card_id, field)]))
            return
        if rel_diff(eng, wiki_val) <= tol:
            return
        # History may explain a stale table: a dated % change whose result
        # matches the engine.
        for date, factor, sent in pct_changes(hist, stat_words) if stat_words else []:
            if rel_diff(eng, wiki_val * factor) <= tol:
                rows.append(dict(field=field, engine=eng, wiki=wiki_val, status="explained",
                                 note=f"table predates {date}: x{factor:.2f} = {wiki_val*factor:.0f}"))
                return
        rows.append(dict(field=field, engine=eng, wiki=wiki_val, status="mismatch", note=note))

    check("cost", stats["cost"], num(attrs.get("Cost")))
    if not is_spell:
        spawner = is_building and ("Spawn Speed" in attrs or "Production Speed" in attrs)
        hp = pick(l11, f"{base} Hitpoints", "Hitpoints")
        life = seconds(attrs.get("Lifetime"))
        lost = pick(l11, f"{base} Hitpoints lost per second", "Hitpoints lost per second")
        if spawner and f"{base} Hitpoints" not in l11 and lost and life:
            hp = round(lost * life)   # the table's Hitpoints column is the spawned unit's
        check("hp", stats["hp"], hp, ("hitpoint",))
        dmg = pick(l11, f"{base} Damage", f"{base} Area Damage", "Damage", "Area Damage")
        if not spawner and (stats["damage"] > 0 or dmg):
            check("damage", stats["damage"], dmg, ("damage",))
        hs = num(attrs.get("Hit Speed"))
        if hs and stats["attack_cooldown_ticks"] > 0:
            check("hit_speed_s", stats["hit_speed_s"], hs, ("hit speed", "attack speed", "attack time"))
        rng = attrs.get("Range") or allattrs.get("Range")
        if rng and stats["attack_range"] > 0:
            lo, hi = range_pair(rng)
            check("range", stats["attack_range"], hi, ("range",))
            if lo is not None:
                check("min_range", stats["min_attack_range"], lo, ("range",))
        if not is_building:
            sp = num(attrs.get("Speed"))
            if sp and stats["speed_wiki_tiles_per_min"] > 0:
                check("speed_tier", round(stats["speed_wiki_tiles_per_min"], 1), sp, ("speed",), tol=0.05)
            cnt = attrs.get("Count")
            if cnt and cnt.strip().startswith("x"):
                c = num(cnt)
                if c and c != stats["count"]:
                    rows.append(dict(field="count", engine=stats["count"], wiki=c, status="mismatch", note=""))
        dep = num(attrs.get("Deploy Time"))
        check("deploy_s", stats["deploy_ticks"] / 10.0, dep, ("deploy",))
        tgt = attrs.get("Target", "")
        if "Air" in tgt and not stats["targets_air"]:
            rows.append(dict(field="targets_air", engine=False, wiki=True, status="mismatch", note=tgt))
        if tgt.strip() in ("Ground", "Buildings") and stats["targets_air"]:
            rows.append(dict(field="targets_air", engine=True, wiki=False, status="mismatch", note=tgt))
        if is_building:
            check("lifetime_s", stats["lifetime_ticks"] / 10.0, life, ("lifetime",))
        spl = num(attrs.get("Splash Radius"))
        if spl and not stats["spawn_effect_radius"]:
            check("splash_radius", stats["splash_radius"], spl, ("radius",))
        death = stats.get("death_effect") or {}
        dd = num(l11.get("Death Damage"))
        if dd and death.get("damage") is not None:
            check("death_damage", death["damage"], dd, ("death damage", "damage"))
        sh = num(l11.get("Shield Hitpoints"))
        if sh:
            check("shield_hp", stats["shield_hp"], sh, ("shield",))
        cd = num(l11.get("Charge Damage"))
        if cd and stats["charge_multiplier"] > 1:
            check("charge_damage", round(stats["damage"] * stats["charge_multiplier"]), cd, ("charge", "damage"))
        spd = num(l11.get("Spawn Damage"))
        if spd and stats["spawn_effect_damage"]:
            check("spawn_damage", stats["spawn_effect_damage"], spd, ("spawn damage", "damage"))
    else:
        dmg = pick(l11, f"{base} Area Damage", "Area Damage", "Damage")
        if dmg is not None and stats["damage"] > 0:
            check("damage", stats["damage"], dmg, ("damage",))
        rad = num(attrs.get("Radius"))
        if rad and stats["spell_radius"] > 0 and (stats["damage"] > 0 or (card_id, "radius") in NOT_COMPARABLE):
            check("radius", stats["spell_radius"], rad, ("radius",))
        ct = num(l11.get("Crown Tower Damage"))
        if ct is not None and stats["damage"] > 0:
            eng_ct = stats["spell_crown_tower_damage"]
            eng_ct = stats["damage"] if eng_ct < 0 else eng_ct
            check("crown_tower_damage", eng_ct, ct, ("crown tower",))
        if stats["roll_range"] > 0:
            check("roll_range", stats["roll_range"], num(attrs.get("Range")), ("range",))
            check("roll_width", stats["roll_width"], num(attrs.get("Width")), ("width",))
    return rows


def wiki_inconsistencies(entry):
    """Derived-column cross-checks (typos such as the Trusty Turret's HP)."""
    out = []
    for t in entry.get("level11", []):
        v = t["values"]
        for k in list(v):
            if not k.endswith("Hitpoints lost per second") and not k.endswith("Hitpoints Lost per Second"):
                continue
            prefix = k[: -len("Hitpoints lost per second")]
            hpk = prefix + "Hitpoints"
            life = None
            for a in entry.get("attributes", []):
                for ak, av in a["values"].items():
                    if "Lifetime" in ak:
                        life = num(av)
            hp, lost = num(v.get(hpk)), num(v.get(k))
            if hp and lost and life and rel_diff(hp, lost * life) > 0.03:
                out.append(f"{hpk} {hp:g} vs {k} {lost:g} x lifetime {life:g} s = {lost*life:g}")
    return out


# ------------------------------------------------------------------ groups --
def meta_ids(base_by_name):
    meta = json.load(open(META))
    ids = set()
    for d in meta["decks"]:
        for c in d["cards"]:
            if c in base_by_name:
                ids.add(base_by_name[c])
    return ids


def nav_live_variants(wiki):
    """Every '<Card>(Evolution)' / '<Card>(Hero)' in the wiki navigation box."""
    raw = os.path.join(ROOT, "data", "wiki", "raw", "Cannon_Evolution.html.json")
    sys.path.insert(0, HERE)
    from wikiparse import html_text
    t = html_text(json.load(open(raw))["parse"]["text"]["*"])
    i = t.find("Troop Cards")
    j = t.find("Tower Troops", i) if t.find("Tower Troops", i) > 0 else i + 8000
    nav = t[i:j]
    evo, hero = set(), set()
    for item in re.split(r"\s*•\s*|\n", nav):
        item = item.strip()
        nm = re.sub(r"\(.*", "", item).strip()
        if not nm:
            continue
        if "(Evolution)" in item:
            evo.add(nm)
        if "(Hero)" in item:
            hero.add(nm)
    return evo, hero


# -------------------------------------------------------------------- main --
MECHANICS = """\
## Mechanics missing or wrong

Status is after patch 0003. "Fixed" means implemented in the engine;
"TODO" means listed only.

Group (a), our deck. All of these were implemented:
- **Hero Musketeer, Trusty Turret.** Fixed: turret 1536 HP, 140 damage at a
  0.5 s hit speed, range 4.0, air and ground, 10 s lifetime, 2 s deploy.
  It lands 3 tiles forward after a 1 s delay, with 204 landing damage, and
  the ability can be used once per deployment (it was a 22 s cooldown).
  - Before: 200 HP, 90 damage, spawned at her feet, 220-tick cooldown, no limit.
  - The wiki's 721 turret HP is a typo: 153.6 HP lost per second over 10 s is 1536.
  - Not on the wiki: the landing radius (2.0 assumed) and whether she pauses
    during the delay (she does not in the engine).
- **Evolved Cannon, deploy barrage.** Fixed: 9 impacts of 304 (89 to Crown
  Towers), radius 2, knockback 1.0.
  - Before: the barrage never fired (the building factory never called the
    deploy-burst code), and it was one 202-damage burst.
  - Assumed: two rows (5 + 4) across the arena in front of the Cannon. The
    12/2025 -8% may not be reflected in the wiki table's 304.
- **Ice Spirit and Evolved Ice Spirit.** Fixed: the hit now splashes (1.5,
  evolved 2.0) and freezes every splash victim, not only the primary target
  (`CardStats::withSplashOnHit`); freeze 1.1 s (was 1.0 s). HP stays 215, the
  table value (L1 83 x 2.585).
- **Ice Golem.** Fixed: damage and death damage 92 (was 84); death nova slows
  30% for 2.5 s (was 35% for 3 s).
- **Hero Ice Golem.** Fixed: range 1.2; Snowstorm single use, 1 s start delay,
  final blast slows rather than stuns. Its 84 damage follows the Hero page.
- **Evolved Skeletons.** Fixed: every spawned Evolved Skeleton now also
  multiplies (before only the three original ones did); cap 8 alive.
- **Fireball and The Log.** Fixed: Crown Tower damage 159 and 35 (was full).
- **First-hit speed.** TODO: no general first-hit delay in the engine.

Groups (b) and (c), numbers fixed in patch 0003 (each line commented in
`CardRegistry.h` with its wiki page): Crown Tower damage for 13 spells; building
lifetimes (Tesla 25 s, Goblin Cage 20 s, Goblin Drill 10 s, Elixir Collector
110 s; `CardStats::withLifetime`, every building was 30 s); deploy times (Golem
3 s, Princess 1.2 s, X-Bow and Mortar 3.5 s); Mortar hit 4.7 s (History);
splash radii (Bomber, Valkyrie, Dark Prince, Sparky, Princess, Witch, Heal
Spirit, Mortar); speed tiers, ranges, hit speeds, HP and damage; costs (Void,
Monk 5); 9 Evolutions to 1 cycle.

Groups (b) and (c), mechanics listed as TODO only:
- Hero Knight's 512 shield on Triumphant Taunt.
- Void: the 4/8/2026 rework beyond its cost (hit speed and tier damage).
- Goblin Curse damage amplification and slow; Earthquake's building bonus.
- Spirit Empress's second form and Goblinstein's Doctor (two-unit cards).
- Knockback on most spells and deploy effects; projectile travel time.
- Bomber, Mortar and other splash shots apply splash at the target, not at a
  thrown landing point.
"""


def main():
    wiki = json.load(open(WIKI))["cards"]
    base_ids = cre.get_all_card_ids()
    base_by_name = {cre.get_card_info(i)["name"]: i for i in base_ids}
    all_ids = sorted(set(base_ids) | set(EVOLUTION_IDS))
    grp_b = meta_ids(base_by_name) - set(GROUP_A)

    results = {}
    for cid in all_ids:
        entry = wiki.get(str(cid))
        if not entry or entry.get("error"):
            results[cid] = dict(name=cre.get_card_info(cid)["name"], error=(entry or {}).get("error", "no page"))
            continue
        s = cre.get_card_stats(cid)
        stats = s["evolved"] if s["is_evolution"] else s["base"]
        rows = compare_stats(cid, s["name"], stats, entry)
        if s["is_evolution"]:
            cyc = num(merged(entry, "attributes").get("Cycles"))
            if cyc and cyc != s["evolution_cycle_threshold"]:
                rows.append(dict(field="cycles", engine=s["evolution_cycle_threshold"], wiki=cyc,
                                 status="mismatch", note=""))
        for r in rows:  # rows built outside check() (cycles, counts) honour it too
            if r["status"] == "mismatch" and (cid, r["field"]) in NOT_COMPARABLE:
                r.update(status="not comparable", note=NOT_COMPARABLE[(cid, r["field"])])
        results[cid] = dict(name=s["name"], page=entry["page"], url=entry["url"], rows=rows,
                            wiki_inconsistencies=wiki_inconsistencies(entry),
                            history_2026=[x for _, x in history_changes(entry.get("history", ""))
                                          if x and "/2026" in x.split(",")[0]])

    def group(cid):
        return "a" if cid in GROUP_A else ("b" if cid in grp_b else "c")

    with open(OUT_JSON, "w") as f:
        json.dump({str(k): dict(v, group=group(k)) for k, v in results.items()}, f, indent=1)

    # ---- markdown
    def label(cid):
        r = results[cid]
        tag = " (Evolution)" if cid in EVOLUTION_IDS else ""
        return f"{r['name']}{tag} [{cid}]"

    lines = ["# Card stats audit: engine vs wiki (level 11)", "",
             "Generated by `card_audit/audit.py` from `data/wiki/level11.json` "
             "(Fandom wiki, fetched by `card_audit/fetch_wiki.py`) and "
             "`clash_royale_env.get_card_stats`. A mismatch is a relative difference "
             f"over {TOL:.0%}. *Explained* rows are cases where the table predates a "
             "dated History change and the engine already matches the post-change "
             "value; these are not counted.", "",
             "Values are shown **after patch 0003** unless the section says *before*.", ""]
    for g, title in (("a", "Group (a): our deck, its base cards and the alternatives"),
                     ("b", "Group (b): cards in meta_decks.json"),
                     ("c", "Group (c): everything else")):
        ids = [c for c in all_ids if group(c) == g] + ([c for c in HERO_IDS] if g == "a" and False else [])
        if g == "a":
            ids = [c for c in GROUP_A if c in results or c in HERO_IDS]
        mm = [(c, r) for c in ids for r in results.get(c, {}).get("rows", []) if r["status"] == "mismatch"]
        lines += [f"## {title}", "", f"{len(mm)} mismatches across {len(set(c for c, _ in mm))} cards.", "",
                  "| card | field | engine | wiki | note | source |", "|---|---|---|---|---|---|"]
        for c in ids:
            r = results.get(c)
            if r is None:
                continue
            if "error" in r:
                lines.append(f"| {label(c)} | (no wiki data) | | | {r['error']} | |")
                continue
            for row in r["rows"]:
                if row["status"] == "mismatch" or g == "a":
                    lines.append(f"| {label(c)} | {row['field']} | {row['engine']:g} | {row['wiki']:g} | "
                                 f"{row['status']}{': ' + row['note'] if row['note'] else ''} | [{r['page']}]({r['url']}) |"
                                 if not isinstance(row['engine'], bool) else
                                 f"| {label(c)} | {row['field']} | {row['engine']} | {row['wiki']} | {row['note']} | [{r['page']}]({r['url']}) |")
            for w in r["wiki_inconsistencies"]:
                lines.append(f"| {label(c)} | wiki typo? | | | {w} | [{r['page']}]({r['url']}) |")
        lines.append("")

    # Heroes: hand-checked (their ability stats live in secondary tables).
    lines += ["## Heroes (ability checks)", ""]
    for hid in HERO_IDS:
        e = wiki.get(str(hid), {})
        s = cre.get_card_stats(hid)["base"]
        rows = compare_stats(hid, s["name"], s, e) if e and not e.get("error") else []
        mm = [r for r in rows if r["status"] == "mismatch"]
        lines.append(f"- **{s['name']} [{hid}]** ability cost {s['ability_cost']:g}, cooldown "
                     f"{s['ability_cooldown_ticks']} ticks, uses {s['ability_uses_limit']}; "
                     f"{len(mm)} base-stat mismatches"
                     + ("".join(f"; {r['field']} {r['engine']:g} vs {r['wiki']:g}" for r in mm
                                if not isinstance(r['engine'], bool)))
                     + (f" ([wiki]({e.get('url')}))" if e else ""))
    lines.append("")

    before_path = os.path.join(HERE, "audit_results_before.json")
    if os.path.exists(before_path):
        before = json.load(open(before_path))
        lines += ["## Before patch 0003: every mismatch found", "",
                  "Snapshot `card_audit/audit_results_before.json`, taken on the unpatched "
                  "engine (patches 0001 and 0002 only). *After* is the engine value now; "
                  "rows whose *after* still differs are explained in the notes above "
                  "(not comparable) or listed as TODO.", ""]
        for g in "abc":
            rows = [(int(c), k, r) for c, k in before.items() if k.get("group") == g
                    for r in k.get("rows", []) if r["status"] == "mismatch"]
            rows.sort(key=lambda x: x[0])
            lines += [f"### Group ({g}): {len(rows)} mismatches before", "",
                      "| card | field | engine before | wiki | after | source |", "|---|---|---|---|---|---|"]
            for c, k, r in rows:
                now = next((x for x in results.get(c, {}).get("rows", []) if x["field"] == r["field"]), None)
                nv = now["engine"] if now else None
                nv = f"{nv:g}" if isinstance(nv, float) else nv
                after = "fixed" if now is None else (f"{nv}" + (f" ({now['status']})" if now["status"] != "mismatch" else " (TODO)"))
                ev = r["engine"]
                ev = f"{ev:g}" if isinstance(ev, float) else ev
                lines.append(f"| {k['name']} [{c}] | {r['field']} | {ev} | {r['wiki']:g} | {after} | [{k['page']}]({k['url']}) |")
            lines.append("")

    lines.append(MECHANICS)

    evo_live, hero_live = nav_live_variants(wiki)
    evo_engine = {cre.get_card_info(i)["name"] for i in EVOLUTION_IDS}
    hero_engine = {cre.get_card_info(i)["name"].replace("Hero ", "") for i in HERO_IDS}
    lines += ["## Live Evolutions and Heroes the engine lacks", "",
              "From the wiki's card navigation box (troop list; spells and buildings "
              "checked by hand).", "",
              "- Missing Evolutions: " + ", ".join(sorted(evo_live - evo_engine)) or "none",
              "- Missing Heroes: " + ", ".join(sorted(hero_live - hero_engine)) or "none", ""]
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines))
    counts = {g: sum(1 for c in all_ids if group(c) == g
                     for r in results.get(c, {}).get("rows", []) if r["status"] == "mismatch")
              for g in "abc"}
    print("mismatches per group:", counts)
    print("missing evolutions:", sorted(evo_live - evo_engine))
    print("missing heroes:", sorted(hero_live - hero_engine))


if __name__ == "__main__":
    main()
