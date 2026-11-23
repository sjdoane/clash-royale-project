"""Fetch Clash Royale Fandom wiki pages for every card id the engine registers
and save the parsed level-11 numbers to data/wiki/level11.json.

  .venv/bin/python card_audit/fetch_wiki.py [--refresh]

Raw API responses are cached under data/wiki/raw/ (gitignored); a cached page
is never re-requested unless --refresh. Requests are spaced >= 0.6 s apart
(under the 2 req/s budget) with a neutral User-Agent and no personal data.
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
# CR_ENGINE_PY points at another engine build's python_ai (a scratch tree).
ENGINE_PY = os.environ.get("CR_ENGINE_PY", os.path.join(ROOT, "engine", "python_ai"))
sys.path.insert(0, ENGINE_PY)
from wikiparse import parse_tables, html_text  # noqa: E402
import clash_royale_env as cre  # noqa: E402

RAW = os.path.join(ROOT, "data", "wiki", "raw")
OUT = os.path.join(ROOT, "data", "wiki", "level11.json")
API = "https://clashroyale.fandom.com/api.php"
UA = "cr-robot-card-audit/0.1"
MIN_INTERVAL = 0.6
# Derived from the engine, never copied: a new Evolution or Hero id is picked
# up without an edit here.
EVOLUTION_IDS = tuple(cre.get_evolution_card_ids())
HERO_IDS = tuple(i for i in sorted(cre.get_all_card_ids()) if cre.get_card_info(i)["is_hero"])

# Engine name -> wiki page title where they differ.
PAGE_ALIASES = {
    "Mini PEKKA": "Mini P.E.K.K.A.",
}

_last = [0.0]


def page_url(page, prop="text"):
    q = {"action": "parse", "page": page, "prop": prop, "format": "json", "redirects": "1"}
    return API + "?" + urllib.parse.urlencode(q)


def human_url(page):
    return "https://clashroyale.fandom.com/wiki/" + urllib.parse.quote(page.replace(" ", "_"))


def cache_path(page, prop):
    fn = page.replace("/", "_").replace(" ", "_")
    return os.path.join(RAW, f"{fn}.{'html' if prop == 'text' else prop}.json")


def fetch(page, prop="text", refresh=False):
    path = cache_path(page, prop)
    if os.path.exists(path) and not refresh:
        with open(path) as f:
            return json.load(f)
    wait = MIN_INTERVAL - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(page_url(page, prop), headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
    finally:
        _last[0] = time.time()
    os.makedirs(RAW, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    return data


def page_for(card_id, name):
    base = PAGE_ALIASES.get(name, name)
    if card_id in HERO_IDS:
        b = name[len("Hero "):] if name.startswith("Hero ") else name
        return PAGE_ALIASES.get(b, b) + "/Hero"
    if card_id in EVOLUTION_IDS:
        return base + "/Evolution"
    return base


def extract(html):
    """Attribute tables -> {header: value}; statistics tables -> level-11 row."""
    tables = parse_tables(html)
    out = {"attributes": [], "level11": [], "history": ""}
    for t in tables:
        tid = t["id"] or ""
        rows = t["rows"]
        if not rows:
            continue
        if tid.startswith("unit-attributes-table"):
            # Header row, value row (sometimes repeated pairs).
            attrs = {}
            for i in range(0, len(rows) - 1, 2):
                for h, v in zip(rows[i], rows[i + 1]):
                    attrs[h] = v
            out["attributes"].append({"table": tid, "context": t["context"], "values": attrs})
        elif tid.startswith("unit-statistics-table"):
            header = rows[0]
            for r in rows[1:]:
                if r and r[0].strip() == "11":
                    out["level11"].append({"table": tid, "context": t["context"],
                                           "values": dict(zip(header, r))})
                    break
    text = html_text(html)
    m = re.search(r"\nHistory\[\]\s*\n(.*?)(\n(Trivia|Gallery)\[\]|\nWiki Navigation)", text, re.S)
    if m:
        out["history"] = m.group(1).strip()
    return out


def engine_cards():
    import clash_royale_env as c
    ids = sorted(set(c.get_all_card_ids()) | set(EVOLUTION_IDS) | set(HERO_IDS))
    return [(i, c.get_card_info(i)["name"]) for i in ids]


def main():
    refresh = "--refresh" in sys.argv
    result = {"_source": API, "_fetched": time.strftime("%Y-%m-%d"), "cards": {}}
    for cid, name in engine_cards():
        page = page_for(cid, name)
        entry = {"engine_name": name, "page": page, "url": human_url(page)}
        try:
            data = fetch(page, "text", refresh)
            if "error" in data:
                entry["error"] = data["error"].get("info", "error")
            else:
                entry.update(extract(data["parse"]["text"]["*"]))
        except Exception as e:  # network error: record, keep going
            entry["error"] = repr(e)
        result["cards"][str(cid)] = entry
        flag = entry.get("error") or f"{len(entry.get('level11', []))} L11 rows"
        print(f"{cid:4d} {page:40s} {flag}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
