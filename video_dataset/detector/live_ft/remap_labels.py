"""Remap live label files written with the alphabetical class order (live_ft/r3_names.json) to the
detector's own class order (data/det/classes_round2.json[:198]). Refuses to run twice (marker file)."""
import json, sys
from pathlib import Path
old = json.load(open(sys.argv[1])); new = json.load(open(sys.argv[2]))[:198]
assert len(old) == 198 and sorted(old) == sorted(new)
m = {i: new.index(n) for i, n in enumerate(old)}
for d in sys.argv[3:]:
    d = Path(d); mark = d / "REMAPPED_TO_DETECTOR_ORDER"
    assert not mark.exists(), f"{d} already remapped"
    n = rows = 0
    for p in sorted((d / "labels" / "train").glob("*.txt")):
        out = []
        for line in p.read_text().splitlines():
            c, rest = line.split(" ", 1)
            out.append(f"{m[int(c)]} {rest}"); rows += 1
        p.write_text("\n".join(out) + ("\n" if out else "")); n += 1
    (d / "names.json").write_text(json.dumps(new))
    mark.write_text("class ids remapped from the alphabetical order to classes_round2.json[:198]\n")
    for c in d.rglob("*.cache"):
        c.unlink()
    print(d, n, "files", rows, "rows")
