"""Our player's plays from a recorded match video, read off the card tray.

    .venv/bin/python video_dataset/tray.py VIDEO GEO.json [--t0 S --t1 S] [--out plays.json]

The tray is the same in every video (549x976 frame via capture/wda.IosGeometry):
four hand slots, the next card and the elixir bar. A play shows as

  - the slot going EMPTY (flat dark blue, grey std ~7 against 55-75 for a card)
    for about a second before the next card slides in;
  - the elixir bar dropping by the card's cost at the same moment.

Card identity: the slot crops of one video are clustered (contrast-normalised,
so a dimmed unaffordable card joins its bright self), and each cluster is then
named once by its pooled correlation with RoyaleAPI card art in every form
(data/card_art/cards-150; -ev1, -hero). One crop against the art alone is too
weak (Hog vs Cannon 0.49 / 0.48); a cluster of hundreds is not. Clusters are
assigned to distinct deck cards, so each card is named once.

Output per play: t (s in the video), slot, card_key, cost, elixir before and
after, `elixir_ok` (the drop matches the cost within 1), `refill` (the card
that slid in) and `confident` (an elixir drop or a different refill).

Hand-checked on 4s1o1sk-hPM 40-220 s against the in-game name labels: 41
events, 37 real; every real play had the right base card; all clear false
positives (a banner over the tray, a cancelled drag) are non-`confident`; one
play missed. Evolution: a charged Evo card sits in the hand in a purple frame (diamond tab
on top), so `evolved` is read per play from the frame, not the art, and the
key gets -ev1 exactly when the played copy was the Evolution. Hero forms
(Hero Musketeer) come from the art and the deck; the hero ability is on the
board, not in the tray (detector). Placement tiles
come from the unit detector (video_dataset/detector/), joined on t.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine" / "perception"))

from capture.wda import IosGeometry  # noqa: E402
from readers.elixir import ElixirBarReader  # noqa: E402

ART = ROOT / "data" / "card_art" / "cards-150"
# 2.6 cards in every form the game has (RoyaleAPI keys) and their cost.
DECK_FORMS = {
    "hog-rider": ["hog-rider"],
    "musketeer": ["musketeer", "musketeer-ev1", "musketeer-hero"],
    "cannon": ["cannon", "cannon-ev1"],
    "ice-golem": ["ice-golem", "ice-golem-hero"],
    "skeletons": ["skeletons", "skeletons-ev1"],
    "ice-spirit": ["ice-spirit", "ice-spirit-ev1"],
    "the-log": ["the-log"],
    "fireball": ["fireball"],
}
COST = {"hog-rider": 4, "musketeer": 4, "cannon": 3, "ice-golem": 2,
        "skeletons": 1, "ice-spirit": 1, "the-log": 2, "fireball": 4}

# Tray geometry in the 549x976 frame (measured on 4s1o1sk-hPM).
SLOT_X = [(128, 216), (230, 320), (333, 425), (435, 527)]
SLOT_Y, SLOT_H = 812, 113          # resting card; a selected card lifts ~20 px
LIFT_SEARCH = range(788, 816, 3)
ELIXIR_ROI = (145, 952, 388, 6)
EMPTY_STD = 20.0                   # grey std of the inner slot below this = empty
JOIN_CORR = 0.80                   # crop joins a cluster above this correlation
LEFTOVER_CORR = 0.75               # a secondary cluster takes a primary's name above this
CLAIM_MIN = 0.45                   # art score below which a cluster names nothing (resting: 0.60-0.94)
# A charged Evolution sits in the hand in a purple frame with a diamond tab on
# top; the same card uncharged has the plain frame. Fraction of purple pixels in
# the frame ring: 0.38-0.48 charged, <= 0.05 otherwise (4s1o1sk-hPM, 8 frames).
EVO_PURPLE = 0.20
# The "next" card preview (Suivant) left of the hand: the card that slides into
# an emptied slot is the one shown here just before.
NEXT_BOX = (28, 908, 74, 964)      # x0, y0, x1, y1
MIN_VOTES = 3                      # fewer named crops than this: trust the preview
MIN_SEG = 3                        # samples (0.3 s) a card must rest to count as dealt


def feat(img: np.ndarray) -> np.ndarray:
    """Card art above the cost badge, colour, contrast-normalised, flattened."""
    h, w = img.shape[:2]
    c = img[int(h * .16):int(h * .58), int(w * .22):int(w * .78)]
    # Coarse and blurred: a resting card sits a few pixels off from one deal to
    # the next, and at 32x36 that alone split one card into several clusters.
    c = cv2.GaussianBlur(c, (0, 0), 2.5)
    c = cv2.resize(c, (12, 14), interpolation=cv2.INTER_AREA).astype(np.float32)
    c = (c - c.mean()) / (c.std() + 1e-6)
    return c.ravel()


def art_refs() -> dict[str, np.ndarray]:
    refs = {}
    for forms in DECK_FORMS.values():
        for k in forms:
            a = cv2.imread(str(ART / f"{k}.png"), cv2.IMREAD_UNCHANGED)
            if a is None:
                continue
            al = a[:, :, 3:] / 255.0
            rgb = (a[:, :, :3] * al + np.array([120, 60, 20]) * (1 - al)).astype(np.uint8)
            refs[k] = feat(rgb)
    return refs


@dataclass
class Sample:
    t: float
    elixir: float | None
    empty: list[bool]
    cluster: list[int]          # -1 when empty
    evo: list[bool]             # slot shows the charged-Evolution frame
    next_cluster: int = -1
    clock: float | None = None          # seconds left on the match clock (1 Hz)      # online cluster of the next-card preview


def evo_frame(fr: np.ndarray, x0: int, x1: int) -> float:
    """Fraction of purple pixels in the ring around a resting slot (tab included)."""
    c = fr[SLOT_Y - 25:SLOT_Y + SLOT_H + 4, max(0, x0 - 6):x1 + 6]
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1], hsv[..., 2]
    ring = np.zeros(c.shape[:2], bool)
    ring[:, :10] = ring[:, -10:] = ring[:22, :] = ring[-8:, :] = True
    purple = (h >= 125) & (h <= 160) & (s > 90) & (v > 110)
    return float(purple[ring].mean())


def scan(video: Path, geo: IosGeometry, t0: float, t1: float | None, hz: float = 10.0):
    """Sample the tray at `hz`; cluster slot crops online. Returns samples, centroids."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
    step = max(1, int(round(fps / hz)))
    el = ElixirBarReader(ELIXIR_ROI)
    sums: list[np.ndarray] = []
    counts: list[int] = []
    samples: list[Sample] = []
    nsums: list[np.ndarray] = []           # the preview's own clusters
    i = 0
    while True:
        if not cap.grab():
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t1 is not None and t > t1:
            break
        i += 1
        if (i - 1) % step:
            continue
        ok, f = cap.retrieve()
        if not ok:
            continue
        try:
            fr = geo.frame(f)
        except Exception:                                   # noqa: BLE001
            continue
        r = el.read(fr)
        empty, clus, evo = [], [], []
        for x0, x1 in SLOT_X:
            inner = fr[SLOT_Y:SLOT_Y + SLOT_H, x0 + 8:x1 - 8]
            e = float(cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY).std()) < EMPTY_STD
            empty.append(e)
            evo.append(not e and evo_frame(fr, x0, x1) >= EVO_PURPLE)
            if e:
                clus.append(-1)
                continue
            # Resting position only. A selected card is lifted and glowing; its crop
            # forms its own clusters, left unnamed, so the played card is the last
            # RESTING identity seen in the slot.
            v = feat(fr[SLOT_Y:SLOT_Y + SLOT_H, x0:x1])
            best = (-2.0, -1)
            for c, sm in enumerate(sums):
                cen = sm / counts[c]
                sc = float(v @ cen) / (np.linalg.norm(cen) * len(v) ** .5 + 1e-6)
                if sc > best[0]:
                    best = (sc, c)
            sc, c = best
            if sc < JOIN_CORR:
                sums.append(v.copy())
                counts.append(1)
                c = len(sums) - 1
            else:
                sums[c] += v
                counts[c] += 1
            clus.append(c)
        conf = r.confidence if r.value is not None else 0.0
        nx0, ny0, nx1, ny1 = NEXT_BOX
        nf = feat(fr[ny0:ny1, nx0:nx1])
        nb = (-2.0, -1)
        for c, sm in enumerate(nsums):
            sc = float(nf @ sm) / (np.linalg.norm(sm) * len(nf) ** .5 + 1e-6)
            if sc > nb[0]:
                nb = (sc, c)
        if nb[0] < JOIN_CORR:
            nsums.append(nf.copy())
            nb = (1.0, len(nsums) - 1)
        else:
            nsums[nb[1]] += nf
        clock = None
        if len(samples) % CLOCK_EVERY == 0:
            v, c = _hud().read_timer(fr)
            clock = float(v) if v is not None and c > 0 else None
        samples.append(Sample(t, r.value if conf > 0.5 else None, empty, clus, evo, nb[1], clock))
    cents = [s / n for s, n in zip(sums, counts)]
    return samples, cents, counts


def name_clusters(cents, counts, refs, min_count: int = 20,
                  evo_frac: dict[int, float] | None = None) -> dict[int, str]:
    """Assign each sizeable cluster a deck card (one per base card), then its best form.

    A charged Evolution shows the Evolution's own ART (Evo Cannon is a purple
    cannon) inside the purple frame, and the base art once the charge is spent,
    so one card has two resting looks. Clusters mostly in the purple frame are
    named among the -ev1 forms, separately from the plain ones.
    """
    evo_frac = evo_frac or {}
    evo_big = [c for c, n in enumerate(counts) if n >= min_count and evo_frac.get(c, 0) > 0.5]
    names_evo: dict[int, str] = {}
    taken_evo: set[str] = set()
    pairs = sorted(((float((cents[c] / (np.linalg.norm(cents[c]) / len(cents[c]) ** .5 + 1e-6)
                             * refs[k]).mean()), c, k)
                    for c in evo_big for k in refs if k.endswith("-ev1")), reverse=True)
    for sc, c, k in pairs:
        if c in names_evo or k in taken_evo:
            continue
        names_evo[c] = k
        taken_evo.add(k)
    big = [c for c, n in enumerate(counts) if n >= min_count and c not in names_evo]
    score = {}
    for c in big:
        v = cents[c] / (np.linalg.norm(cents[c]) / len(cents[c]) ** .5 + 1e-6)
        for k, r in refs.items():
            score[c, k] = float((v * r).mean())
    names: dict[int, str] = {}
    taken: set[str] = set()
    # Biggest cluster first, each claiming only its OWN best card. A card rests
    # in the tray far longer than it is lifted, dimmed or sliding, so its resting
    # look is its biggest cluster. Greedy by score instead let a small lifted
    # cluster that happened to score 0.65 take Hog Rider from the 8,527-sample
    # resting Hog (0.60) on uVf1EUszTIM. A cluster whose best card is taken is a
    # variant of it, never its second choice.
    plain = {b: [k for k in f if not k.endswith("-ev1")] for b, f in DECK_FORMS.items()}
    base_of: dict[int, str] = {}
    for c in sorted(big, key=lambda c: -counts[c]):
        best = max(((score[c, k], b) for b, forms in plain.items()
                    for k in forms if (c, k) in score), default=None)
        if best is None or best[0] < CLAIM_MIN or best[1] in taken:
            continue
        base_of[c] = best[1]
        taken.add(best[1])
    # Leftover clusters: the same card seen differently (dimmed, mid-slide) joins
    # the primary cluster it correlates with; anything else (a lifted card) stays unnamed.
    primary = dict(base_of)
    for c in big:
        if c in primary:
            continue
        u = cents[c] / (np.linalg.norm(cents[c]) + 1e-6)
        sims = {p: float(u @ (cents[p] / (np.linalg.norm(cents[p]) + 1e-6))) for p in primary}
        if sims:
            p = max(sims, key=sims.get)
            if sims[p] >= LEFTOVER_CORR:
                base_of[c] = primary[p]
    # One form per base card in a video: pool the form scores over its clusters.
    for base in set(base_of.values()):
        cs = [c for c in base_of if base_of[c] == base]
        forms = [k for k in plain[base] if all((c, k) in score for c in cs)]
        form = max(forms, key=lambda k: sum(counts[c] * score[c, k] for c in cs))
        for c in cs:
            names[c] = form
    names.update(names_evo)
    return names


def _hud():
    """The detector's HUD reader (match clock), imported on first use."""
    if "hud" not in sys.modules:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "detector"))
    import hud
    return hud


CLOCK_EVERY = 10        # samples between clock reads (1 Hz at 10 Hz)
NEW_MATCH_CLOCK = 150   # a reset to >= 2:30 left is a new match; overtime resets to 2:00
CLOCK_LOST_S = 4.0      # no clock for this long: not a battle screen (result, menus)


def match_ids(samples: list[Sample]) -> list[int]:
    """Number the battles from the match clock; -1 where no battle is on screen.

    Measured on 4s1o1sk-hPM: every reset to the 2:00 overtime clock follows 0:01,
    every new match reads 2:38-2:54 on its first legible frame, and the result
    screen between matches shows no clock."""
    out, m, last, last_t = [], -1, None, None
    for smp in samples:
        if smp.clock is not None:
            if m < 0 or (smp.clock >= NEW_MATCH_CLOCK and smp.clock > last + 20):
                m += 1
            last, last_t = smp.clock, smp.t
        live = last_t is not None and smp.t - last_t <= CLOCK_LOST_S
        out.append(m if live else -1)
    return out


def own_plays(video_path, geo_path, t0: float = 0.0, t1: float | None = None,
              scanned=None) -> list[dict]:
    """`scanned` is a previous scan()'s result for the same video and window."""
    geo = IosGeometry.load(str(geo_path))
    samples, cents, counts = scanned or scan(Path(video_path), geo, t0, t1)
    match = match_ids(samples)
    ev_n: dict[int, list[int]] = {}
    for smp in samples:
        for c, e in zip(smp.cluster, smp.evo):
            if c >= 0:
                ev_n.setdefault(c, [0, 0])[e] += 1
    evo_frac = {c: v[1] / sum(v) for c, v in ev_n.items()}
    names = name_clusters(cents, counts, art_refs(), evo_frac=evo_frac)
    base_of = {c: next(b for b, f in DECK_FORMS.items() if k in f) for c, k in names.items()}
    n = len(samples)
    # The preview is never lifted or dimmed, so its own clusters are clean; each
    # is named by which card it showed before strongly voted slot segments.
    def segments(s):
        i = 0
        while i < n:
            if samples[i].empty[s]:
                i += 1
                continue
            j = i
            while j < n and not samples[j].empty[s]:
                j += 1
            votes: dict[str, int] = {}
            for k in range(i, j):
                b = base_of.get(samples[k].cluster[s])
                if b:
                    votes[b] = votes.get(b, 0) + 1
            yield i, j, votes
            i = j
    co: dict[int, dict[str, int]] = {}
    for s in range(4):
        for i, j, votes in segments(s):
            if i == 0 or sum(votes.values()) < MIN_VOTES:
                continue
            b = max(votes, key=votes.get)
            for k in range(max(0, i - 5), i):
                d = co.setdefault(samples[k].next_cluster, {})
                d[b] = d.get(b, 0) + 1
    next_name = {c: max(d, key=d.get) for c, d in co.items()}

    def preview(i: int) -> str | None:
        """The card in the next-card preview just before sample i."""
        votes: dict[str, int] = {}
        for k in range(max(0, i - 5), i):
            b = next_name.get(samples[k].next_cluster)
            if b:
                votes[b] = votes.get(b, 0) + 1
        return max(votes, key=votes.get) if votes else None
    check = {"agree": 0, "disagree": 0, "from_preview": 0}
    # One card occupies a slot from the moment it slides in until it is played,
    # so each occupancy segment is named by vote over its named crops; a lifted
    # or dimmed crop (its own unnamed cluster) inherits the segment's name.
    seg_name = [[None] * 4 for _ in range(n)]
    seg_evo = [[False] * 4 for _ in range(n)]
    seg_voted = [[False] * 4 for _ in range(n)]
    gone = [list(smp.empty) for smp in samples]     # empty, or only a blink of a card
    for s in range(4):
        for i, j, votes in segments(s):
            b = max(votes, key=votes.get) if votes else None
            # Measured: the preview read just before the card lands matches the
            # vote 36/39 on 4s1o1sk-hPM; read when the slot emptied, 26/39.
            pv = preview(i) if i > 0 else None
            if sum(votes.values()) < MIN_VOTES:
                if pv is not None:
                    b = pv
                    check["from_preview"] += 1
            elif pv is not None and i > 0:
                check["agree" if pv == b else "disagree"] += 1
            ev = sum(samples[k].evo[s] for k in range(i, j)) > 0.5 * (j - i)
            voted = sum(votes.values()) >= MIN_VOTES
            if j - i < MIN_SEG:
                # A card sliding to the next slot crosses this one for a blink.
                b, voted = None, False
                for k in range(i, j):
                    gone[k][s] = True
            for k in range(i, j):
                seg_name[k][s], seg_evo[k][s], seg_voted[k][s] = b, ev, voted
    plain_form = {}
    for c, k in names.items():
        if not k.endswith("-ev1"):
            plain_form.setdefault(base_of[c], k)
    def form(b, ev):
        if b is None:
            return None
        if ev and b + "-ev1" in DECK_FORMS[b]:
            return b + "-ev1"
        return plain_form.get(b, b)
    plays = []
    for s in range(4):
        prev_card, prev_i, prev_evo, i = None, None, False, 0
        held = None          # (name, i, evo) of the last voted card still in this slot
        while i < n:
            smp = samples[i]
            if not gone[i][s]:
                if seg_voted[i][s]:
                    held = (seg_name[i][s], i, seg_evo[i][s])
                    prev_card, prev_i, prev_evo = held
                elif held:
                    # A lifted or half-dragged card clusters away from its own art
                    # and an unvoted name for it is a guess; nothing was played
                    # since the vote, so it is still the voted card.
                    prev_card, prev_i, prev_evo = held
                elif seg_name[i][s]:
                    prev_card, prev_i, prev_evo = seg_name[i][s], i, seg_evo[i][s]
                i += 1
                continue
            # Slot empty: the card was picked up. It was played if the elixir drops
            # by about its cost within 1.5 s and a DIFFERENT card fills the slot.
            if prev_card is None:
                i += 1
                continue
            j = i
            while j < n and gone[j][s]:
                j += 1
            refill = next((seg_name[k][s] for k in range(j, min(j + 10, n))
                           if seg_name[k][s]), None)
            refill_voted = any(seg_voted[k][s] for k in range(j, min(j + 10, n)))
            base = prev_card
            before = [x.elixir for x in samples[max(0, i - 6):i] if x.elixir is not None]
            eb = before[-1] if before else None
            drop_t, ea = None, None
            if eb is not None:
                # Through the whole empty spell: a held drag can last seconds.
                for k in range(i, min(max(i + 15, j + 5), n)):
                    e = samples[k].elixir
                    if e is not None and eb - e >= COST[base] - 1.0:
                        drop_t, ea = samples[k].t, e
                        break
            # The cycle never deals a played card straight back, so the same
            # card refilling its own slot is a drag that was let go, whatever
            # the elixir did meanwhile (another slot may have spent it).
            same_back = refill == prev_card and refill_voted
            # Without an elixir drop only a voted, different refill shows a play.
            if match[i] < 0:
                same_back = True            # not a battle screen
            if not same_back and (drop_t is not None or (refill_voted and refill != prev_card)):
                held = None
                # The frame, not the art, says whether this play is the Evolution.
                key = form(base, prev_evo)
                plays.append({"t": round(drop_t if drop_t is not None else smp.t, 2),
                              "slot": s, "match": match[i], "t_empty": smp.t,
                              "card_key": key, "base": base,
                              "evolved": prev_evo, "cost": COST[base],
                              "elixir_before": eb, "elixir_after": ea,
                              "elixir_ok": drop_t is not None, "refill": form(refill, False) if refill else None,
                              # Hand-checked on 4s1o1sk-hPM 40-220 s: every false
                              # positive had neither an elixir drop nor a refill.
                              "empty_s": round(samples[min(j, n - 1)].t - smp.t, 2),
                              "hand": [seg_name[i - 1][k] if i else None for k in range(4)],
                              "voted": bool(prev_i is not None and seg_voted[prev_i][s]),
                              "confident": drop_t is not None or (
                                  refill_voted and refill not in (None, prev_card))})
            prev_card = None
            i = j
    # Three or more slots emptying at once is the hand being dealt or cleared
    # (match start or end), never plays: one finger places one card at a time.
    te = sorted(p["t_empty"] for p in plays)
    plays = [p for p in plays
             if sum(abs(x - p["t_empty"]) <= 0.3 for x in te) < 3]
    plays.sort(key=lambda p: p["t"])
    plays = cycle_pass(plays, form)
    own_plays.check = check
    return plays


MIN_EMPTY_S = 0.5     # a real play empties its slot 0.9-2.3 s; drags and banners <= 0.4 s


def cycle_pass(plays: list[dict], form) -> list[dict]:
    """Clash Royale's cycle: the hand is the deck minus the last four cards played.

    So a play's card is not among the last four confident plays and not in any
    other slot. A play breaking that is relabelled when exactly one card fits,
    else marked not confident. Also drops a second play from the same slot
    short empties (drags, banners).
    """
    out: list[dict] = []
    hist: list[str] = []
    cur = None
    for p in plays:
        if p["match"] != cur:              # a new battle deals a fresh hand
            cur, hist = p["match"], []
        p["cycle_ok"] = True
        if p["empty_s"] < MIN_EMPTY_S:
            p["confident"] = False
        if p["confident"]:
            others = {b for k, b in enumerate(p.pop("hand")) if k != p["slot"] and b}
            allowed = set(DECK_FORMS) - set(hist[-4:]) - others
            if p["base"] not in allowed:
                p["cycle_ok"] = False
                # A vote over the slot's own crops outranks the cycle (a missed
                # play upstream breaks the cycle, not the vote); a preview-only
                # name is relabelled when exactly one card fits.
                if p["voted"]:
                    pass
                elif len(allowed) == 1:
                    b = allowed.pop()
                    p.update(base=b, cost=COST[b], card_key=form(b, p["evolved"]),
                             relabelled=True)
                else:
                    p["confident"] = False
            if p["confident"]:
                hist.append(p["base"])
        else:
            p.pop("hand")
        out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("geo", type=Path)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    plays = own_plays(a.video, a.geo, a.t0, a.t1)
    for p in plays:
        print(p)
    ok = sum(p["elixir_ok"] for p in plays)
    conf = sum(p["confident"] for p in plays)
    print(f"{len(plays)} plays, {ok} with a matching elixir drop, {conf} confident; "
          f"preview vs vote: {own_plays.check}")
    if a.out:
        a.out.write_text(json.dumps(plays, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
