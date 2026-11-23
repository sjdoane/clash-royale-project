"""Score the hand-checked baseline (baseline_labels.json) per video and overall."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def score(entries):
    c = defaultdict(int)
    for e in entries:
        c["frames"] += 1
        c["fp"] += e["fp"]
        c["miss"] += e["miss"]
        for cls, side in e["dets"]:
            c["tp"] += 1
            c["cls_" + cls] += 1
            c["side_" + side] += 1
    judged = c["cls_ok"] + c["cls_form"] + c["cls_wrong"]
    sides = c["side_ok"] + c["side_wrong"]
    return {
        "frames": c["frames"], "tp": c["tp"], "fp": c["fp"], "miss": c["miss"],
        "precision": c["tp"] / max(c["tp"] + c["fp"], 1),
        "recall": c["tp"] / max(c["tp"] + c["miss"], 1),
        "cls_strict": c["cls_ok"] / max(judged, 1),
        "cls_base": (c["cls_ok"] + c["cls_form"]) / max(judged, 1),
        "form_wrong": c["cls_form"], "wrong_cls": c["cls_wrong"], "cls_unsure": c["cls_unsure"],
        "side_acc": c["side_ok"] / max(sides, 1), "side_wrong": c["side_wrong"],
        # strict end-to-end: a unit counts only if found, right card+form, right side
        "e2e_recall": sum(1 for e in entries for cl, sd in e["dets"] if cl == "ok" and sd == "ok")
                      / max(c["tp"] + c["miss"], 1),
    }


def main():
    lab = json.loads((HERE / "baseline_labels.json").read_text())
    lab.pop("_doc")
    by = defaultdict(list)
    for k, v in lab.items():
        by[k.rsplit("_", 1)[0]].append(v)
    rows = [(vid, score(es)) for vid, es in sorted(by.items())]
    rows.append(("ALL", score(list(lab.values()))))
    hdr = "| video | frames | TP | FP | miss | precision | recall | class acc (strict) | class acc (base card) | form wrong | wrong class | unsure | side acc | strict e2e recall |"
    print(hdr)
    print("|" + "---|" * hdr.count("|")[:-1] if False else "|---" * (hdr.count("|") - 1) + "|")
    for vid, s in rows:
        print(f"| {vid} | {s['frames']} | {s['tp']} | {s['fp']} | {s['miss']} | {s['precision']:.2f} | {s['recall']:.2f} | "
              f"{s['cls_strict']:.2f} | {s['cls_base']:.2f} | {s['form_wrong']} | {s['wrong_cls']} | {s['cls_unsure']} | "
              f"{s['side_acc']:.2f} | {s['e2e_recall']:.2f} |")


if __name__ == "__main__":
    main()
