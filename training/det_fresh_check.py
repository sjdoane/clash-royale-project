#!/usr/bin/env python3
"""det_track_check.py on the two FRESH test videos (LekrcVeQjiw, kaLMeKpsuGY) instead of the
frames every detector decision so far was made on.

    .venv-det/bin/python training/det_fresh_check.py W.pt [--vs REF] [--imgsz 896] [--json out]

Same arguments, tables and bootstrap as det_track_check.py; only the frame set and the cache
directory (runs/det_track_check_fresh/) differ. The labels are agent-made, pre-filled from the
reference B, pending a human check (live_ppo/eval/labels_units_fresh_README.md).

Use it to CONFIRM a choice made elsewhere, once per final candidate. Do not pick between
checkpoints with it: every look turns these frames into tuning data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import det_track_check as TC  # noqa: E402

FRESH = ("vid_LekrcVeQjiw", "vid_kaLMeKpsuGY")
TC.DC.SRCS = FRESH
TC.GROUPS = (("pooled", FRESH), ("LekrcVeQjiw", FRESH[:1]), ("kaLMeKpsuGY", FRESH[1:]))
TC.CACHE = TC.REPO / "runs" / "det_track_check_fresh"

if __name__ == "__main__":
    TC.main()
