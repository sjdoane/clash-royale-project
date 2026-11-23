"""Print every ElixirLedger.update (reading, timestamp, booked spend) on a replay."""
import os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
os.environ.pop("CLASH_DECK", None)
sys.path[:0] = [str(REPO / "engine/perception"), str(REPO / "engine"), str(REPO / "engine/python_ai")]
import live.mvp_loop as L
from live import elixir_ledger as EL
orig = EL.ElixirLedger.update
def upd(self, elixir, now=None, multiplier=1.0):
    before = self.spent
    orig(self, elixir, now=now, multiplier=multiplier)
    print(f"LEDGER now={now!r} elixir={elixir!r} last={self._last} spent {before:.0f}->{self.spent:.0f} plays={self.plays[-1:] if self.spent != before else ''}", flush=True)
EL.ElixirLedger.update = upd
sys.argv = ["mvp_loop", "--policy", "scripted", "--frames", sys.argv[1], "--seconds", "400"]
L.main()
