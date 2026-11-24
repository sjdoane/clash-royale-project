# Engine patches

These are the simulator changes as they were developed, one file per change. They are **already
applied** on the `gen2` branch of the engine fork
([sjdoane/ClashRoyaleAi](https://github.com/sjdoane/ClashRoyaleAi/tree/gen2)), which is the `engine/`
submodule, so you do not need to apply anything. They are kept here so each change can be read
on its own. Each change is also one commit on that branch, and its notes are in the fork's
`.claude/CLAUDE.md` and `perception/UPSTREAM_REQUESTS.md`.

- `0001-0003` (this folder) and `pending/0004-0019`: the baseline fixes (the first commit on `gen2`).
- `pending/0030-0060`, `0080-0087`: one change each (thrown-spell flight, first-hit timing, body sizes and masses, bridge crossing, building pull (0086), river crossing (0087), ...).
- The tag `gen2-data` (patch 0046) is the engine the behaviour-cloning datasets were built with.

Since patch 0004 a match ends at 5:00 with triple elixir from 4:00. A 6:00 end means the engine was built from upstream `main` or the submodule was not initialised: run `git submodule update --init --recursive` and `./setup.sh`.
After pulling a change that moves the `engine` submodule, run the same two commands to rebuild.
