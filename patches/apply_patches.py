#!/usr/bin/env python3
"""Apply (or undo) the local patches to the engine submodule.

The engine lives in engine/ as a git submodule pinned to an upstream commit,
so fixes we need that are not upstream yet live here as plain `git diff`
files and are applied into the submodule's working tree:

    python patches/apply_patches.py            # apply every patches/*.patch
    python patches/apply_patches.py --reverse  # undo them again
    python patches/apply_patches.py --check    # report, change nothing

Each patch is skipped when it is already applied, so running this twice is
safe. A patch that neither applies nor is already applied (upstream moved
under it) is reported and the script exits non-zero.
"""
import argparse
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(os.path.dirname(HERE), "engine")


def git(*args, check=False):
    return subprocess.run(["git", "-C", ENGINE, *args],
                          capture_output=True, text=True, check=check)


def applies(patch, reverse=False):
    args = ["apply", "--check"] + (["--reverse"] if reverse else []) + [patch]
    return git(*args).returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reverse", action="store_true", help="undo the patches")
    ap.add_argument("--check", action="store_true", help="only report status")
    args = ap.parse_args()

    if not os.path.isfile(os.path.join(ENGINE, "CMakeLists.txt")):
        sys.exit(f"engine submodule is empty at {ENGINE}; run "
                 "'git submodule update --init --recursive' first")

    patches = sorted(glob.glob(os.path.join(HERE, "*.patch")))
    if not patches:
        sys.exit(f"no *.patch files in {HERE}")

    failed = 0
    for patch in patches:
        name = os.path.basename(patch)
        already = applies(patch, reverse=True)   # reverse applies cleanly
        fresh = applies(patch)                   # forward applies cleanly
        if args.check:
            state = ("applied" if already else "not applied" if fresh
                     else "CONFLICT")
            print(f"{name}: {state}")
            failed += state == "CONFLICT"
            continue
        if args.reverse:
            if not already:
                print(f"{name}: not applied, nothing to undo")
                continue
            git("apply", "--reverse", patch, check=True)
            print(f"{name}: reversed")
            continue
        if already:
            print(f"{name}: already applied")
        elif fresh:
            git("apply", patch, check=True)
            print(f"{name}: applied")
        else:
            print(f"{name}: does not apply and is not already applied -- "
                  f"the engine moved under it; re-derive the patch")
            failed += 1
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
