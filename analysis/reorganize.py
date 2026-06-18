#!/usr/bin/env python3
"""Migrate existing flat clips into the descriptive name + subfolder scheme.

Reads each clip's .json sidecar, builds the new path
  results/a{angle}_p{pol}_s{sep}/{label}_a{angle}_p{pol}_f{foff}_s{sep}_{ts}.npy
and moves the .npy + .json together.

DRY-RUN by default (prints what it would do). Add --apply to actually move.

    uv run python reorganize.py                 # preview
    uv run python reorganize.py --apply         # do it
"""
import argparse
import glob
import json
import os
import re
import shutil


def _tok(v, default="na"):
    s = re.sub(r"[^A-Za-z0-9.+-]", "", str(v).strip())
    return s if s else default


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="../results")
    ap.add_argument("--apply", action="store_true", help="actually move (default: preview)")
    args = ap.parse_args()

    moves, skipped = [], 0
    for js in sorted(glob.glob(os.path.join(args.results, "**", "*.json"), recursive=True)):
        npy = js[:-5] + ".npy"
        if not os.path.exists(npy):
            skipped += 1
            continue
        m = json.load(open(js))
        label = m.get("label", "clip")
        a, pol, foff, sep = (_tok(m.get(k)) for k in ("angle", "pol", "foff", "sep"))
        ts = m.get("timestamp") or os.path.basename(npy).split("_")[-1].split(".")[0]
        subdir = os.path.join(args.results, f"a{a}_p{pol}_s{sep}")
        newbase = os.path.join(subdir, f"{label}_a{a}_p{pol}_f{foff}_s{sep}_{ts}")
        if os.path.abspath(newbase + ".npy") == os.path.abspath(npy):
            continue  # already in the right place/name
        moves.append((npy, js, subdir, newbase))

    if not moves:
        print(f"Nothing to move ({skipped} sidecars had no .npy).")
        return

    for npy, js, subdir, newbase in moves:
        print(f"{os.path.relpath(npy, args.results)}  ->  "
              f"{os.path.relpath(newbase + '.npy', args.results)}")
        if args.apply:
            os.makedirs(subdir, exist_ok=True)
            shutil.move(npy, newbase + ".npy")
            shutil.move(js, newbase + ".json")

    print(f"\n{len(moves)} clips " + ("moved." if args.apply else
          "would move. Re-run with --apply to do it."))


if __name__ == "__main__":
    main()