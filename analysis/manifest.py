#!/usr/bin/env python3
"""Index all captured clips into a readable table (and optional CSV).

Reads every .json sidecar under results/ (recursively) and prints one row per
clip with its conditions -- so you can see what's what at a glance instead of
decoding timestamps. Works on old flat files and new descriptive ones alike,
since it reads the sidecar, not the filename.

    uv run python manifest.py                 # print table
    uv run python manifest.py --csv results/manifest.csv
"""
import argparse
import csv
import glob
import json
import os

COLS = ["label", "angle", "pol", "foff", "sep", "frames", "timestamp", "file"]


def collect(results_dir):
    rows = []
    for js in sorted(glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True)):
        try:
            m = json.load(open(js))
        except Exception:
            continue
        npy = js[:-5] + ".npy"
        rows.append({
            "label": m.get("label", ""),
            "angle": m.get("angle", ""),
            "pol": m.get("pol", ""),
            "foff": m.get("foff", ""),
            "sep": m.get("sep", ""),
            "frames": m.get("n_frames", ""),
            "timestamp": m.get("timestamp", ""),
            "file": os.path.relpath(npy, results_dir),
            "note": m.get("note", ""),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="../results", help="results directory")
    ap.add_argument("--csv", default=None, help="also write a CSV manifest")
    ap.add_argument("--sort", default="file", choices=COLS, help="sort key")
    args = ap.parse_args()

    rows = collect(args.results)
    if not rows:
        print(f"No sidecars found under {args.results}")
        return
    rows.sort(key=lambda r: str(r[args.sort]))

    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in COLS}
    line = "  ".join(c.ljust(widths[c]) for c in COLS)
    print(line); print("-" * len(line))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in COLS))
    print(f"\n{len(rows)} captures")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS + ["note"])
            w.writeheader(); w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()