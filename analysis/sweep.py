#!/usr/bin/env python3
"""Cross-folder sweep: aggregate metrics across capture subfolders and compare.

Drilling INTO one configuration -> analyze_folder.py.
Comparing ACROSS configurations -> this tool. It reads every clip from every
subfolder under results/, computes raw-domain metrics, and plots how
interference varies with one variable, split by another -- e.g. corruption vs
frequency offset, one line per pointing angle.

    uv run python sweep.py                                   # all folders, defaults
    uv run python sweep.py --group angle --x foff --metric corrupt_sample_frac
    uv run python sweep.py --group angle,pol --metric interfered_frame_rate
    uv run python sweep.py ../results/a*_p0_s30              # only some folders

Writes master_summary.csv (every clip, all fields + metrics), a grouped line
plot, and a pivot CSV (group x x-value -> metric). Metrics are raw-domain only
(no rsp), so this runs without rendering and needs no display.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import interference as ix

FIELDS = ["angle", "pol", "foff", "sep"]
METRICS = ["corrupt_sample_frac", "saturated_frac", "affected_chirp_frac",
           "interfered_frame_rate", "inr_db"]


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def gather(folders, results_root, k):
    """Return (clip_rows, baseline_rows). Recomputes metrics from raw clips."""
    clip_rows, base_rows = [], []
    jsons = []
    for f in folders:
        jsons += glob.glob(os.path.join(f, "*.json"))
    for js in sorted(jsons):
        npy = js[:-5] + ".npy"
        if not os.path.exists(npy):
            continue
        m = json.load(open(js))
        cm = ix.dataset_metrics(np.load(npy), k=k)
        row = {fld: m.get(fld, "") for fld in FIELDS}
        row.update({mk: cm[mk] for mk in METRICS})
        row["clip"] = os.path.relpath(npy, results_root)
        (base_rows if m.get("label") == "baseline" else clip_rows).append(row)
        print(f"  measured {row['clip']}")
    return clip_rows, base_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="*", help="subfolders or globs (default: all under --results)")
    ap.add_argument("--results", default="../results")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--x", default="foff", choices=FIELDS, help="x-axis variable")
    ap.add_argument("--group", default="angle", help="series split: field or comma list (e.g. angle,pol)")
    ap.add_argument("--metric", default="corrupt_sample_frac", choices=METRICS)
    args = ap.parse_args()

    folders = args.folders or [d for d in sorted(glob.glob(os.path.join(args.results, "*")))
                               if os.path.isdir(d)]
    if not folders:
        print(f"No subfolders found under {args.results}")
        return
    print(f"scanning {len(folders)} folder(s)...")
    clips, baselines = gather(folders, args.results, args.k)
    if not clips:
        print("No clips found.")
        return
    os.makedirs(args.outdir, exist_ok=True)

    # ---- master CSV ----
    cols = ["clip"] + FIELDS + METRICS
    mpath = os.path.join(args.outdir, "master_summary.csv")
    with open(mpath, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        w.writerows([{c: r.get(c, "") for c in cols} for r in clips])
    print(f"\nwrote {mpath}  ({len(clips)} clips)")

    # ---- grouped line plot ----
    group_fields = args.group.split(",")
    series = {}
    for r in clips:
        x = num(r[args.x])
        if x is None:
            continue
        key = tuple(str(r[g]) for g in group_fields)
        series.setdefault(key, []).append((x, r[args.metric]))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for key, pts in sorted(series.items()):
        pts.sort()
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        label = ", ".join(f"{g}={v}" for g, v in zip(group_fields, key))
        ax.plot(xs, ys, "o-", label=label)
    if baselines:
        bmean = float(np.mean([b[args.metric] for b in baselines]))
        ax.axhline(bmean, ls=":", color="0.5", label="baseline (mean)")
    ax.set_xlabel(args.x); ax.set_ylabel(args.metric)
    ax.set_title(f"{args.metric} vs {args.x}, by {args.group}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plot = os.path.join(args.outdir, f"sweep_{args.metric}_vs_{args.x}.png")
    fig.savefig(plot, dpi=120); plt.close(fig)
    print(f"wrote {plot}")

    # ---- pivot CSV: group (rows) x x-value (cols) -> metric ----
    xvals = sorted({num(r[args.x]) for r in clips if num(r[args.x]) is not None})
    piv = os.path.join(args.outdir, f"pivot_{args.metric}.csv")
    with open(piv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([args.group] + [f"{args.x}={x:g}" for x in xvals])
        for key in sorted(series):
            lut = {x: y for x, y in series[key]}
            w.writerow([", ".join(f"{g}={v}" for g, v in zip(group_fields, key))]
                       + [lut.get(x, "") for x in xvals])
    print(f"wrote {piv}")

    # ---- console preview ----
    print(f"\n{args.metric} by {args.group} across {args.x}:")
    for key in sorted(series):
        pts = sorted(series[key])
        label = ", ".join(f"{g}={v}" for g, v in zip(group_fields, key))
        print(f"  {label}: " + "  ".join(f"{args.x}={x:g}:{y:.2e}" for x, y in pts))


if __name__ == "__main__":
    main()