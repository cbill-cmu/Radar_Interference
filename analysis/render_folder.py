#!/usr/bin/env python3
"""Analyze a whole capture subfolder (one baseline + its clips) in one command.

Point it at a results subfolder (e.g. results/a90_p0_s30). It finds the baseline
and every clip, then for each clip:
  * renders comparison_mean.png + comparison_worst.png vs the baseline
    (and frame_strip.png / clip_video.* if you pass --strip / --video)
  * computes raw-domain interference metrics
and writes a per-folder summary: a printed table, summary.csv

    uv run python render_folder.py ../results/a90_p0_s30
    uv run python render_folder.py ../results/a90_p0_s30 --strip --video

Outputs go to  <outdir>/<folder-name>/  (default outdir = current directory).
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

import render as rc          # sibling modules in analysis/
import interference as ix

METRIC_KEYS = ["corrupt_sample_frac", "saturated_frac", "affected_chirp_frac",
               "interfered_frame_rate", "inr_db"]


def find_clips(folder):
    baseline, clips = None, []
    for js in sorted(glob.glob(os.path.join(folder, "*.json"))):
        npy = js[:-5] + ".npy"
        if not os.path.exists(npy):
            continue
        m = json.load(open(js))
        if m.get("label") == "baseline" or os.path.basename(npy).startswith("baseline"):
            baseline = (npy, m)
        else:
            clips.append((npy, m))
    return baseline, clips


def _foff_num(m):
    try:
        return float(m.get("foff"))
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="a results subfolder (baseline + clips)")
    ap.add_argument("--outdir", default=".", help="where to write analysis outputs")
    ap.add_argument("--rsp", default="AWR1843AOP")
    ap.add_argument("--k", type=float, default=8.0, help="MAD threshold for metrics")
    ap.add_argument("--no-metrics", action="store_true", help="renders only")
    ap.add_argument("--strip", action="store_true")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--panels", choices=["rd", "ra", "both"], default="rd")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--format", choices=["auto", "gif", "mp4"], default="auto")
    args = ap.parse_args()

    folder = args.folder.rstrip("/")
    baseline, clips = find_clips(folder)
    if not clips:
        print(f"No clips found in {folder}")
        return
    name = os.path.basename(folder)
    out = os.path.join(args.outdir, name)
    os.makedirs(out, exist_ok=True)
    print(f"folder: {folder}\nbaseline: {os.path.basename(baseline[0]) if baseline else 'NONE'}"
          f"\nclips: {len(clips)}\n")

    need_rsp = not args.no_render
    rsp = rc.make_rsp(args.rsp) if need_rsp else None

    base_frames = np.load(baseline[0]) if baseline else None
    base_metrics = (ix.dataset_metrics(base_frames, k=args.k)
                    if (base_frames is not None and not args.no_metrics) else None)
    if need_rsp and base_frames is not None:
        b_rd, b_ra, _ = rc.reduce_frames(rsp, base_frames, "mean")

    rows = []
    # Baseline as a reference row in the summary.
    if base_metrics is not None:
        rows.append({"clip": "BASELINE", "foff": baseline[1].get("foff", ""),
                     **{k: base_metrics[k] for k in METRIC_KEYS}})

    for npy, m in clips:
        cname = os.path.splitext(os.path.basename(npy))[0]
        cdir = os.path.join(out, cname)
        clip = np.load(npy)

        if need_rsp:
            os.makedirs(cdir, exist_ok=True)
            rows_mean = ([("baseline (mean)", b_rd, b_ra)] if base_frames is not None else [])
            c_rd, c_ra, _ = rc.reduce_frames(rsp, clip, "mean")
            rows_mean.append(("clip (mean)", c_rd, c_ra))
            rc.comparison_figure(rows_mean, os.path.join(cdir, "comparison_mean.png"))

            rows_worst = ([("baseline (mean)", b_rd, b_ra)] if base_frames is not None else [])
            w_rd, w_ra, wi = rc.reduce_frames(rsp, clip, "max")
            rows_worst.append((f"clip (worst #{wi})", w_rd, w_ra))
            rc.comparison_figure(rows_worst, os.path.join(cdir, "comparison_worst.png"))

            if args.strip or args.video:
                idx, rds, ras = rc.per_frame(rsp, clip, args.panels, args.stride)
                trig = rc._trigger_pos(idx, m.get("pre_frames"))
                if args.strip:
                    rc.make_strip(idx, rds, True, os.path.join(cdir, "frame_strip.png"),
                                  args.cols, trig)
                if args.video:
                    fp = float(m.get("radar_cfg", {}).get("frame_period", 50.0))
                    rc.write_video(idx, rds, ras, args.panels,
                                   os.path.join(cdir, "clip_video"),
                                   args.fps, True, fp, trig, args.format)

        row = {"clip": cname, "foff": m.get("foff", "")}
        if not args.no_metrics:
            cm = ix.dataset_metrics(clip, k=args.k)
            row.update({k: cm[k] for k in METRIC_KEYS})
        rows.append(row)

    # ---- summary table + CSV ----
    if not args.no_metrics:
        cols = ["clip", "foff"] + METRIC_KEYS
        def fmt(v):
            return f"{v:.3e}" if isinstance(v, float) and abs(v) < 1 else \
                   (f"{v:.2f}" if isinstance(v, float) else str(v))
        w = {c: max(len(c), *(len(fmt(r.get(c, ""))) for r in rows)) for c in cols}
        hdr = "  ".join(c.ljust(w[c]) for c in cols)
        print(hdr); print("-" * len(hdr))
        for r in rows:
            print("  ".join(fmt(r.get(c, "")).ljust(w[c]) for c in cols))
        csv_path = os.path.join(out, "summary.csv")
        with open(csv_path, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=cols); wr.writeheader()
            wr.writerows([{c: r.get(c, "") for c in cols} for r in rows])
        print(f"\nwrote {csv_path}")

if __name__ == "__main__":
    main()