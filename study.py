#!/usr/bin/env python3
"""Automated interference sweep.

For each aggressor setting it: starts aggressor.py as a subprocess, lets it
settle, captures victim frames, stops the aggressor, computes metrics, and
appends a row to results.csv. A baseline (aggressor off) is recorded first.

The aggressor and victim are fully independent: the aggressor talks only to
its own serial port (no DCA), the victim owns the DCA. So they run at once.

Example -- sweep the aggressor chirp slope (the knob that sets how the
interferer's sweep crosses the victim IF band):

    uv run python study.py \
        --config victim.yaml \
        --aggressor-port /dev/serial/by-id/<AGGRESSOR_ENHANCED> \
        --sweep freq_slope --values 30 50 67 80 \
        --n 300

Other useful sweeps:
  --sweep frame_period --values 40 50 55 70   (timing overlap)
  --sweep frequency    --values 76 77         (band offset)
For a power/INR sweep, vary physical separation or antenna orientation by
hand and record each as a separate run (CLI power control isn't exposed).
"""
import argparse
import csv
import subprocess
import sys
import time

import numpy as np

import interference as ix


def run_aggressor(port, sweep_key, value, extra):
    cmd = [sys.executable, "aggressor.py", "--port", port,
           f"--{sweep_key.replace('_', '-')}", str(value)]
    for k, v in extra.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    # New process group so we can signal it cleanly.
    return subprocess.Popen(cmd)


def stop_aggressor(proc, settle=2.0):
    proc.terminate()           # SIGTERM -> aggressor.py calls radar.stop()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(settle)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="victim.yaml")
    ap.add_argument("--aggressor-port", required=True)
    ap.add_argument("--sweep", default="freq_slope",
                    help="aggressor.py arg to vary, e.g. freq_slope, "
                         "frame_period, frequency.")
    ap.add_argument("--values", nargs="+", required=True,
                    help="values for the swept parameter.")
    ap.add_argument("--n", type=int, default=300, help="frames per condition.")
    ap.add_argument("--settle", type=float, default=3.0,
                    help="seconds to let the aggressor warm up before capture.")
    ap.add_argument("--out-prefix", default="run")
    ap.add_argument("--csv", default="results.csv")
    args = ap.parse_args()

    cfg = ix.load_cfg(args.config)
    rows = []

    def record(label, frames_path):
        m = ix.dataset_metrics(np.load(frames_path))
        row = {"condition": label,
               "noise_floor_db": round(m["noise_floor_db"], 2),
               "spike_frac": m["spike_frac"],
               "kurtosis": round(m["kurtosis"], 3),
               "affected_frame_rate": round(m["affected_frame_rate"], 4),
               "snr_db": round(m["snr_db"], 2)}
        rows.append(row)
        ix._print_metrics(label, m)
        return row

    # 1) Baseline: aggressor OFF.
    print("=== baseline (aggressor off) ===")
    base_path = f"{args.out_prefix}_baseline.npy"
    np.save(base_path, ix.capture_frames(cfg, args.n))
    base = record("baseline", base_path)

    # 2) Sweep.
    for val in args.values:
        label = f"{args.sweep}={val}"
        print(f"\n=== {label} ===")
        proc = run_aggressor(args.aggressor_port, args.sweep, val, extra={})
        time.sleep(args.settle)
        path = f"{args.out_prefix}_{args.sweep}_{val}.npy"
        try:
            np.save(path, ix.capture_frames(cfg, args.n))
        finally:
            stop_aggressor(proc)
        record(label, path)

    # 3) Write CSV.
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {args.csv}")

    # 4) Optional plot of noise-floor rise vs swept value.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        swept = [r for r in rows if r["condition"] != "baseline"]
        xs = [r["condition"].split("=")[1] for r in swept]
        nf = [r["noise_floor_db"] - base["noise_floor_db"] for r in swept]
        af = [r["affected_frame_rate"] * 100 for r in swept]
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(xs, nf, "o-", color="tab:red"); ax1.set_ylabel("noise-floor rise (dB)", color="tab:red")
        ax1.set_xlabel(args.sweep)
        ax2 = ax1.twinx()
        ax2.plot(xs, af, "s--", color="tab:blue"); ax2.set_ylabel("affected frames (%)", color="tab:blue")
        plt.title("Interference vs " + args.sweep)
        plt.tight_layout(); plt.savefig("results.png", dpi=110)
        print("wrote results.png")
    except ImportError:
        print("(matplotlib not installed; skipped results.png)")


if __name__ == "__main__":
    main()