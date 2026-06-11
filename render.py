#!/usr/bin/env python3
"""Render saved clips into range-Doppler / range-azimuth graphs for comparison.

Uses the same processing as demo.py / monitor.py (xwr.rsp). Given a clip and an
optional baseline, produces side-by-side panels so you can compare what the
interference did to the victim's view.

Because interference is intermittent, two reductions are produced:
  * comparison_mean.png  -- mean over frames (the typical picture)
  * comparison_worst.png -- baseline (mean) vs the single most-energetic clip
                            frame (the captured event itself)

Usage:
  uv run python render_clip.py clips/clip_XXduration.npy \
      --baseline clips/baseline_YY.npy --rsp AWR1843AOP
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_rsp(rsp_name, azimuth=128):
    from xwr.rsp import numpy as xwr_rsp
    return getattr(xwr_rsp, rsp_name)(window=False, size={"azimuth": azimuth})


def frame_panels(rsp, frame):
    """One raw frame -> (rd, ra), exactly as demo.py computes them."""
    dear = np.abs(rsp(frame[None, ...]))
    rd = np.swapaxes(np.mean(dear, axis=(0, 2, 3)), 0, 1)   # (range, doppler)
    ra = np.swapaxes(np.mean(dear, axis=(0, 1, 2)), 0, 1)   # (range, azimuth)
    return rd, ra


def reduce_frames(rsp, frames, how="mean"):
    """Reduce a stack of raw frames to a single (rd, ra) pair."""
    rds, ras, energy = [], [], []
    for f in frames:
        rd, ra = frame_panels(rsp, f)
        rds.append(rd); ras.append(ra); energy.append(float(rd.sum()))
    rds, ras = np.array(rds), np.array(ras)
    if how == "mean":
        return rds.mean(0), ras.mean(0), int(np.argmax(energy))
    elif how == "max":
        i = int(np.argmax(energy))
        return rds[i], ras[i], i
    raise ValueError(how)


def _panel(ax, data, xlab, title, vmin, vmax):
    im = ax.imshow(20 * np.log10(data + 1e-6), cmap="viridis", aspect="auto",
                   origin="lower", vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlab); ax.set_ylabel("Range"); ax.set_title(title)
    return im


def _clim(*arrays):
    d = np.concatenate([20 * np.log10(a.ravel() + 1e-6) for a in arrays])
    return float(np.percentile(d, 1)), float(np.percentile(d, 99.5))


def comparison_figure(rows, out):
    """rows: list of (label, rd, ra). Shared color limits per column."""
    n = len(rows)
    rd_lim = _clim(*[r[1] for r in rows])
    ra_lim = _clim(*[r[2] for r in rows])
    fig, axs = plt.subplots(n, 2, figsize=(11, 4.4 * n), squeeze=False)
    for i, (label, rd, ra) in enumerate(rows):
        im0 = _panel(axs[i][0], rd, "Doppler", f"{label} | Range-Doppler", *rd_lim)
        im1 = _panel(axs[i][1], ra, "Azimuth", f"{label} | Range-Azimuth", *ra_lim)
        fig.colorbar(im0, ax=axs[i][0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axs[i][1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--rsp", default="AWR1843AOP")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    rsp = make_rsp(args.rsp)
    clip = np.load(args.clip)
    base = np.load(args.baseline) if args.baseline else None

    # Mean comparison (typical picture).
    rows_mean = []
    if base is not None:
        b_rd, b_ra, _ = reduce_frames(rsp, base, "mean")
        rows_mean.append(("baseline (mean)", b_rd, b_ra))
    c_rd, c_ra, _ = reduce_frames(rsp, clip, "mean")
    rows_mean.append(("clip (mean)", c_rd, c_ra))
    comparison_figure(rows_mean, os.path.join(args.outdir, "comparison_mean.png"))

    # Worst-frame comparison (the captured event).
    rows_worst = []
    if base is not None:
        rows_worst.append(("baseline (mean)", b_rd, b_ra))
    w_rd, w_ra, wi = reduce_frames(rsp, clip, "max")
    rows_worst.append((f"clip (worst frame #{wi})", w_rd, w_ra))
    comparison_figure(rows_worst, os.path.join(args.outdir, "comparison_worst.png"))


if __name__ == "__main__":
    main()