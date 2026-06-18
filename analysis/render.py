#!/usr/bin/env python3
"""Render saved clips into range-Doppler / range-azimuth graphs.

Modes (combinable):
  (default)  comparison_mean.png + comparison_worst.png  -- stills vs baseline
  --strip    a contact sheet: one thumbnail per frame (the whole event at a glance)
  --video    a GIF (or MP4 if ffmpeg is present): the clip played back over time

Because a clip is a TIME SEQUENCE of frames (the seconds before and after your
trigger), --strip and --video keep the time axis instead of collapsing it. The
color scale is fixed across the whole clip so interference visibly appears and
fades; the trigger frame (from the .json sidecar's pre_frames) is marked.

Usage:
  uv run python render_clip.py clips/clip_X.npy --baseline clips/baseline_Y.npy
  uv run python render_clip.py clips/clip_X.npy --video --strip
  uv run python render_clip.py clips/clip_X.npy --video --panels both --fps 5
"""
import argparse
import json
import math
import os
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Processing (same as demo.py / monitor.py)
# --------------------------------------------------------------------------- #
def make_rsp(rsp_name, azimuth=128):
    from xwr.rsp import numpy as xwr_rsp
    return getattr(xwr_rsp, rsp_name)(window=False, size={"azimuth": azimuth})


def frame_panels(rsp, frame):
    dear = np.abs(rsp(frame[None, ...]))
    rd = np.swapaxes(np.mean(dear, axis=(0, 2, 3)), 0, 1)   # (range, doppler)
    ra = np.swapaxes(np.mean(dear, axis=(0, 1, 2)), 0, 1)   # (range, azimuth)
    return rd, ra


def _db(a, db):
    return 20 * np.log10(a + 1e-6) if db else a


def _clim(arrays, db):
    v = np.concatenate([_db(a, db).ravel() for a in arrays])
    return float(np.percentile(v, 1)), float(np.percentile(v, 99.5))


# --------------------------------------------------------------------------- #
# Stills vs baseline (mean + worst) -- unchanged behavior
# --------------------------------------------------------------------------- #
def reduce_frames(rsp, frames, how="mean"):
    rds, ras, energy = [], [], []
    for f in frames:
        rd, ra = frame_panels(rsp, f)
        rds.append(rd); ras.append(ra); energy.append(float(rd.sum()))
    rds, ras = np.array(rds), np.array(ras)
    if how == "mean":
        return rds.mean(0), ras.mean(0), int(np.argmax(energy))
    i = int(np.argmax(energy))
    return rds[i], ras[i], i


def _panel(ax, data, xlab, title, vmin, vmax):
    im = ax.imshow(20 * np.log10(data + 1e-6), cmap="viridis", aspect="auto",
                   origin="lower", vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlab); ax.set_ylabel("Range"); ax.set_title(title)
    return im


def comparison_figure(rows, out):
    n = len(rows)
    rd_lim = _clim([r[1] for r in rows], db=True)
    ra_lim = _clim([r[2] for r in rows], db=True)
    fig, axs = plt.subplots(n, 2, figsize=(11, 4.4 * n), squeeze=False)
    for i, (label, rd, ra) in enumerate(rows):
        im0 = _panel(axs[i][0], rd, "Doppler", f"{label} | Range-Doppler", *rd_lim)
        im1 = _panel(axs[i][1], ra, "Azimuth", f"{label} | Range-Azimuth", *ra_lim)
        fig.colorbar(im0, ax=axs[i][0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axs[i][1], fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    print(f"saved -> {out}")


# --------------------------------------------------------------------------- #
# Time-sequence outputs (strip + video)
# --------------------------------------------------------------------------- #
def per_frame(rsp, frames, which, stride):
    idx = list(range(0, len(frames), stride))
    rds, ras = [], []
    for i in idx:
        rd, ra = frame_panels(rsp, frames[i])
        if which in ("rd", "both"):
            rds.append(rd)
        if which in ("ra", "both"):
            ras.append(ra)
    return idx, rds, ras


def _trigger_pos(idx, pre_frames):
    """Position within idx of the first frame at/after the trigger, or None."""
    if pre_frames is None:
        return None
    for p, i in enumerate(idx):
        if i >= pre_frames:
            return p
    return None


def make_strip(idx, rds, db, out, cols, trig_pos):
    cl = _clim(rds, db)
    n = len(rds)
    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axs = np.array(axs).reshape(rows, cols)
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    for m, rd in enumerate(rds):
        ax = axs[m // cols, m % cols]
        ax.imshow(_db(rd, db), cmap="viridis", aspect="auto", origin="lower",
                  vmin=cl[0], vmax=cl[1])
        is_trig = (trig_pos is not None and m == trig_pos)
        ax.set_title(f"{idx[m]}{' *TRIG' if is_trig else ''}", fontsize=7,
                     color="red" if is_trig else "0.3")
        if is_trig:
            for sp in ax.spines.values():
                sp.set_color("red"); sp.set_linewidth(2)
    for m in range(n, rows * cols):
        axs[m // cols, m % cols].axis("off")
    fig.suptitle("Range-Doppler frame strip  (* = trigger frame)")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    print(f"saved -> {out}")


def write_video(idx, rds, ras, which, out_base, fps, db, frame_period_ms,
                trig_pos, fmt):
    import matplotlib.animation as animation
    ncols = 2 if which == "both" else 1
    fig, axs = plt.subplots(1, ncols, figsize=(5 * ncols, 4.2), squeeze=False)
    axs = axs[0]
    ims, col = [], 0
    if which in ("rd", "both"):
        cl = _clim(rds, db)
        im = axs[col].imshow(_db(rds[0], db), cmap="viridis", aspect="auto",
                             origin="lower", vmin=cl[0], vmax=cl[1])
        axs[col].set_xlabel("Doppler"); axs[col].set_ylabel("Range")
        axs[col].set_title("Range-Doppler"); ims.append(("rd", im)); col += 1
    if which in ("ra", "both"):
        cl = _clim(ras, db)
        im = axs[col].imshow(_db(ras[0], db), cmap="viridis", aspect="auto",
                             origin="lower", vmin=cl[0], vmax=cl[1])
        axs[col].set_xlabel("Azimuth"); axs[col].set_ylabel("Range")
        axs[col].set_title("Range-Azimuth"); ims.append(("ra", im))
    sup = fig.suptitle("")
    fig.tight_layout()

    def update(k):
        j = 0
        if which in ("rd", "both"):
            ims[j][1].set_data(_db(rds[k], db)); j += 1
        if which in ("ra", "both"):
            ims[j][1].set_data(_db(ras[k], db)); j += 1
        t = idx[k] * frame_period_ms / 1000.0
        tag = "   <-- TRIGGER" if (trig_pos is not None and k == trig_pos) else ""
        sup.set_text(f"frame {idx[k]}   t = {t:.2f} s{tag}")
        return [im for _, im in ims] + [sup]

    n = len(idx)
    anim = animation.FuncAnimation(fig, update, frames=n, blit=False)

    use_mp4 = (fmt == "mp4") or (fmt == "auto" and shutil.which("ffmpeg"))
    try:
        if use_mp4:
            out = out_base + ".mp4"
            anim.save(out, writer=animation.FFMpegWriter(fps=fps))
        else:
            out = out_base + ".gif"
            anim.save(out, writer=animation.PillowWriter(fps=fps))
        print(f"saved -> {out}  ({n} frames @ {fps} fps)")
    except Exception as e:
        # Fall back to GIF if MP4/ffmpeg failed; report missing-Pillow clearly.
        if use_mp4:
            print(f"MP4 failed ({e}); falling back to GIF")
            try:
                out = out_base + ".gif"
                anim.save(out, writer=animation.PillowWriter(fps=fps))
                print(f"saved -> {out}  ({n} frames @ {fps} fps)")
            except Exception as e2:
                print(f"GIF also failed: {e2}\n(install Pillow:  uv add pillow)")
        else:
            print(f"GIF failed: {e}\n(install Pillow:  uv add pillow)")
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------- #
def _load_sidecar(clip_path):
    side = clip_path[:-4] + ".json" if clip_path.endswith(".npy") else clip_path + ".json"
    if os.path.exists(side):
        try:
            return json.load(open(side))
        except Exception:
            return {}
    return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--rsp", default="AWR1843AOP")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--strip", action="store_true", help="contact sheet of frames")
    ap.add_argument("--video", action="store_true", help="GIF/MP4 over time")
    ap.add_argument("--panels", choices=["rd", "ra", "both"], default="rd",
                    help="which panel(s) in strip/video (strip always uses rd)")
    ap.add_argument("--fps", type=float, default=5.0, help="video playback fps (slow=easier to see)")
    ap.add_argument("--stride", type=int, default=1, help="render every Nth frame")
    ap.add_argument("--cols", type=int, default=10, help="columns in the strip")
    ap.add_argument("--format", choices=["auto", "gif", "mp4"], default="auto")
    ap.add_argument("--linear", action="store_true", help="linear scale (default dB)")
    args = ap.parse_args()

    db = not args.linear
    rsp = make_rsp(args.rsp)
    clip = np.load(args.clip)
    meta = _load_sidecar(args.clip)
    pre_frames = meta.get("pre_frames")
    fp_ms = float(meta.get("radar_cfg", {}).get("frame_period", 50.0))

    did_default = not (args.strip or args.video)

    if did_default:
        base = np.load(args.baseline) if args.baseline else None
        rows_mean = []
        if base is not None:
            b_rd, b_ra, _ = reduce_frames(rsp, base, "mean")
            rows_mean.append(("baseline (mean)", b_rd, b_ra))
        c_rd, c_ra, _ = reduce_frames(rsp, clip, "mean")
        rows_mean.append(("clip (mean)", c_rd, c_ra))
        comparison_figure(rows_mean, os.path.join(args.outdir, "comparison_mean.png"))
        rows_worst = []
        if base is not None:
            rows_worst.append(("baseline (mean)", b_rd, b_ra))
        w_rd, w_ra, wi = reduce_frames(rsp, clip, "max")
        rows_worst.append((f"clip (worst frame #{wi})", w_rd, w_ra))
        comparison_figure(rows_worst, os.path.join(args.outdir, "comparison_worst.png"))

    if args.strip or args.video:
        which = args.panels
        # strip needs rd; ensure we compute it.
        compute = "both" if (args.video and which == "both") else \
                  ("rd" if (args.strip and which != "ra") else which)
        if args.strip and "rd" not in (compute, "both"):
            compute = "both"
        idx, rds, ras = per_frame(rsp, clip, compute, args.stride)
        trig_pos = _trigger_pos(idx, pre_frames)
        if args.strip:
            make_strip(idx, rds, db, os.path.join(args.outdir, "frame_strip.png"),
                       args.cols, trig_pos)
        if args.video:
            write_video(idx, rds, ras, which,
                        os.path.join(args.outdir, "clip_video"),
                        args.fps, db, fp_ms, trig_pos, args.format)


if __name__ == "__main__":
    main()