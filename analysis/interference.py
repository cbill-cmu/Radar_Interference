#!/usr/bin/env python3
"""Raw-domain radar interference detection & quantification (xwr + DCA1000).

Why raw domain
--------------
FMCW interference is a localized time-domain event: when the aggressor's sweep
crosses the victim's IF passband it injects a short, high-amplitude, broadband
burst into a run of fast-time ADC samples (often saturating the ADC). A range
FFT *spreads* that burst across all range bins, destroying its localization and
hiding it inside a vague "noise floor". So we detect on the raw samples instead,
which keeps the interference exactly where it lives and lets us label it.

We use three independent raw-domain signatures (no range/Doppler transform):

  1. SATURATION    - samples pinned at/near the int16 rail (hard ADC clipping).
  2. IMPULSIVENESS - after a fast-time high-pass (first difference), interferer
                     bursts are sharp edges while real targets are smooth
                     sinusoids; robust median/MAD threshold flags burst samples.
  3. CHIRP OUTLIER - within a frame the static scene repeats chirp-to-chirp;
                     subtract the per-frame median chirp and asynchronous
                     interference lights up as residual outliers.

Deinterleaving uses the repo's validated `xwr.rsp.iqiq_from_iiqq` (the stream is
physically QQII; getting this wrong silently swaps I/Q), so I/Q ordering is
correct by construction.

Subcommands
-----------
  smoke    short capture + sanity stats + (optional) raw chirp plot.
  capture  record N frames to .npy.
  pairs    record clean (aggressor off) then prompt + record interfered, so the
           two captures share one static scene. Writes <prefix>_clean.npy and
           <prefix>_interfered.npy.
  analyze  raw-domain interference metrics for one or two captures (+ optional
           per-frame interference-rate plot).
"""
import argparse
import sys

import numpy as np


# --------------------------------------------------------------------------- #
# Deinterleave (uses the repo's validated routine; falls back if rsp absent)
# --------------------------------------------------------------------------- #
def deinterleave(raw: np.ndarray) -> np.ndarray:
    """IIQQ int16 (..., 2N)  ->  complex64 (..., N), correct QQII ordering."""
    try:
        from xwr.rsp import iq_from_iiqq
        return iq_from_iiqq(raw.astype(np.int16)).astype(np.complex64)
    except Exception:
        # Mirror of xwr.rsp.iq_from_iiqq (sample_swap=False), so this module
        # still runs if the rsp backend isn't importable.
        s = (*raw.shape[:-1], raw.shape[-1] // 2)
        iq = np.zeros(s, dtype=np.complex64)
        iq[..., 0::2] = 1j * raw[..., 0::4] + raw[..., 2::4]
        iq[..., 1::2] = 1j * raw[..., 1::4] + raw[..., 3::4]
        return iq


# --------------------------------------------------------------------------- #
# Core raw-domain detector
# --------------------------------------------------------------------------- #
def interference_masks(iq: np.ndarray, k: float = 6.0, sat_level: int = 32000):
    """Per-sample interference flags for one frame.

    Args:
        iq: complex64 (chirps, tx, rx, N), fast-time last.
        k: robust threshold in MADs for the impulse detector.
        sat_level: |I| or |Q| above this counts as ADC saturation.

    Returns:
        dict of boolean masks (each shaped like a fast-time signal) plus the
        high-pass residual magnitude used for thresholding.
    """
    # 1) Saturation: either quadrature pinned near the rail.
    sat = (np.abs(iq.real) >= sat_level) | (np.abs(iq.imag) >= sat_level)

    # 2) Impulsiveness: high-pass along fast time kills smooth target beats,
    #    leaving sharp interferer edges. First difference is a cheap high-pass.
    d = np.abs(np.diff(iq, axis=-1, prepend=iq[..., :1]))
    med = np.median(d)
    mad = np.median(np.abs(d - med)) + 1e-9
    impulse = d > (med + k * mad)

    # 3) Chirp-to-chirp outlier: static scene repeats across chirps (axis 0);
    #    asynchronous interference does not. Residual vs per-frame median chirp.
    ref = np.median(iq, axis=0, keepdims=True)
    resid = np.abs(iq - ref)
    rmed = np.median(resid)
    rmad = np.median(np.abs(resid - rmed)) + 1e-9
    chirp_outlier = resid > (rmed + k * rmad)

    return {"sat": sat, "impulse": impulse, "chirp_outlier": chirp_outlier,
            "hp": d}


def frame_metrics(iq: np.ndarray, k: float = 6.0, sat_level: int = 32000) -> dict:
    """Scalar interference quantities for one frame."""
    m = interference_masks(iq, k=k, sat_level=sat_level)
    nsamp = iq.size
    # A sample is "interfered" if any raw-domain detector flags it.
    flagged = m["sat"] | m["impulse"] | m["chirp_outlier"]

    # Per-chirp: which (chirp, tx, rx) rows contain any flagged sample, and the
    # mean run-length of contiguous flagged samples (burst width) along fast time.
    chirp_hit = flagged.any(axis=-1)                      # (chirps, tx, rx)
    burst_widths = _mean_run_length(flagged)

    # Interference-to-quiet amplitude ratio: how far the worst burst sticks out
    # above the typical high-passed level (a unit-free "how strong" number, dB).
    hp = m["hp"]
    inr_db = 20.0 * np.log10((hp.max() + 1e-9) / (np.median(hp) + 1e-9))

    return {
        "corrupt_sample_frac": float(flagged.mean()),
        "saturated_frac": float(m["sat"].mean()),
        "affected_chirp_frac": float(chirp_hit.mean()),
        "mean_burst_width": float(burst_widths),
        "inr_db": float(inr_db),
    }


def _mean_run_length(mask: np.ndarray) -> float:
    """Mean length of contiguous True runs along the last axis (0 if none)."""
    flat = mask.reshape(-1, mask.shape[-1]).astype(np.int8)
    runs, n = [], flat.shape[-1]
    # Find run boundaries per row via diff on padded mask.
    padded = np.concatenate([np.zeros((flat.shape[0], 1), np.int8), flat,
                             np.zeros((flat.shape[0], 1), np.int8)], axis=1)
    d = np.diff(padded, axis=1)
    starts = np.argwhere(d == 1)
    ends = np.argwhere(d == -1)
    if len(starts) == 0:
        return 0.0
    # starts/ends are aligned row-wise by construction.
    lengths = ends[:, 1] - starts[:, 1]
    return float(lengths.mean())


def dataset_metrics(frames: np.ndarray, k: float = 6.0,
                    affected_thresh: float = 1e-4) -> dict:
    """Aggregate raw-domain metrics over a stack of frames (N, chirps, tx, rx, 2N)."""
    per = [frame_metrics(deinterleave(f), k=k) for f in frames]
    keys = per[0].keys()
    agg = {kk: float(np.median([d[kk] for d in per])) for kk in keys}
    corr = np.array([d["corrupt_sample_frac"] for d in per])
    agg["interfered_frame_rate"] = float(np.mean(corr > affected_thresh))
    agg["_per_frame"] = per
    return agg


# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #
def capture_frames(cfg: dict, n_frames: int, settle: int = 10) -> np.ndarray:
    import xwr
    system = xwr.XWRSystem(**cfg)
    q = system.qstream(numpy=True)
    frames, got = [], 0
    try:
        while got < n_frames + settle:
            f = q.get()
            if f is None:
                break
            got += 1
            if got > settle:
                frames.append(f.copy())
    finally:
        system.stop()
    if not frames:
        raise RuntimeError("No frames captured. Check IP, rmem_max, serial "
                           "port, and that the victim config is valid.")
    return np.stack(frames)


def load_cfg(path: str) -> dict:
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def cmd_smoke(args):
    cfg = load_cfg(args.config)
    frames = capture_frames(cfg, args.n)
    print(f"captured {frames.shape[0]} frames, frame shape {frames.shape[1:]}, "
          f"dtype {frames.dtype}")
    np.save(args.out, frames)
    print(f"saved -> {args.out}")
    m = dataset_metrics(frames)
    print(f"raw-domain baseline stats: corrupt_sample_frac={m['corrupt_sample_frac']:.2e} "
          f"saturated_frac={m['saturated_frac']:.2e} "
          f"interfered_frame_rate={m['interfered_frame_rate']*100:.1f}%")
    # Optional: plot one raw chirp magnitude so you can eyeball the signal.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        iq = deinterleave(frames[0])
        plt.figure(figsize=(7, 3))
        plt.plot(np.abs(iq[0, 0, 0]))
        plt.xlabel("fast-time sample"); plt.ylabel("|IQ|")
        plt.title("Smoke test: one raw chirp (rx0)")
        plt.tight_layout(); plt.savefig("smoke_raw_chirp.png", dpi=110)
        print("saved -> smoke_raw_chirp.png")
    except ImportError:
        print("(matplotlib not installed; skipped PNG)")


def cmd_capture(args):
    cfg = load_cfg(args.config)
    frames = capture_frames(cfg, args.n)
    np.save(args.out, frames)
    print(f"captured {frames.shape[0]} frames -> {args.out}")


def cmd_pairs(args):
    cfg = load_cfg(args.config)
    print("Capturing CLEAN set -- make sure the aggressor is OFF.")
    input("Press Enter when ready...")
    np.save(f"{args.prefix}_clean.npy", capture_frames(cfg, args.n))
    print(f"saved -> {args.prefix}_clean.npy")
    print("\nNow START the aggressor (keep the scene identical), then:")
    input("Press Enter once the aggressor is transmitting...")
    np.save(f"{args.prefix}_interfered.npy", capture_frames(cfg, args.n))
    print(f"saved -> {args.prefix}_interfered.npy")
    print("\nAnalyze with:\n  uv run python interference.py analyze "
          f"{args.prefix}_clean.npy {args.prefix}_interfered.npy")


def _print(name, m):
    print(f"[{name}]  corrupt_samples={m['corrupt_sample_frac']:.2e}  "
          f"saturated={m['saturated_frac']:.2e}  "
          f"affected_chirps={m['affected_chirp_frac']*100:.1f}%  "
          f"interfered_frames={m['interfered_frame_rate']*100:.1f}%  "
          f"mean_burst={m['mean_burst_width']:.1f} smp  INR={m['inr_db']:.1f} dB")


def cmd_analyze(args):
    base = dataset_metrics(np.load(args.files[0]), k=args.k)
    _print(args.files[0], base)
    intf = None
    if len(args.files) > 1:
        intf = dataset_metrics(np.load(args.files[1]), k=args.k)
        _print(args.files[1], intf)
        print("\n--- interfered vs baseline ---")
        print(f"corrupt-sample fraction : {base['corrupt_sample_frac']:.2e} "
              f"-> {intf['corrupt_sample_frac']:.2e}")
        print(f"saturated fraction      : {base['saturated_frac']:.2e} "
              f"-> {intf['saturated_frac']:.2e}")
        print(f"affected-chirp fraction : {base['affected_chirp_frac']*100:.1f}% "
              f"-> {intf['affected_chirp_frac']*100:.1f}%")
        print(f"interfered-frame rate   : {base['interfered_frame_rate']*100:.1f}% "
              f"-> {intf['interfered_frame_rate']*100:.1f}%")
        print(f"INR (burst over quiet)  : {base['inr_db']:.1f} -> {intf['inr_db']:.1f} dB")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 3))
            b = [d["corrupt_sample_frac"] for d in base["_per_frame"]]
            plt.plot(b, label=args.files[0])
            if intf is not None:
                i = [d["corrupt_sample_frac"] for d in intf["_per_frame"]]
                plt.plot(i, label=args.files[1])
            plt.xlabel("frame"); plt.ylabel("corrupt-sample fraction")
            plt.title("Per-frame interference (drifts in/out if unsynced)")
            plt.legend(); plt.tight_layout(); plt.savefig("interference_timeline.png", dpi=110)
            print("\nsaved -> interference_timeline.png")
        except ImportError:
            print("(matplotlib not installed; skipped plot)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("smoke"); s.add_argument("--config", default="victim.yaml")
    s.add_argument("--n", type=int, default=100); s.add_argument("--out", default="smoke.npy")
    s.set_defaults(func=cmd_smoke)

    c = sub.add_parser("capture"); c.add_argument("--config", default="victim.yaml")
    c.add_argument("--n", type=int, default=300); c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_capture)

    p = sub.add_parser("pairs"); p.add_argument("--config", default="victim.yaml")
    p.add_argument("--n", type=int, default=500); p.add_argument("--prefix", default="pair")
    p.set_defaults(func=cmd_pairs)

    a = sub.add_parser("analyze"); a.add_argument("files", nargs="+")
    a.add_argument("--k", type=float, default=6.0, help="MAD threshold for detectors.")
    a.add_argument("--plot", action="store_true")
    a.set_defaults(func=cmd_analyze)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())