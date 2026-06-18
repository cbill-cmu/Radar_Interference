#!/usr/bin/env python3
"""Transmit-only interferer (aggressor), configured from a YAML file.

Reads aggressor.yaml (same structure as victim.yaml) so you don't retype every
chirp parameter. The sweep variables that change per run can be overridden on
the command line without editing the file -- e.g. --frequency for a
frequency-offset sweep:

    uv run python aggressor.py --config aggressor.yaml
    uv run python aggressor.py --config aggressor.yaml --frequency 77.1   # +100 MHz offset

No DCA/capture card involved -- serial + 5 V power only. The aggressor radiates
chirps; its LVDS stream goes nowhere, which is fine. Ctrl-C (or SIGTERM) stops
it cleanly; a hard kill can leave the radar transmitting until power-cycled.
"""
import argparse
import logging
import signal
import sys
import time

import yaml


def load_config(path, overrides):
    """Load radar config from YAML; apply any non-None CLI overrides on top."""
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    radar = dict(cfg["radar"])
    for key, val in overrides.items():
        if val is not None:
            radar[key] = val
    return radar


def start_aggressor(radar):
    """Instantiate the radar class named by `device` and start transmitting."""
    import xwr.radar as xradar
    device = radar.get("device", "AWR1843")
    RadarClass = getattr(xradar, device)          # generic: AWR1843, AWR1642, ...
    r = RadarClass(port=radar["port"], name="aggressor")
    r.setup(
        frequency=radar["frequency"],
        idle_time=radar["idle_time"],
        adc_start_time=radar["adc_start_time"],
        ramp_end_time=radar["ramp_end_time"],
        tx_start_time=radar["tx_start_time"],
        freq_slope=radar["freq_slope"],
        adc_samples=radar["adc_samples"],
        sample_rate=radar["sample_rate"],
        frame_length=radar["frame_length"],
        frame_period=radar["frame_period"],
    )
    r.start()
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="settings/aggressor.yaml",
                    help="YAML config (same structure as victim.yaml)")
    # Per-run overrides (optional). Anything left out uses the YAML value.
    ap.add_argument("--port", default=None, help="override config port")
    ap.add_argument("--frequency", type=float, default=None, help="override start freq (GHz)")
    ap.add_argument("--freq-slope", type=float, default=None, help="override slope (MHz/us)")
    ap.add_argument("--frame-period", type=float, default=None, help="override frame period (ms)")
    ap.add_argument("--duration", type=float, default=0.0, help="seconds; 0 = until Ctrl-C")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [aggressor] %(message)s")

    radar = load_config(args.config, {
        "port": args.port,
        "frequency": args.frequency,
        "freq_slope": args.freq_slope,
        "frame_period": args.frame_period,
    })

    r = start_aggressor(radar)
    logging.info("transmitting: freq=%.3f GHz, slope=%.3f MHz/us, frame_period=%.1f ms",
                 radar["frequency"], radar["freq_slope"], radar["frame_period"])

    stopped = {"done": False}

    def shutdown(*_):
        if not stopped["done"]:
            stopped["done"] = True
            try:
                r.stop()
            finally:
                logging.info("aggressor stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if args.duration > 0:
        time.sleep(args.duration)
        shutdown()
    else:
        signal.pause()


if __name__ == "__main__":
    main()