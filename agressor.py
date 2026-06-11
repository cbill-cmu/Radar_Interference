#!/usr/bin/env python3
"""Transmit-only interferer (aggressor) for the radar interference study.

The aggressor needs NO capture card. It only needs power + USB serial.
We configure it over its control serial port and issue sensorStart; it then
radiates FMCW chirps into the scene. Its LVDS stream goes nowhere (no DCA
attached), which is fine -- the TX still transmits.

Run this in its own terminal/process while `interference.py` captures the
victim in another:

    uv run python aggressor.py --port /dev/serial/by-id/<AGGRESSOR_ENHANCED>

By default the aggressor uses the victim's known-valid 256x64 timing but with
a DIFFERENT chirp slope and a slightly different frame period. The slope
mismatch makes its sweep cross the victim's IF passband (=> clear broadband
interference bursts), and the frame-period offset makes those bursts drift
through the capture instead of landing in the same place every frame.

Ctrl-C (or SIGTERM) stops the radar cleanly. If you kill -9 it instead, the
radar may keep transmitting until power-cycled.
"""
import argparse
import logging
import signal
import sys
import time

from xwr.radar import AWR1843


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Transmit-only radar interferer.")
    p.add_argument("--port", required=True,
                   help="aggressor control serial port (by-id Enhanced port).")
    # Defaults mirror the valid victim 256x64 profile EXCEPT freq_slope and
    # frame_period, so the config stays inside the radar's timing limits.
    p.add_argument("--frequency", type=float, default=77.0)
    p.add_argument("--freq-slope", type=float, default=50.0,
                   help="MHz/us. Differ from the victim (67.012) to get "
                        "broadband interference bursts.")
    p.add_argument("--idle-time", type=float, default=6.0)
    p.add_argument("--adc-start-time", type=float, default=5.7)
    p.add_argument("--ramp-end-time", type=float, default=34.0)
    p.add_argument("--tx-start-time", type=float, default=1.0)
    p.add_argument("--adc-samples", type=int, default=256)
    p.add_argument("--sample-rate", type=int, default=10000)
    p.add_argument("--frame-length", type=int, default=64)
    p.add_argument("--frame-period", type=float, default=55.0,
                   help="ms. Offset from the victim (50.0) so interference "
                        "events drift across frames.")
    p.add_argument("--duration", type=float, default=0.0,
                   help="seconds to transmit; 0 = until Ctrl-C.")
    return p


def start_aggressor(args) -> AWR1843:
    """Configure and start the interferer. Returns the radar handle."""
    radar = AWR1843(port=args.port, name="aggressor")
    radar.setup(
        frequency=args.frequency,
        idle_time=args.idle_time,
        adc_start_time=args.adc_start_time,
        ramp_end_time=args.ramp_end_time,
        tx_start_time=args.tx_start_time,
        freq_slope=args.freq_slope,
        adc_samples=args.adc_samples,
        sample_rate=args.sample_rate,
        frame_length=args.frame_length,
        frame_period=args.frame_period,
    )
    radar.start()  # sensorStart -> begins transmitting
    return radar


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [aggressor] %(message)s")

    radar = start_aggressor(args)
    logging.info("transmitting: slope=%.3f MHz/us, frame_period=%.1f ms",
                 args.freq_slope, args.frame_period)

    stopped = {"done": False}

    def shutdown(*_):
        if not stopped["done"]:
            stopped["done"] = True
            try:
                radar.stop()
            finally:
                logging.info("aggressor stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if args.duration > 0:
        time.sleep(args.duration)
        shutdown()
    else:
        signal.pause()  # sleep until a signal arrives


if __name__ == "__main__":
    main()