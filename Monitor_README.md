# Browser monitor + retroactive clip capture

No X server needed. The monitor captures the victim continuously, renders the
live view as an image served over HTTP, and lets you save a raw clip spanning a
few seconds **before and after** the moment you click.

## 1. Start the monitor (radar box, victim on the DCA1000)

```bash
cd ~/xwr
uv run python Collection/monitor.py \
  --config Collection/victim.yaml --rsp AWR1843AOP --device AWR1843 \
  --buffer-seconds 3 --post-seconds 3 --outdir Collection/clips
```

`--buffer-seconds` is the pre-trigger window, `--post-seconds` the post-trigger
window. Both are tunable; raise them if events are brief or your reaction is slow.

## 2. Open the page (laptop browser)

Go to `http://<radar-host>:8000` (find the box IP with `hostname -I` if the
name doesn't resolve, e.g. `http://192.168.x.x:8000`). You'll see two live
panels — **Range-Doppler** and **Range-Azimuth** — exactly like demo.py, on a
dB scale by default (uncheck "dB scale" for demo's linear view).

## 3. Capture a baseline (aggressor OFF)

Fill in the condition fields (angle, polarization roll, freq offset, separation,
note), then click **Save BASELINE**. These fields are recorded in a JSON sidecar
next to every clip, so each capture is self-describing.

## 4. Capture interference (aggressor ON)

Start the aggressor in another terminal, watch the panels, and when you see
something wrong click **Save CLIP** (or press **spacebar**). The status line
shows the buffer filling and confirms each save. The clip contains gap-free raw
frames from before and after your click.

## 5. Compare later

```bash
uv run python Collection/render_clip.py \
  Collection/clips/clip_<ts>.npy \
  --baseline Collection/clips/baseline_<ts>.npy --rsp AWR1843AOP \
  --outdir Collection/clips
```

Produces `comparison_mean.png` (typical picture) and `comparison_worst.png`
(baseline vs the single most-energetic clip frame — the captured event).

## Notes

- Live view uses the latest frame; saved clips use a gap-free `qstream` buffer,
  so analysis is faithful even though the on-screen view may skip frames.
- One process owns the DCA. Run **only** the monitor on the victim (not the
  monitor and a separate capture at once).
- The aggressor is independent (serial only); start/stop it as before.
- For the 2D sweep (4 angles x 4 polarizations x 4-5 freq offsets), set each
  physical angle/polarization by hand, take a baseline + clips per freq offset,
  and label every capture via the page fields. Automating the freq-offset
  stepping of the aggressor is the next script (sweep.py) when you want it.