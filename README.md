# Radar mutual-interference study (TI AWR1843AOPEVM x 2, xwr + DCA1000)

Victim = radar on the DCA1000 (recorded). Aggressor = second radar, transmit
only (serial + 5 V, no capture card). You watch the victim live in a browser,
clip raw frames (a few seconds before/after) when you see interference, and
render/quantify those clips afterward.

---

## Repo layout

```
Radar_Interference/
|-- README.md  pyproject.toml  .gitignore
|-- victim.example.yaml      # committed templates (real *.yaml are gitignored)
|-- aggressor.example.yaml
|-- victim.yaml              # real victim config w/ this box's port
|-- aggressor.yaml           # real aggressor config w/ this box's port
|-- aggressor.py             # HARDWARE: transmit-only interferer (reads aggressor.yaml)
|-- monitor.py               # HARDWARE: victim live view + clip capture -> results/
|-- analysis/
|   |-- render_clip.py        # stills / frame strip / video from a clip
|   `-- interference.py       # raw-domain corruption metrics
`-- results/                 # RAW captured clips (.npy) + sidecars (.json)
    `-- .gitkeep
```

`results/` holds raw data; `analysis/` holds the scripts you run on it and the
rendered outputs. Raw `.npy` and renders (`.png/.gif/.mp4`) are gitignored; the
`.json` sidecars (condition labels) are kept as the experiment record. Both
radars are configured from YAML (`victim.yaml`, `aggressor.yaml`); copy each
from its `.example.yaml` and set the board's `if00` port.

---

## One-time setup (per boot / reconnect)

```bash
# 1. DCA network (no ifconfig on modern Ubuntu)
RADAR_IF=$(ip -o link | awk -F': ' '/enp|eth/{print $2; exit}')
sudo ip addr add 192.168.33.30/24 dev $RADAR_IF && sudo ip link set $RADAR_IF up
echo 6291456 | sudo tee /proc/sys/net/core/rmem_max
sudo chmod 777 /dev/ttyUSB* 2>/dev/null

# 2. Identify the two boards (Enhanced = if00 = control port)
ls -l /dev/serial/by-id/
#   victim if00     -> put in victim.yaml (radar.port)
#   aggressor if00  -> put in aggressor.yaml (radar.port)
```

First-time only, create the real configs from the templates and set the ports:
```bash
cp victim.example.yaml victim.yaml
cp aggressor.example.yaml aggressor.yaml
# then edit radar.port in each
```

DCA1000 powered from its own 5 V jack (switch on DC_JACK_5V_IN). Both radars in
functional mode (S2.2 up, others down).

---

## The experiment: one cell at a time

The study is a 2D sweep -- pointing **angle** {0, 90, 180, 270 deg} x **frequency
offset** (4-5 values) -- repeated for each **polarization** (board roll). One
"cell" = one (angle, polarization, freq-offset) combination.

### A. Start the session (once per sitting)

Terminal 1 (tmux pane), victim live view:
```bash
cd ~/.../Radar_Interference
uv run python monitor.py --config victim.yaml --buffer-seconds 3 --post-seconds 3
```
Open `http://<box-LAN-ip>:8000` (get it with `hostname -I`; use the 172.x, not
192.168.33.x). Leave this running for the whole session.

### B. For each PHYSICAL configuration (set by hand)

1. Physically set the **angle** and **polarization (roll)** of the boards.
2. In the browser, fill the condition fields: angle, pol, separation.
3. **Aggressor OFF.** Click **Save BASELINE**. (One baseline per physical config.)

### C. For each FREQUENCY OFFSET within that configuration (the actual runs)

1. Start the aggressor at this offset. Base chirp params live in
   `aggressor.yaml`; override only the sweep variable (start frequency) per run.
   Offset = aggressor frequency - victim's 77.0 GHz:
   ```bash
   # Terminal 2 (separate tmux pane).
   pkill -f aggressor.py                      # clear any stranded instance first
   uv run python aggressor.py --frequency 77.1   # e.g. +100 MHz offset
   ```
   Wait for `transmitting:`. (Errno5 / TimeoutError -> power-cycle that board.)
2. Set the `foff` field in the browser to match.
3. Watch the live panels. When you see interference (full-screen smear =
   mismatched-slope wipeout; localized fake peak = ghost target), **Save CLIP**
   (or spacebar). Grab a few clips if events are intermittent.
4. **Stop the aggressor** (Ctrl-C in its pane; wait for `aggressor stopped`).
5. Next offset -> repeat C.

Everything you clip lands in `results/` as `clip_<timestamp>.npy` + `.json`
(the sidecar records angle/pol/foff/sep/note). ~16 physical configs x 4-5
offsets.

---

## Analysis (after capture)

```bash
cd ~/.../Radar_Interference/analysis

# Side-by-side stills vs baseline:
uv run python render_clip.py ../results/clip_<ts>.npy --baseline ../results/baseline_<ts>.npy

# Frame strip (whole event at a glance) + slowed video:
uv run python render_clip.py ../results/clip_<ts>.npy --strip --video --fps 5

# Quantify "usable vs corrupted" (raw-domain metrics):
uv run python interference.py analyze ../results/baseline_<ts>.npy ../results/clip_<ts>.npy --k 8
```

Renders (`comparison_*.png`, `frame_strip.png`, `clip_video.gif/mp4`) are written
into `analysis/`. A clip with a fully-lit frame among clean neighbors is the
normal slope-mismatch signature: that frame is corrupted (broadband wipeout),
its neighbors usable. The fraction of corrupted frames is your answer to
"can these run simultaneously."

---

## Gotchas

- Pin explicit `by-id` ports for both boards (auto-detect is ambiguous with two).
- Run the monitor OR a capture script on the victim, never both (one owns the DCA).
- tmux: detaching leaves the aggressor transmitting AND holding its port; stop it
  with Ctrl-C (or `pkill -f aggressor.py`) before starting a new one.
- A frozen board (after Errno5/Timeout) needs a power-cycle, then re-chmod.
- DCA1000 is 5 V only -- never 12 V.