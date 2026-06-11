# Radar mutual-interference study (TI AWR1843AOPEVM × 2, xwr + DCA1000)

A harness for studying mutual interference between two FMCW mmWave radars. One
radar is the **victim** (wired to the DCA1000 capture card, data recorded and
analyzed); the other is the **aggressor** (transmits only, no capture card).
Built on the [`xwr`](https://github.com/RadarML/xwr) stack for radar control,
DCA1000 capture, and signal processing.

---

## 1. Refresher — how the whole thing works

**Why two roles.** The DCA1000 captures raw IQ from exactly one radar over LVDS.
So the victim is the one on the capture card; the aggressor needs only USB
serial + 5 V power and just radiates chirps into the scene. Its energy couples
into the victim's receiver — that coupling is the interference under study.

**What interference does (physics).** When the aggressor's frequency sweep
crosses the victim's IF passband it injects a short, high-amplitude, broadband
burst into a run of fast-time ADC samples (often saturating the ADC). Because
the two radars are unsynchronized with different frame periods, these events
drift in and out across frames. In processed views this appears as a flickering
raised noise floor and/or ghost targets; in the raw data it appears as localized
impulsive bursts and clipping.

**The study (current goal).** A 2D sweep over **pointing angle** {0, 90, 180,
270 deg} x **frequency offset** (4-5 values), with **polarization** (board roll)
as a second, separately-labeled physical variable. The governing question is
*not* "how many dB" but **"is the raw victim data still usable, or corrupted
beyond recovery?"** Approach: capture a clean baseline you can see, turn on the
aggressor, and when you see something wrong in the live view, clip the raw data
(with a few seconds before/after) and compare against baseline. The
classification of interference is *derived from* these observed clip pairs
rather than assumed up front.

---

## 2. Hardware

- 2x AWR1843AOPEVM, both flashed with the mmW demo firmware, both in functional
  mode (S2.2 up, others down).
- 1x DCA1000EVM, powered from its **own 5 V** barrel jack (switch on
  `DC_JACK_5V_IN` -- the AOP does not back-feed 5 V like the BoosterPack boards).
- Victim <-> DCA1000 over the LVDS ribbon + Ethernet to the host (192.168.33.x).
- Both radars on USB serial. The AOP uses a CP2105 dual UART -> `ttyUSB*`; the
  **Enhanced** (`if00`) node is the control port. Identify each board by its
  stable `/dev/serial/by-id/...` path (serial numbers differ per board).

---

## 3. Repo layout

```
radar-interference/
|-- README.md              # this file
|-- pyproject.toml         # deps (numpy, pyyaml, matplotlib, xwr)
|-- .gitignore             # excludes raw data; keeps JSON sidecars
|-- victim.example.yaml    # config template (committed)
|-- victim.yaml            # real config w/ this box's port (gitignored)
|-- aggressor.py           # transmit-only interferer (xwr.radar.AWR1843)
|-- monitor.py             # browser live view + retroactive clip capture
|-- render_clip.py         # render baseline-vs-clip comparison graphs
|-- interference.py        # raw-domain interference detection / quantification
|-- study.py               # automated frequency/parameter sweep (legacy metrics)
`-- clips/                 # captured data + sidecars (data gitignored)
```

**Script reference**

- `aggressor.py` -- configures the second radar over serial and starts it
  transmitting. No DCA involved. Differ its `--freq-slope`/`--frequency` from the
  victim to provoke interference. Ctrl-C stops it cleanly.
- `monitor.py` -- runs the victim through the same processing as the repo's
  `demo.py` (range-Doppler + range-azimuth via `xwr.rsp`) and serves the live
  panels to your browser over HTTP. Keeps a rolling buffer of raw frames; a
  button/spacebar saves a clip spanning seconds before and after the click, plus
  a JSON sidecar of the conditions. This is the primary data-collection tool.
- `render_clip.py` -- turns a saved clip (+ optional baseline) into side-by-side
  comparison graphs (`comparison_mean.png`, `comparison_worst.png`).
- `interference.py` -- raw-domain detector/quantifier (saturation, high-pass
  impulsiveness, chirp-to-chirp outliers) for *measuring* corruption once you
  know what you're looking for. Subcommands: `smoke`, `capture`, `pairs`,
  `analyze`.
- `study.py` -- orchestrates an automated sweep; predates the clip workflow and
  logs the older noise-style metrics (update its CSV columns if you want it to
  log the raw-domain metrics instead).

---

## 4. Setup

```bash
git clone <your-repo-url> radar-interference
cd radar-interference
uv sync                          # installs deps incl. xwr (see Fallback if this fails)
cp victim.example.yaml victim.yaml
ls -l /dev/serial/by-id/         # put the victim's if00 path in victim.yaml
```

**OS setup (per boot).** Modern Ubuntu has no `ifconfig`; use `ip`:

```bash
RADAR_IF=$(ip -o link | awk -F': ' '/enp|eth/{print $2; exit}')
sudo ip addr add 192.168.33.30/24 dev $RADAR_IF && sudo ip link set $RADAR_IF up
echo 6291456 | sudo tee /proc/sys/net/core/rmem_max
sudo chmod 777 /dev/ttyUSB* 2>/dev/null
```

**Fallback (if `uv sync` can't install xwr):** keep developing inside your `xwr`
clone -- put this repo's scripts in a folder there and run with
`uv run --project ~/xwr python <script>.py`, exactly as during development. The
git-dependency in `pyproject.toml` is the cleaner target, but the in-clone path
is the proven one.

---

## 5. Workflow

**Phase 1 -- smoke test (aggressor off).** Confirm capture works:
```bash
uv run python interference.py smoke --config victim.yaml
```
Move a metal object in front of the victim; raw stats should react.

**Phase 2 -- aggressor up (separate terminal / tmux pane).**
```bash
uv run python aggressor.py --port /dev/serial/by-id/<AGGRESSOR_if00> --freq-slope 50.0
```
Wait for `transmitting:`. If `start()` throws `Errno 5`, power-cycle that board.

**Phase 3 -- live monitor + baseline + clips (primary).**
```bash
uv run python monitor.py --config victim.yaml --buffer-seconds 3 --post-seconds 3
```
Open `http://<radar-host>:8000`. With the aggressor off, fill the condition
fields and click **Save BASELINE**. Turn the aggressor on, watch, and **Save
CLIP** (or spacebar) when you see interference.

**Phase 4 -- analyze.**
```bash
uv run python render_clip.py clips/clip_<ts>.npy --baseline clips/baseline_<ts>.npy
uv run python interference.py analyze clips/baseline_<ts>.npy clips/clip_<ts>.npy --k 8
```

**The 2D sweep.** For each physical (angle, polarization) you set by hand, take a
baseline + clips across the freq offsets, labeling every capture in the page
fields. ~16 physical placements x 4-5 offsets. Automating the aggressor's
freq-offset stepping is the next script (`sweep.py`).

---

## 6. Data management

Raw clips are large (~0.75 MB/frame; a 6 s clip ~ 90 MB), so **raw data never
goes in git**. The `.gitignore` excludes `*.npy` and `*.png` but **keeps the
`clips/**/*.json` sidecars** -- those are tiny and form the experiment record
(what conditions were captured, when). Store the bulk `.npy`/`.png` outside the
repo (a data drive, lab NAS, or object storage); reference them by the sidecar
filenames. Commit `uv.lock` for reproducible installs. Keep the real
`victim.yaml` out of git (it holds this box's serial path); commit only
`victim.example.yaml`.

Organize captures by condition, e.g. `clips/angle090_pol000/` or rely on the
sidecar fields -- pick one and be consistent so the dataset stays interpretable.

---

## 7. Putting it in a repo -- exact steps

```bash
cd radar-interference          # the folder containing these files
git init
git add README.md pyproject.toml .gitignore victim.example.yaml \
        aggressor.py monitor.py render_clip.py interference.py study.py \
        clips/.gitkeep
git commit -m "Radar mutual-interference harness: aggressor, monitor, analysis"
# create an empty repo on GitHub/GitLab first, then:
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

Confirm the data exclusion worked before pushing:
```bash
git status --porcelain          # should show NO *.npy / *.png files
git check-ignore clips/test.npy # should print the path (i.e. it IS ignored)
```

---

## 8. Gotchas (consolidated)

- Two AOP boards -> auto-detect is ambiguous; always pin explicit `by-id` ports
  for both victim (in `victim.yaml`) and aggressor (`--port`).
- Only one process can own the DCA: run the monitor **or** a capture script on
  the victim, never both at once. The aggressor is independent.
- A bad aggressor `start()` can freeze that board -> power-cycle to recover.
- Live browser view uses the latest frame and is pull-throttled (~2/s); saved
  clips use a gap-free `qstream` buffer, so analysis stays faithful.
- Run scripts from the repo root so cross-imports resolve.
- The DCA1000 is 5 V only -- never feed it 12 V.