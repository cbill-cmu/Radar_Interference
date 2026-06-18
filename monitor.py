#!/usr/bin/env python3
"""Browser-based live radar monitor with retroactive (pre/post) clip capture.

Runs the victim through the same processing as the repo's `demo.py`
(XWRSystem capture + xwr.rsp range-Doppler / range-azimuth), but instead of an
X-forwarded matplotlib window it serves the two panels as an image to your
laptop browser over HTTP -- no X server needed.

Workflow
--------
1. Start this on the radar box (victim wired to the DCA1000):
     uv run python monitor.py --config victim.yaml --rsp AWR1843AOP --device AWR1843
2. On your laptop, open  http://<radar-host>:8000  (e.g. http://radar1:8000).
3. With the aggressor OFF, fill in the condition fields and click "Save BASELINE".
4. Turn the aggressor ON. Watch the live panels. When you see something wrong,
   click "Save CLIP" (or press the spacebar). It saves the rolling buffer of raw
   frames from a few seconds BEFORE your click plus a few seconds AFTER, so you
   catch the event despite reaction lag.
5. Render comparisons later with render_clip.py.

The buffer keeps RAW int16 frames (gap-free, via qstream) so saved clips are
faithful for analysis; the browser view is rendered from the latest frame.
"""
import argparse
import io
import json
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Shared state between the capture thread and the HTTP server
# --------------------------------------------------------------------------- #
class State:
    def __init__(self, pre_frames, post_frames, outdir, cfg, rsp_name):
        self.lock = threading.Lock()
        self.ring = deque(maxlen=pre_frames)     # rolling PAST buffer (raw int16)
        self.pre_frames = pre_frames
        self.post_frames = post_frames
        self.latest = None                       # most recent raw frame (for view)
        self.frame_count = 0
        self.fps = 0.0
        self.pending = None                      # active clip capture, or None
        self.last_saved = None
        self.outdir = outdir
        self.cfg = cfg
        self.rsp_name = rsp_name
        self.db = True
        os.makedirs(outdir, exist_ok=True)


def capture_loop(state, cfg, stop_event):
    import xwr
    log = logging.getLogger("monitor.capture")
    system = xwr.XWRSystem(**cfg)
    q = system.qstream(numpy=True)
    t0, n = time.perf_counter(), 0
    try:
        while not stop_event.is_set():
            f = q.get()
            if f is None:
                break
            g = f.copy()
            with state.lock:
                state.ring.append(g)
                state.latest = g
                state.frame_count += 1
                if state.pending is not None:
                    state.pending["future"].append(g)
                    if len(state.pending["future"]) >= state.pending["need"]:
                        p = state.pending
                        state.pending = None
                        threading.Thread(target=_save_clip, args=(state, p),
                                         daemon=True).start()
            n += 1
            dt = time.perf_counter() - t0
            if dt >= 2.0:
                state.fps = n / dt
                n, t0 = 0, time.perf_counter()
    finally:
        system.stop()
        log.info("capture stopped")


def _save_clip(state, p):
    log = logging.getLogger("monitor.save")
    frames = np.stack(p["past"] + p["future"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(state.outdir, f"{p['label']}_{ts}")
    np.save(base + ".npy", frames)
    meta = dict(p["meta"])
    meta.update(label=p["label"], timestamp=ts,
                n_frames=int(frames.shape[0]),
                pre_frames=len(p["past"]), post_frames=len(p["future"]),
                frame_shape=list(frames.shape[1:]),
                rsp=state.rsp_name, radar_cfg=state.cfg.get("radar", {}))
    with open(base + ".json", "w") as fh:
        json.dump(meta, fh, indent=2)
    with state.lock:
        state.last_saved = base + ".npy"
    log.info("saved clip %s.npy (%d frames)", base, frames.shape[0])


# --------------------------------------------------------------------------- #
# Processing -- mirrors demo.py exactly (range-Doppler + range-azimuth)
# --------------------------------------------------------------------------- #
class Renderer:
    """Dual-panel (range-Doppler + range-azimuth) renderer on a BACKGROUND thread.

    The render loop runs independently at a target rate and writes the latest
    PNG into a cache; the HTTP handler returns those cached bytes instantly, so
    a browser request NEVER waits on rsp/matplotlib. Render cost only limits how
    fast the picture updates -- it can never make the page laggy.
    """
    def __init__(self, rsp_name, target_fps=4.0, azimuth=64, dpi=72):
        from xwr.rsp import numpy as xwr_rsp
        # Smaller azimuth FFT than demo's 128 keeps the second panel affordable.
        self.rsp = getattr(xwr_rsp, rsp_name)(window=False, size={"azimuth": azimuth})
        self._fig, self._axs = plt.subplots(1, 2, figsize=(9, 3.8))
        self._ims = [None, None]
        self._axs[0].set_xlabel("Doppler"); self._axs[0].set_ylabel("Range")
        self._axs[0].set_title("Range-Doppler")
        self._axs[1].set_xlabel("Azimuth"); self._axs[1].set_ylabel("Range")
        self._axs[1].set_title("Range-Azimuth")
        self._fig.tight_layout()
        self._dpi = dpi
        self._interval = 1.0 / target_fps
        self._png = None
        self._lock = threading.Lock()

    def _panels(self, frame):
        # demo.py processing: full DRAE, then mean to RD and RA.
        dear = np.abs(self.rsp(frame[None, ...]))
        rd = np.swapaxes(np.mean(dear, axis=(0, 2, 3)), 0, 1)   # (range, doppler)
        ra = np.swapaxes(np.mean(dear, axis=(0, 1, 2)), 0, 1)   # (range, azimuth)
        return rd, ra

    def _render(self, frame, db):
        rd, ra = self._panels(frame)
        if db:
            rd = 20 * np.log10(rd + 1e-6); ra = 20 * np.log10(ra + 1e-6)
        for i, data in enumerate((rd, ra)):
            if self._ims[i] is None:
                self._ims[i] = self._axs[i].imshow(
                    data, cmap="viridis", aspect="auto", origin="lower")
            else:
                self._ims[i].set_data(data)
            self._ims[i].set_clim(vmin=float(np.min(data)), vmax=float(np.max(data)))
        buf = io.BytesIO()
        self._fig.savefig(buf, format="png", dpi=self._dpi)
        return buf.getvalue()

    def render_loop(self, state, stop_event):
        log = logging.getLogger("monitor.render")
        while not stop_event.is_set():
            t0 = time.perf_counter()
            with state.lock:
                frame = state.latest
                db = state.db
            if frame is not None:
                try:
                    png = self._render(frame, db)
                    with self._lock:
                        self._png = png
                except Exception as e:
                    log.warning("render error: %s", e)
            dt = time.perf_counter() - t0
            time.sleep(max(0.0, self._interval - dt))

    def latest_png(self):
        with self._lock:
            return self._png


# --------------------------------------------------------------------------- #
# HTTP server
# --------------------------------------------------------------------------- #
PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Radar interference monitor</title>
<style>
 body{font-family:system-ui,sans-serif;margin:16px;background:#111;color:#eee}
 #view{width:100%;max-width:1000px;border:1px solid #333;background:#000}
 .row{margin:8px 0} label{display:inline-block;width:120px}
 input{background:#222;color:#eee;border:1px solid #444;padding:4px;width:160px}
 button{padding:10px 18px;font-size:16px;margin-right:10px;cursor:pointer}
 #clip{background:#b33;color:#fff;border:0} #base{background:#357;color:#fff;border:0}
 #status{font-family:monospace;color:#9c9;white-space:pre}
</style></head><body>
<h2>Radar interference monitor</h2>
<img id="view" src="/frame.png"><br>
<div class="row"><label><input type="checkbox" id="db" checked> dB scale</label></div>
<div class="row"><label>Pointing angle</label><input id="angle" placeholder="deg, e.g. 90">
  <label>Polarization</label><input id="pol" placeholder="roll deg, e.g. 0"></div>
<div class="row"><label>Freq offset</label><input id="foff" placeholder="MHz, e.g. 100">
  <label>Separation</label><input id="sep" placeholder="cm, e.g. 30"></div>
<div class="row"><label>Note</label><input id="note" placeholder="free text" style="width:340px"></div>
<div class="row">
  <button id="clip">Save CLIP (spacebar)</button>
  <button id="base">Save BASELINE</button>
</div>
<div id="status">connecting...</div>
<script>
 const v=document.getElementById('view');
 // Cached bytes -> requests are instant; refresh on a steady timer.
 function refresh(){const im=new Image();
   im.onload=()=>{v.src=im.src;}; im.src='/frame.png?t='+Date.now();}
 setInterval(refresh,300);
 document.getElementById('db').onchange=function(){
   fetch('/config?db='+(this.checked?1:0));};
 function meta(){return 'angle='+encodeURIComponent(angle.value)+'&pol='+encodeURIComponent(pol.value)
   +'&foff='+encodeURIComponent(foff.value)+'&sep='+encodeURIComponent(sep.value)
   +'&note='+encodeURIComponent(note.value);}
 function trig(label){fetch('/trigger?label='+label+'&'+meta()).then(r=>r.text()).then(t=>{status.textContent=t;});}
 document.getElementById('clip').onclick=()=>trig('clip');
 document.getElementById('base').onclick=()=>trig('baseline');
 document.addEventListener('keydown',e=>{if(e.code==='Space'){e.preventDefault();trig('clip');}});
 setInterval(()=>fetch('/status').then(r=>r.json()).then(s=>{
   status.textContent='frames='+s.frames+'  fps='+s.fps.toFixed(1)
     +'  buffer='+s.buffer_fill+'/'+s.pre+'  '+(s.pending?'[CAPTURING POST...]':'idle')
     +(s.last_saved?('\\nlast saved: '+s.last_saved):'');}),1000);
</script></body></html>"""


def make_handler(state, renderer):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence per-request logging
            pass

        def _send(self, code, ctype, body):
            try:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # client navigated away / refreshed mid-transfer

        def do_GET(self):
            u = urlparse(self.path)
            qs = parse_qs(u.query)
            if u.path == "/":
                self._send(200, "text/html", PAGE.encode())
            elif u.path == "/frame.png":
                png = renderer.latest_png()  # cached; never blocks on rendering
                if png is None:
                    self._send(503, "text/plain", b"no frame yet"); return
                self._send(200, "image/png", png)
            elif u.path == "/config":
                if "db" in qs:
                    with state.lock:
                        state.db = qs["db"][0] != "0"
                self._send(200, "text/plain", b"ok")
            elif u.path == "/status":
                with state.lock:
                    s = dict(frames=state.frame_count, fps=state.fps,
                             buffer_fill=len(state.ring), pre=state.pre_frames,
                             pending=state.pending is not None,
                             last_saved=state.last_saved)
                self._send(200, "application/json", json.dumps(s).encode())
            elif u.path == "/trigger":
                label = qs.get("label", ["clip"])[0]
                meta = {k: qs.get(k, [""])[0] for k in ("angle", "pol", "foff", "sep", "note")}
                with state.lock:
                    if state.pending is not None:
                        self._send(200, "text/plain", b"busy: a clip is already being captured")
                        return
                    if not state.ring:
                        self._send(200, "text/plain", b"no frames buffered yet")
                        return
                    state.pending = dict(past=list(state.ring), future=[],
                                         need=state.post_frames, label=label, meta=meta)
                    fill = len(state.ring)
                self._send(200, "text/plain",
                           f"{label}: saving {fill} past + {state.post_frames} future frames".encode())
            else:
                self._send(404, "text/plain", b"not found")
    return H


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="victim.yaml")
    ap.add_argument("--rsp", default="AWR1843AOP")
    ap.add_argument("--device", default="AWR1843")
    ap.add_argument("--buffer-seconds", type=float, default=3.0, help="pre-trigger buffer")
    ap.add_argument("--post-seconds", type=float, default=3.0, help="post-trigger capture")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)-16s %(message)s")
    import yaml
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    cfg["radar"]["device"] = args.device

    fp_ms = float(cfg["radar"]["frame_period"])
    pre = max(1, round(args.buffer_seconds * 1000.0 / fp_ms))
    post = max(1, round(args.post_seconds * 1000.0 / fp_ms))
    logging.getLogger("monitor").info(
        "buffer: %d pre + %d post frames (%.0f ms/frame)", pre, post, fp_ms)

    state = State(pre, post, args.outdir, cfg, args.rsp)
    renderer = Renderer(args.rsp)

    stop = threading.Event()
    cap = threading.Thread(target=capture_loop, args=(state, cfg, stop), daemon=True)
    cap.start()
    rnd = threading.Thread(target=renderer.render_loop, args=(state, stop), daemon=True)
    rnd.start()

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(state, renderer))
    logging.getLogger("monitor").info("open http://<radar-host>:%d in your browser", args.port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        logging.getLogger("monitor").info("shutting down")
    finally:
        stop.set()
        srv.shutdown()


if __name__ == "__main__":
    main()