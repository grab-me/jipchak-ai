"""FastAPI MJPEG 스트리밍 — D435 + pipeline 추론 결과를 브라우저로.

구조:
    [capture+pipeline thread] → latest_vis (Lock) → [/stream MJPEG] → browser <img>

실행:
    uvicorn 09_web_stream:app --host 0.0.0.0 --port 8000
또는 직접:
    python 09_web_stream.py

브라우저:
    http://localhost:8000/          (간단한 뷰어 페이지)
    http://localhost:8000/stream    (raw MJPEG)
    http://localhost:8000/state     (현재 추론 결과 JSON — det/grasp 카운트 등)
    http://localhost:8000/docs      (OpenAPI)
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline import GraspPipeline, _serialize  # noqa: E402

W, H, FPS = 640, 480, 30
DOLL_CLASSES = {"bird", "teddy bear"}
TEDDY_BEAR_COCO = 88
JPEG_QUALITY = 80

# ───── 공유 상태 (capture thread → HTTP handlers) ─────
_state_lock = threading.Lock()
_latest_vis: Optional[np.ndarray] = None      # BGR (cv2 인코딩용)
_latest_state: dict = {"running": False}
_stop_flag = threading.Event()


def jet_bgr(score: float) -> tuple:
    s = max(0.0, min(1.0, float(score)))
    r = int(255 * max(0, min(1, 1.5 - abs(4 * s - 3))))
    g = int(255 * max(0, min(1, 1.5 - abs(4 * s - 2))))
    b = int(255 * max(0, min(1, 1.5 - abs(4 * s - 1))))
    return (b, g, r)


def draw_overlay(rgb: np.ndarray, out: dict) -> np.ndarray:
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for d in out["detections"]:
        x1, y1, x2, y2 = [int(b) for b in d["bbox"]]
        is_doll = d["class_name"] in DOLL_CLASSES
        col = (0, 255, 0) if is_doll else (60, 60, 200)
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2 if is_doll else 1)
        if is_doll:
            cv2.putText(vis, f"{d['class_name']} {d['score']:.2f}",
                        (x1, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, col, 1, cv2.LINE_AA)
    for i, g in enumerate(out["grasps"]):
        x, y = int(g["x"]), int(g["y"])
        score = float(g["score"])
        r = int(6 + score * 16)
        col = jet_bgr(score)
        cv2.circle(vis, (x, y), r, col, -1)
        cv2.circle(vis, (x, y), r, (255, 255, 255), 2)
        z_str = f"{g['z']:.2f}m" if g["z"] is not None else "N/A"
        cv2.putText(vis, f"#{i} q{score:.2f} z{z_str}",
                    (x + r + 3, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def capture_loop():
    """별도 thread — D435 캡처 + pipeline 추론 + 시각화 → _latest_vis."""
    global _latest_vis, _latest_state

    print("[capture] loading pipeline...", flush=True)
    pipe = GraspPipeline(device="cpu")

    print("[capture] starting D435...", flush=True)
    rs_pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, W, H, rs.format.rgb8, FPS)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    profile = rs_pipe.start(cfg)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)

    for _ in range(15):
        rs_pipe.wait_for_frames()

    print("[capture] warmup model...", flush=True)
    dummy = np.zeros((H, W, 3), dtype=np.uint8)
    dummy_d = np.zeros((H, W), dtype=np.float32)
    pipe.predict(dummy, dummy_d, depth_raw=dummy_d, grasp_top_k=1)

    with _state_lock:
        _latest_state["running"] = True
    print("[capture] live", flush=True)

    fps_hist: list[float] = []
    try:
        while not _stop_flag.is_set():
            frames = align.process(rs_pipe.wait_for_frames())
            color_f = frames.get_color_frame()
            depth_f = frames.get_depth_frame()
            if not color_f or not depth_f:
                continue

            rgb = np.asarray(color_f.get_data())
            depth_raw = np.asarray(depth_f.get_data()).astype(np.float32)
            depth_m = depth_raw * depth_scale

            t0 = time.perf_counter()
            result = pipe.predict(
                rgb, depth_m,
                depth_raw=depth_raw,
                grasp_top_k=5,
                det_conf=0.4,
                det_class_filter={TEDDY_BEAR_COCO},
            )
            out = _serialize(result)
            t_total = (time.perf_counter() - t0) * 1000

            vis = draw_overlay(rgb, out)

            fps_hist.append(1000 / max(t_total, 1e-3))
            if len(fps_hist) > 30:
                fps_hist.pop(0)
            avg_fps = sum(fps_hist) / len(fps_hist)

            line1 = (f"Det: {out['det_ms']:.0f}ms  Grasp: {out['grasp_ms']:.0f}ms  "
                     f"Total: {t_total:.0f}ms  ({avg_fps:.1f} fps)")
            line2 = (f"Objects: {out['n_detections']}  Grasps: {out['n_grasps']}  "
                     f"depth valid: {(depth_m > 0).mean()*100:.0f}%")
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 50), (0, 0, 0), -1)
            cv2.putText(vis, line1, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, line2, (8, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (200, 200, 200), 1, cv2.LINE_AA)

            with _state_lock:
                _latest_vis = vis
                _latest_state = {
                    "running": True,
                    "fps": round(avg_fps, 1),
                    "det_ms": out["det_ms"],
                    "grasp_ms": out["grasp_ms"],
                    "total_ms": round(t_total, 1),
                    "n_detections": out["n_detections"],
                    "n_grasps": out["n_grasps"],
                    "depth_valid_pct": round(float((depth_m > 0).mean()) * 100, 1),
                    "grasps": out["grasps"],
                    "detections": out["detections"],
                }
    finally:
        rs_pipe.stop()
        with _state_lock:
            _latest_state["running"] = False
        print("[capture] stopped", flush=True)


# ───── FastAPI ─────
_thread: Optional[threading.Thread] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _thread
    _thread = threading.Thread(target=capture_loop, daemon=True)
    _thread.start()
    yield
    _stop_flag.set()


app = FastAPI(title="JipChak D435 Stream", lifespan=lifespan)


async def _mjpeg_generator():
    boundary = b"--frame"
    while True:
        with _state_lock:
            frame = _latest_vis
        if frame is None:
            await asyncio.sleep(0.05)
            continue
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            await asyncio.sleep(1 / FPS)
            continue
        yield (boundary + b"\r\nContent-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
               + jpeg.tobytes() + b"\r\n")
        await asyncio.sleep(1 / FPS)


@app.get("/stream")
def stream():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/state")
def state():
    with _state_lock:
        return dict(_latest_state)


@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!doctype html><html><head><meta charset="utf-8"><title>JipChak D435</title>
<style>
body { background:#111; color:#ddd; font-family: sans-serif; margin:0; padding:16px; }
h1 { font-size:18px; margin:0 0 8px 0; }
img { max-width:100%; border:1px solid #333; }
#state { font-size:13px; margin-top:8px; white-space:pre; color:#9cf; }
</style></head><body>
<h1>JipChak — D435 + pipeline</h1>
<img src="/stream" alt="live"/>
<div id="state">loading...</div>
<script>
async function tick(){
  try {
    const r = await fetch('/state');
    document.getElementById('state').textContent = JSON.stringify(await r.json(), null, 2);
  } catch(e){}
  setTimeout(tick, 500);
}
tick();
</script>
</body></html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
