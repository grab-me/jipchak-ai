"""실시간 라이브 캡처 (D435) + pipeline 추론 + cv2 시각화.

매 frame:
- D435 RGB + aligned depth 받음
- pipeline.predict() 호출 (det_conf=0.4, teddy bear 만)
- cv2 창에 detection bbox + grasp 점

종료: 'q' 또는 ESC
저장: 's' 키
"""
import os
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline import GraspPipeline, _serialize  # noqa: E402

W, H, FPS = 640, 480, 30
DOLL_CLASSES = {"bird", "teddy bear"}
TEDDY_BEAR_COCO = 88


def jet_bgr(score: float) -> tuple:
    s = max(0.0, min(1.0, float(score)))
    r = int(255 * max(0, min(1, 1.5 - abs(4 * s - 3))))
    g = int(255 * max(0, min(1, 1.5 - abs(4 * s - 2))))
    b = int(255 * max(0, min(1, 1.5 - abs(4 * s - 1))))
    return (b, g, r)


def draw_overlay(rgb, out):
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    for d in out["detections"]:
        x1, y1, x2, y2 = [int(b) for b in d["bbox"]]
        is_doll = d["class_name"] in DOLL_CLASSES
        col = (0, 255, 0) if is_doll else (60, 60, 200)
        thick = 2 if is_doll else 1
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, thick)
        if is_doll:
            label = f"{d['class_name']} {d['score']:.2f}"
            cv2.putText(vis, label, (x1, max(10, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    for i, g in enumerate(out["grasps"]):
        x, y = int(g["x"]), int(g["y"])
        score = float(g["score"])
        r = int(6 + score * 16)
        col = jet_bgr(score)
        cv2.circle(vis, (x, y), r, col, -1)
        cv2.circle(vis, (x, y), r, (255, 255, 255), 2)
        z_str = f"{g['z']:.2f}m" if g["z"] is not None else "N/A"
        cv2.putText(vis, f"#{i} q{score:.2f} z{z_str}",
                    (x + r + 3, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return vis


def main():
    print("[1/4] Loading pipeline (모델 cold start ~수십초)...", flush=True)
    pipe = GraspPipeline(device="cpu")

    print("[2/4] Initializing D435...", flush=True)
    rs_pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, W, H, rs.format.rgb8, FPS)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    profile = rs_pipe.start(cfg)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    print(f"  depth_scale={depth_scale}  (raw → m)")

    # auto-exposure 안정
    for _ in range(15):
        rs_pipe.wait_for_frames()

    print("[3/4] Warming up models (첫 추론 JIT)...", flush=True)
    dummy_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    dummy_depth = np.zeros((H, W), dtype=np.float32)
    t_warm = time.perf_counter()
    pipe.predict(dummy_rgb, dummy_depth, depth_raw=dummy_depth, grasp_top_k=1)
    print(f"  warmup done in {time.perf_counter() - t_warm:.1f}s", flush=True)

    print("[4/4] Live loop. Keys: q/ESC=quit, s=save, f=fullscreen toggle", flush=True)

    WINDOW_NAME = "JipChak Realtime (D435)"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 960)

    fps_hist = []
    save_count = 0
    fullscreen = False

    try:
        while True:
            frames = align.process(rs_pipe.wait_for_frames())
            color_f = frames.get_color_frame()
            depth_f = frames.get_depth_frame()
            if not color_f or not depth_f:
                continue

            rgb = np.asarray(color_f.get_data())                          # uint8 RGB
            depth_raw = np.asarray(depth_f.get_data()).astype(np.float32)  # raw units
            depth_m = depth_raw * depth_scale                              # meters

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

            fps_hist.append(1000 / t_total)
            if len(fps_hist) > 30:
                fps_hist.pop(0)
            avg_fps = sum(fps_hist) / len(fps_hist)

            line1 = (
                f"Detection: {out['det_ms']:.0f}ms   "
                f"Grasp: {out['grasp_ms']:.0f}ms   "
                f"Total: {t_total:.0f}ms   ({avg_fps:.1f} frames/sec)"
            )
            line2 = (
                f"Objects: {out['n_detections']}   "
                f"Grasp candidates: {out['n_grasps']}   "
                f"depth valid: {(depth_m > 0).mean()*100:.0f}%"
            )
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 50), (0, 0, 0), -1)
            cv2.putText(vis, line1, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, line2, (8, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (200, 200, 200), 1, cv2.LINE_AA)

            h = vis.shape[0]
            cv2.rectangle(vis, (0, h - 70), (vis.shape[1], h), (0, 0, 0), -1)
            cv2.putText(vis, "lime box = teddy bear (det_conf >= 0.4)",
                        (8, h - 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(vis, "circle = grasp candidate (size/color = quality)",
                        (8, h - 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(vis,
                        "q = quality 0~1 (higher = better grasp)   "
                        "z = depth in meters (camera to point)",
                        (8, h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.40, (180, 180, 180), 1, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                save_count += 1
                fname = f"08_frame_{save_count:03d}.png"
                cv2.imwrite(fname, vis)
                print(f"  saved: {fname}")
            if key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                )
    finally:
        rs_pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
