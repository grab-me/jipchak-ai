"""실시간 라이브 캡처 + pipeline 추론 + cv2 시각화.

매 frame 마다:
- Astra RGB+depth 받음
- pipeline.predict() 호출
- cv2 창에 detection bbox + grasp 점 그림

종료: 'q' 또는 ESC
저장: 's' 키 (현재 frame PNG)
"""
import os
import sys
import time

import cv2
import numpy as np
from openni import openni2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline import GraspPipeline, _serialize  # noqa: E402

OPENNI_REDIST = r"C:\Users\SSAFY\Desktop\OpenNI_2.3.0.86\Win64-Release\tools\NiViewer"
DOLL_CLASSES = {"bird", "teddy bear"}


def jet_bgr(score: float) -> tuple:
    """0~1 score → jet 컬러맵 BGR (cv2 용)."""
    s = max(0.0, min(1.0, float(score)))
    # 간단한 jet 근사
    r = int(255 * max(0, min(1, 1.5 - abs(4 * s - 3))))
    g = int(255 * max(0, min(1, 1.5 - abs(4 * s - 2))))
    b = int(255 * max(0, min(1, 1.5 - abs(4 * s - 1))))
    return (b, g, r)  # BGR


def draw_overlay(rgb, out):
    """RGB(numpy) 위에 detection bbox + grasp 점 그려서 BGR 반환."""
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # detection bbox
    for d in out["detections"]:
        x1, y1, x2, y2 = [int(b) for b in d["bbox"]]
        is_doll = d["class_name"] in DOLL_CLASSES
        col = (0, 255, 0) if is_doll else (60, 60, 200)
        thick = 2 if is_doll else 1
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, thick)
        if is_doll:
            label = f"{d['class_name']} {d['score']:.2f}"
            cv2.putText(
                vis, label, (x1, max(10, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA,
            )

    # grasp 점
    for i, g in enumerate(out["grasps"]):
        x, y = int(g["x"]), int(g["y"])
        score = float(g["score"])
        r = int(6 + score * 16)
        col = jet_bgr(score)
        cv2.circle(vis, (x, y), r, col, -1)
        cv2.circle(vis, (x, y), r, (255, 255, 255), 2)
        z_str = f"{g['z']:.2f}m" if g["z"] is not None else "N/A"
        cv2.putText(
            vis, f"#{i} q{score:.2f} z{z_str}",
            (x + r + 3, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return vis


def main():
    print("[1/4] Loading pipeline (모델 cold start ~수십초)...", flush=True)
    pipe = GraspPipeline(device="cpu")

    print("[2/4] Initializing Astra...", flush=True)
    openni2.initialize(OPENNI_REDIST)
    dev = openni2.Device.open_any()
    color_s = dev.create_color_stream()
    depth_s = dev.create_depth_stream()
    color_s.start()
    depth_s.start()

    print("[3/4] Warming up models (첫 추론 JIT)...", flush=True)
    dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_depth = np.zeros((480, 640), dtype=np.float32)
    t_warm = time.perf_counter()
    pipe.predict(dummy_rgb, dummy_depth, depth_raw=dummy_depth, grasp_top_k=1)
    print(f"  warmup done in {time.perf_counter() - t_warm:.1f}s", flush=True)

    print("[4/4] Live loop. Keys: q/ESC=quit, s=save, f=fullscreen toggle", flush=True)

    # 창 크기 조절 가능 + 초기 2배 크기 (640x480 → 1280x960)
    WINDOW_NAME = "JipChak Realtime"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 960)

    fps_hist = []
    save_count = 0
    fullscreen = False

    try:
        while True:
            cf = color_s.read_frame()
            df = depth_s.read_frame()

            rgb = np.frombuffer(cf.get_buffer_as_uint8(), dtype=np.uint8).reshape(
                (cf.height, cf.width, 3)
            ).copy()
            depth_raw = np.frombuffer(df.get_buffer_as_uint16(), dtype=np.uint16).reshape(
                (df.height, df.width)
            )
            depth_m = depth_raw.astype(np.float32) / 1000.0

            t0 = time.perf_counter()
            result = pipe.predict(
                rgb, depth_m,
                depth_raw=depth_raw.astype(np.float32),
                grasp_top_k=5,
            )
            out = _serialize(result)
            t_total = (time.perf_counter() - t0) * 1000

            # 시각화
            vis = draw_overlay(rgb, out)

            # latency 라벨 (실측 vs pipeline 자체 측정 둘 다)
            fps_hist.append(1000 / t_total)
            if len(fps_hist) > 30:
                fps_hist.pop(0)
            avg_fps = sum(fps_hist) / len(fps_hist)

            # === 상단 정보 (latency / 카운트) ===
            line1 = (
                f"Detection: {out['det_ms']:.0f}ms   "
                f"Grasp: {out['grasp_ms']:.0f}ms   "
                f"Total: {t_total:.0f}ms   ({avg_fps:.1f} frames/sec)"
            )
            line2 = (
                f"Objects detected: {out['n_detections']}   "
                f"Grasp candidates: {out['n_grasps']}"
            )
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 50), (0, 0, 0), -1)
            cv2.putText(vis, line1, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, line2, (8, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (200, 200, 200), 1, cv2.LINE_AA)

            # === 하단 범례 ===
            h = vis.shape[0]
            cv2.rectangle(vis, (0, h - 70), (vis.shape[1], h), (0, 0, 0), -1)
            cv2.putText(vis, "lime box = doll detected (bird/teddy bear)",
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
            if key in (ord("q"), 27):  # q or ESC
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
        color_s.stop()
        depth_s.stop()
        openni2.unload()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
