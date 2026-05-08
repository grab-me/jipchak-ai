"""D435 인형 데이터 자동 수집 — fine-tune 용 RGB+depth 페어 저장.

두 가지 모드:
  - 'space' 키: 수동 캡처 (1장씩 직접 샷 잡기)
  - 'a' 키: auto 모드 토글 (1초마다 자동 캡처, 인형 배치 바꾸면서 연속 수집)

저장:
  D435-test/dataset/rgb/000001.png      (8-bit RGB, 라벨링 input)
  D435-test/dataset/depth/000001.png    (16-bit mm, 옵션 활용)
  D435-test/dataset/meta.csv            (filename, depth_valid_pct, timestamp)

종료: q / ESC
"""
import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs

W, H, FPS = 640, 480, 30
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "dataset")
RGB_DIR = os.path.join(OUT_DIR, "rgb")
DEPTH_DIR = os.path.join(OUT_DIR, "depth")
META_PATH = os.path.join(OUT_DIR, "meta.csv")

AUTO_INTERVAL_SEC = 1.0
MIN_DEPTH_VALID_PCT = 50.0  # 너무 낮으면 캡처 거부 (렌즈 가림 / IR 안 잡힘)


def next_index() -> int:
    if not os.path.exists(RGB_DIR):
        return 1
    existing = [f for f in os.listdir(RGB_DIR) if f.endswith(".png")]
    if not existing:
        return 1
    nums = []
    for f in existing:
        try:
            nums.append(int(os.path.splitext(f)[0]))
        except ValueError:
            pass
    return (max(nums) + 1) if nums else 1


def main():
    os.makedirs(RGB_DIR, exist_ok=True)
    os.makedirs(DEPTH_DIR, exist_ok=True)

    new_meta = not os.path.exists(META_PATH)
    meta_f = open(META_PATH, "a", newline="", encoding="utf-8")
    meta_w = csv.writer(meta_f)
    if new_meta:
        meta_w.writerow(["index", "filename", "timestamp", "depth_valid_pct"])

    print("Initializing D435...", flush=True)
    rs_pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, W, H, rs.format.rgb8, FPS)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    profile = rs_pipe.start(cfg)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)

    for _ in range(15):
        rs_pipe.wait_for_frames()

    idx = next_index()
    print(f"Starting at index {idx}. Keys: SPACE=capture, a=auto toggle, q/ESC=quit")

    auto_mode = False
    last_auto = 0.0
    captured_this_session = 0

    cv2.namedWindow("collect", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("collect", 1280, 960)

    try:
        while True:
            frames = align.process(rs_pipe.wait_for_frames())
            color_f = frames.get_color_frame()
            depth_f = frames.get_depth_frame()
            if not color_f or not depth_f:
                continue

            rgb = np.asarray(color_f.get_data())
            depth_raw = np.asarray(depth_f.get_data())  # uint16, raw units (= mm at scale 0.001)
            depth_valid_pct = float((depth_raw > 0).mean()) * 100

            vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.rectangle(vis, (0, 0), (W, 60), (0, 0, 0), -1)
            mode_str = "AUTO ON" if auto_mode else "manual"
            cv2.putText(vis, f"#{idx-1} captured this session: {captured_this_session}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, f"mode: {mode_str}   depth valid: {depth_valid_pct:.0f}%",
                        (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0) if auto_mode else (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow("collect", vis)

            now = time.time()
            do_capture = False
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                do_capture = True
            elif key == ord("a"):
                auto_mode = not auto_mode
                last_auto = now
                print(f"  auto = {auto_mode}")
            elif key in (ord("q"), 27):
                break
            elif auto_mode and (now - last_auto) >= AUTO_INTERVAL_SEC:
                do_capture = True
                last_auto = now

            if do_capture:
                if depth_valid_pct < MIN_DEPTH_VALID_PCT:
                    print(f"  skip (depth_valid {depth_valid_pct:.0f}% < {MIN_DEPTH_VALID_PCT})")
                    continue
                fname = f"{idx:06d}.png"
                ts = datetime.now().isoformat(timespec="seconds")
                # cv2 는 BGR 로 저장 — RGB 입력이라 변환 필요
                cv2.imwrite(os.path.join(RGB_DIR, fname),
                            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(DEPTH_DIR, fname), depth_raw)
                meta_w.writerow([idx, fname, ts, f"{depth_valid_pct:.1f}"])
                meta_f.flush()
                captured_this_session += 1
                idx += 1
                print(f"  saved {fname}  (depth {depth_valid_pct:.0f}%)")
    finally:
        rs_pipe.stop()
        meta_f.close()
        cv2.destroyAllWindows()
        print(f"Done. Total this session: {captured_this_session}")
        print(f"Dataset dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
