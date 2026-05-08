"""D435 1프레임 캡처 → RGB-aligned depth + 통계 + PNG 저장.

astra-test/02 와 동일 포맷으로 출력 — Astra(invalid 30%, max 614mm) 와 직접 비교.
align(rs.stream.color) 로 RGB-Depth HW 정합 (Astra 의 BAD_PARAMETER 우회).
"""
import numpy as np
import pyrealsense2 as rs
from PIL import Image

W, H, FPS = 640, 480, 30

pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, W, H, rs.format.rgb8, FPS)
cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
profile = pipeline.start(cfg)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()  # m / unit, D435 보통 0.001
print(f"depth_scale: {depth_scale}  (1 unit = {depth_scale*1000:.3f} mm)")

align = rs.align(rs.stream.color)  # depth → color frame 으로 정합

try:
    # 워밍업 (auto-exposure 안정)
    for _ in range(10):
        pipeline.wait_for_frames()

    frames = align.process(pipeline.wait_for_frames())
    color_f = frames.get_color_frame()
    depth_f = frames.get_depth_frame()

    color_arr = np.asarray(color_f.get_data())                 # uint8 (H, W, 3) RGB
    depth_arr = np.asarray(depth_f.get_data())                  # uint16 (H, W) raw units

    # mm 단위로 환산 (Astra 와 동일 단위로 맞춤)
    depth_mm = (depth_arr.astype(np.float32) * depth_scale * 1000.0).astype(np.uint16)
finally:
    pipeline.stop()

print(f"\n=== RGB ===")
print(f"  shape : {color_arr.shape}")
print(f"  dtype : {color_arr.dtype}")
print(f"  range : [{color_arr.min()}, {color_arr.max()}]")

print(f"\n=== Depth (mm) ===")
print(f"  shape : {depth_mm.shape}")
print(f"  dtype : {depth_mm.dtype}")
print(f"  range : [{depth_mm.min()}, {depth_mm.max()}] mm")
valid = int((depth_mm > 0).sum())
total = depth_mm.size
print(f"  valid pixels: {valid:,}/{total:,} ({100*valid/total:.1f}%)")
print(f"  invalid pixels: {100*(1-valid/total):.1f}%")
if valid > 0:
    v = depth_mm[depth_mm > 0]
    print(f"  valid range : [{v.min()}, {v.max()}] mm   (median {int(np.median(v))})")

Image.fromarray(color_arr).save("rgb.png")
Image.fromarray(depth_mm).save("depth.png")  # 16-bit PNG, mm
print(f"\nsaved: rgb.png, depth.png")
