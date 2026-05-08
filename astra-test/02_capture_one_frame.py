"""Astra 1프레임 캡처 → numpy + PNG 저장 + spec 출력."""
import numpy as np
from PIL import Image
from openni import openni2

OPENNI_REDIST = r"C:\Users\SSAFY\Desktop\OpenNI_2.3.0.86\Win64-Release\tools\NiViewer"

openni2.initialize(OPENNI_REDIST)
dev = openni2.Device.open_any()

# Image registration ON — depth/RGB 같은 픽셀 = 같은 위치 (저번 "registration off" 해결)
try:
    dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
    print("Image registration: DEPTH_TO_COLOR enabled")
except Exception as e:
    print(f"registration warning: {e}")

# stream 생성 + start
color_stream = dev.create_color_stream()
depth_stream = dev.create_depth_stream()
color_stream.start()
depth_stream.start()

# 워밍업 (자동 노출 안정)
for _ in range(5):
    color_stream.read_frame()
    depth_stream.read_frame()

# 진짜 1프레임
color_frame = color_stream.read_frame()
depth_frame = depth_stream.read_frame()

# numpy 변환
color_buf = color_frame.get_buffer_as_uint8()
color_arr = np.frombuffer(color_buf, dtype=np.uint8).reshape(
    (color_frame.height, color_frame.width, 3)
)

depth_buf = depth_frame.get_buffer_as_uint16()
depth_arr = np.frombuffer(depth_buf, dtype=np.uint16).reshape(
    (depth_frame.height, depth_frame.width)
)

# spec 출력
print(f"\n=== RGB ===")
print(f"  shape : {color_arr.shape}")
print(f"  dtype : {color_arr.dtype}")
print(f"  range : [{color_arr.min()}, {color_arr.max()}]")

print(f"\n=== Depth ===")
print(f"  shape : {depth_arr.shape}")
print(f"  dtype : {depth_arr.dtype}")
print(f"  range : [{depth_arr.min()}, {depth_arr.max()}] mm")
valid = int((depth_arr > 0).sum())
total = depth_arr.size
print(f"  valid pixels: {valid:,}/{total:,} ({100*valid/total:.1f}%)")
print(f"  invalid pixels: {100*(1-valid/total):.1f}%  ← 너무 높으면 인형 잡기 위험")

# 저장
Image.fromarray(color_arr).save("rgb.png")
Image.fromarray(depth_arr).save("depth.png")  # 16-bit PNG (raw mm)
print(f"\nsaved: rgb.png, depth.png")

color_stream.stop()
depth_stream.stop()
openni2.unload()