"""depth stream 만 켜서 진짜 데이터 들어오는지 확인."""
import numpy as np
from openni import openni2

OPENNI_REDIST = r"C:\Users\SSAFY\Desktop\OpenNI_2.3.0.86\Win64-Release\tools\NiViewer"
openni2.initialize(OPENNI_REDIST)
dev = openni2.Device.open_any()

depth = dev.create_depth_stream()

# 지원되는 video mode 목록
print("Depth video modes:")
for i, m in enumerate(depth.get_sensor_info().videoModes):
    print(f"  [{i}] {m.resolutionX}x{m.resolutionY} @ {m.fps}fps fmt={m.pixelFormat}")

depth.start()

# 워밍업 길게 + 매 프레임 통계
for i in range(20):
    f = depth.read_frame()
    arr = np.frombuffer(f.get_buffer_as_uint16(), dtype=np.uint16).reshape(
        (f.height, f.width)
    )
    valid = int((arr > 0).sum())
    print(f"  frame {i+1:2d}: range [{arr.min()},{arr.max()}] "
          f"valid {valid:,}/{arr.size:,} ({100*valid/arr.size:.1f}%)")

depth.stop()
openni2.unload()