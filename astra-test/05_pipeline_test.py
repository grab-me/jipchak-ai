"""Astra 캡처 데이터 → pipeline.py 첫 추론 (인형 도메인)."""
import os
import sys

import numpy as np
from PIL import Image

# pipeline.py 가 있는 jipchak-ai 루트 sys.path 에 추가
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline import GraspPipeline, _serialize  # noqa: E402

# Astra 데이터 로드
rgb = np.array(Image.open("rgb.png"))
depth_raw = np.array(Image.open("depth.png")).astype(np.float32)
depth_m = depth_raw / 1000.0  # Astra mm → m

print(f"RGB: {rgb.shape}")
print(f"depth: {depth_raw.shape}  range "
      f"{depth_raw[depth_raw>0].min():.0f}~{depth_raw.max():.0f}mm")

print("\nLoading pipeline (Detection + Grasp)...")
pipe = GraspPipeline(device="cpu")

print("\nPredicting...")
result = pipe.predict(rgb, depth_m, depth_raw=depth_raw, grasp_top_k=5)
out = _serialize(result)

print("\n=== Detection ===")
print(f"  total: {out['n_detections']}")
for d in out["detections"][:15]:
    bbox = [int(b) for b in d["bbox"]]
    print(f"  {d['class_name']:15s} conf={d['score']:.3f}  bbox={bbox}")

print("\n=== Grasp ===")
print(f"  total: {out['n_grasps']}")
for i, g in enumerate(out["grasps"]):
    z = f"{g['z']:.3f}m" if g["z"] is not None else "N/A"
    print(f"  #{i}  ({int(g['x']):3d}, {int(g['y']):3d})  z={z}  score={g['score']:.3f}")

print(f"\nlatency: det={out['det_ms']}ms  grasp={out['grasp_ms']}ms  "
      f"total={out['total_ms']}ms")
print(f"mask_used: {out['mask_used']}")
