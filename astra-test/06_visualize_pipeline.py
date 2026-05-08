"""pipeline.py 결과 시각화 — RGB + Detection bbox + Grasp 점.

좌: 전체 detection bbox (인형 클래스 lime 강조, 기타 red 옅게)
우: grasp 후보 점 (Q 비례 크기/색)
"""
import os
import sys

import matplotlib
matplotlib.use("TkAgg")  # GUI 표시. 만약 막히면 "Agg" 로 변경 (저장만)

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline import GraspPipeline, _serialize  # noqa: E402

DOLL_CLASSES = {"bird", "teddy bear"}  # 인형으로 간주할 COCO 클래스

# 데이터 로드
rgb = np.array(Image.open("rgb.png"))
depth_raw = np.array(Image.open("depth.png")).astype(np.float32)
depth_m = depth_raw / 1000.0

print("Loading pipeline + predicting...")
pipe = GraspPipeline(device="cpu")
result = pipe.predict(rgb, depth_m, depth_raw=depth_raw, grasp_top_k=5)
out = _serialize(result)

print(f"  det={out['det_ms']}ms  grasp={out['grasp_ms']}ms  total={out['total_ms']}ms")
print(f"  detections={out['n_detections']}  grasps={out['n_grasps']}")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 좌: Detection
ax = axes[0]
ax.imshow(rgb)
n_doll = 0
for d in out["detections"]:
    x1, y1, x2, y2 = d["bbox"]
    is_doll = d["class_name"] in DOLL_CLASSES
    color = "lime" if is_doll else "red"
    alpha = 0.9 if is_doll else 0.3
    lw = 2 if is_doll else 1
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=lw, edgecolor=color, facecolor="none", alpha=alpha,
    )
    ax.add_patch(rect)
    if is_doll:
        ax.text(
            x1, max(0, y1 - 4),
            f"{d['class_name']} {d['score']:.2f}",
            color=color, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
        )
        n_doll += 1
ax.set_title(
    f"Detection — total {out['n_detections']}  (doll classes: {n_doll})"
)
ax.axis("off")

# 우: Grasp
ax = axes[1]
ax.imshow(rgb)
for i, g in enumerate(out["grasps"]):
    x, y = int(g["x"]), int(g["y"])
    score = float(g["score"])
    z = g["z"]
    z_str = f"z={z:.3f}m" if z is not None else "z=N/A"

    msize = 8 + score * 18
    ax.plot(
        x, y, "o",
        markersize=msize,
        markerfacecolor=plt.cm.jet(score),
        markeredgecolor="white",
        markeredgewidth=2,
        alpha=0.95,
    )
    ax.text(
        x + 8, y + 4,
        f"#{i}  q={score:.2f}\n{z_str}",
        color="white", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.75),
    )
ax.set_title(f"Grasp — {out['n_grasps']} candidates (size/color = Q)")
ax.axis("off")

fig.suptitle(
    f"Astra capture → pipeline inference   "
    f"det={out['det_ms']}ms  grasp={out['grasp_ms']}ms  total={out['total_ms']}ms  "
    f"({1000/out['total_ms']:.1f} FPS)",
    fontsize=12,
)
fig.tight_layout()

fig.savefig("06_pipeline_result.png", dpi=110, bbox_inches="tight")
print("saved: 06_pipeline_result.png")
plt.show()
