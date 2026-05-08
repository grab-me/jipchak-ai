"""depth 시각화 + RGB 위에 alpha overlay."""
import matplotlib

matplotlib.use("TkAgg")  # GUI 표시. 저장만 원하면 "Agg"
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

rgb = np.array(Image.open("rgb.png"))
depth = np.array(Image.open("depth.png"))  # 16-bit, mm 단위

# 통계
valid = depth > 0
print(f"depth valid: {valid.sum():,}/{depth.size:,} ({100*valid.mean():.1f}%)")
print(f"valid range: {depth[valid].min()} ~ {depth[valid].max()} mm")

# 시각화: depth 를 mm 단위 컬러맵 (0 인 invalid 픽셀은 검정 처리)
depth_disp = depth.astype(np.float32)
depth_disp[~valid] = np.nan  # invalid → 컬러맵에서 검정

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(rgb)
axes[0].set_title("RGB")
axes[0].axis("off")

im = axes[1].imshow(depth_disp, cmap="jet")
axes[1].set_title(f"Depth (mm)  valid {100*valid.mean():.1f}%")
axes[1].axis("off")
plt.colorbar(im, ax=axes[1], fraction=0.046)

axes[2].imshow(rgb)
axes[2].imshow(depth_disp, cmap="jet", alpha=0.5)
axes[2].set_title("RGB + Depth overlay")
axes[2].axis("off")

fig.tight_layout()
fig.savefig("visualize.png", dpi=110, bbox_inches="tight")
print("saved: visualize.png")
plt.show()