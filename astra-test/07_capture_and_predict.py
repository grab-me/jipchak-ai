"""캡처 + pipeline 추론 + 시각화 한 번에 (단발).

매번 실행하면 카메라에서 1프레임 받고 분석. 인형 위치 바꿔가며 빠른 반복.
"""
import os
import sys

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from openni import openni2
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline import GraspPipeline, _serialize  # noqa: E402

OPENNI_REDIST = r"C:\Users\SSAFY\Desktop\OpenNI_2.3.0.86\Win64-Release\tools\NiViewer"
DOLL_CLASSES = {"bird", "teddy bear"}


def capture_one():
    """Astra 1프레임 RGB+depth 받기."""
    openni2.initialize(OPENNI_REDIST)
    dev = openni2.Device.open_any()
    color_s = dev.create_color_stream()
    depth_s = dev.create_depth_stream()
    color_s.start()
    depth_s.start()

    # 워밍업
    for _ in range(5):
        color_s.read_frame()
        depth_s.read_frame()

    cf = color_s.read_frame()
    df = depth_s.read_frame()

    rgb = np.frombuffer(cf.get_buffer_as_uint8(), dtype=np.uint8).reshape(
        (cf.height, cf.width, 3)
    ).copy()
    depth_raw = np.frombuffer(df.get_buffer_as_uint16(), dtype=np.uint16).reshape(
        (df.height, df.width)
    ).copy()

    color_s.stop()
    depth_s.stop()
    openni2.unload()
    return rgb, depth_raw


def visualize(rgb, out, output_path="07_result.png"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Detection
    ax = axes[0]
    ax.imshow(rgb)
    n_doll = 0
    for d in out["detections"]:
        x1, y1, x2, y2 = d["bbox"]
        is_doll = d["class_name"] in DOLL_CLASSES
        c = "lime" if is_doll else "red"
        alpha = 0.9 if is_doll else 0.3
        lw = 2 if is_doll else 1
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=lw, edgecolor=c, facecolor="none", alpha=alpha,
        )
        ax.add_patch(rect)
        if is_doll:
            ax.text(
                x1, max(0, y1 - 4),
                f"{d['class_name']} {d['score']:.2f}",
                color=c, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
            )
            n_doll += 1
    ax.set_title(f"Detection — total {out['n_detections']} (doll: {n_doll})")
    ax.axis("off")

    # Grasp
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
            markeredgecolor="white", markeredgewidth=2,
            alpha=0.95,
        )
        ax.text(
            x + 8, y + 4, f"#{i} q={score:.2f}\n{z_str}",
            color="white", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.75),
        )
    ax.set_title(f"Grasp — {out['n_grasps']} candidates")
    ax.axis("off")

    fig.suptitle(
        f"Astra capture → pipeline   "
        f"det={out['det_ms']}ms  grasp={out['grasp_ms']}ms  "
        f"total={out['total_ms']}ms  ({1000/out['total_ms']:.1f} FPS)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    print(f"saved: {output_path}")
    plt.show()


def main():
    print("[1/3] Capturing from Astra...")
    rgb, depth_raw = capture_one()
    depth_m = depth_raw.astype(np.float32) / 1000.0
    print(f"  RGB {rgb.shape}  depth {depth_raw.shape}  "
          f"range {depth_raw[depth_raw>0].min()}~{depth_raw.max()}mm")

    # 저장 (디버깅 용도)
    Image.fromarray(rgb).save("rgb.png")
    Image.fromarray(depth_raw).save("depth.png")

    print("[2/3] Loading pipeline + predicting...")
    pipe = GraspPipeline(device="cpu")
    result = pipe.predict(
        rgb, depth_m,
        depth_raw=depth_raw.astype(np.float32),
        grasp_top_k=5,
    )
    out = _serialize(result)
    print(f"  total {out['total_ms']}ms  "
          f"det {out['n_detections']}  grasp {out['n_grasps']}")

    print("[3/3] Visualizing...")
    visualize(rgb, out)


if __name__ == "__main__":
    main()
