"""GR-ConvNet 결과 합성 시각화.

흩어진 PNG (rgb / quality / grasp) 를 한 장으로 겹쳐 의미 직관 강화.
RGB 위에 Q맵을 alpha overlay + grasp 박스 + 중심점 + 점수 라벨.

Usage:
  python visualize_combined.py \
      --network trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98 \
      --rgb_path ../graspnet-baseline/doc/example_data/color.png \
      --depth_path ../graspnet-baseline/doc/example_data/depth.png \
      --output results/combined_cornell_ch32.png
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from inference.post_process import post_process_output  # noqa: E402
from utils.data.camera_data import CameraData  # noqa: E402
from utils.dataset_processing.grasp import detect_grasps  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--network", required=True)
    p.add_argument("--rgb_path", required=True)
    p.add_argument("--depth_path", required=True)
    p.add_argument("--output_size", type=int, default=224,
                   help="cornell=224, jacquard=300")
    p.add_argument("--n_grasps", type=int, default=5)
    p.add_argument("--output", default="results/combined.png")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--alpha", type=float, default=0.45,
                   help="Q맵 overlay 투명도")
    args = p.parse_args()

    device = torch.device(args.device)

    # 1. 데이터 로드
    rgb = np.array(Image.open(args.rgb_path))
    depth = np.expand_dims(np.array(Image.open(args.depth_path)), axis=2)
    H, W = rgb.shape[:2]

    cam = CameraData(width=W, height=H, output_size=args.output_size,
                     include_depth=True, include_rgb=True)
    x, _, _ = cam.get_data(rgb=rgb, depth=depth)
    rgb_crop = cam.get_rgb(rgb, norm=False)  # (output_size, output_size, 3) uint8

    # 2. 모델 추론
    net = torch.load(args.network, map_location=device)
    net.eval()
    x = x.to(device)
    with torch.no_grad():
        pred = net.predict(x)
        q_img, ang_img, width_img = post_process_output(
            pred["pos"], pred["cos"], pred["sin"], pred["width"]
        )

    # 3. grasp 후보 추출
    grasps = detect_grasps(q_img, ang_img, width_img=width_img,
                           no_grasps=args.n_grasps)

    # 4. 합성 plot
    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb_crop)
    qim = ax.imshow(q_img, cmap="jet", alpha=args.alpha, vmin=0, vmax=1)
    plt.colorbar(qim, ax=ax, fraction=0.046, pad=0.04, label="Q (grasp quality)")

    for i, g in enumerate(grasps):
        gr = g.as_gr  # GraspRectangle, points (4, 2) = (y, x)
        pts = gr.points
        # 박스 외곽선 (4점 + 닫기 위해 첫점 다시)
        ys = pts[[0, 1, 2, 3, 0], 0]
        xs = pts[[0, 1, 2, 3, 0], 1]
        ax.plot(xs, ys, linewidth=2)

        # 중심점 + 라벨
        cy, cx = g.center
        ax.plot(cx, cy, "wo", markersize=8, markeredgecolor="black")
        ax.text(
            cx + 6, cy + 6,
            f"{i}: q={q_img[cy, cx]:.2f}\nθ={g.angle:+.2f}",
            color="white", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.7),
        )

    ax.set_title(
        f"GR-ConvNet  |  {os.path.basename(args.network)}  |  "
        f"input={args.output_size}x{args.output_size}",
        fontsize=11,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    print(f"saved: {args.output}")
    print(f"top {len(grasps)} grasps:")
    for i, g in enumerate(grasps):
        cy, cx = g.center
        print(f"  [{i}] center=({cx},{cy})  q={q_img[cy, cx]:.3f}  "
              f"angle={g.angle:+.3f}rad  width(px)={g.length:.1f}")


if __name__ == "__main__":
    main()
