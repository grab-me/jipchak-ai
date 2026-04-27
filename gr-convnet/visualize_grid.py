"""GR-ConvNet pretrained 4종 동시 비교 (2×2 grid).

같은 RGB-D 입력에 cornell ch16/ch32 + jacquard d/rgbd 를 각자 native input size 로
실행해 Q맵·grasp 박스를 한 PNG 에 합쳐 모델별 일관성/차이를 한눈에.

각 패널은 자기 input_size 로 center crop 된 영역만 보여준다 (cornell=224, jacquard=300).
따라서 패널마다 보이는 영역의 픽셀 범위가 다르다 — 동일 영역 비교가 아니라 모델 native 동작 비교.

Usage:
  python visualize_grid.py
  python visualize_grid.py --device cpu --n_grasps 3
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


# 각 모델은 trained-models/ 하위 가장 높은 IoU epoch 사용
MODELS = [
    {
        "label": "cornell ch16 (RGBD, 224)",
        "checkpoint": "trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch16/epoch_30_iou_0.97",
        "output_size": 224,
        "use_rgb": True,
        "use_depth": True,
    },
    {
        "label": "cornell ch32 (RGBD, 224)",
        "checkpoint": "trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98",
        "output_size": 224,
        "use_rgb": True,
        "use_depth": True,
    },
    {
        "label": "jacquard d (depth-only, 300)",
        "checkpoint": "trained-models/jacquard-d-grconvnet3-drop0-ch32/epoch_50_iou_0.94",
        "output_size": 300,
        "use_rgb": False,
        "use_depth": True,
    },
    {
        "label": "jacquard rgbd (RGBD, 300)",
        "checkpoint": "trained-models/jacquard-rgbd-grconvnet3-drop0-ch32/epoch_48_iou_0.93",
        "output_size": 300,
        "use_rgb": True,
        "use_depth": True,
    },
]


def run_one(net, device, rgb, depth, output_size, use_rgb, use_depth, n_grasps):
    H, W = rgb.shape[:2]
    cam = CameraData(
        width=W, height=H, output_size=output_size,
        include_depth=use_depth, include_rgb=use_rgb,
    )
    x, _, _ = cam.get_data(rgb=rgb, depth=depth)
    # depth-only/rgb-only 분기에서 CameraData 가 batch dim 을 안 붙이는 경우 방어
    if x.dim() == 3:
        x = x.unsqueeze(0)
    rgb_crop = cam.get_rgb(rgb, norm=False)

    x = x.to(device)
    with torch.no_grad():
        pred = net.predict(x)
        q_img, ang_img, width_img = post_process_output(
            pred["pos"], pred["cos"], pred["sin"], pred["width"]
        )

    grasps = detect_grasps(
        q_img, ang_img, width_img=width_img, no_grasps=n_grasps
    )
    return rgb_crop, q_img, grasps


def plot_panel(ax, rgb_crop, q_img, grasps, label, alpha=0.45):
    ax.imshow(rgb_crop)
    ax.imshow(q_img, cmap="jet", alpha=alpha, vmin=0, vmax=1)

    for g in grasps:
        gr = g.as_gr
        pts = gr.points
        ys = pts[[0, 1, 2, 3, 0], 0]
        xs = pts[[0, 1, 2, 3, 0], 1]
        ax.plot(xs, ys, linewidth=1.5)
        cy, cx = g.center
        ax.plot(cx, cy, "wo", markersize=5, markeredgecolor="black")

    qmax = float(q_img.max()) if q_img.size > 0 else 0.0
    ax.set_title(f"{label}  |  n={len(grasps)}  Q_max={qmax:.2f}", fontsize=10)
    ax.axis("off")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rgb_path", default="../graspnet-baseline/doc/example_data/color.png")
    p.add_argument("--depth_path", default="../graspnet-baseline/doc/example_data/depth.png")
    p.add_argument("--n_grasps", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default="results/grid_4models.png")
    args = p.parse_args()

    device = torch.device(args.device)

    rgb = np.array(Image.open(args.rgb_path))
    depth = np.expand_dims(np.array(Image.open(args.depth_path)), axis=2)
    print(f"input  rgb: {rgb.shape}, depth: {depth.shape}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    for ax, m in zip(axes.flatten(), MODELS):
        print(f"  running {m['label']} ...")
        net = torch.load(m["checkpoint"], map_location=device)
        net.eval()
        rgb_crop, q_img, grasps = run_one(
            net, device, rgb, depth,
            m["output_size"], m["use_rgb"], m["use_depth"], args.n_grasps,
        )
        plot_panel(ax, rgb_crop, q_img, grasps, m["label"])
        del net  # free GPU mem before next model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fig.suptitle(
        f"GR-ConvNet pretrained 4 modes  |  input: {os.path.basename(args.rgb_path)}",
        fontsize=12,
    )
    fig.tight_layout()

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output, dpi=110, bbox_inches="tight")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
