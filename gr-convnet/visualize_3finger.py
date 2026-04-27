"""3-finger 관점 시각화 — antipodal 박스/각도 무시, grasp center 만 표시.

우리 사용 케이스 (3 DOF 수직 하강) 에서는 cos/sin/width 출력이 의미 없다.
모델 출력에서 (x, y, Q) 만 추출하고 depth 로 z 채워 (x, y, z, Q) 후보 점을
RGB 위에 plot.

cornell ch16 vs ch32 같은 입력으로 좌우 비교 (우리가 채택 후보로 좁힌 두 모델).

Usage:
  python visualize_3finger.py
  python visualize_3finger.py --device cpu
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from inference.post_process import post_process_output  # noqa: E402
from utils.data.camera_data import CameraData  # noqa: E402
from utils.dataset_processing.grasp import detect_grasps  # noqa: E402


CHECKPOINTS = [
    ("cornell ch16", "trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch16/epoch_30_iou_0.97"),
    ("cornell ch32", "trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98"),
]
OUTPUT_SIZE = 224


def run_one(net, device, rgb, depth, n_grasps):
    H, W = rgb.shape[:2]
    cam = CameraData(width=W, height=H, output_size=OUTPUT_SIZE,
                     include_depth=True, include_rgb=True)
    x, _, _ = cam.get_data(rgb=rgb, depth=depth)
    rgb_crop = cam.get_rgb(rgb, norm=False)

    x = x.to(device)
    with torch.no_grad():
        pred = net.predict(x)
        q_img, ang_img, width_img = post_process_output(
            pred["pos"], pred["cos"], pred["sin"], pred["width"]
        )
    grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=n_grasps)
    return rgb_crop, q_img, grasps


def plot_panel(ax, rgb_crop, q_img, grasps, depth_m, label):
    """점 + 좌표 라벨만. antipodal 박스/각도 무시 (3-finger 관점)."""
    ax.imshow(rgb_crop)
    ax.imshow(q_img, cmap="jet", alpha=0.40, vmin=0, vmax=1)

    H, W = depth_m.shape
    top = (H - OUTPUT_SIZE) // 2
    left = (W - OUTPUT_SIZE) // 2

    print(f"\n  [{label}] candidates (3-finger view):")
    for i, g in enumerate(grasps):
        cy, cx = g.center  # crop 좌표
        q = float(q_img[cy, cx])

        # 원본 이미지 좌표
        orig_y = int(cy + top)
        orig_x = int(cx + left)
        z = float(depth_m[orig_y, orig_x]) if 0 <= orig_y < H and 0 <= orig_x < W else 0.0
        z_str = f"z={z:.3f}m" if z > 0 else "z=N/A"

        print(f"    #{i}  pixel=({orig_x:4d},{orig_y:4d})  {z_str}  q={q:.3f}")

        # 점 (Q 비례 크기)
        msize = 8 + q * 14  # 8~22
        ax.plot(cx, cy, "o",
                markersize=msize,
                markerfacecolor=plt.cm.jet(q),
                markeredgecolor="white",
                markeredgewidth=2,
                alpha=0.95)
        ax.text(cx + 9, cy + 4,
                f"#{i} q={q:.2f}\n{z_str}",
                color="white", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="black", alpha=0.75))

    ax.set_title(f"{label}  |  n={len(grasps)}  Q_max={q_img.max():.2f}",
                 fontsize=11)
    ax.axis("off")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rgb_path", default="../graspnet-baseline/doc/example_data/color.png")
    p.add_argument("--depth_path", default="../graspnet-baseline/doc/example_data/depth.png")
    p.add_argument("--meta_path", default="../graspnet-baseline/doc/example_data/meta.mat")
    p.add_argument("--n_grasps", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default="results/3finger_cornell_compare.png")
    args = p.parse_args()

    device = torch.device(args.device)

    rgb = np.array(Image.open(args.rgb_path))
    depth_raw = np.expand_dims(np.array(Image.open(args.depth_path)), axis=2)

    # depth -> meters 변환 (GraspNet meta.mat 의 factor_depth 사용)
    meta = scio.loadmat(args.meta_path)
    factor_depth = float(meta["factor_depth"].squeeze())
    depth_m = depth_raw[..., 0].astype(np.float32) / factor_depth
    print(f"input rgb: {rgb.shape}  depth: {depth_raw.shape}  factor_depth={factor_depth}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, (label, ckpt) in zip(axes, CHECKPOINTS):
        net = torch.load(ckpt, map_location=device)
        net.eval()
        rgb_crop, q_img, grasps = run_one(
            net, device, rgb, depth_raw, args.n_grasps
        )
        plot_panel(ax, rgb_crop, q_img, grasps, depth_m, label)
        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fig.suptitle(
        "3-finger view: Q + center only (antipodal box/angle ignored)",
        fontsize=12,
    )
    fig.tight_layout()

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output, dpi=110, bbox_inches="tight")
    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
