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
from grasp_to_3dof import grasp_to_3dof  # noqa: E402


CHECKPOINTS = [
    ("cornell ch16", "trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch16/epoch_30_iou_0.97"),
    ("cornell ch32", "trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98"),
]
OUTPUT_SIZE = 224


def run_one(net, device, rgb, depth_raw_2d, depth_m_2d, n_grasps):
    """모델 입력엔 raw depth (학습 분포 유지), 후처리 z 추출엔 m 단위."""
    H, W = rgb.shape[:2]
    cam = CameraData(width=W, height=H, output_size=OUTPUT_SIZE,
                     include_depth=True, include_rgb=True)
    depth_in = depth_raw_2d[..., None]  # (H, W, 1) raw — CameraData 내부에서 자체 정규화
    x, _, _ = cam.get_data(rgb=rgb, depth=depth_in)
    rgb_crop = cam.get_rgb(rgb, norm=False)

    x = x.to(device)
    with torch.no_grad():
        pred = net.predict(x)
        q_img, _, _ = post_process_output(
            pred["pos"], pred["cos"], pred["sin"], pred["width"]
        )

    grasps = grasp_to_3dof(
        q_img, depth_m_2d,
        original_size=(W, H),
        crop_size=OUTPUT_SIZE,
        top_k=n_grasps,
    )
    return rgb_crop, q_img, grasps


def plot_panel(ax, rgb_crop, q_img, grasps, label):
    """점 + 좌표 라벨만. antipodal 박스/각도 무시 (3-finger 관점)."""
    ax.imshow(rgb_crop)
    ax.imshow(q_img, cmap="jet", alpha=0.40, vmin=0, vmax=1)

    print(f"\n  [{label}] candidates (3-finger view):")
    for i, g in enumerate(grasps):
        cx_crop = g.metadata["crop_x"]
        cy_crop = g.metadata["crop_y"]
        z_str = f"z={g.z:.3f}m" if g.z is not None else "z=N/A"

        print(f"    #{i}  pixel=({int(g.x):4d},{int(g.y):4d})  "
              f"{z_str}  q={g.score:.3f}")

        msize = 8 + g.score * 14  # 8~22
        ax.plot(cx_crop, cy_crop, "o",
                markersize=msize,
                markerfacecolor=plt.cm.jet(g.score),
                markeredgecolor="white",
                markeredgewidth=2,
                alpha=0.95)
        ax.text(cx_crop + 9, cy_crop + 4,
                f"#{i} q={g.score:.2f}\n{z_str}",
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
    depth_raw_2d = np.array(Image.open(args.depth_path))  # (H, W) 16-bit raw

    # depth -> meters (GraspNet meta.mat factor_depth)
    meta = scio.loadmat(args.meta_path)
    factor_depth = float(meta["factor_depth"].squeeze())
    depth_m_2d = depth_raw_2d.astype(np.float32) / factor_depth
    print(f"input rgb: {rgb.shape}  depth: {depth_raw_2d.shape}  "
          f"factor_depth={factor_depth}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, (label, ckpt) in zip(axes, CHECKPOINTS):
        net = torch.load(ckpt, map_location=device)
        net.eval()
        rgb_crop, q_img, grasps = run_one(
            net, device, rgb, depth_raw_2d, depth_m_2d, args.n_grasps
        )
        plot_panel(ax, rgb_crop, q_img, grasps, label)
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
