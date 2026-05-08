"""Cornell dataset 무작위 샘플 시각화 — ch16 robustness 정성 검증.

n_samples 장 무작위 추출 → RGB + Q맵 alpha overlay + grasp center 점.
같은 모델로 다양한 입력에서 일관성 확인 (n=1 → n=여러 해소).

Usage:
  python visualize_cornell_samples.py --n_samples 5
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.feature import peak_local_max

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from inference.post_process import post_process_output  # noqa: E402
from utils.data import get_dataset  # noqa: E402

DEFAULT_CKPT = (
    "trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch16/"
    "epoch_30_iou_0.97"
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--network", default=DEFAULT_CKPT)
    p.add_argument("--dataset-path",
                   default=os.path.expanduser("~/datasets/cornell-grasp"))
    p.add_argument("--n_samples", type=int, default=5)
    p.add_argument("--n_grasps", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/cornell_samples_ch16.png")
    args = p.parse_args()

    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)

    Dataset = get_dataset("cornell")
    dataset = Dataset(
        args.dataset_path, output_size=224,
        include_depth=True, include_rgb=True,
        random_rotate=False, random_zoom=False,
    )
    print(f"Cornell dataset size: {dataset.length}")

    indices = random.sample(range(dataset.length), args.n_samples)
    print(f"sampled indices: {indices}")

    net = torch.load(args.network, map_location=device)
    net.eval()

    fig, axes = plt.subplots(1, args.n_samples,
                             figsize=(5 * args.n_samples, 5))
    if args.n_samples == 1:
        axes = [axes]

    print(f"\nResults (model: {os.path.basename(args.network)}):")
    for ax, idx in zip(axes, indices):
        x, _, _, _, _ = dataset[idx]
        x = x.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = net.predict(x)
            q_img, ang_img, _ = post_process_output(
                pred["pos"], pred["cos"], pred["sin"], pred["width"]
            )

        rgb_crop = dataset.get_rgb(idx, rot=0, zoom=1.0, normalise=False)

        local_max = peak_local_max(
            q_img, min_distance=20,
            threshold_abs=0.2,
            num_peaks=args.n_grasps,
        )

        print(f"  idx={idx:4d}  Q_max={q_img.max():.3f}  n={len(local_max)}")

        ax.imshow(rgb_crop)
        ax.imshow(q_img, cmap="jet", alpha=0.40, vmin=0, vmax=1)
        for (yy, xx) in local_max:
            qv = float(q_img[yy, xx])
            ax.plot(xx, yy, "o",
                    markersize=8 + qv * 14,
                    markerfacecolor=plt.cm.jet(qv),
                    markeredgecolor="white",
                    markeredgewidth=2,
                    alpha=0.95)
            ax.text(xx + 6, yy + 4, f"{qv:.2f}",
                    color="white", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="black", alpha=0.7))
        ax.set_title(f"idx={idx}  Q_max={q_img.max():.2f}  n={len(local_max)}",
                     fontsize=9)
        ax.axis("off")

    fig.suptitle(
        f"Cornell random samples — {os.path.basename(args.network)} "
        f"(n={args.n_samples}, seed={args.seed})",
        fontsize=11,
    )
    fig.tight_layout()

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output, dpi=110, bbox_inches="tight")
    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
