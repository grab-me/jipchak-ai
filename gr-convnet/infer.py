"""단일 진입점 — RGB-D → 3 DOF grasp 후보.

메인 레포(jipchak)에서 두 가지 방식으로 호출:

1) subprocess (Spring Boot 등에서 호출):
    python infer.py --rgb color.png --depth depth.png --output result.json

2) Python import (다른 Python 코드에서):
    import sys; sys.path.insert(0, '/path/to/jipchak-ai/gr-convnet')
    from infer import GraspInfer

    infer = GraspInfer(device='cuda:0')   # 또는 'cpu'
    candidates = infer.predict(rgb, depth_m, depth_raw=depth_raw, top_k=5)
    # candidates: list[Grasp3DoF] — (x, y, z, score, metadata)

출력 JSON:
{
  "model": "cornell-randsplit-rgbd-grconvnet3-drop1-ch16",
  "device": "cuda:0",
  "load_ms": 152.3,
  "inference_ms": 3.1,
  "n_candidates": 5,
  "candidates": [
    {"x": 634, "y": 283, "z": 0.413, "score": 0.864},
    ...
  ]
}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from grasp_to_3dof import Grasp3DoF, grasp_to_3dof  # noqa: E402
from inference.post_process import post_process_output  # noqa: E402
from utils.data.camera_data import CameraData  # noqa: E402

DEFAULT_CHECKPOINT = os.path.join(
    ROOT, "trained-models",
    "cornell-randsplit-rgbd-grconvnet3-drop1-ch16",
    "epoch_30_iou_0.97",
)


class GraspInfer:
    """단일 입력 grasp 추론 — 모델 한 번 로드 후 반복 호출 가능."""

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str = "cuda:0",
        output_size: int = 224,
    ) -> None:
        self.device = torch.device(device)
        self.output_size = output_size
        self.checkpoint = checkpoint
        self.net = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.net.eval()

    def predict(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        *,
        depth_raw: Optional[np.ndarray] = None,
        top_k: int = 5,
        peak_min_distance: int = 20,
        peak_threshold: float = 0.2,
        workspace_mask: Optional[np.ndarray] = None,
    ) -> list[Grasp3DoF]:
        """
        :param rgb: (H, W, 3) uint8
        :param depth_m: (H, W) float32 — meter 단위 (z 좌표 추출용)
        :param depth_raw: (H, W) — 모델 입력용 raw 분포 유지값. 없으면 depth_m*1000
        :param top_k: 최대 후보 수
        :param workspace_mask: (H, W) bool — True 영역만 grasp 후보 허용 (Detection AI bbox 통합 시)
        :return: list[Grasp3DoF] — confidence 내림차순
        """
        H, W = rgb.shape[:2]
        if depth_raw is None:
            depth_raw = (depth_m * 1000.0).astype(np.float32)

        depth_in = depth_raw[..., None] if depth_raw.ndim == 2 else depth_raw

        cam = CameraData(
            width=W, height=H, output_size=self.output_size,
            include_depth=True, include_rgb=True,
        )
        x, _, _ = cam.get_data(rgb=rgb, depth=depth_in)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)

        with torch.no_grad():
            pred = self.net.predict(x)
            q_img, _, _ = post_process_output(
                pred["pos"], pred["cos"], pred["sin"], pred["width"]
            )

        return grasp_to_3dof(
            q_img, depth_m,
            original_size=(W, H),
            crop_size=self.output_size,
            top_k=top_k,
            peak_min_distance=peak_min_distance,
            peak_threshold=peak_threshold,
            workspace_mask=workspace_mask,
        )


def candidates_to_dict(candidates: list[Grasp3DoF]) -> list[dict]:
    """JSON 직렬화용 — Grasp3DoF 의 metadata 는 디버깅용이라 제외."""
    out = []
    for c in candidates:
        out.append({
            "x": float(c.x),
            "y": float(c.y),
            "z": (float(c.z) if c.z is not None else None),
            "score": float(c.score),
        })
    return out


def _load_rgbd(rgb_path: str, depth_path: str, factor_depth: float):
    rgb = np.array(Image.open(rgb_path))
    depth_raw = np.array(Image.open(depth_path)).astype(np.float32)
    depth_m = depth_raw / factor_depth
    return rgb, depth_m, depth_raw


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rgb", required=True, help="RGB 이미지 경로")
    p.add_argument("--depth", required=True, help="Depth 이미지 경로 (16-bit png/tiff)")
    p.add_argument("--factor-depth", type=float, default=1000.0,
                   help="raw depth → m 변환 인자 (RealSense/Kinect 보통 1000)")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--device", default="cuda:0", help="cuda:0 또는 cpu")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--output", default="-",
                   help="JSON 출력 경로. '-' 면 stdout")
    args = p.parse_args()

    rgb, depth_m, depth_raw = _load_rgbd(args.rgb, args.depth, args.factor_depth)

    t0 = time.perf_counter()
    infer = GraspInfer(checkpoint=args.checkpoint, device=args.device)
    load_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    candidates = infer.predict(rgb, depth_m, depth_raw=depth_raw, top_k=args.top_k)
    inference_ms = (time.perf_counter() - t0) * 1000

    result = {
        "model": os.path.basename(os.path.dirname(args.checkpoint)),
        "checkpoint": os.path.basename(args.checkpoint),
        "device": args.device,
        "input": {
            "rgb": args.rgb,
            "depth": args.depth,
            "factor_depth": args.factor_depth,
            "size": [int(rgb.shape[1]), int(rgb.shape[0])],
        },
        "load_ms": round(load_ms, 2),
        "inference_ms": round(inference_ms, 2),
        "n_candidates": len(candidates),
        "candidates": candidates_to_dict(candidates),
    }

    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output == "-":
        print(payload)
    else:
        with open(args.output, "w") as f:
            f.write(payload)
        print(f"saved: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
