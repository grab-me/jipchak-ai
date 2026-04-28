"""두 AI 통합 wrapper.

Detection (SSDlite-MobileNetV3) → bbox → workspace_mask → Grasp (GR-ConvNet ch16)
→ 3 DOF (x, y, z, score) 후보.

이게 인형뽑기 시스템의 진짜 추론 진입점. 메인 레포(jipchak)는 이 pipeline 의
predict() 만 호출하면 됨.

CLI:
    python pipeline.py --rgb color.png --depth depth.png

Library:
    from pipeline import GraspPipeline
    p = GraspPipeline(device='cuda:0')
    out = p.predict(rgb, depth_m, depth_raw=depth_raw, top_k=5)
    # out = {'detections': [...], 'grasps': [Grasp3DoF, ...], 'mask_used': bool}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np
import scipy.io as scio
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "gr-convnet"))
sys.path.insert(0, os.path.join(ROOT, "detection"))

from detect import DetectInfer, Detection  # noqa: E402
from infer import GraspInfer  # noqa: E402

# 인형뽑기 시나리오에서 의미 있는 COCO 클래스 (사전학습 SSDlite 한정)
# fine-tune 후엔 자체 클래스 사용 — 이건 도메인 전 PoC 용
DOLL_LIKE_COCO = {
    88,  # teddy bear (가장 직접적)
    # 다른 인형/장난감 클래스가 COCO 80에 없음. fine-tune 시 자체 라벨 사용.
}


def bboxes_to_mask(
    bboxes: list[tuple],
    image_shape: tuple[int, int],
    expand_ratio: float = 0.0,
) -> np.ndarray:
    """bbox 리스트 → (H, W) bool workspace_mask.

    :param bboxes: list of (x1, y1, x2, y2) 픽셀
    :param image_shape: (H, W)
    :param expand_ratio: bbox 크기 대비 확장 (0.1 = 10% 더 크게 = grasp 후보 영역 살짝 넓힘)
    """
    H, W = image_shape
    mask = np.zeros((H, W), dtype=bool)
    for x1, y1, x2, y2 in bboxes:
        if expand_ratio > 0:
            w = x2 - x1
            h = y2 - y1
            x1 -= w * expand_ratio / 2
            y1 -= h * expand_ratio / 2
            x2 += w * expand_ratio / 2
            y2 += h * expand_ratio / 2
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(W, int(x2))
        y2 = min(H, int(y2))
        mask[y1:y2, x1:x2] = True
    return mask


class GraspPipeline:
    """Detection → workspace mask → Grasp 통합 추론.

    두 모델 startup 시 1회 로드, 매 predict() 에 재사용.
    """

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device
        self.detector = DetectInfer(device=device)
        self.grasper = GraspInfer(device=device)

    def predict(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        *,
        depth_raw: Optional[np.ndarray] = None,
        det_conf: float = 0.05,
        det_top_k: int = 20,
        det_class_filter: Optional[set[int]] = None,
        grasp_top_k: int = 5,
        bbox_expand: float = 0.1,
    ) -> dict:
        """
        :param det_class_filter: COCO id 집합. None 이면 전체 클래스 (fine-tune 전엔 None 권장 — teddy bear 만 있어 너무 좁음)
        :param bbox_expand: bbox 영역을 약간 확장 (grasp 후보가 bbox 경계에서 잘림 방지)
        :return: {
            'detections': [...dict...],   # bbox/class/score
            'grasps': [...Grasp3DoF...],  # x, y, z, score
            'mask_used': bool,             # workspace_mask 적용했는지
            'det_ms': float, 'grasp_ms': float, 'total_ms': float,
        }
        """
        H, W = rgb.shape[:2]
        if depth_raw is None:
            depth_raw = (depth_m * 1000.0).astype(np.float32)

        # 1. Detection
        t0 = time.perf_counter()
        detections = self.detector.predict(
            rgb, conf=det_conf, top_k=det_top_k, class_filter=det_class_filter,
        )
        det_ms = (time.perf_counter() - t0) * 1000

        # 2. workspace_mask 생성
        if detections:
            bboxes = [d.bbox for d in detections]
            workspace_mask = bboxes_to_mask(bboxes, (H, W), expand_ratio=bbox_expand)
            mask_used = True
        else:
            workspace_mask = None
            mask_used = False

        # 3. Grasp (workspace_mask 적용)
        t0 = time.perf_counter()
        grasps = self.grasper.predict(
            rgb, depth_m,
            depth_raw=depth_raw,
            top_k=grasp_top_k,
            workspace_mask=workspace_mask,
        )
        grasp_ms = (time.perf_counter() - t0) * 1000

        return {
            "detections": detections,
            "grasps": grasps,
            "mask_used": mask_used,
            "det_ms": round(det_ms, 2),
            "grasp_ms": round(grasp_ms, 2),
            "total_ms": round(det_ms + grasp_ms, 2),
        }


def _serialize(out: dict) -> dict:
    """JSON 직렬화 가능한 형태로 변환."""
    return {
        "det_ms": out["det_ms"],
        "grasp_ms": out["grasp_ms"],
        "total_ms": out["total_ms"],
        "mask_used": out["mask_used"],
        "n_detections": len(out["detections"]),
        "detections": [
            {
                "bbox": list(d.bbox),
                "class_id": d.class_id,
                "class_name": d.class_name,
                "score": d.score,
            }
            for d in out["detections"]
        ],
        "n_grasps": len(out["grasps"]),
        "grasps": [
            {
                "x": float(g.x),
                "y": float(g.y),
                "z": (float(g.z) if g.z is not None else None),
                "score": float(g.score),
            }
            for g in out["grasps"]
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rgb", required=True)
    p.add_argument("--depth", required=True)
    p.add_argument("--meta", default=None,
                   help="meta.mat (factor_depth 추출용). 없으면 --factor-depth 사용")
    p.add_argument("--factor-depth", type=float, default=1000.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--det-conf", type=float, default=0.05)
    p.add_argument("--det-top-k", type=int, default=20)
    p.add_argument("--grasp-top-k", type=int, default=5)
    p.add_argument("--bbox-expand", type=float, default=0.1)
    p.add_argument("--doll-only", action="store_true",
                   help="teddy bear (COCO 88) 클래스만 detection 으로 사용")
    p.add_argument("--output", default="-")
    args = p.parse_args()

    rgb = np.array(Image.open(args.rgb).convert("RGB"))
    depth_raw = np.array(Image.open(args.depth)).astype(np.float32)
    if args.meta:
        meta = scio.loadmat(args.meta)
        factor_depth = float(meta["factor_depth"].squeeze())
    else:
        factor_depth = args.factor_depth
    depth_m = depth_raw / factor_depth

    cls_filter = DOLL_LIKE_COCO if args.doll_only else None

    print(f"Loading models on {args.device}...", file=sys.stderr)
    t0 = time.perf_counter()
    pipeline = GraspPipeline(device=args.device)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  loaded in {load_ms:.1f} ms", file=sys.stderr)

    # Warmup (cold start 영향 회피 위해 1회 dummy 호출)
    pipeline.predict(rgb, depth_m, depth_raw=depth_raw,
                     det_conf=args.det_conf,
                     det_top_k=args.det_top_k,
                     grasp_top_k=args.grasp_top_k,
                     det_class_filter=cls_filter,
                     bbox_expand=args.bbox_expand)

    # 진짜 측정
    out = pipeline.predict(
        rgb, depth_m, depth_raw=depth_raw,
        det_conf=args.det_conf,
        det_top_k=args.det_top_k,
        grasp_top_k=args.grasp_top_k,
        det_class_filter=cls_filter,
        bbox_expand=args.bbox_expand,
    )

    result = {
        "models": {
            "detection": "ssdlite320_mobilenet_v3_large",
            "grasp": "gr-convnet-cornell-ch16",
        },
        "device": args.device,
        "input": {
            "rgb": args.rgb, "depth": args.depth,
            "factor_depth": factor_depth,
            "size": [int(rgb.shape[1]), int(rgb.shape[0])],
        },
        "load_ms": round(load_ms, 2),
        "doll_only": args.doll_only,
        **_serialize(out),
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
