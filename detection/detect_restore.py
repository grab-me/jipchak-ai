"""Detection AI 단일 진입점.

채택 모델: torchvision `ssdlite320_mobilenet_v3_large` (BSD-3, COCO 80 클래스).
Grasp AI 의 `infer.py` 와 동일한 CLI/Library 패턴.

CLI:
    python detect.py --rgb color.png --output result.json

Library:
    from detect import DetectInfer
    detector = DetectInfer(device='cuda:0')
    detections = detector.predict(rgb_array, conf=0.05, top_k=20)
    # detections: list[Detection] — bbox, class_id, class_name, score

출력 JSON:
{
  "model": "ssdlite320_mobilenet_v3_large",
  "device": "cuda:0",
  "load_ms": 1234,
  "inference_ms": 37.1,
  "n_detections": 5,
  "detections": [
    {"bbox": [x1, y1, x2, y2], "class_id": 88, "class_name": "teddy bear", "score": 0.91},
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
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.transforms.functional import to_tensor

# COCO 80 클래스 (torchvision detection 모델 표준)
COCO_CLASSES = [
    "__background__",
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella",
    "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A",
    "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


@dataclass
class Detection:
    """단일 detection 결과."""

    bbox: tuple  # (x1, y1, x2, y2) 픽셀
    class_id: int
    class_name: str
    score: float


class DetectInfer:
    """SSDlite-MobileNetV3 추론 — startup 시 1회 로드 후 재사용."""

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = torch.device(device)
        self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            weights="DEFAULT"
        )
        self.model.to(self.device).eval()

    def predict(
        self,
        rgb: np.ndarray,
        conf: float = 0.05,
        top_k: int = 20,
        class_filter: Optional[set[int]] = None,
    ) -> list[Detection]:
        """
        :param rgb: (H, W, 3) uint8
        :param conf: confidence 임계값
        :param top_k: 최대 후보 수
        :param class_filter: 허용할 class_id 집합 (None 이면 모두). 인형뽑기에선 보통 {88: teddy bear, 47: cup, 84: book...} 등
        :return: confidence 내림차순 후보 리스트
        """
        x = to_tensor(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            r = self.model(x)

        boxes = r[0]["boxes"].cpu().numpy()
        scores = r[0]["scores"].cpu().numpy()
        labels = r[0]["labels"].cpu().numpy()

        results: list[Detection] = []
        for box, score, label in zip(boxes, scores, labels):
            if score < conf:
                continue
            cls_id = int(label)
            if class_filter is not None and cls_id not in class_filter:
                continue
            cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else "unknown"
            results.append(
                Detection(
                    bbox=tuple(float(v) for v in box),
                    class_id=cls_id,
                    class_name=cls_name,
                    score=float(score),
                )
            )

        results.sort(key=lambda d: d.score, reverse=True)
        return results[:top_k]


def detections_to_dict(detections: list[Detection]) -> list[dict]:
    return [
        {
            "bbox": list(d.bbox),
            "class_id": d.class_id,
            "class_name": d.class_name,
            "score": d.score,
        }
        for d in detections
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rgb", required=True, help="RGB 이미지 경로")
    p.add_argument("--device", default="cuda:0", help="cuda:0 또는 cpu")
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--output", default="-",
                   help="JSON 출력 경로. '-' 면 stdout")
    args = p.parse_args()

    rgb = np.array(Image.open(args.rgb).convert("RGB"))

    t0 = time.perf_counter()
    detector = DetectInfer(device=args.device)
    load_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    detections = detector.predict(rgb, conf=args.conf, top_k=args.top_k)
    inference_ms = (time.perf_counter() - t0) * 1000

    result = {
        "model": "ssdlite320_mobilenet_v3_large",
        "device": args.device,
        "input": {"rgb": args.rgb, "size": [int(rgb.shape[1]), int(rgb.shape[0])]},
        "load_ms": round(load_ms, 2),
        "inference_ms": round(inference_ms, 2),
        "n_detections": len(detections),
        "detections": detections_to_dict(detections),
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
