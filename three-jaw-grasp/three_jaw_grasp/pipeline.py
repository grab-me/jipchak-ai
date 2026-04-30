import numpy as np
from typing import Any, List, Optional
from .candidate import GraspCandidate
from .adapters import BaseGraspAdapter
from .evaluator import ThreeJawEvaluator
from .detector import ObjectDetector

class GraspPipeline:
    """
    외부 모델의 출력을 받아 최적의 3발 파지를 선택하는 전체 파이프라인.
    """
    def __init__(
        self,
        model: Any,  # 사용자가 로드한 외부 모델
        adapter: BaseGraspAdapter,
        evaluator: Optional[ThreeJawEvaluator] = None,
        detector: Optional[ObjectDetector] = None,
        is_3d: bool = False
    ):
        self.model = model
        self.adapter = adapter
        self.evaluator = evaluator or ThreeJawEvaluator()
        self.detector = detector
        self.is_3d = is_3d

    def predict_best(self, rgb: np.ndarray, depth: np.ndarray, filename_prefix: str = None) -> GraspCandidate:
        """
        1. (YOLO) 객체 디텍터로 바운딩 박스 추론 및 Crop
        2. 외부 파지 모델 추론
        3. 어댑터를 통한 포맷 변환 및 원본 좌표 복원
        4. 평가기를 통한 최적 파지 선택
        """
        # 1. Object Detection (YOLO)
        xmin, ymin, xmax, ymax = 0, 0, rgb.shape[1], rgb.shape[0]
        crop_rgb, crop_depth = rgb, depth
        
        if self.detector:
            xmin, ymin, xmax, ymax = self.detector.detect(rgb, filename_prefix=filename_prefix)
            crop_rgb = rgb[ymin:ymax, xmin:xmax]
            crop_depth = depth[ymin:ymax, xmin:xmax] if depth is not None else None

        # 2. 파지 모델 추론 (Cropped Image 기반)
        try:
            raw_output = self.model.predict(crop_rgb, crop_depth, crop_box=(xmin, ymin, xmax, ymax))
        except TypeError:
            raw_output = self.model.predict(crop_rgb, crop_depth)
        
        # 3. 포맷 변환
        candidates = self.adapter.adapt(raw_output)
        
        if not candidates:
            raise ValueError("탐지된 파지 후보가 없습니다.")

        # 좌표 복원 (Un-crop)
        # 3D 모델(GraspNet)은 point cloud를 다루기 때문에 모델 내부에서 crop_box를 기반으로 
        # 원본 좌표계의 3D(X,Y,Z) 값을 직접 계산하도록 책임을 넘깁니다.
        if self.detector and (xmin > 0 or ymin > 0) and not self.is_3d:
            orig_h, orig_w = rgb.shape[:2]
            for c in candidates:
                c.center_x += xmin
                c.center_y += ymin
                
                # 마스크가 존재할 경우, 원본 이미지 해상도로 패딩(복원)
                if getattr(c, 'mask', None) is not None:
                    padded_mask = np.zeros((orig_h, orig_w), dtype=c.mask.dtype)
                    crop_h, crop_w = c.mask.shape
                    y_end = min(ymin + crop_h, orig_h)
                    x_end = min(xmin + crop_w, orig_w)
                    padded_mask[ymin:y_end, xmin:x_end] = c.mask[:(y_end-ymin), :(x_end-xmin)]
                    c.mask = padded_mask

        # 4. 최적 파지 선택
        best_grasp = self.evaluator.select_best(candidates, depth=depth)
        
        return best_grasp
