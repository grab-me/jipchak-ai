import numpy as np
from typing import List, Optional
from .candidate import GraspCandidate

class FeatureExtractor:
    """
    GraspCandidate를 평가 모델용 특징 벡터로 변환.
    """
    FEATURE_DIM = 15

    def extract(self, grasp: GraspCandidate, depth: Optional[np.ndarray] = None) -> np.ndarray:
        """
        단일 파지 후보에 대한 특징 추출.
        """
        # 기본 특징 (7개)
        features = [
            grasp.width,
            grasp.original_score,
            grasp.center_z,
            grasp.center_x,
            grasp.center_y,
            np.sin(grasp.angle),
            np.cos(grasp.angle)
        ]

        # Depth 기반 추가 특징 (8개)
        if depth is not None:
            depth_patch = self._get_depth_patch(grasp, depth)
            features.extend([
                np.mean(depth_patch),
                np.std(depth_patch),
                np.min(depth_patch),
                np.max(depth_patch),
                np.median(depth_patch),
                np.percentile(depth_patch, 25),
                np.percentile(depth_patch, 75),
                float(np.any(depth_patch < 0.01))  # 바닥 충돌 위험 여부
            ])
        else:
            features.extend([0.0] * 8)

        return np.array(features, dtype=np.float32)

    def extract_batch(self, candidates: List[GraspCandidate], depth: Optional[np.ndarray] = None) -> np.ndarray:
        """
        학습 또는 추론 시 배치 처리를 위한 특징 추출.
        """
        features_list = [self.extract(c, depth) for c in candidates]
        return np.stack(features_list)

    def _get_depth_patch(self, grasp: GraspCandidate, depth: np.ndarray, size: int = 20) -> np.ndarray:
        """
        파지 중심 주변의 depth 맵 패치 추출.
        주 좌표계 불일치 시 보정 로직이 필요할 수 있음.
        """
        h, w = depth.shape
        # 단순화를 위해 center_x, center_y를 픽셀 좌표로 가정 (현실적으로는 좌표계 변환 필요)
        cx, cy = int(grasp.center_x), int(grasp.center_y)
        
        x1 = max(0, cx - size // 2)
        x2 = min(w, cx + size // 2)
        y1 = max(0, cy - size // 2)
        y2 = min(h, cy + size // 2)
        
        patch = depth[y1:y2, x1:x2]
        if patch.size == 0:
            return np.zeros((size, size))
        return patch
