from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np

@dataclass
class GraspCandidate:
    """
    표준 파지 후보 데이터 클래스.
    모든 모델의 출력은 이 포맷으로 변환되어 파이프라인에서 처리됨.
    """
    center_x: float
    center_y: float
    center_z: float
    width: float
    angle: float  # Radian
    original_score: float
    mask: Optional[np.ndarray] = None
    box: Optional[List[float]] = None
    raw: Dict[str, Any] = field(default_factory=dict)  # 원본 모델의 추가 정보 저장용

    def __post_init__(self):
        # 필요 시 데이터 유효성 검사 추가
        pass
