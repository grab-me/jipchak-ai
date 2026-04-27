import numpy as np
from typing import Any, List, Optional
from .candidate import GraspCandidate
from .adapters import BaseGraspAdapter
from .evaluator import ThreeJawEvaluator

class GraspPipeline:
    """
    외부 모델의 출력을 받아 최적의 3발 파지를 선택하는 전체 파이프라인.
    """
    def __init__(
        self,
        model: Any,  # 사용자가 로드한 외부 모델
        adapter: BaseGraspAdapter,
        evaluator: Optional[ThreeJawEvaluator] = None
    ):
        self.model = model
        self.adapter = adapter
        self.evaluator = evaluator or ThreeJawEvaluator()

    def predict_best(self, rgb: np.ndarray, depth: np.ndarray) -> GraspCandidate:
        """
        1. 외부 모델 추론
        2. 어댑터를 통한 포맷 변환
        3. 평가기를 통한 최적 파지 선택
        """
        # 외부 모델의 predict 메서드는 사용자 정의에 따라 다를 수 있음 (인터페이스 규약 준수 권장)
        raw_output = self.model.predict(rgb, depth)
        
        # 포맷 변환
        candidates = self.adapter.adapt(raw_output)
        
        if not candidates:
            raise ValueError("탐지된 파지 후보가 없습니다.")

        # 최적 파지 선택
        best_grasp = self.evaluator.select_best(candidates, depth=depth)
        
        return best_grasp
