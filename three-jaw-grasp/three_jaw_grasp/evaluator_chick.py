import numpy as np
import math
from typing import Optional

from .candidate import GraspCandidate
from .evaluator import ThreeJawEvaluator
from .factory import EvaluatorFactory

@EvaluatorFactory.register("chick")
class ChickEvaluator(ThreeJawEvaluator):
    """
    병아리 인형 전용 파지 평가기.
    - 최대 폭(Width) 90mm 제한.
    - 파지 중심이 물체 마스크 무게중심에 가까울수록 점수 가산.
    """
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        super().__init__(config_path=config_path, **kwargs)
        
        # 하드웨어/인형 스펙 강제 오버라이딩
        # 병아리 인형 스펙: 높이 130mm, 폭 90mm, 길이 110mm
        # 이미지의 pixel 단위를 0.001 (1mm = 1px) 등 스케일로 맞추거나,
        # 어댑터에서 pixel을 그대로 넘기므로 여기서 px 기준으로 비교.
        # 예시: 90mm = 90px (근사치, 실제 카메라/해상도에 따라 변환 비율 수정 필요)
        self.pixel_per_mm = 1.0 # 1mm = 1px로 가정
        
        self.config['gripper']['min_width'] = 10 * self.pixel_per_mm
        self.config['gripper']['max_width'] = 90 * self.pixel_per_mm
        self.config['gripper']['ideal_width'] = 50 * self.pixel_per_mm
        
        # 가중치 재분배: 중심 거리가 매우 중요함
        w = self.config['evaluation']['rule_weights']
        # 중심 밀착도를 새로 추가하기 위해 가중치를 줄입니다.
        total_w = sum(w.values())
        w['center'] = 0.40  # 중심 밀착도 40% 할당
        w['width']  = 0.25
        w['score']  = 0.20
        w['height'] = 0.10
        w['stability'] = 0.05
        
    def _center_fitness(self, g: GraspCandidate) -> float:
        """
        물체 마스크의 무게중심(Centroid)과 파지점(center_x, center_y) 간의 거리 점수화.
        가까울수록 1.0, 멀어질수록 0.0
        """
        if getattr(g, 'mask', None) is None:
            return 1.0 # 마스크가 없으면 기본값
            
        import cv2
        mask = g.mask
        M = cv2.moments(mask)
        if M["m00"] != 0:
            mc_x = M["m10"] / M["m00"]
            mc_y = M["m01"] / M["m00"]
        else:
            return 0.5
            
        dist = math.sqrt((g.center_x - mc_x)**2 + (g.center_y - mc_y)**2)
        
        # 거리가 0이면 1.0, 50픽셀 이상 벗어나면 0점
        max_dist = 50.0 
        return float(max(0.0, 1.0 - (dist / max_dist)))

    def _calculate_rule_score(
        self,
        g: GraspCandidate,
        depth: Optional[np.ndarray]
    ) -> float:
        """기존 점수 계산식에 center_fitness 추가"""
        w = self.config['evaluation']['rule_weights']
        
        score = (
            w.get('width', 0)     * self._width_fitness(g)   +
            w.get('score', 0)     * self._original_score(g)   +
            w.get('height', 0)    * self._height_fitness(g)   +
            w.get('stability', 0) * self._stability(g)        +
            w.get('symmetry', 0)  * self._symmetry(g)         +
            w.get('center', 0)    * self._center_fitness(g)
        )
        score *= self._mask_fitness(g)
        return float(min(0.90, score))

    def score_detail(self, g: GraspCandidate, depth: Optional[np.ndarray] = None) -> dict:
        details = super().score_detail(g, depth)
        
        w = self.config['evaluation']['rule_weights']
        rc = self._center_fitness(g)
        
        # 기존 딕셔너리 업데이트
        details['center'] = {'raw': rc, 'weighted': w.get('center', 0) * rc}
        
        # total 재계산
        rm = self._mask_fitness(g)
        base_total = sum(d['weighted'] for k, d in details.items() if k not in ['total', 'mask'])
        total = base_total * rm
        details['total'] = float(min(0.90, total))
        
        return details
