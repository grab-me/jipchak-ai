import torch
import torch.nn as nn
import yaml
import numpy as np
import math
from typing import List, Optional
from .candidate import GraspCandidate
from .feature_extractor import FeatureExtractor


class GraspScoreMLP(nn.Module):
    """
    학습된 파지 성공 확률을 계산하는 MLP 모델.

    구조:
        [FEATURE_DIM] → Linear(64) → ReLU → Dropout(0.2)
                      → Linear(32) → ReLU
                      → Linear(1)  → Sigmoid  → 파지 성공 확률 [0, 1]
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class ThreeJawEvaluator:
    """
    3발 집게에 최적화된 파지 후보를 선택하는 평가기.

    Rule-based와 MLP 기반 평가를 모두 지원하며,
    개별 평가 함수를 override하여 커스터마이징할 수 있습니다.

    Attributes
    ----------
    extractor : FeatureExtractor
        특징 추출기. MLP 모드에서 사용.
    config    : dict
        집게 스펙 및 평가 가중치. YAML 또는 기본값 사용.
    model     : GraspScoreMLP | None
        학습된 MLP 모델. None이면 Rule-based 모드.

    Override 가능한 메서드 (커스터마이징 포인트):
        _width_fitness(g)   — 파지 폭 적합도
        _height_fitness(g)  — 높이 적합도
        _stability(g)       — 수직 접근 안정성
        _symmetry(g)        — 3발 120° 대칭 적합도
        _original_score(g)  — 원본 모델 신뢰도
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        model_path: Optional[str] = None,
        extractor: Optional[FeatureExtractor] = None
    ):
        self.extractor = extractor or FeatureExtractor()
        self.config = self._load_config(config_path)
        self.model = None

        if model_path:
            self.model = GraspScoreMLP(self.extractor.FEATURE_DIM)
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.eval()

    # ------------------------------------------------------------------
    # 공개 메서드
    # ------------------------------------------------------------------

    def select_best(
        self,
        candidates: List[GraspCandidate],
        depth: Optional[np.ndarray] = None
    ) -> GraspCandidate:
        """
        후보 리스트 중 가장 높은 점수의 파지를 반환.

        Parameters
        ----------
        candidates : list[GraspCandidate]
        depth      : np.ndarray [H, W] float32 (미터). 없으면 None.

        Returns
        -------
        GraspCandidate — 최고 점수 후보 1개
        """
        if not candidates:
            raise ValueError("평가할 파지 후보가 없습니다.")

        if self.model:
            scores = self._mlp_scores(candidates, depth)
        else:
            scores = [self._calculate_rule_score(c, depth) for c in candidates]

        best_idx = int(np.argmax(scores))
        return candidates[best_idx]

    def score_all(
        self,
        candidates: List[GraspCandidate],
        depth: Optional[np.ndarray] = None
    ) -> List[float]:
        """
        모든 후보의 점수를 리스트로 반환 (디버깅/시각화용).
        """
        if self.model:
            return list(self._mlp_scores(candidates, depth))
        return [self._calculate_rule_score(c, depth) for c in candidates]

    # ------------------------------------------------------------------
    # 내부 점수 계산 — override 가능
    # ------------------------------------------------------------------

    def _width_fitness(self, g: GraspCandidate) -> float:
        """
        파지 폭 적합도.
        이상적 폭(ideal_width)에 가까울수록 높은 점수.
        허용 범위(min~max) 밖이면 0.

        Override 예시:
            def _width_fitness(self, g):
                return 1.0 if 0.03 <= g.width <= 0.07 else 0.0
        """
        spec = self.config['gripper']
        if g.width < spec['min_width'] or g.width > spec['max_width']:
            return 0.0
        sigma = (spec['max_width'] - spec['min_width']) / 2.0
        return float(np.exp(-0.5 * ((g.width - spec['ideal_width']) / sigma) ** 2))

    def _height_fitness(self, g: GraspCandidate) -> float:
        """
        파지 높이 적합도.
        충돌 임계값(collision_threshold_z)보다 높으면 1.0.
        그 이하이면 선형으로 감소.

        Override 예시 (특정 높이 범위만 허용):
            def _height_fitness(self, g):
                return 1.0 if 0.05 <= g.center_z <= 0.40 else 0.0
        """
        threshold = self.config['gripper']['collision_threshold_z']
        if g.center_z >= threshold:
            return 1.0
        if g.center_z <= 0:
            return 0.0
        return float(g.center_z / threshold)

    def _stability(self, g: GraspCandidate) -> float:
        """
        수직 접근 안정성.

        3발 집게가 위에서 수직으로 내려올 때 가장 안정적.
        집게가 Z축 자유 회전 가능 시, 파지 각도와 집게 3발 배치
        (0°, 60°, 120°... 60° 주기)가 얼마나 잘 정렬되는지로 판단.

        수식:
            period    = π / 3               (60° 주기)
            residual  = angle mod period    → [0, π/3)
            deviation = min(residual, period - residual)  → [0, π/6]
            stability = 1 - deviation / (π/6)             → [0, 1]

            → angle이 0°, 60°, 120°, ... 에 가까울수록 1.0
            → 30°(π/6) 편차일 때 0.0

        Override 예시 (하드웨어 제약으로 특정 각도만 허용):
            def _stability(self, g):
                return 1.0 if abs(g.angle) < 0.2 else 0.5
        """
        period    = math.pi / 3.0                     # 60°
        residual  = abs(g.angle % period)              # [0, π/3)
        deviation = min(residual, period - residual)   # [0, π/6]
        max_dev   = period / 2.0                       # π/6
        return float(1.0 - deviation / max_dev)

    def _symmetry(self, g: GraspCandidate) -> float:
        """
        3발 집게의 120° 대칭 구조 적합도.

        3발 집게는 0°, 120°, 240° 방향으로 발이 배치되므로,
        파지 각도가 이 배치와 잘 정렬될수록 안정적.

        수식: cos(3 × angle)²
            → angle = 0, π/3, 2π/3, π ... 에서 최대 (1.0)
            → angle = π/6, π/2, ... 에서 최소 (0.0)

        Override 예시 (4발 집게의 경우):
            def _symmetry(self, g):
                return math.cos(4 * g.angle) ** 2
        """
        return float(math.cos(3.0 * g.angle) ** 2)

    def _original_score(self, g: GraspCandidate) -> float:
        """
        원본 탐지 모델의 신뢰도 점수.

        반환값을 [0, 1]로 클램핑하여 가중합 범위를 보장합니다.
        모델 출력이 [0, 1] 범위를 벗어나는 경우 경고 없이 클램핑됩니다.
        (예: logit 스코어나 [0, 100] 범위 모델 연동 시 주의)

        Override 예시 (특정 임계값 이하 무시):
            def _original_score(self, g):
                return g.original_score if g.original_score > 0.5 else 0.0
        """
        return float(min(1.0, max(0.0, g.original_score)))

    # ------------------------------------------------------------------
    # 내부 구현
    # ------------------------------------------------------------------

    def _calculate_rule_score(
        self,
        g: GraspCandidate,
        depth: Optional[np.ndarray]
    ) -> float:
        """개별 기준 점수를 가중합하여 최종 Rule-based 점수 계산."""
        w = self.config['evaluation']['rule_weights']
        score = (
            w['width']     * self._width_fitness(g)   +
            w['score']     * self._original_score(g)   +
            w['height']    * self._height_fitness(g)   +
            w['stability'] * self._stability(g)        +
            w['symmetry']  * self._symmetry(g)
        )
        return float(score)

    def score_detail(self, g: GraspCandidate, depth: Optional[np.ndarray] = None) -> dict:
        """
        단일 후보의 세부 점수를 딕셔너리로 반환 (디버깅 / 시각화용).

        Returns
        -------
        dict — 각 기준별 원점수 및 가중 점수, 총점 포함
            {
              'width'    : {'raw': float, 'weighted': float},
              'score'    : {'raw': float, 'weighted': float},
              'height'   : {'raw': float, 'weighted': float},
              'stability': {'raw': float, 'weighted': float},
              'symmetry' : {'raw': float, 'weighted': float},
              'total'    : float
            }
        """
        w  = self.config['evaluation']['rule_weights']
        rw = self._width_fitness(g)
        rs = self._original_score(g)
        rh = self._height_fitness(g)
        rb = self._stability(g)
        ry = self._symmetry(g)
        total = (w['width']*rw + w['score']*rs + w['height']*rh
                 + w['stability']*rb + w['symmetry']*ry)
        return {
            'width'    : {'raw': rw, 'weighted': w['width']     * rw},
            'score'    : {'raw': rs, 'weighted': w['score']     * rs},
            'height'   : {'raw': rh, 'weighted': w['height']    * rh},
            'stability': {'raw': rb, 'weighted': w['stability'] * rb},
            'symmetry' : {'raw': ry, 'weighted': w['symmetry']  * ry},
            'total'    : total,
        }

    def _mlp_scores(
        self,
        candidates: List[GraspCandidate],
        depth: Optional[np.ndarray]
    ) -> np.ndarray:
        """MLP 모델로 모든 후보의 성공 확률 계산."""
        features = self.extractor.extract_batch(candidates, depth)
        with torch.no_grad():
            tensor = torch.from_numpy(features)
            scores = self.model(tensor).squeeze().numpy()
        # 후보 1개일 때 squeeze()가 0-d 텐서를 반환하므로 1D 보장
        return np.atleast_1d(scores)

    def _load_config(self, path: Optional[str]) -> dict:
        if path:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        # 기본값: 인형뽑기 기계용 (3발 120° 고정 → symmetry=0)
        # 실제 집게 스펙 측정 후 config/gripper_spec.yaml을 작성하고 config_path로 전달 권장
        return {
            'gripper': {
                'min_width': 0.01,
                'max_width': 0.10,
                'ideal_width': 0.05,
                'collision_threshold_z': 0.02
            },
            'evaluation': {
                'rule_weights': {
                    'width':     0.35,  # 물리 제약 — 가장 중요
                    'score':     0.30,  # 탐지 신뢰도
                    'height':    0.25,  # 바닥 충돌 방지
                    'stability': 0.10,  # 접근 각도 정렬
                    'symmetry':  0.00,  # 3발 고정 120° → 평가 불필요
                }
            }
        }
