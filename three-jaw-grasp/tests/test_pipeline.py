"""
유닛 테스트: Adapter → FeatureExtractor → Evaluator 전체 데이터 흐름 검증

실행 방법:
    cd three-jaw-grasp
    python -m pytest tests/test_pipeline.py -v
    또는
    python tests/test_pipeline.py
"""

import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from three_jaw_grasp.candidate import GraspCandidate
from three_jaw_grasp.adapters import GraspGroupAdapter, RectGraspAdapter, PoseArrayAdapter
from three_jaw_grasp.feature_extractor import FeatureExtractor
from three_jaw_grasp.evaluator import ThreeJawEvaluator
from three_jaw_grasp.pipeline import GraspPipeline


# =====================================================================
# 헬퍼
# =====================================================================

def make_dummy_depth(h=480, w=640, value=0.5) -> np.ndarray:
    return np.full((h, w), value, dtype=np.float32)


# =====================================================================
# 1. GraspCandidate 기본 검증
# =====================================================================

def test_grasp_candidate_creation():
    g = GraspCandidate(
        center_x=100.0, center_y=200.0, center_z=0.05,
        width=0.04, angle=0.0, original_score=0.9, raw={}
    )
    assert g.center_x == 100.0
    assert g.width == 0.04
    print("[PASS] test_grasp_candidate_creation")


# =====================================================================
# 2. GraspGroupAdapter
# =====================================================================

def test_graspgroup_adapter():
    class MockGrasp:
        def __init__(self, x, y, z, width, theta, score):
            self.x, self.y, self.z = x, y, z
            self.width = width
            self.theta = theta
            self.score = score

    raw = [
        MockGrasp(0.1, 0.2, 0.05, 0.04, 0.0, 0.9),
        MockGrasp(0.3, 0.4, 0.10, 0.06, 1.0, 0.7),
    ]
    adapter = GraspGroupAdapter()
    candidates = adapter.adapt(raw)

    assert len(candidates) == 2
    assert candidates[0].center_x == 0.1
    assert candidates[0].original_score == 0.9
    assert isinstance(candidates[1].angle, float)
    print("[PASS] test_graspgroup_adapter")


# =====================================================================
# 3. RectGraspAdapter (GR-ConvNet 포맷)
# =====================================================================

def test_rect_grasp_adapter_basic():
    """GR-ConvNet 출력 → GraspCandidate 변환 검증"""
    rect_pred = np.array([80.0, 90.0, 160.0, 170.0])   # [x1, y1, x2, y2]
    cls_score = np.zeros(20)
    cls_score[5] = 2.0   # 최대 bin = 5

    raw_output = {'rect_pred': rect_pred, 'cls_score': cls_score, 'depth': None}
    adapter = RectGraspAdapter()
    candidates = adapter.adapt(raw_output)

    assert len(candidates) == 1
    c = candidates[0]

    # 중심 좌표 확인
    assert abs(c.center_x - 120.0) < 1e-5, f"Expected 120.0, got {c.center_x}"
    assert abs(c.center_y - 130.0) < 1e-5, f"Expected 130.0, got {c.center_y}"

    # 각도 bin 5 → 라디안 변환: (5/20) * π - π/2 = π/4 - π/2 = -π/4
    expected_angle = (5 / 20) * math.pi - math.pi / 2
    assert abs(c.angle - expected_angle) < 1e-5, f"Expected {expected_angle:.4f}, got {c.angle:.4f}"

    # center_z = 0 (depth=None)
    assert c.center_z == 0.0
    print("[PASS] test_rect_grasp_adapter_basic")


def test_rect_grasp_adapter_with_depth():
    """depth 맵이 있을 때 center_z 추출 검증"""
    rect_pred = np.array([100.0, 100.0, 200.0, 200.0])
    cls_score = np.zeros(20); cls_score[0] = 1.0

    depth = make_dummy_depth(value=0.42)

    raw_output = {'rect_pred': rect_pred, 'cls_score': cls_score, 'depth': depth}
    adapter = RectGraspAdapter()
    candidates = adapter.adapt(raw_output)

    assert abs(candidates[0].center_z - 0.42) < 1e-4, f"Expected 0.42, got {candidates[0].center_z}"
    print("[PASS] test_rect_grasp_adapter_with_depth")


def test_rect_grasp_adapter_score_range():
    """원본 신뢰도가 softmax 결과로 [0, 1] 범위인지 검증"""
    rect_pred = np.array([0.0, 0.0, 100.0, 100.0])
    cls_score = np.random.randn(20)

    raw_output = {'rect_pred': rect_pred, 'cls_score': cls_score, 'depth': None}
    adapter = RectGraspAdapter()
    candidates = adapter.adapt(raw_output)

    score = candidates[0].original_score
    assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
    print("[PASS] test_rect_grasp_adapter_score_range")


# =====================================================================
# 4. PoseArrayAdapter
# =====================================================================

def test_pose_array_adapter():
    raw = [
        {'translation': [0.1, 0.2, 0.3], 'rotation': None, 'width': 0.05, 'score': 0.8},
        {'translation': [0.5, 0.6, 0.7], 'rotation': None, 'width': 0.06, 'score': 0.6},
    ]
    adapter = PoseArrayAdapter()
    candidates = adapter.adapt(raw)

    assert len(candidates) == 2
    assert candidates[0].center_z == 0.3
    assert candidates[0].angle == 0.0    # rotation=None → 0
    print("[PASS] test_pose_array_adapter")


# =====================================================================
# 5. FeatureExtractor
# =====================================================================

def test_feature_extractor_dim():
    """추출 벡터 차원이 FEATURE_DIM과 일치하는지 검증"""
    extractor = FeatureExtractor()
    g = GraspCandidate(100.0, 200.0, 0.05, 0.04, 0.0, 0.9, {})
    features = extractor.extract(g)
    assert features.shape == (extractor.FEATURE_DIM,), \
        f"Expected ({extractor.FEATURE_DIM},), got {features.shape}"
    assert features.dtype == np.float32
    print("[PASS] test_feature_extractor_dim")


def test_feature_extractor_batch():
    """배치 추출 형태 검증"""
    extractor = FeatureExtractor()
    candidates = [
        GraspCandidate(100.0, 200.0, 0.05, 0.04, 0.0, 0.9, {}),
        GraspCandidate(150.0, 250.0, 0.08, 0.06, 1.0, 0.7, {}),
    ]
    depth = make_dummy_depth()
    features = extractor.extract_batch(candidates, depth)
    assert features.shape == (2, extractor.FEATURE_DIM), \
        f"Expected (2, {extractor.FEATURE_DIM}), got {features.shape}"
    print("[PASS] test_feature_extractor_batch")


# =====================================================================
# 6. ThreeJawEvaluator — Rule-based
# =====================================================================

def test_evaluator_selects_best():
    """좋은 후보를 올바르게 선택하는지 검증"""
    evaluator = ThreeJawEvaluator()
    candidates = [
        GraspCandidate(100.0, 100.0, 0.05, 0.04, 0.0, 0.9, {}),  # 우수
        GraspCandidate(200.0, 200.0, 0.01, 0.10, 0.5, 0.3, {}),  # 낮은 높이 + 낮은 신뢰도
        GraspCandidate(300.0, 300.0, 0.00, 0.04, 0.0, 0.8, {}),  # center_z=0 → 바닥 충돌
    ]
    best = evaluator.select_best(candidates)
    assert best is candidates[0], "가장 좋은 후보를 선택하지 못함"
    print("[PASS] test_evaluator_selects_best")


def test_evaluator_score_all_length():
    """score_all()이 후보 수와 같은 길이를 반환하는지 검증"""
    evaluator = ThreeJawEvaluator()
    candidates = [
        GraspCandidate(100.0, 100.0, 0.05, 0.04, 0.0, 0.9, {}),
        GraspCandidate(200.0, 200.0, 0.05, 0.04, 0.5, 0.7, {}),
        GraspCandidate(300.0, 300.0, 0.05, 0.04, 1.0, 0.5, {}),
    ]
    scores = evaluator.score_all(candidates)
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.5 for s in scores), f"Unexpected score range: {scores}"
    print("[PASS] test_evaluator_score_all_length")


def test_evaluator_stability_not_constant():
    """_stability()가 각도에 따라 다른 값을 반환하는지 (1.0 고정이 아닌지) 검증"""
    evaluator = ThreeJawEvaluator()
    g1 = GraspCandidate(0, 0, 0.05, 0.04, 0.0,           0.9, {})
    g2 = GraspCandidate(0, 0, 0.05, 0.04, math.pi / 6,   0.9, {})
    s1 = evaluator._stability(g1)
    s2 = evaluator._stability(g2)
    assert s1 != s2, f"_stability()가 모든 각도에서 동일한 값을 반환함: {s1}"
    print(f"[PASS] test_evaluator_stability_not_constant  (angle=0 → {s1:.3f}, angle=π/6 → {s2:.3f})")


def test_evaluator_width_out_of_range():
    """허용 범위 밖의 폭은 width_fitness=0 이어야 함"""
    evaluator = ThreeJawEvaluator()
    g_too_small = GraspCandidate(0, 0, 0.05, 0.001, 0.0, 0.9, {})  # min_width=0.01보다 작음
    g_too_large = GraspCandidate(0, 0, 0.05, 0.999, 0.0, 0.9, {})  # max_width=0.10보다 큼
    assert evaluator._width_fitness(g_too_small) == 0.0
    assert evaluator._width_fitness(g_too_large) == 0.0
    print("[PASS] test_evaluator_width_out_of_range")


def test_evaluator_single_candidate():
    """후보가 1개뿐일 때도 정상 동작하는지 검증"""
    evaluator = ThreeJawEvaluator()
    candidates = [GraspCandidate(100.0, 100.0, 0.05, 0.04, 0.0, 0.9, {})]
    best = evaluator.select_best(candidates)
    assert best is candidates[0]
    print("[PASS] test_evaluator_single_candidate")


# =====================================================================
# 7. GraspPipeline 통합 테스트 (Mock 모델)
# =====================================================================

def test_pipeline_end_to_end_graspgroup():
    """GraspGroup 포맷 Mock 모델 → 파이프라인 정상 동작 검증"""
    class MockModel:
        def predict(self, rgb, depth):
            class G:
                pass
            results = []
            for x, y, z, w, t, s in [
                (0.10, 0.20, 0.05, 0.04, 0.0, 0.9),
                (0.30, 0.40, 0.03, 0.08, 0.5, 0.6),
                (0.50, 0.60, 0.00, 0.05, 0.0, 0.8),  # z=0 → 위험
            ]:
                g = G()
                g.x, g.y, g.z, g.width, g.theta, g.score = x, y, z, w, t, s
                results.append(g)
            return results

    from three_jaw_grasp.adapters import GraspGroupAdapter
    rgb   = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = make_dummy_depth()

    pipeline = GraspPipeline(
        model=MockModel(),
        adapter=GraspGroupAdapter(),
        evaluator=ThreeJawEvaluator()
    )
    best = pipeline.predict_best(rgb, depth)

    assert isinstance(best, GraspCandidate)
    assert best.original_score == 0.9, "신뢰도가 가장 높고 안전한 후보가 선택되어야 함"
    print(f"[PASS] test_pipeline_end_to_end_graspgroup  best=({best.center_x}, {best.center_y}, z={best.center_z})")


def test_pipeline_end_to_end_rect():
    """GR-ConvNet 포맷 Mock 모델 → 파이프라인 정상 동작 검증"""
    class MockGRConvNet:
        def predict(self, rgb, depth):
            rect_pred = np.array([80.0, 90.0, 160.0, 170.0])
            cls_score = np.zeros(20); cls_score[5] = 2.0
            return {'rect_pred': rect_pred, 'cls_score': cls_score, 'depth': depth}

    from three_jaw_grasp.adapters import RectGraspAdapter
    rgb   = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = make_dummy_depth(value=0.30)

    pipeline = GraspPipeline(
        model=MockGRConvNet(),
        adapter=RectGraspAdapter(),
        evaluator=ThreeJawEvaluator()
    )
    best = pipeline.predict_best(rgb, depth)

    assert isinstance(best, GraspCandidate)
    assert abs(best.center_z - 0.30) < 1e-4
    print(f"[PASS] test_pipeline_end_to_end_rect  best=(cx={best.center_x:.1f}, cy={best.center_y:.1f}, z={best.center_z:.2f}, angle={best.angle:.3f})")


# =====================================================================
# 실행
# =====================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Three-Jaw Grasp Pipeline 유닛 테스트")
    print("=" * 60)

    # GraspCandidate
    test_grasp_candidate_creation()

    # Adapters
    test_graspgroup_adapter()
    test_rect_grasp_adapter_basic()
    test_rect_grasp_adapter_with_depth()
    test_rect_grasp_adapter_score_range()
    test_pose_array_adapter()

    # FeatureExtractor
    test_feature_extractor_dim()
    test_feature_extractor_batch()

    # Evaluator
    test_evaluator_selects_best()
    test_evaluator_score_all_length()
    test_evaluator_stability_not_constant()
    test_evaluator_width_out_of_range()
    test_evaluator_single_candidate()

    # Pipeline 통합
    test_pipeline_end_to_end_graspgroup()
    test_pipeline_end_to_end_rect()

    print("=" * 60)
    print("모든 테스트 통과!")
    print("=" * 60)
