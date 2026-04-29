"""
GR-ConvNet (robotic-grasping-cornell) 연동 예시

사용 방법:
    1. three-jaw-grasp 폴더에서 실행
    2. robotic-grasping-cornell 모델 경로를 GRCONVNET_ROOT로 설정
    3. python examples/grconv_format.py

GR-ConvNet 모델 출력 포맷:
    model(img) → (rect_pred, cls_score)
        rect_pred : [batch, 4]   — [x1, y1, x2, y2] (픽셀 좌표)
        cls_score : [batch, 20]  — 각도 bin별 점수 (logits)
"""

import sys
import os
import numpy as np
import torch

# ── 경로 설정 ──────────────────────────────────────────────────────────
THIS_DIR       = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT  = os.path.dirname(THIS_DIR)          # three-jaw-grasp/
GRCONVNET_ROOT = os.path.join(os.path.dirname(os.path.dirname(PIPELINE_ROOT)),
                               'robotic-grasping-cornell')  # 실제 경로에 맞게 수정

sys.path.insert(0, PIPELINE_ROOT)
sys.path.insert(0, GRCONVNET_ROOT)

from three_jaw_grasp import GraspPipeline, RectGraspAdapter, ThreeJawEvaluator


# =====================================================================
# GR-ConvNet 래퍼 (predict 인터페이스 구현)
# =====================================================================

class GRConvNetWrapper:
    """
    robotic-grasping-cornell의 GraspNet을 파이프라인 인터페이스에 맞게 래핑.

    predict(rgb, depth) → {'rect_pred': ..., 'cls_score': ..., 'depth': depth}
    """

    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        # GR-ConvNet 프로젝트의 network.py 필요
        try:
            from network import GraspNet
        except ImportError:
            raise ImportError(
                f"network.py를 찾을 수 없습니다. GRCONVNET_ROOT를 확인하세요: {GRCONVNET_ROOT}"
            )

        self.device = torch.device(device)
        self.model  = GraspNet()

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.model.eval()
        print(f"[GRConvNetWrapper] 모델 로드 완료: {checkpoint_path}")

    def predict(self, rgb: np.ndarray, depth: np.ndarray):
        """
        Parameters
        ----------
        rgb   : np.ndarray [H, W, 3] uint8
        depth : np.ndarray [H, W]    float32 (미터)

        Returns
        -------
        dict — RectGraspAdapter가 처리할 수 있는 포맷
        """
        # GR-ConvNet 입력 전처리 (GraspDataset과 동일한 정규화 적용 권장)
        # 여기서는 단순 예시용 — 실제 사용 시 grasp_dataset.py의 전처리 로직 참고
        img_tensor = torch.from_numpy(
            rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            rect_pred, cls_score = self.model(img_tensor)

        return {
            'rect_pred': rect_pred.squeeze().cpu().numpy(),    # [4]
            'cls_score': cls_score.squeeze().cpu().numpy(),    # [20]
            'depth'    : depth
        }


# =====================================================================
# 실행 예시
# =====================================================================

def run_demo():
    CHECKPOINT = os.path.join(GRCONVNET_ROOT, 'models', 'model_49.ckpt')

    if not os.path.exists(CHECKPOINT):
        print(f"[경고] 체크포인트 없음: {CHECKPOINT}")
        print("Mock 데이터로 대신 실행합니다.\n")
        _run_mock_demo()
        return

    # 실제 모델 사용
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model   = GRConvNetWrapper(CHECKPOINT, device=device)
    adapter = RectGraspAdapter()
    evaluator = ThreeJawEvaluator(
        config_path=os.path.join(PIPELINE_ROOT, 'config', 'gripper_spec.yaml')
    )
    pipeline = GraspPipeline(model=model, adapter=adapter, evaluator=evaluator)

    # 더미 이미지로 테스트 (실제 영상으로 교체 예정)
    rgb   = np.zeros((224, 224, 3), dtype=np.uint8)
    depth = np.ones((224, 224), dtype=np.float32) * 0.30   # 30cm

    best = pipeline.predict_best(rgb, depth)
    _print_result(best)


def _run_mock_demo():
    """체크포인트 없이 Mock 데이터로 검증"""
    class MockGRConvNet:
        def predict(self, rgb, depth):
            rect_pred = np.array([80.0, 90.0, 160.0, 170.0])
            cls_score = np.zeros(20); cls_score[3] = 3.0
            return {'rect_pred': rect_pred, 'cls_score': cls_score, 'depth': depth}

    pipeline = GraspPipeline(
        model=MockGRConvNet(),
        adapter=RectGraspAdapter(),
        evaluator=ThreeJawEvaluator()
    )

    rgb   = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.ones((480, 640), dtype=np.float32) * 0.30

    best = pipeline.predict_best(rgb, depth)
    _print_result(best)


def _print_result(best):
    import math
    print("=" * 50)
    print("  3발 집게 최적 파지 위치 (GR-ConvNet)")
    print("=" * 50)
    print(f"  중심 픽셀    : ({best.center_x:.1f}, {best.center_y:.1f})")
    print(f"  깊이 (center_z): {best.center_z:.3f} m")
    print(f"  파지 폭       : {best.width:.1f} px  ※ 미터 변환은 카메라 내부파라미터 필요")
    print(f"  집게 회전 각도: {math.degrees(best.angle):.1f}°")
    print(f"  원본 신뢰도   : {best.original_score:.3f}")
    print(f"  각도 bin      : {best.raw.get('angle_bin', 'N/A')}")
    print("=" * 50)


if __name__ == '__main__':
    run_demo()
