"""
GR-ConvNet + Three-Jaw Grasp Pipeline 시각화 데모

실행 방법:
    cd three-jaw-grasp
    python examples/visualize_grasp.py

동작:
    1. dataset/Images_Test/Cropped320_rgd/ 에서 이미지를 순서대로 불러옴
    2. GR-ConvNet 모델로 파지 위치 추론
    3. 이미지 위에 3발 집게 위치를 시각화 (중심점 + 3발 팔)
    4. 각 기준별 점수도 함께 출력
    5. 창을 닫으면 다음 이미지로 이동
"""

import sys
import os
import glob
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ── 경로 설정 ──────────────────────────────────────────────────────────
THIS_DIR       = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT  = os.path.dirname(THIS_DIR)
GRCONVNET_ROOT = os.path.join(os.path.dirname(os.path.dirname(PIPELINE_ROOT)),
                               'robotic-grasping-cornell')
DATASET_DIR    = os.path.join(PIPELINE_ROOT, 'dataset', 'Images_Test', 'Cropped320_rgd')

sys.path.insert(0, PIPELINE_ROOT)
sys.path.insert(0, GRCONVNET_ROOT)

from three_jaw_grasp import GraspPipeline, RectGraspAdapter, ThreeJawEvaluator
from three_jaw_grasp.candidate import GraspCandidate


# =====================================================================
# GR-ConvNet 래퍼 (GR-ConvNet 입력: 224×224 정규화 이미지)
# =====================================================================

class GRConvNetWrapper:
    """
    robotic-grasping-cornell GraspNet을 파이프라인 인터페이스로 래핑.
    이미지를 GR-ConvNet 입력 포맷으로 전처리 후 추론.
    """
    IMG_SIZE = 224   # GR-ConvNet 입력 해상도

    # ImageNet 정규화 파라미터 (grasp_dataset.py와 동일)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        try:
            from network import GraspNet
        except ImportError:
            raise ImportError(f"network.py를 찾을 수 없습니다: {GRCONVNET_ROOT}")

        self.device = torch.device(device)
        self.net    = GraspNet()
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.net.load_state_dict(ckpt['model'])
        self.net.to(self.device)
        self.net.eval()
        print(f"[GRConvNet] 모델 로드 완료: {checkpoint_path}")

    def predict(self, rgb: np.ndarray, depth: np.ndarray):
        """
        rgb   : [H, W, 3] uint8 (원본 크기 무관 — 내부에서 224×224로 리사이즈)
        depth : [H, W] float32 m (depth=None 허용)
        """
        # 1. 리사이즈 + 정규화
        img_pil  = Image.fromarray(rgb).resize((self.IMG_SIZE, self.IMG_SIZE))
        img_arr  = np.array(img_pil, dtype=np.float32) / 255.0     # [0, 1]
        img_norm = (img_arr - self.MEAN) / self.STD                 # 정규화
        img_t    = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)  # [1,3,H,W]

        # 2. 추론
        with torch.no_grad():
            rect_pred, cls_score = self.net(img_t)

        rect = rect_pred.squeeze().cpu().numpy()    # [4]
        cls  = cls_score.squeeze().cpu().numpy()    # [20]

        # 3. rect 좌표를 원본 이미지 크기로 역변환
        oh, ow = rgb.shape[:2]
        scale_x = ow / self.IMG_SIZE
        scale_y = oh / self.IMG_SIZE
        rect_scaled = np.array([
            rect[0] * scale_x,
            rect[1] * scale_y,
            rect[2] * scale_x,
            rect[3] * scale_y,
        ], dtype=np.float32)

        # 4. depth 리사이즈
        depth_resized = None
        if depth is not None:
            from PIL import Image as PILImage
            depth_pil = PILImage.fromarray(depth).resize((ow, oh), PILImage.NEAREST)
            depth_resized = np.array(depth_pil, dtype=np.float32)

        return {
            'rect_pred': rect_scaled,
            'cls_score': cls,
            'depth'    : depth_resized
        }


# =====================================================================
# 3발 집게 시각화 함수
# =====================================================================

def draw_three_jaw_grasp(ax, grasp: GraspCandidate, detail: dict, img_shape):
    """
    이미지 위에 3발 집게 파지 위치를 시각화.

    표시 요소:
        - 노란 점: 집게 중심점
        - 빨강/초록/파랑 선: 3발 팔 (120° 간격)
        - 빨강 원: 집게가 벌어지는 반경
        - 흰 텍스트: 각도 및 점수 정보
    """
    cx, cy   = grasp.center_x, grasp.center_y
    angle    = grasp.angle          # 라디안
    width_px = grasp.width          # 픽셀 단위 반경

    # 반경이 너무 작으면 최소 반경 보장
    radius = max(width_px, 15.0)

    # 3발 각도: base_angle 기준 0°, 120°, 240°
    base_angle = angle - math.pi / 2   # 집게 기준축 (위쪽 방향)
    arm_angles = [base_angle + i * (2 * math.pi / 3) for i in range(3)]
    colors     = ['#FF4444', '#44FF44', '#4488FF']
    labels     = ['Jaw 1', 'Jaw 2', 'Jaw 3']

    # 3발 팔 그리기
    for arm_angle, color, label in zip(arm_angles, colors, labels):
        fx = cx + radius * math.cos(arm_angle)
        fy = cy + radius * math.sin(arm_angle)
        ax.plot([cx, fx], [cy, fy], color=color, linewidth=2.5,
                solid_capstyle='round', zorder=4, label=label)
        ax.scatter(fx, fy, color=color, s=50, zorder=5)

    # 파지 반경 원
    circle = plt.Circle((cx, cy), radius, color='yellow', fill=False,
                         linewidth=1.5, linestyle='--', alpha=0.6, zorder=3)
    ax.add_patch(circle)

    # 중심점
    ax.scatter(cx, cy, color='yellow', s=80, zorder=6, marker='*')

    # 점수 텍스트
    score_text = (
        f"Total: {detail['total']:.3f}\n"
        f"width: {detail['width']['weighted']:.2f}  "
        f"score: {detail['score']['weighted']:.2f}\n"
        f"height: {detail['height']['weighted']:.2f}  "
        f"stab: {detail['stability']['weighted']:.2f}\n"
        f"angle: {math.degrees(angle):.1f}°  z: {grasp.center_z:.3f}m"
    )
    ax.text(5, 15, score_text,
            color='white', fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))


# =====================================================================
# 메인 실행
# =====================================================================

def run_visualization(max_images: int = 10, save_dir: str = None):
    """
    dataset 이미지를 순서대로 불러와 3발 집게 파지 위치를 시각화.

    Parameters
    ----------
    max_images : 최대 표시 이미지 수
    save_dir   : 결과 이미지 저장 경로 (None이면 화면만 표시)
    """
    # ── 이미지 파일 목록 ──
    image_paths = sorted(glob.glob(os.path.join(DATASET_DIR, '*.png')))
    if not image_paths:
        print(f"[오류] 이미지가 없습니다: {DATASET_DIR}")
        return

    print(f"[INFO] 이미지 {len(image_paths)}장 발견. 최대 {max_images}장 처리.")

    # ── 파이프라인 구성 ──
    checkpoint = os.path.join(GRCONVNET_ROOT, 'models', 'model_49.ckpt')
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'

    if os.path.exists(checkpoint):
        model = GRConvNetWrapper(checkpoint, device=device)
    else:
        print(f"[경고] 체크포인트 없음 ({checkpoint}). Mock 모델로 실행.")
        model = _MockModel()

    config_path = os.path.join(PIPELINE_ROOT, 'config', 'gripper_spec.yaml')
    pipeline  = GraspPipeline(
        model     = model,
        adapter   = RectGraspAdapter(),
        evaluator = ThreeJawEvaluator(config_path=config_path)
    )
    evaluator = pipeline.evaluator

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # ── 이미지 순회 ──
    for i, img_path in enumerate(image_paths[:max_images]):
        fname = os.path.basename(img_path)
        print(f"\n[{i+1}/{min(max_images, len(image_paths))}] {fname}")

        # 이미지 로드
        rgb = np.array(Image.open(img_path).convert('RGB'))
        h, w = rgb.shape[:2]

        # depth는 없으므로 0.3m 균일 더미로 대체 (실제 카메라 연동 시 교체)
        depth = np.full((h, w), 0.30, dtype=np.float32)

        # 파이프라인 추론
        try:
            best  = pipeline.predict_best(rgb, depth)
            detail = evaluator.score_detail(best)
        except Exception as e:
            print(f"  [오류] {e}")
            continue

        # 결과 출력
        print(f"  중심: ({best.center_x:.1f}, {best.center_y:.1f})  "
              f"z={best.center_z:.3f}m  angle={math.degrees(best.angle):.1f}°")
        print(f"  점수: total={detail['total']:.3f}  "
              f"width={detail['width']['weighted']:.2f}  "
              f"score={detail['score']['weighted']:.2f}  "
              f"height={detail['height']['weighted']:.2f}")

        # ── 시각화 ──
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.imshow(rgb)
        ax.set_title(f"{fname}\nTotal Score: {detail['total']:.3f}", fontsize=10)
        ax.axis('off')

        draw_three_jaw_grasp(ax, best, detail, rgb.shape)

        # 범례
        patches = [
            mpatches.Patch(color='#FF4444', label='Jaw 1'),
            mpatches.Patch(color='#44FF44', label='Jaw 2'),
            mpatches.Patch(color='#4488FF', label='Jaw 3'),
        ]
        ax.legend(handles=patches, loc='lower right', fontsize=8,
                  facecolor='black', labelcolor='white', framealpha=0.6)

        plt.tight_layout()

        if save_dir:
            save_path = os.path.join(save_dir, f'grasp_{fname}')
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"  저장: {save_path}")

        plt.show()
        plt.close()

    print("\n[완료]")


class _MockModel:
    """체크포인트 없을 때 사용하는 Mock 모델."""
    def predict(self, rgb, depth):
        h, w = rgb.shape[:2]
        rect = np.array([w*0.3, h*0.3, w*0.7, h*0.7], dtype=np.float32)
        cls  = np.zeros(20, dtype=np.float32); cls[5] = 2.0
        return {'rect_pred': rect, 'cls_score': cls, 'depth': depth}


# =====================================================================
# 진입점
# =====================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='3발 집게 파지 위치 시각화')
    parser.add_argument('--max',  type=int, default=10,   help='최대 이미지 수 (기본: 10)')
    parser.add_argument('--save', type=str, default=None, help='결과 저장 폴더 (생략 시 화면 표시만)')
    args = parser.parse_args()

    run_visualization(max_images=args.max, save_dir=args.save)
