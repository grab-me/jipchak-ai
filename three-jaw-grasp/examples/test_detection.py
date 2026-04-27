"""
GR-ConvNet 실제 객체 인식 검증 테스트

GraspDataset(Cornell Dataset)의 실제 이미지와 어노테이션을 사용해
모델 추론이 올바르게 동작하는지 확인합니다.

실행:
    cd three-jaw-grasp
    python examples/test_detection.py
"""

import sys
import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

THIS_DIR      = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.dirname(THIS_DIR)
GRCONVNET_ROOT = os.path.join(os.path.dirname(os.path.dirname(PIPELINE_ROOT)),
                               'robotic-grasping-cornell')
DATASET_PATH  = os.path.join(PIPELINE_ROOT, 'dataset', 'grasp')

sys.path.insert(0, PIPELINE_ROOT)
sys.path.insert(0, GRCONVNET_ROOT)

from three_jaw_grasp import GraspPipeline, RectGraspAdapter, ThreeJawEvaluator


# ─── GR-ConvNet 래퍼 (visualize_grasp.py와 동일) ─────────────────────
class GRConvNetWrapper:
    IMG_SIZE = 224
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, checkpoint_path, device='cpu'):
        from network import GraspNet
        self.device = torch.device(device)
        self.net = GraspNet()
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.net.load_state_dict(ckpt['model'])
        self.net.to(self.device)
        self.net.eval()

    def predict(self, rgb, depth):
        img = Image.fromarray(rgb).resize((self.IMG_SIZE, self.IMG_SIZE))
        img_arr  = np.array(img, dtype=np.float32) / 255.0
        img_norm = (img_arr - self.MEAN) / self.STD
        img_t = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            rect_pred, cls_score = self.net(img_t)

        rect = rect_pred.squeeze().cpu().numpy()
        cls  = cls_score.squeeze().cpu().numpy()

        oh, ow = rgb.shape[:2]
        sx, sy = ow / self.IMG_SIZE, oh / self.IMG_SIZE
        rect_scaled = np.array([rect[0]*sx, rect[1]*sy, rect[2]*sx, rect[3]*sy], dtype=np.float32)

        return {'rect_pred': rect_scaled, 'cls_score': cls, 'depth': depth}


# ─── 어노테이션(GT) 로드 ────────────────────────────────────────────
def load_annotation(dataset_path, index):
    """
    GT 파지 사각형 로드.
    형식: cls x1 y1 x2 y2 (224×224 기준 픽셀)
    """
    ann_file = os.path.join(dataset_path, 'Annotations', index + '.txt')
    if not os.path.exists(ann_file) or os.stat(ann_file).st_size == 0:
        return None
    rects = []
    with open(ann_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls = int(parts[0])
                x1, y1, x2, y2 = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                rects.append({'cls': cls, 'rect': [x1, y1, x2, y2]})
    return rects if rects else None


def load_image_indices(dataset_path, split='test'):
    """test.txt 또는 train.txt에서 이미지 인덱스 로드."""
    set_file = os.path.join(dataset_path, 'ImageSets', split + '.txt')
    if not os.path.exists(set_file):
        # ImageSets 없으면 Images 폴더에서 직접 수집
        img_dir = os.path.join(dataset_path, 'Images_Test', 'Cropped320_rgd')
        if os.path.exists(img_dir):
            names = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith('.png')]
            return names, img_dir
        return [], None
    with open(set_file) as f:
        indices = [x.strip() for x in f.readlines()]
    img_dir = os.path.join(dataset_path, 'Images')
    return indices, img_dir


# ─── 시각화 ────────────────────────────────────────────────────────
def draw_prediction(ax, rgb, grasp, gt_rects=None):
    ax.imshow(rgb)
    ax.set_axis_off()

    cx, cy   = grasp.center_x, grasp.center_y
    angle    = grasp.angle
    radius   = max(grasp.width, 15.0)

    base_angle = angle - math.pi / 2
    jaw_colors = ['#FF4444', '#44FF44', '#4488FF']
    for i, color in enumerate(jaw_colors):
        a = base_angle + i * (2 * math.pi / 3)
        fx = cx + radius * math.cos(a)
        fy = cy + radius * math.sin(a)
        ax.plot([cx, fx], [cy, fy], color=color, lw=2.5, solid_capstyle='round', zorder=4)
        ax.scatter(fx, fy, color=color, s=40, zorder=5)

    circle = plt.Circle((cx, cy), radius, color='yellow', fill=False,
                         lw=1.5, linestyle='--', alpha=0.7, zorder=3)
    ax.add_patch(circle)
    ax.scatter(cx, cy, color='yellow', s=80, marker='*', zorder=6)

    # GT 파지 사각형 표시
    if gt_rects:
        for gt in gt_rects[:3]:
            x1, y1, x2, y2 = gt['rect']
            # 224→실제 크기 스케일
            h, w = rgb.shape[:2]
            sx, sy = w / 224, h / 224
            rect_patch = mpatches.FancyArrowPatch(
                (x1*sx, y1*sy), (x2*sx, y2*sy),
                arrowstyle='-', color='cyan', lw=1.5, linestyle=':', zorder=2, alpha=0.8
            )
            ax.add_patch(rect_patch)
            # GT 중심
            gcx = (x1 + x2) / 2 * sx
            gcy = (y1 + y2) / 2 * sy
            ax.scatter(gcx, gcy, color='cyan', s=30, marker='+', zorder=6, lw=1.5)


def print_detection_result(idx, fname, grasp, detail, gt_rects):
    print(f"\n[{idx}] {fname}")
    print(f"  예측 중심   : ({grasp.center_x:.1f}, {grasp.center_y:.1f})  "
          f"angle={math.degrees(grasp.angle):.1f}°  z={grasp.center_z:.3f}m")
    print(f"  총점        : {detail['total']:.3f}")
    print(f"    width    = {detail['width']['raw']:.3f}  (weighted {detail['width']['weighted']:.3f})")
    print(f"    score    = {detail['score']['raw']:.3f}  (weighted {detail['score']['weighted']:.3f})")
    print(f"    height   = {detail['height']['raw']:.3f}  (weighted {detail['height']['weighted']:.3f})")
    print(f"    stability= {detail['stability']['raw']:.3f}  (weighted {detail['stability']['weighted']:.3f})")
    if gt_rects:
        gt = gt_rects[0]
        gx = (gt['rect'][0] + gt['rect'][2]) / 2
        gy = (gt['rect'][1] + gt['rect'][3]) / 2
        h, w = 320, 320  # 이미지 기준 (추후 실제 이미지 크기로 대체)
        dist = math.sqrt((grasp.center_x - gx)**2 + (grasp.center_y - gy)**2)
        print(f"  GT 중심     : ({gx:.1f}, {gy:.1f})  |  예측-GT 거리: {dist:.1f}px")
    else:
        print(f"  GT 어노테이션: 없음")


# ─── 메인 ──────────────────────────────────────────────────────────
def run_detection_test(max_images=5, save_dir=None):
    checkpoint = os.path.join(GRCONVNET_ROOT, 'models', 'model_49.ckpt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(checkpoint):
        print(f"[오류] 체크포인트 없음: {checkpoint}")
        return

    print(f"[GRConvNet] 모델 로드 중...")
    model     = GRConvNetWrapper(checkpoint, device)
    adapter   = RectGraspAdapter()
    evaluator = ThreeJawEvaluator(config_path=os.path.join(PIPELINE_ROOT, 'config', 'gripper_spec.yaml'))
    pipeline  = GraspPipeline(model=model, adapter=adapter, evaluator=evaluator)

    # 이미지 목록 로드
    indices, img_dir = load_image_indices(DATASET_PATH, split='test')
    if not indices:
        print("[오류] 이미지 목록을 찾을 수 없습니다.")
        print(f"  확인 경로: {DATASET_PATH}")
        return

    print(f"[INFO] 테스트 이미지 {len(indices)}장, 최대 {max_images}장 처리 ({device})")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    ok_count   = 0
    fail_count = 0

    for i, idx in enumerate(indices[:max_images]):
        img_path = os.path.join(img_dir, idx + '.png')
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, idx + '.jpg')
        if not os.path.exists(img_path):
            print(f"  [건너뜀] 이미지 없음: {img_path}")
            fail_count += 1
            continue

        rgb   = np.array(Image.open(img_path).convert('RGB'))
        h, w  = rgb.shape[:2]
        depth = np.full((h, w), 0.30, dtype=np.float32)

        gt_rects = load_annotation(DATASET_PATH, idx)

        try:
            best   = pipeline.predict_best(rgb, depth)
            detail = evaluator.score_detail(best)
        except Exception as e:
            print(f"  [오류] {idx}: {e}")
            fail_count += 1
            continue

        print_detection_result(i + 1, idx, best, detail, gt_rects)
        ok_count += 1

        # 시각화
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        draw_prediction(ax, rgb, best, gt_rects)
        title = (f"{idx}\n"
                 f"예측: ({best.center_x:.0f}, {best.center_y:.0f})  "
                 f"angle={math.degrees(best.angle):.0f}°  "
                 f"total={detail['total']:.3f}")
        ax.set_title(title, fontsize=9)

        patches = [
            mpatches.Patch(color='#FF4444', label='Jaw 1 (예측)'),
            mpatches.Patch(color='#44FF44', label='Jaw 2 (예측)'),
            mpatches.Patch(color='#4488FF', label='Jaw 3 (예측)'),
            mpatches.Patch(color='cyan',    label='GT 중심 (+)'),
        ]
        ax.legend(handles=patches, loc='lower right', fontsize=8,
                  facecolor='black', labelcolor='white', framealpha=0.7)
        plt.tight_layout()

        if save_dir:
            out_path = os.path.join(save_dir, f'detect_{idx}.png')
            plt.savefig(out_path, dpi=100, bbox_inches='tight')
            print(f"  저장: {out_path}")

        plt.show()
        plt.close()

    print(f"\n{'='*50}")
    print(f"  결과: {ok_count}장 성공 / {fail_count}장 실패 (총 {ok_count+fail_count}장)")
    print(f"{'='*50}")

    if ok_count > 0 and fail_count == 0:
        print("  [OK] 객체 인식 정상 동작 확인")
    elif fail_count > 0:
        print("  [!] 일부 이미지에서 오류 발생 — 위 로그 확인")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='GR-ConvNet 객체 인식 검증')
    parser.add_argument('--max',  type=int, default=5,    help='처리할 이미지 수 (기본: 5)')
    parser.add_argument('--save', type=str, default=None, help='결과 저장 폴더')
    args = parser.parse_args()

    run_detection_test(max_images=args.max, save_dir=args.save)
