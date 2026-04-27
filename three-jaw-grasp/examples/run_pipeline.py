"""
통합 시뮬레이션 환경 (Run Pipeline)
모든 모델(GR-ConvNet, GraspNet, YOLO)을 옵션으로 선택하여
동일한 평가 파이프라인과 시각화 함수(draw_three_jaw_grasp)를 통과시킵니다.

실행 예:
    python examples/run_pipeline.py --model yolo --max 3 --save examples/output
    python examples/run_pipeline.py --model grconvnet --max 3
    python examples/run_pipeline.py --model graspnet --max 3
"""

import sys
import os
import glob
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

THIS_DIR      = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PIPELINE_ROOT)

from three_jaw_grasp import GraspPipeline, ThreeJawEvaluator
from three_jaw_grasp.adapters import RectGraspAdapter, GraspGroupAdapter, YoloGraspAdapter
from three_jaw_grasp.visualizer import draw_three_jaw_grasp

# GRConvNet 프로젝트 루트를 경로에 추가하여 network 모듈 수입 허용
GRCONVNET_ROOT = os.path.join(os.path.dirname(os.path.dirname(PIPELINE_ROOT)), 'robotic-grasping-cornell')
if os.path.exists(GRCONVNET_ROOT):
    sys.path.insert(0, GRCONVNET_ROOT)

from examples.wrappers import GRConvNetWrapper, GraspNetMockWrapper, YoloMockModel


def main():
    parser = argparse.ArgumentParser(description="Three-Jaw Grasp Pipeline (다중 모델 통합 모드)")
    parser.add_argument('--model', type=str, choices=['grconvnet', 'graspnet', 'yolo'], default='yolo',
                        help="평가할 모델 종류 선택")
    parser.add_argument('--max', type=int, default=5, help='처리할 이미지 수')
    parser.add_argument('--save', type=str, default=None, help='저장 폴더 (예: examples/output)')
    args = parser.parse_args()

    dataset_dir = os.path.join(PIPELINE_ROOT, 'dataset', '01')
    if not os.path.exists(dataset_dir):
        print(f"[오류] 데이터셋 폴더 부재: {dataset_dir}")
        return

    # 1. 모델과 어댑터, 렌더링 설정
    is_3d = False
    intrinsics = None

    if args.model == 'grconvnet':
        # GR-ConvNet 가중치가 있다면 로드, 없으면 dummy
        ckpt_path = os.path.join(GRCONVNET_ROOT, 'models', 'model_49.ckpt')
        model = GRConvNetWrapper(checkpoint_path=ckpt_path if os.path.exists(ckpt_path) else None)
        adapter = RectGraspAdapter()

    elif args.model == 'graspnet':
        model = GraspNetMockWrapper()
        adapter = GraspGroupAdapter()
        is_3d = True
        intrinsics = model.intrinsics

    elif args.model == 'yolo':
        model = YoloMockModel(data_dir=dataset_dir)
        adapter = YoloGraspAdapter()

    # 2. 공통 파이프라인(Evaluator) 조립
    evaluator = ThreeJawEvaluator(config_path=os.path.join(PIPELINE_ROOT, 'config', 'gripper_spec.yaml'))
    pipeline  = GraspPipeline(model=model, adapter=adapter, evaluator=evaluator)

    if args.save:
        timestamp_dir = os.path.join(args.save, f"{args.model}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(timestamp_dir, exist_ok=True)
        print(f"[저장 안내] {timestamp_dir} 에 저장됩니다.")
        args.save = timestamp_dir

    # 3. 데이터셋 순환 실행
    image_paths = sorted(glob.glob(os.path.join(dataset_dir, '*r.png')))
    print(f"\n[{args.model.upper()} 모드] {min(args.max, len(image_paths))}장 테스트 시작\n{'='*50}")

    for i, img_path in enumerate(image_paths[:args.max]):
        fname = os.path.basename(img_path)
        prefix = fname.split('r.png')[0]
        
        rgb = np.array(Image.open(img_path).convert('RGB'))
        h, w = rgb.shape[:2]
        depth = np.full((h, w), 0.30, dtype=np.float32)

        # 모델별 필요 인자에 맞춰 추론. (YOLO는 정답 모사를 위해 파일 prefix가 필요함)
        if args.model == 'yolo':
            raw_output = model.predict(rgb, depth, filename_prefix=prefix)
        else:
            raw_output = model.predict(rgb, depth)

        # 공통 어댑터 및 평가
        candidates = adapter.adapt(raw_output)
        if not candidates:
            print(f"[{i+1}] {fname} -> [건너뜀] 추출된 파지 후보가 없음")
            continue
            
        best = evaluator.select_best(candidates, depth)
        detail = evaluator.score_detail(best)
        
        print(f"[{i+1}] {fname} -> Total Score: {detail['total']:.3f} | Best Center: (X={best.center_x:.2f}, Y={best.center_y:.2f})")

        # 공통 시각화 모듈 (visualizer.py) 호출
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        draw_three_jaw_grasp(ax, rgb, best, detail, 
                             model_name=args.model, is_3d=is_3d, intrinsics=intrinsics)
        plt.tight_layout()

        if args.save:
            out_path = os.path.join(args.save, f"{args.model}_{fname}")
            plt.savefig(out_path, dpi=100, bbox_inches='tight')
        
        # plt.show() # 서버 환경이나 배치 테스트 시에는 끄는 것을 권장합니다.
        plt.close()

if __name__ == '__main__':
    main()
