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


from three_jaw_grasp.factory import ModelFactory, AdapterFactory
from three_jaw_grasp.detector import YoloMockDetector

# 외부 플러그인 로드 (실제 프로덕션 환경에서는 사용자가 작성한 스크립트를 import 하는 방식과 동일)
import external_models.grconvnet.plugin
import external_models.graspnet.plugin

def main():
    parser = argparse.ArgumentParser(description="Three-Jaw Grasp Pipeline (다중 모델 통합 모드)")
    # 등록된 모델 키값을 choices로 동적 할당할 수도 있으나 직관성을 위해 유지
    parser.add_argument('--model', type=str, default='grconvnet',
                        help="평가할 모델 종류 선택 (grconvnet, graspnet 등)")
    parser.add_argument('--dataset', type=str, default='01', 
                        help="테스트할 데이터셋 폴더명 (예: 01, 02 등)")
    parser.add_argument('--max', type=int, default=5, help='처리할 이미지 수')
    parser.add_argument('--save', type=str, default=None, help='저장 폴더 (예: examples/output)')
    args = parser.parse_args()

    dataset_dir = os.path.join(PIPELINE_ROOT, 'dataset', args.dataset)
    if not os.path.exists(dataset_dir):
        print(f"[오류] 데이터셋 폴더 부재: {dataset_dir}")
        return

    # 1. 모델과 어댑터 동적 생성 (Factory Pattern)
    try:
        # 모델별 추가 인자(kwargs) 처리는 팩토리를 통해 유연하게 전달 가능
        if args.model == 'grconvnet':
            ckpt_path = os.path.join(GRCONVNET_ROOT, 'models', 'model_49.ckpt')
            model = ModelFactory.create(args.model, checkpoint_path=ckpt_path if os.path.exists(ckpt_path) else None)
        else:
            model = ModelFactory.create(args.model)
            
        adapter = AdapterFactory.create(args.model)
    except KeyError as e:
        print(f"[오류] {e}")
        return

    # 렌더링 설정
    is_3d = getattr(model, 'is_3d', args.model == 'graspnet')
    intrinsics = getattr(model, 'intrinsics', None)

    # 2. 공통 파이프라인 조립 (YOLO 디텍터 주입)
    evaluator = ThreeJawEvaluator(config_path=os.path.join(PIPELINE_ROOT, 'config', 'gripper_spec.yaml'))
    detector = YoloMockDetector(data_dir=dataset_dir)
    pipeline  = GraspPipeline(model=model, adapter=adapter, evaluator=evaluator, detector=detector, is_3d=is_3d)

    if args.save:
        date_str = datetime.datetime.now().strftime('%Y%m%d')
        base_dir = args.save
        
        # 모델명_날짜_넘버 패턴 찾기
        existing_dirs = glob.glob(os.path.join(base_dir, f"{args.model}_{date_str}_*"))
        max_num = 0
        for d in existing_dirs:
            try:
                num_str = os.path.basename(d).split('_')[-1]
                max_num = max(max_num, int(num_str))
            except ValueError:
                continue
                
        new_num = max_num + 1
        save_dir = os.path.join(base_dir, f"{args.model}_{date_str}_{new_num}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"[저장 안내] {save_dir} 에 저장됩니다.")
        args.save = save_dir

    # 3. 데이터셋 순환 실행
    image_paths = sorted(glob.glob(os.path.join(dataset_dir, '*r.png')))
    print(f"\n[{args.model.upper()} 모드] {min(args.max, len(image_paths))}장 테스트 시작\n{'='*50}")

    for i, img_path in enumerate(image_paths[:args.max]):
        fname = os.path.basename(img_path)
        prefix = fname.split('r.png')[0]
        
        rgb = np.array(Image.open(img_path).convert('RGB'))
        h, w = rgb.shape[:2]
        depth = np.full((h, w), 0.30, dtype=np.float32)

        # 캡슐화된 파이프라인 메서드 실행 (YOLO Crop -> 추론 -> Uncrop -> 평가)
        try:
            best = pipeline.predict_best(rgb, depth, filename_prefix=prefix)
        except ValueError as e:
            print(f"[{i+1}] {fname} -> [건너뜀] {e}")
            continue
            
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
        
        plt.show()
        plt.close() 

if __name__ == '__main__':
    main()
