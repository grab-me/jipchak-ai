import os
import sys
import glob
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import datetime
import random

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PIPELINE_ROOT)

from ultralytics import YOLO
from three_jaw_grasp.pipeline import GraspPipeline
from three_jaw_grasp.factory import EvaluatorFactory
from three_jaw_grasp.adapters import YoloSegGraspAdapter
from three_jaw_grasp.visualizer import draw_three_jaw_grasp

class UltralyticsSegWrapper:
    """
    Ultralytics YOLO Segmentation 모델을 파이프라인에 연결하는 래퍼 클래스
    """
    def __init__(self, pt_path):
        self.model = YOLO(pt_path)
        
    def predict(self, rgb, depth=None, crop_box=None):
        # 추론 (Numpy Array 전달)
        # Ultralytics는 일반적으로 BGR 이미지를 기대하므로 RGB로 변환
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        results = self.model.predict(source=bgr, conf=0.25, verbose=False)
        
        raw_outputs = []
        for r in results:
            print(f"  [Debug] 예측된 박스 개수: {len(r.boxes)}, 마스크 존재 여부: {r.masks is not None}")
            if r.masks is not None:
                # masks.data 는 보통 원래 해상도와 다를 수 있으므로 resize 필요
                masks = r.masks.data.cpu().numpy() # (N, H, W)
                boxes = r.boxes.xyxy.cpu().numpy() # (N, 4)
                scores = r.boxes.conf.cpu().numpy() # (N,)
                
                orig_shape = rgb.shape[:2]
                
                for mask, box, score in zip(masks, boxes, scores):
                    if mask.shape != orig_shape:
                        # cv2.resize는 (width, height) 순서임
                        mask = cv2.resize(mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
                        
                    raw_outputs.append({
                        'mask': mask,
                        'box': box.tolist(),
                        'score': float(score)
                    })
        return raw_outputs

def main():
    model_dir = os.path.join(PIPELINE_ROOT, 'dataset', 'model_chick')
    pt_path = os.path.join(model_dir, 'heart_chick_seg_best.pt')
    img_dir = os.path.join(model_dir, 'test_images_sample')
    
    if not os.path.exists(pt_path):
        print(f"모델 파일이 없습니다: {pt_path}")
        return
        
    print(f"🐥 [병아리 모델] 파이프라인 초기화 중... ({pt_path})")
    model = UltralyticsSegWrapper(pt_path)
    adapter = YoloSegGraspAdapter()
    
    # 전략 패턴(Strategy)을 통해 병아리 전용 평가기 로드
    evaluator = EvaluatorFactory.create("chick", config_path=os.path.join(PIPELINE_ROOT, 'config', 'gripper_spec.yaml'))
    
    # Detector 없이 YOLO Seg 모델 자체를 메인 모델로 사용
    pipeline = GraspPipeline(model=model, adapter=adapter, evaluator=evaluator, detector=None)
    
    # jpg, png, jpeg 등 모든 이미지 확장자 검색
    images = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        images.extend(glob.glob(os.path.join(img_dir, ext)))
    
    # 랜덤하게 5장 샘플링
    test_count = min(5, len(images))
    images = random.sample(images, test_count)
    print(f"테스트 이미지 총 {len(images)}장 중 랜덤하게 {test_count}장을 추출하여 테스트합니다.\n" + "="*50)
    
    base_dir = os.path.join(PIPELINE_ROOT, 'examples', 'output')
    
    # 모델명_날짜_넘버 패턴으로 폴더 생성
    date_str = datetime.datetime.now().strftime('%Y%m%d')
    existing_dirs = glob.glob(os.path.join(base_dir, f"model_chick_{date_str}_*"))
    max_num = 0
    for d in existing_dirs:
        try:
            num_str = os.path.basename(d).split('_')[-1]
            max_num = max(max_num, int(num_str))
        except ValueError:
            continue
            
    new_num = max_num + 1
    save_dir = os.path.join(base_dir, f"model_chick_{date_str}_{new_num}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[저장 안내] {save_dir} 에 결과가 저장됩니다.\n")
            
    for i, img_path in enumerate(images[:5]):
        fname = os.path.basename(img_path)
        rgb = np.array(Image.open(img_path).convert('RGB'))
        h, w = rgb.shape[:2]
        depth = np.full((h, w), 0.30, dtype=np.float32) # Depth 카메라가 없으므로 고정값 사용
        
        print(f"[{i+1}/5] {fname} 처리 중...")
        try:
            best_grasp = pipeline.predict_best(rgb, depth)
            detail = evaluator.score_detail(best_grasp)
            print(f"  -> 최고 파지점: Center({best_grasp.center_x:.1f}, {best_grasp.center_y:.1f}) | Score: {detail['total']:.3f} (Mask 핏: {detail['mask']['weighted']:.2f})")
            
            # 시각화 및 저장 (v2와 동일한 10x10 사이즈로 통일)
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            draw_three_jaw_grasp(ax, rgb, best_grasp, detail, model_name="YOLO_Seg", is_3d=False)
            plt.title(f"Score: {detail['total']:.3f} (Mask Multiplier: {detail['mask']['weighted']:.2f})")
            
            out_path = os.path.join(save_dir, fname)
            plt.savefig(out_path, dpi=100, bbox_inches='tight')
            print(f"  -> 시각화 결과 저장됨: {out_path}\n")
            plt.close()
            
        except Exception as e:
            import traceback
            print(f"  -> 파지 탐지 실패: {e}")
            traceback.print_exc()
            print("\n")
            
    print(f"모든 처리가 완료되었습니다. 결과는 {save_dir} 폴더를 확인해주세요!")

if __name__ == "__main__":
    main()
