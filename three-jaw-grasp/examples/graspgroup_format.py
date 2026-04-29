import numpy as np
import sys
import os

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from three_jaw_grasp import GraspPipeline, GraspGroupAdapter, ThreeJawEvaluator

# 1. 가짜 외부 모델 정의 (인터페이스 예시)
class MockGraspModel:
    def predict(self, rgb, depth):
        # GraspGroupAdapter가 처리할 수 있는 Mock 데이터 반환
        class MockGrasp:
            def __init__(self, x, y, z, width, theta, score):
                self.x, self.y, self.z = x, y, z
                self.width = width
                self.theta = theta
                self.score = score
        
        return [
            MockGrasp(100, 100, 0.05, 0.04, 0.0, 0.9),
            MockGrasp(150, 150, 0.03, 0.08, 0.5, 0.8),
            MockGrasp(200, 200, 0.01, 0.05, 1.0, 0.7),  # 바닥 충돌 위험
        ]

def main():
    # 2. 이미지 및 뎁스 정보 준비
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.ones((480, 640), dtype=np.float32) * 0.1
    
    # 3. 파이프라인 구성
    model = MockGraspModel()
    adapter = GraspGroupAdapter()
    evaluator = ThreeJawEvaluator(
        config_path="config/gripper_spec.yaml"
    )
    
    pipeline = GraspPipeline(model, adapter, evaluator)
    
    # 4. 예측 실행
    best_grasp = pipeline.predict_best(rgb, depth)
    
    print("--- Best Grasp for Three-Jaw Gripper ---")
    print(f"Center: ({best_grasp.center_x}, {best_grasp.center_y}, {best_grasp.center_z})")
    print(f"Width: {best_grasp.width}")
    print(f"Angle: {best_grasp.angle}")
    print(f"Original Score: {best_grasp.original_score}")

if __name__ == "__main__":
    main()
