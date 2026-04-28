"""
GraspNet 모델을 외부 환경에서 Three-Jaw-Grasp 파이프라인에 주입하는 플러그인입니다.
"""
import math
import numpy as np

from three_jaw_grasp.factory import ModelFactory

@ModelFactory.register("graspnet")
class GraspNetMockWrapper:
    class MockGrasp:
        def __init__(self, score, width, x, y, z, theta):
            self.score = score
            self.width = width
            self.x = x
            self.y = y
            self.z = z
            self.theta = theta

    def __init__(self, checkpoint_path=None):
        print("[GraspNet] Mock 래퍼 초기화 완료 (C++ 빌드우회)")
        self.intrinsics = np.array([
            [615.0, 0.0, 320.0],
            [0.0, 615.0, 240.0],
            [0.0, 0.0, 1.0]
        ])

    def predict(self, rgb, depth, crop_box=None, prefix=None):
        fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
        cx, cy = self.intrinsics[0, 2], self.intrinsics[1, 2]
        
        target_z = 0.30
        mock_grasps = []
        
        # Crop 정보를 받아서 실제 원본 좌표를 복원하여 3D를 계산합니다.
        xmin, ymin = 0, 0
        if crop_box:
            xmin, ymin = crop_box[0], crop_box[1]
            
        h, w = rgb.shape[:2]
        
        for i in range(5):
            # 잘린 이미지 내에서 물체가 있을 법한 임의의 픽셀 (Crop의 중앙 부근)
            u_crop = np.random.uniform(w * 0.3, w * 0.7)
            v_crop = np.random.uniform(h * 0.3, h * 0.7)
            
            # 원본 좌표로 변환
            u_orig = u_crop + xmin
            v_orig = v_crop + ymin
            
            # 카메라 내부 파라미터를 이용해 3D(m) 좌표로 투영
            X = (u_orig - cx) * target_z / fx
            Y = (v_orig - cy) * target_z / fy
            
            mock_grasps.append(self.MockGrasp(
                score=np.random.uniform(0.6, 0.99),
                width=0.05, x=X, y=Y, z=target_z,
                theta=np.random.uniform(-math.pi/2, math.pi/2)
            ))
        return mock_grasps
