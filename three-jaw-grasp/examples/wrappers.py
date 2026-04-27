"""
다양한 딥러닝 모델들을 파이프라인에 연결하기 위한 래퍼(Wrapper) 클래스 모음입니다.
실제 사용자는 이 래퍼 내부의 껍데기를 실제 PyTorch 모델 추론 로직으로 교체하여 사용하면 됩니다.
"""
import math
import numpy as np
import torch
from PIL import Image

class GRConvNetWrapper:
    IMG_SIZE = 224
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, checkpoint_path=None, device='cpu'):
        print("[GRConvNet] 모델 초기화 중... (Mock이 아닙니다)")
        try:
            from network import GraspNet
            self.device = torch.device(device)
            self.net = GraspNet()
            if checkpoint_path:
                ckpt = torch.load(checkpoint_path, map_location=self.device)
                self.net.load_state_dict(ckpt['model'])
            self.net.to(self.device)
            self.net.eval()
            self.real_model = True
        except ImportError:
            print("[GRConvNet] 로드 실패, dummy dict 반환 모드로 동작합니다.")
            self.real_model = False

    def predict(self, rgb, depth, prefix=None):
        if not self.real_model:
            return {'rect_pred': np.array([0,0,0,0]), 'cls_score': np.zeros((1,)), 'depth': depth}

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

    def predict(self, rgb, depth, prefix=None):
        fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
        cx, cy = self.intrinsics[0, 2], self.intrinsics[1, 2]
        
        target_z = 0.30
        mock_grasps = []
        for i in range(5):
            rand_u = np.random.uniform(120, 200)
            rand_v = np.random.uniform(120, 200)
            
            X = (rand_u - cx) * target_z / fx
            Y = (rand_v - cy) * target_z / fy
            
            mock_grasps.append(self.MockGrasp(
                score=np.random.uniform(0.6, 0.99),
                width=0.05, x=X, y=Y, z=target_z,
                theta=np.random.uniform(-math.pi/2, math.pi/2)
            ))
        return mock_grasps


class YoloMockModel:
    def __init__(self, data_dir):
        print(f"[YoloMock] YOLO 정답 모사 래퍼 초기화 완료 ({data_dir})")
        self.data_dir = data_dir

    def predict(self, rgb, depth, filename_prefix: str):
        import os
        cpos_file = os.path.join(self.data_dir, f"{filename_prefix}cpos.txt")
        if not os.path.exists(cpos_file):
            return []

        grasps_8pts = []
        with open(cpos_file, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 4):
                if i + 3 >= len(lines): break
                pts = []
                for j in range(4):
                    parts = lines[i+j].strip().split()
                    if len(parts) >= 2:
                        pts.extend([float(parts[0]), float(parts[1])])
                if len(pts) == 8:
                    grasps_8pts.append(pts)

        return grasps_8pts
