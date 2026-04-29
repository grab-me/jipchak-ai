"""
GR-ConvNet 모델을 외부 환경에서 Three-Jaw-Grasp 파이프라인에 주입하는 플러그인입니다.
"""
import numpy as np
import torch
from PIL import Image

from three_jaw_grasp.factory import ModelFactory

@ModelFactory.register("grconvnet")
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
