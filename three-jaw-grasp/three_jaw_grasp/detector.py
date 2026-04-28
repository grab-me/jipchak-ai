import os
from typing import Tuple, List

class ObjectDetector:
    """
    물체를 탐지하여 Bounding Box를 반환하는 객체 인식기 인터페이스
    """
    def detect(self, rgb, **kwargs) -> Tuple[int, int, int, int]:
        """
        :param rgb: 원본 이미지 (H, W, 3)
        :return: (xmin, ymin, xmax, ymax)
        """
        raise NotImplementedError

class YoloMockDetector(ObjectDetector):
    """
    YOLO 26 기반 객체 탐지기를 모사(Mock)하는 클래스.
    실제 가중치가 없으므로 Cornell Dataset의 파지 정답 파일(cpos.txt)을
    읽어 점들을 모두 포함하는 Bounding Box를 임시로 반환합니다.
    """
    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        print("[YoloMockDetector] YOLO 26 객체 탐지기 초기화 완료 (전처리 Crop용)")

    def detect(self, rgb, filename_prefix: str = None) -> Tuple[int, int, int, int]:
        h, w = rgb.shape[:2]
        if not self.data_dir or not filename_prefix:
            # 중앙 50% Crop 임시 반환
            return (w//4, h//4, 3*w//4, 3*h//4)

        cpos_file = os.path.join(self.data_dir, f"{filename_prefix}cpos.txt")
        if not os.path.exists(cpos_file):
            return (0, 0, w, h)

        x_coords = []
        y_coords = []
        with open(cpos_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    x_coords.append(float(parts[0]))
                    y_coords.append(float(parts[1]))

        if not x_coords:
            return (0, 0, w, h)

        # 박스 여유(Padding)를 20픽셀 정도 줍니다.
        xmin = max(0, int(min(x_coords)) - 20)
        ymin = max(0, int(min(y_coords)) - 20)
        xmax = min(w, int(max(x_coords)) + 20)
        ymax = min(h, int(max(y_coords)) + 20)

        # Box 크기가 너무 작으면 (10x10 미만) 안전장치
        if (xmax - xmin) < 10 or (ymax - ymin) < 10:
            return (0, 0, w, h)

        return (xmin, ymin, xmax, ymax)
