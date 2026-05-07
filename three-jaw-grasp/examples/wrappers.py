"""
Local wrapper helpers used by examples.
"""

import os
from typing import List


class YoloMockModel:
    def __init__(self, data_dir: str):
        print(f"[YoloMock] loading Cornell-style cpos from: {data_dir}")
        self.data_dir = data_dir

    def predict(self, rgb, depth, filename_prefix: str = None, **kwargs) -> List[list]:
        if not filename_prefix:
            return []

        cpos_file = os.path.join(self.data_dir, f"{filename_prefix}cpos.txt")
        if not os.path.exists(cpos_file):
            return []

        grasps_8pts = []
        with open(cpos_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for i in range(0, len(lines), 4):
            if i + 3 >= len(lines):
                break
            pts = []
            for j in range(4):
                parts = lines[i + j].strip().split()
                if len(parts) >= 2:
                    pts.extend([float(parts[0]), float(parts[1])])
            if len(pts) == 8:
                grasps_8pts.append(pts)

        return grasps_8pts
