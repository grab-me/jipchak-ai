"""표준 입출력 타입.

어댑터 외부에서 보이는 데이터 형태는 여기 정의한 타입만 사용한다.
모델별 raw 출력은 GraspCandidate.metadata 에 보존.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RGBDInput:
    """단일 RGB-D 프레임."""

    rgb: np.ndarray
    """(H, W, 3) uint8, sRGB."""

    depth: np.ndarray
    """(H, W) float32, 단위 meter. raw 값이면 어댑터에서 환산."""

    intrinsics: Optional[dict] = None
    """{fx, fy, cx, cy}. 일부 모델만 사용."""

    workspace_mask: Optional[np.ndarray] = None
    """(H, W) bool. 작업 영역 한정 (선택)."""

    def __post_init__(self) -> None:
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError(f"rgb must be (H, W, 3), got {self.rgb.shape}")
        if self.depth.ndim != 2:
            raise ValueError(f"depth must be (H, W), got {self.depth.shape}")
        if self.depth.shape != self.rgb.shape[:2]:
            raise ValueError(
                f"rgb {self.rgb.shape[:2]} and depth {self.depth.shape} mismatch"
            )


@dataclass
class GraspCandidate:
    """단일 grasp 후보.

    pixel 좌표 기반. 월드 좌표 변환은 후처리(intrinsics + depth) 단계.
    """

    x: float
    """픽셀 x (열)."""

    y: float
    """픽셀 y (행)."""

    confidence: float
    """0~1, 어댑터 책임 정규화."""

    z: Optional[float] = None
    """depth (meter). 모델 미추정 시 후처리에서 채움."""

    angle: Optional[float] = None
    """antipodal grasp 각도 (rad). 2-finger 모델만."""

    width: Optional[float] = None
    """gripper 폭 (meter). 2-finger 모델만."""

    metadata: dict = field(default_factory=dict)
    """모델별 raw output (디버깅/시각화용)."""
