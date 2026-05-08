"""3-finger center grasp 후처리.

GR-ConvNet 의 Q맵 (224×224 crop) 과 원본 depth (1280×720 m 단위) 에서
3 DOF (x, y, z, score) 후보 점만 추출. antipodal angle/width 는 무시.

위치는 원본 이미지 픽셀 좌표 (1280×720 기준). z 는 m 단위 (depth=0 이면 None).
이 함수가 우리 인형뽑기 시스템에서 모델 출력 → 로봇 좌표로 가는 첫 단계다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from skimage.feature import peak_local_max


@dataclass
class Grasp3DoF:
    """3 DOF grasp 후보 (x, y, z, score)."""

    x: float
    """원본 이미지 픽셀 x (column)."""

    y: float
    """원본 이미지 픽셀 y (row)."""

    z: Optional[float]
    """depth (m). 해당 픽셀 depth=0 이면 None — 후처리에서 보간/제거 필요."""

    score: float
    """0~1 grasp quality (Q맵 값)."""

    metadata: dict = field(default_factory=dict)
    """모델별 raw 정보 보존 (디버깅/시각화)."""


def grasp_to_3dof(
    q_img: np.ndarray,
    depth_full: np.ndarray,
    *,
    original_size: tuple[int, int],
    crop_size: int = 224,
    top_k: int = 10,
    peak_min_distance: int = 20,
    peak_threshold: float = 0.2,
    workspace_mask: Optional[np.ndarray] = None,
) -> list[Grasp3DoF]:
    """모델 Q맵 → 3 DOF grasp 후보 리스트.

    :param q_img: (crop_size, crop_size) — GR-ConvNet `post_process_output` 결과 q_img
    :param depth_full: (H, W) — 원본 depth (m 단위)
    :param original_size: (W, H) — 원본 RGB 크기
    :param crop_size: 모델 입력 crop (cornell=224, jacquard=300)
    :param top_k: 추출할 최대 후보 수
    :param peak_min_distance: 후보 간 최소 픽셀 거리 (NMS 역할)
    :param peak_threshold: q_img 값이 이 이하면 무시
    :param workspace_mask: (H, W) bool — True 영역만 허용 (선택)
    :return: confidence 내림차순 후보 리스트
    """
    W, H = original_size
    if q_img.shape != (crop_size, crop_size):
        raise ValueError(
            f"q_img shape {q_img.shape} != ({crop_size}, {crop_size})"
        )
    if depth_full.shape != (H, W):
        raise ValueError(
            f"depth_full shape {depth_full.shape} != ({H}, {W})"
        )

    local_max = peak_local_max(
        q_img,
        min_distance=peak_min_distance,
        threshold_abs=peak_threshold,
        num_peaks=top_k,
    )

    # CameraData center crop offset 복원
    top = (H - crop_size) // 2
    left = (W - crop_size) // 2

    candidates: list[Grasp3DoF] = []
    for (yy, xx) in local_max:
        score = float(q_img[yy, xx])
        orig_x = int(xx + left)
        orig_y = int(yy + top)

        # 경계 체크
        if not (0 <= orig_y < H and 0 <= orig_x < W):
            continue

        # workspace 필터
        if workspace_mask is not None and not bool(workspace_mask[orig_y, orig_x]):
            continue

        # depth lookup
        z: Optional[float] = None
        d = float(depth_full[orig_y, orig_x])
        if d > 0:
            z = d

        candidates.append(
            Grasp3DoF(
                x=float(orig_x),
                y=float(orig_y),
                z=z,
                score=score,
                metadata={
                    "crop_y": int(yy),
                    "crop_x": int(xx),
                },
            )
        )

    candidates.sort(key=lambda g: g.score, reverse=True)
    return candidates
