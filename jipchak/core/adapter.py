"""GraspAdapter 추상 클래스 — 모든 어댑터의 공통 인터페이스."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from jipchak.core.types import GraspCandidate, RGBDInput


class GraspAdapter(ABC):
    """모델별 어댑터의 추상 베이스.

    구현 시 클래스 변수(name, license, ...)와 load/predict 두 메서드만 채우면
    벤치마크 러너와 사용 코드는 즉시 호환된다.
    """

    name: ClassVar[str]
    """어댑터 식별자 (예: 'grconvnet', 'graspnet', 'yolo-detect')."""

    license: ClassVar[str]
    """SPDX identifier (예: 'BSD-3-Clause', 'CC-BY-NC-SA-4.0')."""

    supports_3finger: ClassVar[bool] = False
    """본질적으로 3-finger center grasp에 적합한 출력을 내는가.
    antipodal 학습 모델은 False (후처리로 (x,y)만 추출해 사용)."""

    requires_depth: ClassVar[bool] = True
    """depth 없이 동작 가능한가 (RGB-only)."""

    @abstractmethod
    def load(self, checkpoint: str, device: str = "cuda:0") -> None:
        """모델 가중치 로드 + device 이동."""

    @abstractmethod
    def predict(self, x: RGBDInput, top_k: int = 10) -> list[GraspCandidate]:
        """추론. 반환 리스트는 confidence 내림차순."""
