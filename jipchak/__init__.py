"""jipchak — 인형뽑기 × AI 오픈소스 프레임워크.

표준 grasp 어댑터 인터페이스로 다양한 모델을 swap 가능하게 한다.
설계 의도는 docs/ARCHITECTURE.md 참고.
"""
from jipchak.core.types import RGBDInput, GraspCandidate
from jipchak.core.adapter import GraspAdapter

__all__ = ["RGBDInput", "GraspCandidate", "GraspAdapter"]
__version__ = "0.0.1"
