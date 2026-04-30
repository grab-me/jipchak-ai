"""
파이프라인의 모델, 어댑터, 학습기(Trainer) 등을 동적으로 주입하고 관리하기 위한 레지스트리(팩토리 패턴) 모듈입니다.
"""
from typing import Callable, Dict, Any, Type

class Registry:
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Any] = {}

    def register(self, name: str) -> Callable:
        """
        클래스/함수를 데코레이터 형태로 레지스트리에 등록합니다.
        사용법:
            @ModelRegistry.register('yolo')
            class YoloModel: ...
        """
        def decorator(obj: Any):
            if name in self._registry:
                print(f"[Warning] '{name}'이(가) {self.name} 레지스트리에 이미 등록되어 있습니다. 덮어씁니다.")
            self._registry[name] = obj
            return obj
        return decorator

    def register_manual(self, name: str, obj: Any):
        """명시적으로 객체를 등록합니다."""
        self._registry[name] = obj

    def get(self, name: str) -> Any:
        """이름으로 등록된 객체(클래스 또는 함수)를 가져옵니다."""
        if name not in self._registry:
            raise KeyError(f"'{name}'이(가) {self.name} 레지스트리에 없습니다. 등록된 목록: {list(self._registry.keys())}")
        return self._registry[name]
        
    def create(self, name: str, *args, **kwargs) -> Any:
        """등록된 클래스를 기반으로 인스턴스를 생성하여 반환합니다."""
        obj_class = self.get(name)
        return obj_class(*args, **kwargs)

# 전역 팩토리 레지스트리
ModelFactory = Registry("Model")
AdapterFactory = Registry("Adapter")
TrainerFactory = Registry("Trainer")
EvaluatorFactory = Registry("Evaluator")
