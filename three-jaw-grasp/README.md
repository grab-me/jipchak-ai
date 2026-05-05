# three-jaw-grasp

모델 출력(GraspNet, GR-ConvNet, YOLO 등)을 공통 포맷으로 변환한 뒤,
집게 평가 로직으로 최적 grasp를 선택하는 파이프라인입니다.

## 핵심 개념

1. `ModelFactory`: 모델 래퍼 등록
2. `AdapterFactory`: 모델별 출력을 `GraspCandidate`로 변환
3. `Evaluator`: 후보 점수 계산(`default`, `chick`)
4. `GraspPipeline`: detect -> predict -> adapt -> evaluate

## jaw 간격(중요)

- **3발 집게**일 때 집게 간 간격은 **120도 고정**입니다.
- 코드에서 `jaw_count=3`이면 자동으로 `360/3=120°` 간격을 사용합니다.
- `jaw_count=2`를 주면 180° 간격 로직으로 동작합니다.

## 빠른 실행

```bash
python examples/run_pipeline.py --model grconvnet --jaw-count 3 --evaluator default
python examples/run_pipeline.py --model grconvnet --jaw-count 3 --evaluator chick
python examples/run_pipeline.py --model graspnet --jaw-count 3 --evaluator default
python examples/run_pipeline.py --model yolo --jaw-count 3 --evaluator chick
```

`--jaw-count 3`이면 120° 고정 3발 집게 평가/시각화가 됩니다.

## 외부 사용자 확장 방법

다른 사용자가 새 모델을 붙일 때 필요한 최소 작업입니다.

1. `external_models/<my_model>/plugin.py` 파일 추가
2. `@ModelFactory.register("<my_model>")`로 모델 클래스 등록
3. `predict(rgb, depth, **kwargs)` 메서드 구현
4. `three_jaw_grasp/adapters.py`에 어댑터 추가
5. `@AdapterFactory.register("<my_model>")` 등록
6. `examples/run_pipeline.py`에서 플러그인 import 추가

예시:

```python
from three_jaw_grasp.factory import ModelFactory

@ModelFactory.register("my_model")
class MyModelWrapper:
    def __init__(self, checkpoint_path=None):
        ...

    def predict(self, rgb, depth, **kwargs):
        return raw_output
```

```python
from three_jaw_grasp.adapters import BaseGraspAdapter
from three_jaw_grasp.factory import AdapterFactory
from three_jaw_grasp.candidate import GraspCandidate

@AdapterFactory.register("my_model")
class MyModelAdapter(BaseGraspAdapter):
    def adapt(self, raw_output):
        return [
            GraspCandidate(
                center_x=..., center_y=..., center_z=...,
                width=..., angle=..., original_score=..., raw={}
            )
        ]
```

## 집게 종류 선택

- 실행 시 `--jaw-count 2|3` 선택 가능
- 3발 고정 시스템이면 `--jaw-count 3`만 사용 권장

## 모델/데이터 경로 수정 포인트

- GR-ConvNet checkpoint: `examples/run_pipeline.py` 내 `GRCONVNET_ROOT`
- YOLO mock 데이터: `dataset/<name>/*r.png`, `*cpos.txt`
- Chick 테스트 모델: `examples/model_chick_v2.py`의 `MODEL_PATH`, `DATASET_DIR`

## `model_chick_v2.py` 테스트

```bash
python examples/model_chick_v2.py
```

필요 패키지:

```bash
pip install ultralytics opencv-python matplotlib numpy pillow
```
