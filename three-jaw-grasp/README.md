# Three-Jaw Grasp Pipeline (Dynamic Factory Pattern)

본 프로젝트는 3발 집게(Three-Jaw Gripper) 파지를 위한 추론 및 평가 파이프라인을 제공합니다. 
최근 리팩토링을 통해 **자료구조 기반 동적 팩토리 패턴**을 도입하여, 코어 모듈을 수정하지 않고도 외부에서 사용자가 만든 독자적인 모델, 어댑터, 학습 로직을 파이프라인에 손쉽게 연동할 수 있게 되었습니다.

## 외부 모델 연동 가이드 (Plugin System)

사용자의 커스텀 모델(예: `EconomicGrasp`)이나 커스텀 학습 파이프라인(예: `YOLO`)을 본 프로젝트에 연동하려면, 파이프라인 코드를 직접 수정할 필요 없이 **팩토리 레지스트리에 등록(Register)**하기만 하면 됩니다.

### 1. 외부 플러그인 작성 예시

자신의 프로젝트(외부 폴더)에 다음과 같은 파이썬 스크립트를 작성합니다.

```python
# 사용자 외부 프로젝트: my_custom_model.py
from three_jaw_grasp.factory import ModelFactory, AdapterFactory, TrainerFactory
from three_jaw_grasp.adapters import BaseGraspAdapter
from three_jaw_grasp.candidate import GraspCandidate

# 1. 커스텀 모델 등록
@ModelFactory.register("economic")
class EconomicGraspModel:
    def predict(self, rgb, depth, **kwargs):
        # 자신만의 독자적인 추론 로직 (예: Tensor나 커스텀 객체 반환)
        raw_output = {"center_x": 100, "center_y": 150, "score": 0.95}
        return raw_output

# 2. 커스텀 어댑터 등록
# 파이프라인이 이해할 수 있는 GraspCandidate 표준 포맷으로 변환해주는 어댑터입니다.
@AdapterFactory.register("economic")
class EconomicGraspAdapter(BaseGraspAdapter):
    def adapt(self, raw_output) -> list[GraspCandidate]:
        return [GraspCandidate(
            center_x=raw_output["center_x"],
            center_y=raw_output["center_y"],
            center_z=0.30,
            width=0.05,
            angle=0.0,
            original_score=raw_output["score"],
            raw=raw_output
        )]

# 3. 커스텀 학습 로직 등록 (선택)
@TrainerFactory.register("economic")
def train_economic_model(data_path, output_path, epochs, **kwargs):
    print("Economic 모델 커스텀 학습을 시작합니다...")
    # 학습 로직
```

### 2. 파이프라인 실행

위와 같이 작성한 플러그인을 파이프라인 실행 스크립트 상단에 `import` 하기만 하면, 기존 `run_pipeline.py`나 `train.py`에서 파라미터만 변경하여 즉시 사용할 수 있습니다.

```bash
# 추론 및 시각화 테스트 (알아서 EconomicGraspModel과 EconomicGraspAdapter가 연결됨)
python examples/run_pipeline.py --model economic

# 커스텀 학습 로직 실행
python train/train.py --model economic --data_path dataset/01 --output_path weights/economic.pth
```

## 디렉토리 구조 (예시)

본 레포지토리는 다음과 같이 코어 파이프라인(`three_jaw_grasp/`)과 외부 플러그인(`external_models/`)이 물리적으로 분리되어 동작함을 보장합니다.

- `three_jaw_grasp/`: 핵심 파이프라인 (평가, 시각화, 파지 추출, YOLO 26 객체 인식 등)
- `external_models/grconvnet/`: GR-ConvNet 전용 모델 플러그인
- `external_models/graspnet/`: GraspNet 전용 모델 플러그인
- `examples/run_pipeline.py`: 통합 실행 환경 (외부 플러그인을 동적 로딩)
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
