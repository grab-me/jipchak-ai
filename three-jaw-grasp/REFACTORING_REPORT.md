# Refactoring Report: Dynamic Factory Pattern 적용기

## 1. 개요 및 목적
기존 `Three-Jaw Grasp` 파이프라인은 모델과 어댑터를 생성할 때 하드코딩된 조건문(`if/elif`)에 크게 의존하고 있었습니다. 이는 새로운 모델(예: 커스텀 YOLO, EconomicGrasp 등)이 추가될 때마다 코어 시스템(`run_pipeline.py`, `train.py`)을 직접 수정해야 하는 문제(개방-폐쇄 원칙 위반)를 발생시켰습니다.

본 리팩토링의 목적은 **"자료구조 기반 동적 팩토리 패턴(Registry Pattern + Factory Pattern)"**을 도입하여 추론 파이프라인과 학습 파이프라인 모두 플러그인(Plugin) 형태로 외부에서 유연하게 주입될 수 있도록 구조를 개선하는 것입니다.

---

## 2. 사용된 디자인 패턴

### 2.1. 레지스트리(Registry) 패턴
- **개념**: 시스템 전역에서 사용할 객체나 클래스를 중앙 딕셔너리(`dict`)에 저장해 두고 관리하는 패턴입니다.
- **적용**: `factory.py`의 `Registry` 클래스를 통해 구현되었습니다. 파이썬의 데코레이터 문법(`@ModelFactory.register("yolo")`)을 사용하여 클래스나 함수 선언과 동시에 레지스트리 딕셔너리에 자동으로 등록되도록 설계했습니다.

### 2.2. 팩토리(Factory) 패턴
- **개념**: 객체 생성 로직을 캡슐화하여, 호출자가 구체적인 클래스 이름을 알지 못해도 문자열 이름만으로 객체를 생성할 수 있게 하는 패턴입니다.
- **적용**: 등록된 레지스트리를 바탕으로 `ModelFactory.create("yolo")`를 호출하면 내부적으로 `dict["yolo"]()`를 실행하여 인스턴스를 반환합니다. 기존의 장황한 `if args.model == ...` 코드를 단 한 줄로 대체했습니다.

### 2.3. 어댑터(Adapter) 패턴 (기존 유지 및 확장)
- **개념**: 서로 다른 인터페이스를 가진 두 클래스를 연결해 주는 패턴입니다.
- **적용**: 외부에서 주입되는 각기 다른 모델들이 내뱉는 다양한 포맷(Tensor, Bounding Box 등)을 파이프라인의 표준 규격인 `GraspCandidate`로 통일하기 위해 사용되었습니다. 리팩토링을 통해 어댑터 또한 팩토리를 통해 동적으로 생성되도록 결합되었습니다.

---

## 3. 주요 리팩토링 사항

### 3.1. 코어 모듈: `factory.py` 도입
- `three_jaw_grasp/factory.py`를 신설하여 `ModelFactory`, `AdapterFactory`, `TrainerFactory` 전역 레지스트리를 제공합니다.

### 3.2. 외부 모델 및 학습 로직 분리 (Plugin System)
파이프라인 외부 사용자의 코드를 시뮬레이션하기 위해 `external_models/` 디렉토리를 신설했습니다.
- **`grconvnet/plugin.py` & `graspnet/plugin.py`**: 기존의 모델별 래퍼 객체를 완전히 독립된 플러그인으로 분리하여 위치시켰습니다.
- **`yolo_grasp_model` 제거 및 파이프라인 전처리 편입**: 기존 YOLO 파지 모델은 제거하고, YOLO 26 디텍터를 파이프라인의 핵심 전처리(Crop) 단계로 통합시켰습니다.

### 3.3. 실행 스크립트 결합도 최소화
- **`examples/run_pipeline.py`**: 조건문을 모두 제거하고 `ModelFactory.create()`, `AdapterFactory.create()`로 대체했습니다.
- **`train/train.py`**: 기존의 복잡한 신경망 코드를 제거하고, 단순히 커맨드라인 아규먼트를 파싱하여 `TrainerFactory.get()`을 통해 적절한 학습 로직에 인자를 전달하는 엔트리포인트(Entry point) 역할로 축소시켰습니다.

---

## 4. 리팩토링 후 기대 효과

1. **플러그 앤 플레이(Plug & Play) 가능**: 사용자가 파이프라인 내부 코드를 몰라도, 자신만의 모델/어댑터/학습코드를 작성하고 데코레이터만 달아주면 파이프라인의 시각화 및 평가 기능을 즉시 활용할 수 있습니다.
2. **코드 유지보수성 향상**: 새로운 딥러닝 모델이 등장하더라도 `run_pipeline.py` 등 핵심 로직은 변경되지 않으므로 안정적인 시스템 확장이 가능해졌습니다.
3. **관심사의 완벽한 분리**: 모델 개발자(YOLO 최적화 담당)와 파이프라인 개발자(시각화 및 3발 집게 로봇 공학 로직 담당)가 코드가 충돌할 우려 없이 완전히 독립된 폴더에서 협업할 수 있습니다.
# Refactoring Report

## 2026-05 Update

### 1) Jaw-count generalization
- `ThreeJawEvaluator`에 `jaw_count` 추가
- `stability/symmetry/mask_collision` 계산을 N-jaw로 일반화
- `jaw_count=3`이면 120도 간격 유지

### 2) Visualizer generalization
- `draw_n_jaw_grasp()` 추가
- `draw_three_jaw_grasp()`는 하위 호환 래퍼로 유지

### 3) Run pipeline CLI update
- `--jaw-count`
- `--evaluator {default,chick}`

### 4) Chick evaluator update
- 측정 가능한 변수 기반 점수 구성 유지
- `jaw_count` 연동

### 5) Example cleanup
- `examples/wrappers.py`를 `YoloMockModel` 중심으로 축소
