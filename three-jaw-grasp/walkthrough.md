# Walkthrough - Three-Jaw Grasp Evaluation Pipeline

## 프로젝트 맥락

본 파이프라인은 **3발 집게를 장착한 인형뽑기 기계**에서 AI가 최적의 파지 위치를 실시간으로 가이드해주는 시스템의 핵심 모듈입니다.

- 기계 내부: 실시간 카메라(또는 스테레오 카메라 2대)로 모니터링
- 사용자: 조이스틱으로 X/Y 이동, AI가 최적 위치 가이드
- 집게: 3D 프린터 제작 3발 집게, Z축 하강/상승으로 물체 파지
- 확장 목표: 산업 크레인, 소각장 크레인 등으로 적용 범위 확대

---

## 구현된 파일 구조 (리팩토링 후)

```
three-jaw-grasp/
├── external_models/         # [NEW] 외부 모델 플러그인 폴더
│   ├── grconvnet/           # GR-ConvNet 전용 모델 플러그인
│   └── graspnet/            # GraspNet 전용 모델 플러그인
├── three_jaw_grasp/         # 코어 파이프라인 (수정 불필요)
│   ├── factory.py           # [NEW] 동적 팩토리 레지스트리
│   ├── pipeline.py          # 팩토리 기반 전체 흐름 제어
│   ├── adapters.py          # 모델 포맷 변환기
│   ├── candidate.py         # GraspCandidate 표준 포맷
│   ├── feature_extractor.py # 평가용 특징 추출
│   └── evaluator.py         # Rule-based / MLP 평가기
├── train/
│   ├── train_mlp.py         # 핵심 평가망(MLP) 학습 로직
│   └── train.py             # [리팩토링] 학습 엔트리포인트 스크립트
├── examples/
│   └── run_pipeline.py      # 통합 시각화/추론 실행 스크립트
├── config/
│   └── gripper_spec.yaml
├── README.md                # 외부 연동 가이드
├── REFACTORING_REPORT.md    # 팩토리 패턴 도입 보고서
└── INTERFACE_CONTRACT.md
```

---

## 주요 구현 사항 (Dynamic Factory Pattern)

기존 파이프라인의 강한 결합도를 해소하고 완벽한 플러그 앤 플레이(Plug & Play)를 지원하기 위해 다음과 같은 변경사항을 적용했습니다.

### 1. `factory.py` (핵심 코어)
- `ModelFactory`, `AdapterFactory`, `TrainerFactory` 3개의 전역 레지스트리 도입.
- 외부 플러그인 스크립트에서 데코레이터(`@ModelFactory.register(...)`)를 사용하여 딕셔너리에 객체를 동적으로 등록.

### 2. 코어 파이프라인 (`three_jaw_grasp/`)
- 기존에 어댑터별로 하드코딩되었던 의존성을 팩토리 레지스트리를 통해 분리.
- 어댑터, 파이프라인 코어는 외부 모델이 어떻게 구현되어 있는지 모른 채 표준 `GraspCandidate` 인터페이스만으로 통신.

### 3. 외부 모델 폴더 (`external_models/`)
- 사용자(연구원 등)가 자신만의 파지 추론 스크립트를 작성하는 독립 공간 보장.
- `grconvnet`, `graspnet` 모듈 예시 플러그인 구현. (객체 인식은 코어 파이프라인의 YOLO 26 디텍터가 사전 수행)

### 4. 통합 실행 환경 (`run_pipeline.py` & `train.py`)
- 모든 조건 분기문(`if model == ...`) 제거.
- `ModelFactory.create(args.model)` 한 줄로 파이프라인이 자동 조립되도록 설계.

---

## 동작 테스트 가이드

### 외부 모델(YOLO) 기반 통합 추론 및 시각화 테스트
```bash
python examples/run_pipeline.py --model yolo --max 1
```

### 외부 학습 파이프라인(MLP) 테스트
```bash
python train/train.py --model mlp --data_path dataset/01/processed_features.npz --output_path weights/three_jaw_mlp.pth
```

> [!TIP]
> 이제 새로운 모델을 도입할 때는 기존 코드를 수정할 필요 없이, `external_models/` 안에 플러그인 코드 하나만 작성하고 `@Factory.register`를 붙여주면 끝입니다!
# Walkthrough

## 1) 실행 흐름

`examples/run_pipeline.py`는 아래 순서로 동작합니다.

1. 모델 생성 (`ModelFactory` 또는 YOLO mock)
2. 어댑터 생성 (`AdapterFactory` 또는 `YoloBoxGraspAdapter`)
3. 평가기 생성 (`default`/`chick`, `jaw_count` 반영)
4. `GraspPipeline.predict_best()` 실행
5. `draw_n_jaw_grasp()`로 시각화

## 2) jaw_count 동작

- `jaw_count=3` -> 발 간격 120도
- `jaw_count=2` -> 발 간격 180도
- 3발 고정 장비는 `--jaw-count 3`로 고정해서 사용

## 3) 기본 실행

```bash
python examples/run_pipeline.py --model grconvnet --jaw-count 3 --evaluator default
python examples/run_pipeline.py --model grconvnet --jaw-count 3 --evaluator chick
```

## 4) 신규 모델 추가 체크리스트

1. `external_models/<name>/plugin.py` 생성
2. `@ModelFactory.register("<name>")` 등록
3. 어댑터 `@AdapterFactory.register("<name>")` 등록
4. `examples/run_pipeline.py`에 plugin import 추가
5. 실행: `python examples/run_pipeline.py --model <name> --jaw-count 3`

## 5) chick 전용 테스트

`examples/model_chick_v2.py`는 YOLOv8-seg + ChickEvaluator 테스트 스크립트입니다.

```bash
python examples/model_chick_v2.py
```

필수 경로:
- 모델: `dataset/model_chick2_v2/best2.pt`
- 이미지: `dataset/model_chick2_v2/test_images_sample/*.png`
