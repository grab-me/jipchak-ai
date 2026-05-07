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
