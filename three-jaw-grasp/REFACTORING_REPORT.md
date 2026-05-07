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
