# Interface Contract

## 1. Model Wrapper Contract

등록 위치: `external_models/<name>/plugin.py`

필수:

```python
@ModelFactory.register("<name>")
class MyModel:
    def __init__(self, checkpoint_path=None, **kwargs):
        ...

    def predict(self, rgb, depth, **kwargs):
        # raw output return
        return raw_output
```

- `rgb`: `np.ndarray[H, W, 3]`, `uint8`
- `depth`: `np.ndarray[H, W]`, `float32` (meter 권장)
- 반환값 형식은 자유. 단, 대응 어댑터가 반드시 해석 가능해야 함.

## 2. Adapter Contract

등록 위치: `three_jaw_grasp/adapters.py`

```python
@AdapterFactory.register("<name>")
class MyAdapter(BaseGraspAdapter):
    def adapt(self, raw_output) -> list[GraspCandidate]:
        return [...]
```

`GraspCandidate` 필수 필드:
- `center_x`, `center_y`, `center_z`
- `width`
- `angle` (radian)
- `original_score` (0~1 권장)
- `raw` (dict)

## 3. Evaluator Contract

기본 평가기:
- `ThreeJawEvaluator(..., jaw_count=3)`
- `jaw_count=3`이면 120도 간격, `jaw_count=2`이면 180도 간격

chick 평가기:
- `ChickEvaluator(..., jaw_count=3)`
- `default` 대비 물리 휴리스틱(목 포착/지지/간섭 여유) 반영

## 4. Pipeline Contract

```python
pipeline = GraspPipeline(
    model=model,
    adapter=adapter,
    evaluator=evaluator,
    detector=detector_or_none,
    is_3d=False,
)
best = pipeline.predict_best(rgb, depth, filename_prefix=None)
```

## 5. Visualizer Contract

```python
draw_n_jaw_grasp(
    ax, rgb, grasp, detail,
    model_name="...",
    jaw_count=3,
    is_3d=False,
    intrinsics=None,
)
```

하위 호환:
- `draw_three_jaw_grasp(...)`는 내부적으로 `jaw_count=3` 호출

## 6. CLI Contract (`examples/run_pipeline.py`)

지원 인자:
- `--model`
- `--dataset`
- `--max`
- `--save`
- `--jaw-count` (2 또는 3)
- `--evaluator` (`default` 또는 `chick`)

예시:

```bash
python examples/run_pipeline.py --model grconvnet --jaw-count 3 --evaluator chick
```
