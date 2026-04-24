# jipchak Architecture

## 핵심 원칙

1. **모델은 어댑터 뒤에 숨긴다.** 사용자는 `GraspAdapter` 인터페이스만 본다. 모델별 의존성, 그리퍼 가정, 좌표계 차이는 어댑터 내부에서 흡수.
2. **표준 입출력 타입을 강제한다.** `RGBDInput` 들어가서 `list[GraspCandidate]` 나온다. 그 외 형태는 어댑터 내부 구현 디테일.
3. **추가는 새 어댑터 1파일로 끝낸다.** 코어 코드 수정 없이 모델을 늘릴 수 있어야 한다.

## 데이터 플로우

```
       RGB-D 카메라 / 파일
              │
              ▼
       [전처리: optional]      ← 어댑터별 입력 정규화는 어댑터 내부
              │
              ▼
        RGBDInput              ← 표준 타입 (jipchak.core.types)
              │
              ▼
   GraspAdapter.predict()      ← 표준 인터페이스 (jipchak.core.adapter)
              │
              ▼
   list[GraspCandidate]        ← 표준 출력 (정렬: confidence desc)
              │
              ▼
    [후처리: 그리퍼 매핑]      ← 3-finger 변환, NMS, 충돌 검사 등은 별도 모듈
              │
              ▼
        로봇 제어 명령
```

## 표준 타입

### RGBDInput
- `rgb`: `(H, W, 3) uint8` — sRGB
- `depth`: `(H, W) float32` — meters (raw 값과 factor 분리해 입력 가능, 어댑터가 내부에서 환산)
- `intrinsics`: `dict` — `{fx, fy, cx, cy}` (선택)
- `workspace_mask`: `(H, W) bool | None` — 작업 영역 한정 (선택)

### GraspCandidate
- `x, y`: `float` — pixel 좌표 (이미지 기준)
- `z`: `float | None` — depth(m). 모델이 직접 추정하지 않으면 후처리 단계에서 채움
- `confidence`: `float` — 0~1 (어댑터가 자체 정규화 책임)
- `angle`: `float | None` — antipodal grasp 각도 (rad). 2-finger 모델만 사용
- `width`: `float | None` — gripper 폭 (m). 2-finger 모델만 사용
- `metadata`: `dict` — 모델별 raw output 보존 (디버깅/시각화용)

## 어댑터 인터페이스

```python
class GraspAdapter(ABC):
    name: str       # "grconvnet", "graspnet", "yolo-detect" 등
    license: str    # SPDX identifier
    supports_3finger: bool   # 본질적으로 3-finger 적합한가
    requires_depth: bool     # depth 없이 동작 가능한가

    def load(self, checkpoint: str, device: str) -> None: ...
    def predict(self, x: RGBDInput, top_k: int = 10) -> list[GraspCandidate]: ...
```

## 그리퍼 매핑

집착 1차 시스템은 **3-finger 수직 하강**(3 DOF). 표준 출력의 `angle`/`width`는 무시하고 `(x, y, z, confidence)`만 사용한다. 이 변환은 어댑터가 아닌 **후처리 모듈**에서 수행해 어댑터는 모델이 본래 출력하는 정보를 그대로 보존한다(향후 다른 그리퍼 사용 시 재활용).

## 벤치마크 러너

`jipchak/benchmark/runner.py`는 어떤 어댑터든 같은 패턴으로 latency/메모리/정성 결과 측정. 새 어댑터 추가 시 벤치마크 코드 재작성 불필요.

## 향후 확장

- 어댑터 plugin discovery (entry_points)
- Async/batched inference 인터페이스
- Fine-tuning 표준 데이터셋 포맷 (3-finger 시연 레이블 포함)
