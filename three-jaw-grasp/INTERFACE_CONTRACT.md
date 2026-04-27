# Three-Jaw Grasp Pipeline — 인터페이스 규약 (Interface Contract)

버전: v0.1.0 | 최종 수정: 2026-04-27

---

## 1. 설계 철학

### 1-1. 프로젝트 맥락

이 파이프라인은 **3발 집게가 달린 기계에서 AI가 최적의 파지 위치를 가이드해주는 시스템**의 핵심 모듈로 설계되었습니다.

**기본 적용 사례 (인형뽑기 기계)**:
- 기계 내부를 실시간 카메라로 모니터링
- 사용자가 조이스틱으로 X/Y 위치 이동, AI가 최적 파지 위치 가이드
- 집게는 Z축으로 하강/상승하여 물체를 잡음
- 3D 프린터로 제작한 3발 집게 사용

**확장 적용 사례 (산업현장)**:
- 산업용 크레인, 소각장 크레인, 물류 자동화 등
- 모델/어댑터/평가기를 교체하여 동일한 파이프라인 재사용 가능

### 1-2. 핵심 설계 원칙

이 파이프라인은 **"최소한의 입력으로도 즉시 동작"** 하면서 **"모든 세부 요소를 교체·확장"** 할 수 있도록 설계되었습니다.

```
(필수)  사용자 모델 → 어댑터 → 특징 추출 → 평가 → 최적 파지 위치 출력
(선택)             ↑         ↑          ↑
               커스텀 어댑터  커스텀 추출기 커스텀 평가기/Config
```

사용자가 제공하는 것이 많을수록 더 정밀한 파이프라인이 되고,
아무것도 커스터마이징하지 않아도 기본 Rule-based 평가로 동작합니다.

---

## 2. 파이프라인이 하는 일

### 2-1. 전체 흐름 요약

외부 Grasp 탐지 모델을 **그대로 가져와서** 사용하되, 그 모델이 탐지한 파지 후보들 중 **3발 집게에 가장 적합한 파지 위치 1개**를 골라줍니다.

```
[RGB + Depth 이미지 입력]
         ↓
[외부 Grasp 탐지 모델]  ← 사용자가 가져오는 모델 (GraspNet, GR-ConvNet 등)
  "이미지에서 잡을 수 있는 후보 N개를 탐지"
  예: 후보 50개 → [(x1,y1,z1,각도1,폭1,점수1), ..., (x50,...)]
         ↓
[어댑터]  ← 우리가 만든 변환기
  "모델마다 다른 출력 포맷 → 우리 표준 포맷(GraspCandidate)으로 통일"
         ↓
[평가기]  ← 우리가 만든 3발 집게 특화 로직
  "N개 후보를 3발 집게 기준으로 점수화 → 가장 높은 1개 선택"
  - 집게 폭에 맞는가? (너무 작거나 크면 못 잡음)
  - 높이가 적절한가? (너무 낮으면 바닥 충돌)
  - 3발 120° 대칭 구조와 잘 맞는가?
  - 원본 모델의 신뢰도는 높은가?
         ↓
[최적 파지 위치 1개 출력]
  center_x, center_y, center_z  ← 집게를 이동시킬 3D 좌표
  angle                         ← 집게를 몇 도로 회전할지
  width                         ← 집게를 얼마나 벌릴지
```

### 2-2. 어댑터(Adapter)란?

어댑터는 **우리가 직접 만든 포맷 변환기**입니다. 외부 오픈소스 라이브러리가 아닙니다.

**왜 필요한가?** 각 Grasp 탐지 모델은 결과를 서로 다른 포맷으로 반환합니다:

```
GraspNet   출력 → GraspGroup 객체  { .x, .y, .z, .theta, .score }
GR-ConvNet 출력 → NumPy 배열      [center_x, center_y, angle, width, score]
AnyGrasp   출력 → 딕셔너리        {'translation': [...], 'score': 0.9}
```

어댑터는 이 서로 다른 포맷을 우리 파이프라인의 내부 표준인 `GraspCandidate`로 변환합니다:

```
GraspNet 출력  →  [GraspGroupAdapter]  →  GraspCandidate(x, y, z, width, angle, score)
GR-ConvNet 출력→  [RectGraspAdapter]   →  GraspCandidate(x, y, z, width, angle, score)
AnyGrasp 출력  →  [PoseArrayAdapter]   →  GraspCandidate(x, y, z, width, angle, score)
```

> **핵심**: 어댑터만 바꾸면 어떤 외부 모델이든 동일한 평가 로직을 적용할 수 있습니다.

### 2-3. 구성 요소 관계도

```
[외부 모델]          우리가 만들지 않음, 사용자가 가져옴
    ↓ 출력
[어댑터]             우리가 만든 코드 (adapters.py)
    ↓ GraspCandidate
[특징 추출기]        우리가 만든 코드 (feature_extractor.py)
    ↓ 숫자 벡터
[평가기]             우리가 만든 코드 (evaluator.py)
    ↓ 최적 후보 1개
[GraspPipeline]      우리가 만든 코드 (pipeline.py) — 전체 흐름 조율

↓ 다른 사람이 확장하는 경우

[파생 파이프라인]    GraspPipeline을 상속해서 우리 코드를 기반으로 확장
```

> **GraspPipeline 상속은 다른 오픈소스 상속이 아닙니다.**  
> 우리가 만든 `GraspPipeline` 클래스를 다른 사용자가 상속하여  
> 자신의 환경(전처리 추가, 후처리 필터링 등)에 맞게 확장하는 것입니다.

---

## 3. 사용자 입력 계층 (Input Levels)

### Level 0 — 모델만 제공 (즉시 사용)

```python
from three_jaw_grasp import GraspPipeline, GraspGroupAdapter

pipeline = GraspPipeline(
    model=YourModel(),          # ← 이것만 필수
    adapter=GraspGroupAdapter() # ← 출력 포맷에 맞는 기본 어댑터 선택
)

best = pipeline.predict_best(rgb, depth)
```

**사용자가 제공해야 하는 것**:
- `YourModel` 클래스: `predict(rgb, depth) -> raw_output` 메서드
- 원하는 기본 어댑터 선택 (GraspGroup / Rect / PoseArray 중 택 1)

**파이프라인이 자동으로 처리하는 것**:
- 기본 특징 추출 (15차원)
- 기본 Rule-based 평가 (내장 가중치 사용)
- 기본 집게 스펙 (min_width=0.01m, max_width=0.10m, ideal=0.05m)

---

### Level 1 — 집게 스펙 & 가중치 조정 (Config 제공)

```python
pipeline = GraspPipeline(
    model=YourModel(),
    adapter=GraspGroupAdapter(),
    evaluator=ThreeJawEvaluator(config_path="config/my_gripper.yaml")
)
```

**사용자가 제공하는 것**: YAML 파일

```yaml
# config/my_gripper.yaml
gripper:
  name: "My Custom Gripper"
  min_width: 0.02      # 집게가 잡을 수 있는 최소 폭 (m)
  max_width: 0.12      # 집게가 잡을 수 있는 최대 폭 (m)
  ideal_width: 0.06    # 가장 안정적인 파지 폭 (m)
  collision_threshold_z: 0.03  # 바닥 충돌 위험 임계값 (m)

evaluation:
  rule_weights:
    width: 0.30        # 파지 폭 적합도 가중치
    score: 0.25        # 원본 모델 신뢰도 가중치
    height: 0.20       # 파지 높이 적합도 가중치
    stability: 0.15    # 수직 접근 안정성 가중치
    symmetry: 0.10     # 3발 120° 대칭 일치도 가중치
```

---

### Level 2 — 커스텀 어댑터 (새로운 모델 포맷 지원)

```python
from three_jaw_grasp import BaseGraspAdapter, GraspCandidate

class MyModelAdapter(BaseGraspAdapter):
    def adapt(self, raw_output) -> list[GraspCandidate]:
        return [
            GraspCandidate(
                center_x=g.position[0],
                center_y=g.position[1],
                center_z=g.position[2],
                width=g.opening,
                angle=g.rotation_z,
                original_score=g.confidence,
                raw={}
            )
            for g in raw_output.grasps
        ]
```

**작성 규칙**:
- `BaseGraspAdapter`를 반드시 상속
- `adapt(raw_output) -> list[GraspCandidate]` 메서드 구현
- 좌표 단위는 **미터(m)**, 각도는 **라디안(rad)**
- `raw` 딕셔너리에 원본 정보를 보존하면 디버깅 편의성 증가

---

### Level 3 — 커스텀 특징 추출기 (도메인 특화)

```python
from three_jaw_grasp import FeatureExtractor
import numpy as np

class MyExtractor(FeatureExtractor):
    FEATURE_DIM = 17  # 기본 15 + 추가 2

    def extract(self, grasp, depth=None):
        base = super().extract(grasp, depth)  # 기본 15차원
        extra = np.array([
            self._contact_area(grasp),   # 예: 접촉 면적 추정
            self._surface_normal(depth)  # 예: 표면 법선 방향
        ], dtype=np.float32)
        return np.concatenate([base, extra])
```

**주의**: `FEATURE_DIM`을 변경하면 MLP 재학습이 필요합니다.  
학습과 추론에 **동일한 extractor 인스턴스**를 사용해야 합니다.

---

### Level 4 — 커스텀 평가기 (평가 기준 완전 교체)

```python
from three_jaw_grasp import ThreeJawEvaluator

class MyEvaluator(ThreeJawEvaluator):
    def _width_fitness(self, g):
        # 원하는 폭 점수 계산 로직으로 교체
        return custom_width_score(g.width)

    def _symmetry(self, g):
        # 특수 집게 구조에 맞는 대칭 점수 교체
        return custom_symmetry_score(g.angle)
```

**Override 가능한 메서드**:
| 메서드 | 설명 | 기본 동작 |
|---|---|---|
| `_width_fitness(g)` | 파지 폭 적합도 | Gaussian 분포 기반 |
| `_height_fitness(g)` | 높이 적합도 | 바닥 충돌 임계값 기반 |
| `_stability(g)` | 수직 접근 안정성 | 미구현 (1.0 고정) |
| `_symmetry(g)` | 3발 대칭 일치도 | `cos(3θ)²` |
| `_original_score(g)` | 원본 모델 신뢰도 | 그대로 반환 |
| `_rule_score(g, depth)` | 전체 룰 기반 점수 | 가중합 계산 |

---

### Level 5 — MLP 학습 및 주입 (완전 커스터마이징)

```python
# 1. 데이터 수집 및 학습
# train/collect_data.py → train/train.py 순서로 실행

# 2. 학습된 모델 주입
pipeline = GraspPipeline(
    model=YourModel(),
    adapter=MyModelAdapter(),
    evaluator=ThreeJawEvaluator(
        model_path="weights/my_mlp.pth",     # 학습된 가중치
        extractor=MyExtractor(),              # 학습 시 사용한 extractor
        config_path="config/my_gripper.yaml" # 집게 스펙
    )
)
```

---

## 3. 필수 인터페이스 — 사용자 모델 규약

### 3.1 모델 인터페이스 (유일한 필수 규약)

```python
class YourGraspModel:
    def predict(self, rgb: np.ndarray, depth: np.ndarray) -> Any:
        """
        Parameters
        ----------
        rgb   : np.ndarray, shape [H, W, 3], dtype uint8
        depth : np.ndarray, shape [H, W],    dtype float32, 단위: 미터(m)

        Returns
        -------
        raw_output : 모델 고유 포맷 (어댑터가 처리함)
        """
        ...
```

**요구사항**:
- `predict` 메서드가 있어야 함 (이름 변경 불가)
- rgb: `[H, W, 3]` uint8
- depth: `[H, W]` float32, 단위는 **미터(m)**
- 반환값 포맷은 자유 (어댑터가 변환함)

---

## 4. GraspCandidate — 파이프라인 내부 표준 포맷

```python
@dataclass
class GraspCandidate:
    center_x:       float  # 파지 중심 X — 어댑터 구현에 따라 픽셀 또는 카메라 좌표계 미터
    center_y:       float  # 파지 중심 Y — 동일
    center_z:       float  # 파지 높이 (미터) — 카메라 또는 바닥 기준, 어댑터 주석에 명시 권장
    width:          float  # 집게가 벌려야 할 폭 (미터)
    angle:          float  # 집게 회전 각도 (라디안, 0 = 기본 방향)
    original_score: float  # 원본 탐지 모델 신뢰도 (0.0 ~ 1.0 권장)
    raw:            dict   # 원본 데이터 보존용 (디버깅·확장 목적)
```

> **좌표계 주의**: `center_x`, `center_y`의 단위(픽셀/미터)는 어댑터마다 다를 수 있습니다.
> 어댑터 구현 시 반드시 docstring에 좌표계와 단위를 명시하세요.
> 모터 제어 연동 시에는 **카메라 좌표 → 모터 좌표 변환 레이어**가 별도로 필요합니다.

> **`original_score` 범위 주의**: 파이프라인 내부에서 `[0, 1]`로 클램핑합니다.  
> 모델이 logit이나 `[0, 100]` 범위 점수를 반환하는 경우 점수가 0 또는 1로 포화되므로,  
> 어댑터에서 softmax 또는 min-max 정규화 후 `original_score`에 넣는 것을 권장합니다.

---

## 5. 평가 기준 설계 원칙

### 5-1. 스코어링은 반드시 파지 실행 전에 수행

```
[카메라 이미지 취득]    ← 파지하기 전
[외부 모델 후보 탐지]   ← 파지하기 전
[평가기: N개 후보 점수화] ← 파지하기 전   ★ 이 파이프라인의 역할
[최적 위치 좌표 반환]   ← 파지하기 전
───────────────────────────────────
[모터 이동 명령]        ← 파이프라인 외부 (사용자 제어 코드)
[집게 하강 및 파지]     ← 파이프라인 외부
```

> `GraspPipeline.predict_best(rgb, depth)`는 판단만 수행하며,  
> 모터 제어 코드는 이 파이프라인에 포함되지 않습니다.

### 5-2. 각 평가 기준의 의미와 설계 근거

| 기준 | 의미 | 설계 근거 | 기본 가중치 |
|---|---|---|---|
| `width` | 집게가 물체를 실제로 잡을 수 있는 폭인가 | 물리 제약 — 범위 밖이면 파지 불가 | 0.35 |
| `score` | 탐지 모델의 이 위치에 대한 신뢰도 | 외부 모델이 이미 대부분의 판단을 수행 | 0.30 |
| `height` | 집게가 바닥에 충돌하지 않는 높이인가 | 안전 마진 — 하드웨어 파손 방지 | 0.25 |
| `stability` | 집게 접근 각도와 3발 배치(60° 주기)의 정렬도 | 수직 접근 시 3발이 균등하게 내려오는가 | 0.10 |
| `symmetry` | 파지 각도와 3발 120° 대칭 구조의 정렬도 | 물체-집게 각도 정렬 | 0.00 |

### 5-3. 3발 120° 고정 집게에 대한 주의사항

3발 발톱이 하드웨어적으로 **항상 120° 균등 배치**인 경우:

```
"발 사이 각도가 균등한가?"  → 항상 120° 고정 → 평가 불필요

"집게를 몇 도 방향으로 회전하면 물체와 잘 맞는가?"  → stability/symmetry가 이것을 평가
```

- **원통형/구형 물체** (인형뽑기 기계 대부분): 어느 각도에서 잡아도 동일 → `symmetry: 0.00`
- **사각형/비대칭 물체**: 특정 각도에서 잡아야 안정적 → `symmetry` 가중치 올리기
- **`stability`**: 집게 자체가 Z축 회전 자유도를 가진 경우에만 의미 있음  
  Z축 고정 집게라면 `stability: 0.00`으로 설정

```yaml
# 인형뽑기 기계 (원통형 인형 위주) 권장값
evaluation:
  rule_weights:
    width:     0.35   # 가장 중요 (물리 제약)
    score:     0.30   # 탐지 신뢰도
    height:    0.25   # 안전 마진
    stability: 0.10   # Z축 회전 가능한 경우만 의미 있음
    symmetry:  0.00   # 원통형 물체라면 0으로 설정
```

### 5-4. 집게 스펙 설정 방법

**반드시 실제 측정값으로 설정해야 합니다.** 기본값은 임시 placeholder입니다.

```yaml
gripper:
  min_width: ???   # 집게 완전 오므림 시 3발 끝단 간 최소 직경 (m) ← 자로 측정
  max_width: ???   # 집게 최대 개방 시 3발 끝단 간 직경 (m)       ← 자로 측정
  ideal_width: ???  # 처음에는 (min + max) / 2 로 설정, 실험 후 조정
  collision_threshold_z: ???  # 카메라 기준 바닥까지 거리 - 안전 여유분 (m)
```

> **스펙 미설정 시 발생하는 문제**: `width_fitness()`가 잘못된 범위로 계산되어  
> 실제로 잡을 수 없는 폭의 후보를 선택하거나, 반대로 잡을 수 있는 후보를 0점 처리함.

### 5-5. 점수 디버깅 방법

```python
# 개별 후보의 세부 점수 확인
evaluator = ThreeJawEvaluator(config_path="config/gripper_spec.yaml")
detail = evaluator.score_detail(best_candidate)

print(f"width     raw={detail['width']['raw']:.3f}   weighted={detail['width']['weighted']:.3f}")
print(f"score     raw={detail['score']['raw']:.3f}   weighted={detail['score']['weighted']:.3f}")
print(f"height    raw={detail['height']['raw']:.3f}   weighted={detail['height']['weighted']:.3f}")
print(f"stability raw={detail['stability']['raw']:.3f}   weighted={detail['stability']['weighted']:.3f}")
print(f"symmetry  raw={detail['symmetry']['raw']:.3f}   weighted={detail['symmetry']['weighted']:.3f}")
print(f"total = {detail['total']:.3f}")
```

## 5. 파생 파이프라인(Derived Pipeline) 가이드

베이스 파이프라인을 기반으로 완전히 새로운 파이프라인을 구현할 경우:

```python
from three_jaw_grasp import GraspPipeline
import numpy as np

class MySpecializedPipeline(GraspPipeline):
    """
    예: 특정 산업 환경에 특화된 파이프라인
    - 전처리: 이미지 노이즈 필터링
    - 후처리: 특정 높이 범위 필터링
    """
    def predict_best(self, rgb, depth):
        # 전처리 (필요 시 오버라이드)
        depth = self._filter_depth(depth)

        # 외부 모델 추론 + 어댑터 변환
        raw_output = self.model.predict(rgb, depth)
        candidates = self.adapter.adapt(raw_output)

        # 후처리: 환경에 맞는 높이 범위 필터링
        candidates = [c for c in candidates if 0.03 < c.center_z < 0.50]

        if not candidates:
            raise ValueError("필터링 후 유효한 파지 후보가 없습니다.")

        return self.evaluator.select_best(candidates, depth=depth)

    def _filter_depth(self, depth: np.ndarray) -> np.ndarray:
        """이상치 depth 값 제거 (환경별 오버라이드 가능)"""
        depth = np.where(depth < 0.01, np.nan, depth)  # 너무 가까운 값 제거
        depth = np.where(depth > 2.0,  np.nan, depth)  # 너무 먼 값 제거
        return depth
```

**산업현장 크레인 확장 예시**:

```python
class IndustrialCranePipeline(GraspPipeline):
    """
    소각장/물류 크레인 특화 파이프라인
    - 대형 물체, 더 넓은 집게 폭
    - 더 높은 안전 마진 적용
    """
    def predict_best(self, rgb, depth):
        raw_output = self.model.predict(rgb, depth)
        candidates = self.adapter.adapt(raw_output)

        # 산업 환경: 소형 물체(폭 < 5cm) 후보 제외
        candidates = [c for c in candidates if c.width >= 0.05]

        return self.evaluator.select_best(candidates, depth=depth)
```

---

## 6. 테스트 데이터 가이드

### 6.1 최소 테스트 (구현 확인용)

별도 데이터셋 없이 NumPy 더미 데이터로 파이프라인 동작 확인:

```python
import numpy as np
rgb   = np.zeros((480, 640, 3), dtype=np.uint8)
depth = np.ones((480, 640), dtype=np.float32) * 0.5  # 50cm 거리

best = pipeline.predict_best(rgb, depth)
```

### 6.2 실물 테스트 (권장 데이터)

| 데이터 종류 | 설명 | 최소 수량 |
|---|---|---|
| RGB 이미지 | 파지 대상 물체의 정면/측면 | 10장 이상 |
| Depth 맵 | 동일 장면의 depth (float32, m) | 동일 |
| 물체 카테고리 | 원통, 구, 박스 등 형상별 분류 | 3종 이상 |
| 파지 성공 라벨 | MLP 학습 시 필요 (0/1 이진) | 선택사항 |

### 6.3 공개 데이터셋 (MLP 학습용)

| 데이터셋 | 형식 | 비고 |
|---|---|---|
| Cornell Grasp Dataset | Rectangle grasp | RectGraspAdapter 호환 |
| Jacquard Dataset | Rectangle grasp | RectGraspAdapter 호환 |
| GraspNet-1Billion | GraspGroup | GraspGroupAdapter 호환 |

---

## 7. 확장 기여 포인트 (Contributing Points)

| 모듈 | 기여 방법 | 난이도 |
|---|---|---|
| `adapters.py` | 새 어댑터 추가 (BaseGraspAdapter 상속) | 낮음 |
| `feature_extractor.py` | 도메인 특화 특징 추가 | 중간 |
| `evaluator.py` | 평가 기준 함수 override | 중간 |
| `config/*.yaml` | 새 집게 스펙 제공 | 낮음 |
| `train/train.py` | 학습 루프 개선, 다중 분류 확장 | 높음 |

---

## 8. 빠른 시작 체크리스트

사용자가 처음 이 파이프라인을 사용할 때의 순서:

- [ ] 1. 사용할 Grasp Detection 모델을 `predict(rgb, depth)` 인터페이스에 맞게 래핑
- [ ] 2. 모델 출력 포맷에 맞는 어댑터 선택 또는 커스텀 어댑터 작성
- [ ] 3. 사용하는 집게의 물리 스펙(`min_width`, `max_width`)을 YAML에 작성
- [ ] 4. `examples/graspgroup_format.py`를 참고해 기본 파이프라인 실행 확인
- [ ] 5. (선택) 평가 가중치를 YAML에서 조정
- [ ] 6. (선택) 데이터 수집 → MLP 학습 → 가중치 주입으로 성능 향상
