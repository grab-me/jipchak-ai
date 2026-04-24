# jipchak — 인형뽑기로 시작하는 집게 × AI 오픈소스 프레임워크

**Goal**: 어떤 grasp 모델(GraspNet, GR-ConvNet, YOLO 기반 detection 등)이든 **표준 인터페이스 하나**로 swap 가능하게 만든다. 인형뽑기를 first use case로 출발해, 사용자가 자기 모델/하드웨어를 끼워 다양한 그리퍼 시스템에 활용할 수 있는 어댑터 프레임워크.

## 왜 이 프레임워크가 필요한가

기존 grasp 모델은 입출력 포맷, 그리퍼 가정(2-finger antipodal vs multi-finger), 좌표계, 의존성이 모두 다르다. 새 모델을 시도할 때마다 통합 코드를 새로 짜야 한다. `jipchak`는 이 통합 비용을 한 번만 치르고, 이후 어댑터 추가만으로 모델을 늘려 간다.

## 레포 구조

```
jipchak-ai/
├── jipchak/              # 프레임워크 본체 (Python 패키지)
│   ├── core/            #   표준 입출력 타입, 어댑터 추상 클래스
│   ├── adapters/        #   모델별 어댑터 (모델당 1파일)
│   └── benchmark/       #   어댑터 무관 latency/정성 평가 러너
├── graspnet-baseline/    # 원본 GraspNet 코드 (참고/실험용)
├── gr-convnet/           # 원본 GR-ConvNet 코드 (참고/실험용)
├── examples/             # 표준 input 샘플 + 사용 예제
└── docs/                 # ARCHITECTURE.md 등 설계 문서
```

원본 모델 레포는 그대로 보존한다. 어댑터는 그 위에 얹는 얇은 레이어다.

## 빠른 시작 (예정)

```python
from jipchak.adapters.grconvnet import GRConvNetAdapter
from jipchak.core.types import RGBDInput

adapter = GRConvNetAdapter(
    checkpoint='gr-convnet/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98',
    device='cuda:0',
)
candidates = adapter.predict(RGBDInput(rgb=..., depth=..., intrinsics=...))
# candidates: list[GraspCandidate(x, y, z, confidence, ...)]
```

## 현재 상태

- ✅ GraspNet baseline 벤치마크 (H200): forward 66.8ms, total 157.5ms, **CPU 실행 불가** (pointnet2 CUDA-only)
- ✅ GR-ConvNet 벤치마크 (H200 GPU 3.17ms / CPU 4-thread 55ms) — 라이선스 BSD-3, CPU 동작 확인
- 🔄 표준 어댑터 인터페이스 설계 중
- ⏳ 어댑터 구현 (GraspNet, GR-ConvNet, YOLO)

자세한 모델 비교/선정 근거는 [docs/MODEL_DECISION.md](docs/MODEL_DECISION.md) (작성 예정).
설계 의도는 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 참고.

## 라이선스

본 레포 코드: TBD (예정 — Apache-2.0 또는 BSD-3 검토 중).
편입한 원본 모델 레포는 각자의 라이선스를 따른다 (GraspNet: CC BY-NC-SA, GR-ConvNet: BSD-3).
