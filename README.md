# jipchak-ai

**프로젝트 "집착(JipChak)"의 AI 모델 실험/검증 레포** (메인 시스템 레포: jipchak)

인형뽑기 기계에 AI 붙여 실시간 성공 확률 분석 + 결과 영상 QR 제공. SSAFY 14기 D108팀.

이 레포는 **AI 모델 선정 / 검증 / fine-tune 준비**에 집중한다. 시스템 통합(Spring Boot/React/하드웨어)은 메인 레포에서.

## 한눈 보기 (어디까지 왔나)

```
[1] 모델 후보 비교            ✅ 끝 — GraspNet 탈락, GR-ConvNet 채택
[2] 정량 검증 (Cornell)       ✅ 끝 — ch16 IoU 0.97 재현
[3] 정성 검증 (Cornell)       ✅ 끝 — 5장 일관성 우수
[4] Fine-tune 파이프라인 검증 ✅ 끝 — --pretrained patch 동작, 50 step IoU 0.955
[5] 도메인 갭 검증            ⏸  보류 — RealSense D405 재구매 대기
[6] 인형 데이터 fine-tune     ⏳ 위 셋업 후
[7] 시스템 통합               ⏳ 메인 레포에서
```

**현재 상태 (2026-04-27)**: 단독 진행 가능한 모델 작업 거의 완료. RealSense 카메라 도착 → 인형 RGB-D 데이터 수집 → fine-tune 이 다음 unblocker.

## 1분 설명 (코치/팀원용)

> "인형뽑기 grasp detection 모델 고르는 단계예요.
>
> 첫 후보 **GraspNet** 은 5가지 이유로 탈락 — ① 라이선스 비상업용 ② 빠른 GPU 에서도 157ms 느림 ③ 출력 6 DOF 인데 우리 크레인은 3 DOF 만 필요 ④ collision detection 80ms 구조적 병목 ⑤ **결정타: EC2 배포 환경(GPU 없음)에서 실행 자체 불가능** (CUDA-only 연산 포함).
>
> 대안 **GR-ConvNet** 채택 — BSD-3 라이선스, 순수 PyTorch, EC2 4-thread CPU 시뮬에서 20ms. 동봉 pretrained 4종 중 **cornell-ch16** 사용 (시각화 결과 ch32 와 동등하면서 2.7배 빠름).
>
> 잠정 채택 상태. 검증한 건 모델이 학습된 환경(Cornell) 에서 정상 동작하는지까지. 인형뽑기 도메인은 RealSense 셋업 후 fine-tune."

## 핵심 개념 (모르면 위 말 어버버)

| 용어 | 한 줄 |
|---|---|
| GR-ConvNet | 우리 모델 (네트워크 아키텍처) |
| Cornell | 학습 **데이터셋** (모델 아님). 흰 테이블 + 단일 사물 885장 |
| antipodal grasp | 양쪽에서 집는 두발 집게 가정. 우리는 세발 → 살짝 mismatch |
| DOF | 자유도. 우리 크레인 = 3 (x, y, z 만; 회전 X) |
| Q맵 | 픽셀별 "잡으면 잘 잡힐 확률" 0~1. 우리가 (x, y) 추출하는 핵심 출력 |
| pretrained / fine-tune | 남이 학습한 가중치 시작점 → 우리 데이터로 추가 학습 |
| latency / FPS | 추론 시간 / 초당 처리 장수. 목표: 200ms 이하 / 5+ FPS |

## 측정 결과 (전체 표는 `BENCHMARKS.md`)

| 모델 | GPU H200 | CPU 4-thread (EC2 sim) | 라이선스 | CPU 동작 |
|---|---|---|---|---|
| GraspNet baseline | 157.5ms | **불가** | CC BY-NC-SA | ❌ |
| **GR-ConvNet ch16** | **3.16ms** | **20.7ms** | **BSD-3** | **✅** |
| GR-ConvNet ch32 | 3.17ms | 55.0ms | BSD-3 | ✅ |

Cornell 정량: ch16 IoU **86/89 = 0.97** (학습 시 발표 수치 재현 → 모델 정상).

## 단서 (잠정인 이유)

- **n=1 → n=5 검증** (Cornell 5장) 했으나 **인형뽑기 도메인은 0장**. RealSense 셋업이 unblocker.
- **Antipodal grasp center ≠ 사물 무게중심**. 모델이 "잡기 쉬워 보이는 위치" 학습한 거지 "들어올릴 수 있는 위치"는 아님. 길쭉한 봉제 인형의 양 끝(머리 등)을 추천할 가능성 → 들다 떨어짐. 인형뽑기 시연 데이터로 **fine-tune 필수**.
- **2-finger 학습 → 3-finger 사용 mismatch**. 출력 4채널 중 (x, y, Q) 만 사용해 우회.
- **EC2 실측 안 함**. 4-thread 시뮬은 H200 서버 CPU 일부 제한. EC2 의 vCPU 마이크로아키텍처 다를 수 있음.

## 디렉토리 구조

```
jipchak-ai/
├── graspnet-baseline/        # 원본 GraspNet (참고/실험용, 탈락 모델)
├── gr-convnet/               # 원본 GR-ConvNet + 우리 도구
│   ├── trained-models/       # pretrained 4종 (cornell ch16/ch32, jacquard d/rgbd)
│   ├── benchmark_grconv.py   # latency 측정
│   ├── visualize_*.py        # 시각화 도구 4종 (combined / grid / 3finger / cornell_samples)
│   ├── grasp_to_3dof.py      # 후처리: Q맵 → (x,y,z,score) 3 DOF 변환
│   ├── infer.py              # 단일 진입점 CLI (디버깅용)
│   ├── infer_server.py       # FastAPI 서버 (운영용, 모델 1회 로드)
│   ├── train_network.py      # 학습 (--pretrained patch 적용)
│   └── logs/                 # fine-tune 검증 결과
├── MODEL_DECISION.md         # 모델 선정 결정문 + 한계
├── BENCHMARKS.md             # 모든 측정 결과 + 재현 명령
├── FINETUNE_PLAN.md          # fine-tune 청사진 + 코드 patch 계획
└── README.md                 # 이 문서
```

## 재현 (서버 SSAFY GPU 환경)

```bash
ssh j-k14d108@jupyter08
conda activate jipchak
cd ~/jipchak-ai-chaemok/gr-convnet

# latency
python benchmark_grconv.py --device cuda:0
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python benchmark_grconv.py --device cpu --n_warmup 3 --n_runs 20

# Cornell 정량 평가 (데이터 ~/datasets/cornell-grasp 미리 다운)
python evaluate.py --network trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch16/epoch_30_iou_0.97 \
    --dataset cornell --dataset-path ~/datasets/cornell-grasp --iou-eval

# 시각화
python visualize_3finger.py        # 3-finger 관점 (ch16 vs ch32)
python visualize_grid.py           # 4 pretrained 비교
python visualize_cornell_samples.py --n_samples 5    # Cornell 무작위 샘플
```

## 메인 레포(jipchak) 통합 가이드

추론은 두 가지 방식. **운영은 서버 모드 필수** — CLI subprocess 는 매 호출마다 cold start 200~600ms 발생.

### CLI (디버깅/단발 호출)
```bash
python gr-convnet/infer.py \
    --rgb color.png --depth depth.png --factor-depth 1000 --top-k 5 \
    --output result.json
```
출력: `gr-convnet/infer.py` 헤더 docstring 참고. 매 호출 = 모델 새로 로드 → 느림.

### 서버 (운영, 권장)
```bash
# 의존성
pip install fastapi uvicorn python-multipart

# 기동
JIPCHAK_DEVICE=cuda:0 python gr-convnet/infer_server.py --port 8080
# 또는 CPU
JIPCHAK_DEVICE=cpu python gr-convnet/infer_server.py --port 8080
```

요청 (Spring Boot/Python/curl 등 어디서든):
```bash
curl -X POST http://localhost:8080/infer \
    -F "rgb=@color.png" \
    -F "depth=@depth.png" \
    -F "factor_depth=1000" \
    -F "top_k=5"
```
응답:
```json
{
  "model": "cornell-randsplit-rgbd-grconvnet3-drop1-ch16",
  "device": "cuda:0",
  "inference_ms": 3.1,
  "n_candidates": 5,
  "candidates": [
    {"x": 634.0, "y": 283.0, "z": 0.413, "score": 0.864},
    ...
  ]
}
```

헬스체크: `GET /health`

### Python 직접 import
같은 Python 프로세스에서 호출하면 가장 단순:
```python
import sys
sys.path.insert(0, "/path/to/jipchak-ai/gr-convnet")
from infer import GraspInfer
import numpy as np
from PIL import Image

infer = GraspInfer(device="cuda:0")  # 1회 로드
rgb = np.array(Image.open("color.png"))
depth_raw = np.array(Image.open("depth.png")).astype(np.float32)
depth_m = depth_raw / 1000.0
candidates = infer.predict(rgb, depth_m, depth_raw=depth_raw, top_k=5)
# candidates: list[Grasp3DoF] (x, y, z, score, metadata)
```

### Latency 가이드 (어떤 모드 선택?)

| 호출 패턴 | 모델 로드 | 추론 (ch16) | 운영 가능? |
|---|---|---|---|
| CLI (subprocess 매번) | 매번 200~600ms | 47ms (CPU) / 229ms (GPU cold) | ❌ |
| 서버 (1회 로드 후 재사용) | 시작 시 1회 | **20ms (CPU 4t) / 3ms (GPU)** | ✅ |
| Python import 재사용 | 1회 | **20ms / 3ms** | ✅ |

→ Spring Boot 메인 레포에서 호출 시 **서버 모드**로 띄워두고 HTTP POST 권장.

## 다음 단계

1. RealSense D405 재구매/셋업 (하드웨어, **다음 unblocker**)
2. 인형뽑기 박스 + 봉제 인형 RGB-D 데이터 수집 (~500~2000장)
3. 레이블링 (인형 중심 클릭 → Cornell `cpos.txt` 형식)
4. fine-tune 실행 (`--pretrained` patch 이미 검증됨, FINETUNE_PLAN.md 참고)
5. 메인 레포(jipchak)와 서버 모드로 통합

## 라이선스

본 레포 코드: TBD (Apache-2.0 또는 BSD-3 검토)
편입한 원본:
- `graspnet-baseline/` — CC BY-NC-SA (실험 참고용, 상업 배포 X)
- `gr-convnet/` — BSD-3 (채택 모델, 상업 OK)
