# JOURNEY — 우리가 어떻게 여기까지 왔나

시간순 일지. 각 Phase 마다 **시도 → 결과 → 깨달음 → 다음 결정**.

본인(이채목)이 코치/팀원에게 "이런 흐름으로 결정했어요" 설명할 때 그대로 사용 가능.

---

## 사전 — 프로젝트 정체성 다시 잡기

**중요한 깨달음 (2026-04-28)**: 우리 프로젝트는 "**로봇 자동 잡기**" 가 아니다.

- 사용자가 직접 조이스틱으로 조작
- AI 는 **분석가 / 도우미** 역할:
  1. 인형 어디 있나 (Detection)
  2. 지금 집게 위치에서 잡으면 성공 확률 (Grasp Q맵 활용)
  3. 어떻게 움직이면 좋은지 피드백 (Q맵 hot spot - 집게 위치)
- 결과 영상 QR 로 사용자에게 제공 (집착 슬로건)

→ 이 프레이밍 잘못 잡으면 모든 작업이 잘못 보임. 지금은 **"두 AI 출력 → 사용자 분석 정보로 변환"** 이 본질.

---

## Phase 1 — Grasp AI 후보 비교 (2026-04 초)

### 시도
"어디를 잡을지" 결정하는 AI 모델 후보. **GraspNet baseline** 부터.

### 결과 — GraspNet 6근거 탈락

| # | 근거 | 의미 |
|---|---|---|
| 1 | 라이선스 CC BY-NC-SA | 상업화 X |
| 2 | H200 GPU 에서도 157ms | 느림 |
| 3 | 6 DOF 출력 | 우리 크레인은 3 DOF 만 필요, 과잉 |
| 4 | collision detection 80ms 병목 | 구조적 |
| 5 | rigid object 강함, soft toy 약함 | 도메인 갭 |
| 6 | **CPU 실행 자체 불가** | pointnet2 가 CUDA-only → EC2 배포 불가능 |

### 깨달음
- **라이선스 ≠ 단순 표기 문제**. 상업화 SaaS 운영 불가.
- **EC2 (GPU 없음, 4vCPU) 가 진짜 배포 환경** = CPU 동작이 절대 조건.
- 6 DOF 출력은 우리가 안 쓰는 정보 — 모델 무게에 비해 가치 없음.

### 다음 결정
**GR-ConvNet** 시도. 라이선스 BSD-3, 순수 PyTorch (CUDA 확장 없음).

---

## Phase 2 — GR-ConvNet 채택 + 검증 (2026-04-24~27)

### 시도
GR-ConvNet 동봉 4개 pretrained 비교 → 채택 → Cornell 데이터셋으로 검증.

### 결과

**4개 비교 (같은 입력)**:
| 모델 | GPU | CPU 4t | 시각화 |
|---|---|---|---|
| cornell-ch32 (큰) | 3.17ms | 55ms | 좋음 |
| **cornell-ch16 (작은)** | **3.16ms** | **20.7ms** | 좋음 (ch32 와 거의 같음) |
| jacquard-rgbd | 측정 X | 측정 X | **부적합** (시뮬 학습) |
| jacquard-d | 측정 X | 측정 X | 부적합 |

→ **cornell-ch16 채택**. ch32 와 grasp 결과 거의 동일하면서 CPU 2.7배 빠름.

**Cornell 데이터셋 정량 평가**:
- ch16 IoU 86/89 = **0.97** (학습 시 발표 수치 정확히 재현)
- 모델 정상 동작 확정.

**Fine-tune 파이프라인 검증**: `train_network.py` 에 `--pretrained` 옵션 추가 → 50 step 학습 동작 확인. 인형 데이터 도착하면 즉시 fine-tune 가능.

### 깨달음
- **Antipodal hot spot ≠ 무게중심**. 모델은 "잡기 쉬워 보이는 위치" 학습 (좁은 부분, 평행면) ≠ "들어올릴 수 있는 위치" (무게중심). 인형 머리만 잡으면 떨어짐 → fine-tune 필수.
- **2-finger 학습 → 3-finger 사용 mismatch**. 출력 4채널 (pos, cos, sin, width) 중 (x, y, Q) 만 사용해 우회.
- "Cornell" 은 **데이터셋** (코넬대, 단일 사물 885장). **모델 아님** (모델 = GR-ConvNet).

### 다음 결정
**운영 인터페이스** 작성: `infer.py` (CLI), `infer_server.py` (FastAPI). 메인 레포(jipchak Spring Boot) 호출 가능 형태.

---

## Phase 3 — 통합 운영 인터페이스 (2026-04-27)

### 시도
FastAPI 서버 — 모델 1회 로드 후 매 요청 재사용 (cold start 회피).

### 결과
- subprocess CLI: 매 호출 200~600ms (cold start)
- **서버 모드 (warmup 후): 5.83ms** (171 FPS)
- → 운영 시 서버 모드 필수.

### 깨달음
- 모델 로드 비용 (~150ms) + CUDA JIT (~200ms) 를 매번 부담하면 안 됨.
- 메인 레포는 HTTP POST 로 서버 호출 (`POST /infer`).
- 라이선스 OK + EC2 운영 가능 + 운영 인터페이스 → **Grasp AI 트랙 완료**.

---

## Phase 4 — Detection AI 후보 비교 (2026-04-27)

### 사전 깨달음
"어디 잡을지" 만으로는 인형뽑기 시스템 절반. **"무엇이 있는지"** 도 필요:
- 인형 영역 마스킹 → Grasp 가 박스 벽/배경에 false positive 안 내게
- 종류 식별 → 결과 영상 QR 에 "곰돌이 인형" 같이 표시
- 카운트 / 분포 분석

### 시도 — 라이선스 분류 + 17개 모델 측정

**라이선스 분류**:
- 🟢 상업 OK: MIT/BSD/Apache (YOLOX, YOLOv6, RT-DETR, torchvision)
- 🟡 상업 X: GPL/AGPL (모든 Ultralytics YOLO — v5/v8/v11/v26)
- 🔴 클라우드 SaaS (Google/AWS/Azure)

**측정 흐름**:
1. **YOLOv8/v11/v26 (AGPL, 측정만)** — n/s/m 사이즈, CPU 4t 47~120ms. 빠르지만 라이선스 X
2. **RT-DETR (Apache, 진짜 후보)** — CPU 244~372ms. **EC2 한계 초과**
3. **위기**: 라이선스 OK + EC2 가능 모델 미발견. AGPL 우회 검토할 뻔
4. **돌파구 — torchvision SSDlite-MobileNetV3 (BSD-3)** — CPU **37ms**, AGPL YOLO 보다도 빠름. 모바일 특화 학습 효과.

### 결과
**SSDlite-MobileNetV3 채택**:
- BSD-3 라이선스
- CPU 4t 37ms
- params 3.4M (가벼움)
- torchvision 기본 패키지 (별도 셋업 0)
- COCO 80 클래스 (`teddy bear` 포함)

### 깨달음
- **라이선스 OK + EC2 가능** 두 조건 동시 만족이 가장 어려운 부분. 다양하게 측정해보지 않았으면 못 찾았을 것.
- **모바일 특화 학습** (MobileNetV3) 이 의외로 cluttered scene 에서 잘 작동. 라이브러리 표준이 가끔 가장 좋은 답.
- "다른 모델 해볼만한 거?" 라는 의문이 핵심 발견 만듦.

---

## Phase 5 — 두 AI 통합 (2026-04-28)

### 시도
`pipeline.py` — Detection → workspace_mask → Grasp 한 번에.

### 결과 — example_data 실측

| | det_ms | grasp_ms | total_ms | n_grasps |
|---|---|---|---|---|
| GPU | 38.6 | 6.0 | **44.6** (22 FPS) | 5 |
| CPU 4t | 97.8 | 11.9 | **109.7** (9 FPS) | 5 |

→ EC2 운영 가능 (200ms 한계 1.8배 마진). 메인 레포는 `GraspPipeline.predict()` 한 번만 호출.

### 깨달음
- example_data 에서 **workspace_mask 효과 없음** — SSDlite false positive `sink` bbox 가 거의 전체 영역 커버. 인형뽑기 도메인 fine-tune + `--doll-only` 필터 후 의미 생김.
- 추정 latency (58ms) vs 실측 (110ms) 차이 = top_k/conf threshold/pipeline 오버헤드. 그래도 운영 가능선 안.

### 다음 결정
- **확률 계산 + 피드백 로직** (사용자 본인 역할 본질) — 5줄 numpy 코드.
- **오픈소스 측면 — 모델 swap 인터페이스 설계** (다음 Phase).
- **RealSense 도착 후** — 인형 데이터 수집 → fine-tune → 통합.

---

## 큰 그림 (한 화면)

```
[Phase 1] GraspNet 탈락 (CPU X)
    ↓
[Phase 2] GR-ConvNet ch16 채택 (BSD-3, 21ms CPU)
    ↓
[Phase 3] 운영 인터페이스 (FastAPI 5.83ms)
    ↓
[Phase 4] Detection AI 비교 → SSDlite (BSD-3, 37ms CPU)
    ↓
[Phase 5] 통합 pipeline.py (110ms CPU, 9 FPS, EC2 가능)
    ↓
[Next]  확률/피드백 로직 + 모델 swap 인터페이스
    ↓
[After RealSense] 인형 데이터 수집 → 두 AI fine-tune → 운영
```

---

## 본인 역할의 본질 (놓치지 말 것)

> "AI 모델 적용 + **확률 계산 로직** 구현"

- AI 모델 적용 ✅ (Phase 1~5 완료)
- **확률 계산 로직 = 다음 단계, 본인이 직접 짤 부분**
  - Q맵 (AI 출력) 에서 집게 위치 픽셀 값 추출 → 성공 확률
  - Q맵 max 위치 - 집게 위치 → 피드백 방향
  - Detection bbox 별 Q 평균 → "이 인형은 70%, 저 인형은 40%"

이건 5줄 numpy 코드 — AI 가 무거운 일 (Q맵 학습) 다 했고, 우리는 그 출력을 사용자에게 보여주는 변환만.

데이터 모이면 단계 2 (별도 AI 로 성공 확률 직접 학습) 가능.
