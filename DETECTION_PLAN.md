# Detection AI 후보 정리 — 인형 인식 트랙

**Last updated**: 2026-04-27
**Status**: 후보 비교 단계 (PoC 미시작)

집게 AI(GR-ConvNet) 가 "**어디를 잡을지**" 라면, Detection AI 는 "**무엇이/어디에 있는지**" 를 담당한다. 두 AI 가 같이 돌아야 인형뽑기 시스템 완성.

## 1. 왜 별도 트랙인가

| | Grasp AI (현재 채택) | Detection AI (이 문서) |
|---|---|---|
| 모델 | GR-ConvNet ch16 | YOLO 류 (미정) |
| 입력 | RGB-D | RGB (depth 선택) |
| 출력 | (x, y, z, score) 잡기 위치 | bbox + 클래스 + confidence |
| 의미 | "여기 잡으면 잘 잡힘" | "여기 인형 있다 / 어떤 종류" |
| 인형뽑기 활용 | 로봇 좌표 명령 | ① 인형 영역 마스킹 ② 종류 식별 ③ 결과 영상 라벨링 |

## 2. 통합 패턴

```
RGB-D 카메라
   ├─→ Detection AI ─→ 인형 bbox / mask / 종류
   │                       │
   │                       ↓
   │                 workspace_mask 생성
   │                       │
   ├─→ Grasp AI (workspace_mask 적용)
   │       │
   │       ↓
   │   (x, y, z, score) 잡기 후보
   │       │
   │       ↓
   │   3-finger 로봇 좌표 변환
   │
   └─→ 결과 영상 + 인형 종류 라벨 → QR 영상 (집착 슬로건 핵심)
```

장점:
- Grasp AI 가 박스 벽/그림자/배경 같은 잘못된 영역 후보를 안 냄 (인형 영역만 검색)
- 사용자에게 "어떤 인형 잡았는지" 보여줄 수 있음 (결과 영상 QR)
- 인형 카운트 / 박스 안 분포 분석 가능 (성공 확률 계산 보조 데이터)

## 3. 라이선스 분류 기준

| 분류 | 라이선스 | 의미 |
|---|---|---|
| 🟢 **상업적 가능** | MIT / BSD / Apache-2.0 | 오픈소스 + 상업 OK (출처 표기만) |
| 🟡 **상업적 불가능 (조건부)** | GPL-3.0 / AGPL-3.0 | 오픈소스이지만 소스공개 의무 + 동일라이선스 전파. 상업 SaaS X (Enterprise 라이선스 별도 구매 시 OK) |
| 🟠 **비상업 라이선스** | CC BY-NC-SA / CC BY-NC | 학술/연구만, 상업 명시적 금지 |
| 🔴 **아예 오픈소스 X** | Proprietary / SaaS | 클라우드 API, 비공개 모델 |

**우리 프로젝트 채택 기준**: 🟢 만. 🟡 는 측정 참고용. 🟠/🔴 제외.

## 4. Detection 모델 전체 표

| 모델 | 출처 | 라이선스 | 분류 | 비고 |
|---|---|---|---|---|
| **YOLOX** | Megvii | Apache-2.0 | 🟢 | **1순위 후보** |
| **YOLOv6** | Meituan | Apache-2.0 | 🟢 | 2순위 |
| **RT-DETR** | Baidu | Apache-2.0 | 🟢 | Transformer, 정확도↑ |
| DETR | Meta | Apache-2.0 | 🟢 | 너무 느림 |
| DINO / Grounding DINO | IDEA Research | Apache-2.0 | 🟢 | text prompt 가능 |
| SAM / SAM2 | Meta | Apache-2.0 | 🟢 | segmentation, 무거움 |
| Faster R-CNN / RetinaNet | torchvision | BSD | 🟢 | 표준, YOLO보다 느림 |
| YOLOv5 | Ultralytics | AGPL-3.0 | 🟡 | |
| YOLOv7 | WongKinYiu | GPL-3.0 | 🟡 | |
| YOLOv8 | Ultralytics | AGPL-3.0 | 🟡 | 측정용으로 사용함 |
| YOLOv9 | WongKinYiu | GPL-3.0 | 🟡 | |
| YOLOv10 | THU-MIG | AGPL-3.0 | 🟡 | |
| YOLOv11 | Ultralytics | AGPL-3.0 | 🟡 | |
| **YOLO26** | Ultralytics | AGPL-3.0 + Enterprise | 🟡 | 2026-01 출시, CPU 43%↑, NMS-free |
| Google Cloud Vision | Google | Proprietary SaaS | 🔴 | 클라우드 호출 |
| AWS Rekognition | AWS | Proprietary SaaS | 🔴 | |
| Azure Computer Vision | MS | Proprietary SaaS | 🔴 | |
| Clarifai / Roboflow Inference | 각 회사 | Proprietary SaaS | 🔴 | |

**Ultralytics 함정**: v5/v8/v11/v26 모두 AGPL-3.0. 인기 압도적이지만 우리 채택 X. **YOLO26 은 Enterprise 라이선스 별도 구매 시 OK** 라 회색지대 — SSAFY/스타트업 무료 사용 시 사실상 X.

## 5. 측정 결과 (Ultralytics YOLO 시리즈, 🟡 측정용)

GraspNet `doc/example_data/color.png` 입력, conf=0.05, warmup 후 평균.

| 모델 | GPU H200 (ms) | CPU 4t (ms) | detections | params |
|---|---|---|---|---|
| yolov8n | 24.0 | 49.6 | 8 | 3.2M |
| yolov8s | 24.9 | 69.4 | 5 | 11.2M |
| yolov8m | 25.7 | 112.7 | 10 | 25.9M |
| yolo11n | 25.7 | **46.5** | 12 | 2.6M |
| yolo11s | 25.8 | 69.2 | 6 | 9.5M |
| yolo26n | 26.2 | 48.2 | 11 | 2.6M |
| yolo26s | 26.4 | 77.0 | 3 | 10.0M |
| yolo26m | 27.2 | 118.5 | 5 | 21.9M |

### 핵심 발견

1. **GPU H200 24~27ms** — 모델 사이즈 차이 묻힘 (GPU가 너무 빠름 + ultralytics `predict()` 호출 오버헤드 포함)
2. **CPU 4-thread (EC2 시뮬)**:
   - n 사이즈 ~47ms (200ms 한계 대비 4배 마진)
   - s 사이즈 ~70ms
   - m 사이즈 ~115ms (마진 1.7배, 여전히 운영 가능)
3. **YOLO26 "CPU 43% 빠름" 주장은 우리 측정에서 안 보임** (PyTorch native vs ONNX export 차이 추정). 같은 사이즈에서 v8 ≈ v11 ≈ v26.
4. **YOLO11n** 이 가장 빠름 (CPU 46.5ms) + detection 수 가장 많음 (12). 의외의 winner. **라이선스 AGPL이라 채택 X**.
5. **Grasp AI(20.7ms) + Detection AI(47ms) 통합 ≈ 67ms** = 15 FPS. 200ms 한계 대비 3배 마진.

### 결론 (Ultralytics 기준)

**latency 검증 완료**. n/s/m 사이즈 모두 EC2 운영 가능. 정확도는 도메인 fine-tune 후 의미 있음 (Grasp AI 와 동일 결론).

→ 진짜 채택은 🟢 후보(YOLOX/YOLOv6) 측정 후 결정. 위 결과는 "이 latency 면 충분히 빠르다" 는 reference.

raw: [detection-poc/yolo_compare.txt](detection-poc/yolo_compare.txt)

## 4. 인형 인식 정확도

### Pretrained 사전학습 데이터셋

대부분 **COCO 2017** (80 클래스) 학습:
- `teddy bear` ← **봉제 인형 일부 커버** ✅
- `person`, `bottle`, `cup` 등 → 박스 안 다른 물체 식별
- 캐릭터 인형(짱구/포켓몬 등), 피규어 → COCO 미커버 ❌

### 단계별 정확도 기대

| 단계 | 정확도 기대 | 작업 |
|---|---|---|
| 1차 (사전학습 그대로) | 봉제 인형은 ~70~80% | 코드만 |
| 2차 (인형뽑기 박스 fine-tune) | 봉제 인형 90%+ | 데이터 ~500장 + 학습 |
| 3차 (캐릭터/종류 분류) | 종류별 정확도 | 데이터 + 클래스별 라벨링 |

→ 1차로 충분히 PoC 가능. 2차부터 도메인 데이터 필수.

## 5. 검증 계획 (집게 AI 와 동일 패턴)

```
[1] 후보 라이선스 필터        ✅ 위 표
[2] 1순위 후보 latency 측정   ⏳ — YOLOX nano GPU/CPU
[3] 정성 검증                  ⏳ — example_data 에서 detection 결과
[4] 인형 도메인 갭            ⏳ — 인형뽑기 박스 사진 (RealSense 셋업 후)
[5] Fine-tune (옵션)          ⏳ — 인형 데이터 모이면
[6] Grasp AI 와 통합           ⏳ — workspace mask 연동
```

각 단계 마무리: latency raw + 시각화 PNG + Cornell 패턴 정량 metric (mAP) 기록.

## 6. 추천 진행 순서

**B-1. YOLOX nano 빠른 PoC (~1~2시간)**
```bash
# 의존성
pip install yolox  # 또는 git clone https://github.com/Megvii-BaseDetection/YOLOX

# 사전학습 가중치
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth

# 같은 example_data 에서 추론
python yolox_demo.py --rgb ../graspnet-baseline/doc/example_data/color.png
```
→ COCO 80 클래스에서 어떤 게 detect 되는지, latency 얼마인지 측정.

**B-2. RT-DETR 비교 (~1시간)**
**B-3. 정량 평가** (COCO val 일부 또는 인형 데이터)

## 7. 통합 시점

지금 시작 → 카메라 도착 시점에 양쪽 모두 준비 완료. 또는 카메라 도착 후 동시 진행 가능. **B-1 PoC 까지는 카메라 무관**.

## 8. Detection AI 도 잠정 후보

| 후보 | 라이선스 | 추정 latency | 비고 |
|---|---|---|---|
| **YOLOX nano** | Apache-2.0 | ~30ms (CPU 4t 추정) | 가장 만만, 1순위 |
| RT-DETR-R18 | Apache-2.0 | ~80ms | 정확도 우선 시 |
| (대체) Grounding DINO | Apache-2.0 | 매우 느림 | text prompt detection — 나중 검토 |
| (대체) SAM | Apache-2.0 | 무거움 | segmentation, 가중 |

## 9. 다음 unblocker

- **B-1 PoC** — 지금 가능. 라이선스 OK 모델 1개의 실 latency 확인이 필요.
- **인형 데이터 수집** — RealSense 셋업 후 (Grasp AI fine-tune 데이터와 같이 모음)

## 10. 메모

- Grasp AI 와 통합 시 **추가 latency**: YOLO ~30ms + GR-ConvNet 6ms = ~36ms ≈ 28 FPS. 200ms 목표 대비 5배 마진.
- COCO 사전학습으로 1차 PoC → 인형 데이터로 2차 fine-tune 흐름이 표준.
- 데이터 수집 시 **YOLO 학습 포맷 (`bbox: x_center y_center w h class`)** 도 같이 레이블링하면 한 번에 두 모델 학습 가능. 레이블링 도구 설계 시 반영.
