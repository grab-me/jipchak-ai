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

## 3. YOLO 후보 비교 (라이선스 우선)

집게 AI 와 같은 기준 — **상업화 가능 라이선스 + CPU 동작 + 적정 latency**.

| 모델 | 라이선스 | 상업화 | 추정 latency (CPU 4t) | 비고 |
|---|---|---|---|---|
| YOLOv8 / v11 (Ultralytics) | AGPL-3.0 | ❌ | 30~50ms (n) | 가장 인기/쉽지만 라이선스 막힘 |
| YOLOv5 (Ultralytics) | AGPL-3.0 / GPL-3.0 | ❌ | ~ | 동일 회사 |
| **YOLOX** (Megvii) | **Apache-2.0** | **✅** | ~30ms (nano) | **1순위 후보** |
| **YOLO-NAS** (Deci → NVIDIA) | Apache-2.0 | △ | ~30ms (s) | 가중치 라이선스 별도 — 확인 필요 |
| YOLOv4 Darknet | variant | △ | ~50ms | 변종마다 다름, 신중 |
| **RT-DETR** (Baidu) | **Apache-2.0** | **✅** | 50~100ms | Transformer, 정확도↑ but 느림 |
| DETR (Meta) | Apache-2.0 | ✅ | 100ms+ | 너무 느림 (실시간 부적합) |

**라이선스 함정**: Ultralytics YOLO (v5/v8/v11) 는 인기 압도적이지만 **AGPL-3.0** 이라 상업 서비스 X. 우리 프로젝트가 SSAFY 자율과제 + 오픈소스 프레임워크 지향이라 **상업 가능 라이선스 강제**.

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
