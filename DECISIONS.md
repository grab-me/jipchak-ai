# DECISIONS — 의사결정 기록

각 분기점마다 **후보 → 평가 → 선택 → 사유 → 결과**. 코치/리뷰어가 "왜 이걸 골랐어요?" 물을 때 흐름 그대로.

---

## D1. Grasp AI 모델 — GraspNet 탈락 → GR-ConvNet 채택

**시점**: 2026-04-24

**후보**:
| 모델 | 라이선스 | GPU latency | CPU |
|---|---|---|---|
| GraspNet baseline | CC BY-NC-SA | 157.5ms | **불가** |
| GR-ConvNet | BSD-3 | 3ms | 21ms (CPU 4t) |

**평가 기준**:
1. 라이선스 (상업화 가능?)
2. CPU 동작 (EC2 환경)
3. Latency (목표 200ms)
4. DOF (3 DOF 만 필요)
5. 도메인 적합 (soft toy)

**선택**: GR-ConvNet

**사유**:
- GraspNet 6근거 탈락 (라이선스 + CPU 불가가 결정타)
- GR-ConvNet 모든 조건 통과

**결과**: 운영 가능 라이선스 OK Grasp AI 후보 1개 확보. Phase 2 진행.

---

## D2. GR-ConvNet pretrained 변종 — Cornell ch16 채택

**시점**: 2026-04-27

**후보 (GR-ConvNet 동봉 4개)**:
| 모델 | 학습 데이터 | 사이즈 | params |
|---|---|---|---|
| cornell-ch32 | Cornell (실 카메라) | 32 채널 | 1.9M |
| cornell-ch16 | Cornell | 16 채널 | 479K |
| jacquard-rgbd | Jacquard (시뮬) | 32 | 1.9M |
| jacquard-d | Jacquard | 32 (depth-only) | 1.9M |

**평가**:
- 시각화 (Cornell ch16/ch32 vs Jacquard 둘): Jacquard 부적합 (sim-to-real 갭)
- ch16 vs ch32 시각화 동일 (같은 좌표 잡음)
- ch16 vs ch32 latency: CPU 4t 21ms vs 55ms (2.7배 차이)

**선택**: cornell-ch16

**사유**:
- Cornell = 실 카메라 학습 → RealSense 와 분포 가까움
- ch16 = ch32 와 정확도 동등 (시각화 검증) + 2.7배 빠름
- params 4배 작음 → 메모리 / 모바일 친화

**결과**: 단일 채택 모델 확정. IoU 0.97 정량 재현 검증.

---

## D3. 운영 인터페이스 형태 — FastAPI 서버

**시점**: 2026-04-27

**후보**:
1. CLI subprocess 호출 (Spring Boot 가 매번 python 실행)
2. FastAPI HTTP 서버 (모델 1회 로드 후 상주)
3. Python import 직접 사용 (Spring Boot 가 Python 이면)

**평가**:
| | latency 첫 호출 | latency 재호출 |
|---|---|---|
| CLI subprocess | 200~600ms | 200~600ms (cold start 매번) |
| FastAPI 서버 | 187ms (1회만) | **5.83ms** |
| Python import | 동일 | 동일 |

**선택**: FastAPI 서버 (옵션 2)

**사유**:
- 메인 레포 = Spring Boot (Python 아님) → 옵션 3 불가
- subprocess 매번 cold start = 운영 X
- FastAPI 가중치 1회 로드 후 매 요청 5.83ms = 171 FPS

**결과**: `infer_server.py` (Grasp), `detect_server.py` (Detection) 두 서버. 메인 레포가 HTTP POST 로 호출.

---

## D4. Detection AI — Ultralytics YOLO 탈락 → torchvision SSDlite 채택

**시점**: 2026-04-28

**후보 (17+ 모델 측정)**:

🟡 AGPL (상업 X — 측정 참고용):
- yolov8n/s/m, yolo11n/s, yolo26n/s/m (Ultralytics)
- yolov9t/c, yolov10n/s/m, yolo12n/s

🟢 Apache (진짜 후보):
- RT-DETR-l/x → 244~372ms (한계 초과)

🟢 BSD (진짜 후보, torchvision):
- **ssdlite320_mobilenet_v3_large** → 37ms ✅
- ssd300_vgg16 → 172ms (한계 근처)
- frcnn_mobilenet → 318ms (X)
- frcnn/fcos/retinanet resnet50 → 1초+ (X)

**평가 기준**:
1. 라이선스 OK (BSD/Apache)
2. CPU 4t < 200ms
3. params 작음
4. 의존성 작음

**선택**: ssdlite320_mobilenet_v3_large

**사유**:
- 17개 중 **세 조건 (라이선스 + 운영 + 가벼움) 동시 만족 유일**
- AGPL YOLO들 (47ms) 보다도 빠름 (37ms) — MobileNetV3 모바일 특화 효과
- torchvision 기본 패키지 → 별도 셋업 0
- COCO 80 클래스 (`teddy bear` 포함)

**결과**: Detection AI 잠정 채택. YOLOX 별도 셋업 사실상 불필요.

---

## D5. Ultralytics YOLO 측정 여부 — 측정만 하고 채택 X

**시점**: 2026-04-28

**갈등**:
- YOLOv8/v11/v26 압도적 인기, 셋업 쉬움 (`pip install ultralytics`)
- 라이선스 AGPL → 우리 채택 X
- "측정해도 의미 없는 거 아닌가?"

**선택**: **측정만** 진행 (참고 reference)

**사유**:
- YOLO 류 latency 스펙트럼 빠르게 확보 (n/s/m 사이즈)
- 라이선스 OK 후보 (RT-DETR/SSDlite) 비교 기준점
- "왜 우리가 안 썼는가" 설명 시 측정 데이터 있어야 설득력

**결과**: 17 모델 표 확보. SSDlite 가 AGPL YOLO 보다도 빠르다는 사실 발견 → 진짜 채택 정당화.

---

## D6. 통합 wrapper 설계 — bbox → workspace_mask → grasp

**시점**: 2026-04-28

**후보 흐름**:
1. **순차 (bbox → mask → grasp)**: Detection 결과로 Grasp 영역 한정
2. 병렬 (Detection / Grasp 독립): 각자 결과 사용자에게 그대로
3. End-to-end (단일 모델로 둘 다): 모델 학습 필요, 데이터 큼

**선택**: 1 (순차)

**사유**:
- Detection 이 인형 영역 한정 → Grasp 가 박스 벽 false positive 안 만듦
- 두 사전학습 모델 그대로 재사용
- 코드 단순 (`bboxes_to_mask()` helper 5줄)

**결과**: `pipeline.py` 작동 검증. 단 example_data 에선 SSDlite false positive `sink` bbox 가 거의 전체 영역 커버해 mask 효과 미미. 인형뽑기 fine-tune 후 의미 생김.

---

## D7. 확률 계산 — numpy 5줄 vs 별도 AI

**시점**: 2026-04-28 (대기 중)

**후보**:
1. **numpy 5줄** — Q맵 픽셀 인덱싱 (`q_img[y, x]`)
2. 별도 AI — "현 상태 → 성공 확률" 분류 모델 학습

**평가**:
| | 정확도 | 학습 데이터 필요 | 즉시 가능 |
|---|---|---|---|
| numpy | Q맵 학습 정확도에 의존 | 0 | ✅ |
| 별도 AI | 도메인 특화로 더 정확 | 수백~수천 시연 | ❌ (데이터 0) |

**선택**: **단계 진행**. 1단계 numpy → 데이터 모이면 2단계 별도 AI

**사유**:
- 데이터 0 인 지금 별도 AI 학습 불가
- numpy 도 사실상 AI 활용 (Q맵 자체가 AI 출력 — 단지 후처리만 numpy)
- 데이터 모이면 그때 업그레이드 자연스러움

**결과**: 미정 (다음 작업).

---

## D8. 어댑터 프레임워크 골격 — 한 차례 시도 → revert → 재시도 대기

**시점**: 2026-04-24 시도, 같은 날 revert. 재시도 = 다음 Phase

**원래 시도**:
- `jipchak/core/types.py` (RGBDInput, GraspCandidate)
- `jipchak/core/adapter.py` (GraspAdapter ABC)
- `jipchak/benchmark/runner.py` (모델 무관 벤치마크)

**revert 사유 (본인 결정)**:
- 모델 검증 부족 (n=1)
- 본인 흐름/개념 이해 갭
- "공부/검증 우선, 추상화는 N개 모델 다뤄본 후"

**현재 상황**:
- Grasp + Detection 두 모델 다뤄봄
- 통합 wrapper (`pipeline.py`) 동작
- 추상화 의미 생긴 시점

**다음 결정 (대기)**: 인터페이스 재설계 (`GraspAdapter`, `DetectAdapter`) → 다른 사람 swap 가능. 오픈소스 프레임워크 본질.

---

## 의사결정 패턴 요약

본인이 의식적으로 따른 기준:
1. **라이선스 우선** (BSD/Apache 만)
2. **CPU 동작 + EC2 latency 한계** 가 절대 조건
3. **다양한 후보 측정** (Ultralytics/RT-DETR/torchvision 17개) — 한두 개 빠르게 결정 X
4. **잠정 채택** 단어 명시 — 검증 부족 시 결정 안 닫음
5. **외부 의존성 최소화** (셋업 부담 ↓ = 팀원 / 외부 기여자 친화)

코치 질문 시 이 5개 기준 + D1~D8 흐름이면 모든 의사결정 설명 가능.
