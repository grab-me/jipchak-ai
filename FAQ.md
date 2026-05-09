# FAQ — 코치/팀원이 물을 만한 질문 + 본인이 그대로 답할 수 있는 답

자기 점검용. **읽고 외우는 게 아니라**, 한 번 읽고 본인이 답할 수 있는지 확인. 막히는 질문이 본인 학습 갭.

---

## 가장 본질적

### Q1. 우리 인형뽑기에 왜 AI 가 필요해요?
A. 집게 자동화 X. **사용자가 직접 조이스틱으로 조작**. AI 는 분석가:
- 인형 어디 있는지 (Detection)
- 지금 집게 위치에서 잡으면 성공 확률 (Grasp Q맵 활용)
- 어떻게 움직이면 좋은지 피드백
- 결과 영상 QR 로 사용자에게 제공

### Q2. AI 두 개 쓰는 이유는?
A. 역할이 다름. **Grasp AI** = 어디 잡으면 잘 잡힐지 (Q맵). **Detection AI** = 무엇이/어디 있는지 (bbox+클래스). 합쳐야 "곰돌이 인형 70%, 토끼 40%" 같은 사용자 분석 정보 만들 수 있음.

### Q3. 확률 계산은 AI 가 해요 우리가 짜요?
A. 둘 다 답이 됨. 지금은 **AI 가 무거운 일 다 했고**(Q맵 학습), 우리는 그 출력에서 집게 위치 픽셀 값 꺼내는 5줄 코드. 알고리즘 짜는 게 아니라 **AI 출력을 사용자에게 보여주는 변환**. 데이터 모이면 별도 AI 로 업그레이드 가능 (성공/실패 시연 수백 번 수집 후).

---

## 모델 선정 — Grasp AI

### Q4. GraspNet 안 쓴 이유?
A. 6근거:
1. 라이선스 CC BY-NC-SA (상업 X)
2. GPU 에서도 157ms 느림
3. 6 DOF 출력 (우리는 3 DOF 만 필요 — 과잉)
4. collision detection 80ms 구조적 병목
5. soft toy 도메인 약함
6. **결정타: pointnet2 가 CUDA-only 라 EC2 (GPU 없음) 에서 실행 자체 불가**

### Q5. 왜 GR-ConvNet?
A. BSD-3 라이선스 ✅, 순수 PyTorch (CUDA 확장 0개) → CPU 동작, GPU 3ms / EC2 4-thread 21ms. 동봉된 4개 pretrained 비교 → cornell-ch16 채택.

### Q6. cornell-ch16 vs ch32 차이?
A. 같은 학습 데이터, 다른 사이즈. ch16 = 479K params, ch32 = 1.9M params. **시각화 결과 거의 동일** (같은 좌표 잡음, 2px 차이). CPU 4t 에선 ch16 이 **2.7배 빠름** → ch16 채택.

### Q7. Cornell 이 모델인가요?
A. **아니, 데이터셋 이름**. 코넬대학교가 만든 grasp 학습용 사진 모음 (885장). 모델 = GR-ConvNet, 학습 데이터 = Cornell. 이걸 헷갈리면 다른 질문에서도 막힘.

### Q8. Jacquard 는 왜 안 썼어요?
A. Jacquard = 시뮬레이션 렌더링 데이터. Cornell = 실 카메라 데이터. 우리 환경 = RealSense 실 카메라 → Cornell 분포에 가까움. Jacquard 모델 시각화 결과 hot spot 위치도 엉뚱하고 박스 크기 비현실적 → sim-to-real 갭 → 탈락.

### Q9. Cornell IoU 0.97 가 무슨 뜻?
A. 학습 시 발표된 모델 정확도 (intersection over union). 우리가 다운로드 받은 Cornell 데이터셋으로 직접 측정해서 86/89 = 0.966 — **학습 시 수치 정확히 재현** = 모델 정상 동작. 우리 환경에서 모델 깨지지 않았음 확정.

---

## 모델 선정 — Detection AI

### Q10. YOLO 안 써요? 더 인기 있던데
A. YOLOv5/v8/v11/v26 (Ultralytics) 다 **AGPL-3.0** 라이선스. 상업 SaaS 운영 X. 우리 SSAFY 자율과제 + 오픈소스 프레임워크라 무료 + 상업 가능 라이선스 강제 → 탈락. (Enterprise 라이선스 별도 구매하면 OK 지만 비용 발생)

### Q11. 그럼 왜 SSDlite?
A. 17개 모델 측정 중 **라이선스 OK + EC2 운영 가능 + 가장 빠름** 조합 유일. 
- 라이선스 BSD-3 (torchvision)
- CPU 4t **37ms** (AGPL YOLO 들보다도 빠름)
- params 3.4M
- torchvision 기본 패키지 (의존성 0)
- COCO 80 클래스 (`teddy bear` 포함)

### Q12. RT-DETR 도 Apache 인데 왜 안 썼어요?
A. 측정해봤는데 CPU 4t 244~372ms — **EC2 200ms 한계 초과**. Transformer 라 무거움. 라이선스 OK 인데 latency X. SSDlite 가 더 빠름.

### Q13. SSDlite 의 정확도 떨어지는 거 아니에요?
A. 도메인 fine-tune 전엔 정확도 비교 의미 작음 (Grasp AI 와 같은 결론). example_data 에서 false positive 많은 것도 이 때문. 인형뽑기 박스 데이터로 fine-tune 후 정확도 확보. **지금 단계 = "라이선스 OK + 운영 가능 + fine-tune 가능" 후보 확보** 가 목표였음.

---

## 통합 / 시스템

### Q14. 두 AI 어떻게 합쳐요?
A. `pipeline.py` 가 wrapper:
```
RGB-D → SSDlite (bbox) → workspace_mask → GR-ConvNet (mask 영역만) → grasp 후보
```
Detection 이 "여기 인형 있다" 영역 한정 → Grasp 가 그 안에서만 잡기 위치 검색. 박스 벽/배경 false positive 제거.

### Q15. 통합 latency 얼마예요?
A. EC2 시뮬 (CPU 4-thread) 기준:
- Detection 97.8ms + Grasp 11.9ms = **109.7ms (9 FPS)**
- 목표 200ms 한계 대비 **1.8배 마진** — 운영 가능

### Q16. EC2 가 정확히 뭐예요?
A. AWS 클라우드 가상 서버. 우리 SSAFY EC2 = 4vCPU, 16GB RAM, **GPU 없음** = 우리 서비스 실제 운영 환경. H200 같은 GPU 서버는 실험용이지 배포 X. 그래서 CPU 동작 + EC2 한계 안 = 절대 조건.

### Q17. 메인 레포 (Spring Boot) 는 어떻게 호출해요?
A. 두 가지:
1. **HTTP 서버 모드 (권장)**: `infer_server.py` (Grasp), `detect_server.py` (Detection) 백그라운드 실행 → Spring Boot 가 POST 요청
2. CLI subprocess: 매 호출 cold start 200~600ms — 운영 X. 디버깅용만.

### Q18. 메인 레포 코드는 본인이 짰어요?
A. (솔직히) "AI 모델 채택 + 통합 wrapper + 운영 인터페이스" 까지 본인 트랙. 메인 레포 (jipchak) 의 Spring Boot/React 부분은 다른 팀원 트랙. 두 레포는 HTTP 로 통신.

---

## 한계 / 단서 (코치가 약점 물을 때)

### Q19. 모델 정확도 어떻게 보장해요?
A. **현재 잠정 채택 상태**. 검증한 건 모델이 학습된 환경 (Cornell/COCO) 에서 정상 동작 한다는 것. **인형뽑기 도메인 정확도는 아직 미검증**. 이유: RealSense D405 카메라 재구매 대기. 도착하면 인형 데이터 수집 → fine-tune → 진짜 검증.

### Q20. Antipodal grasp 이 뭐예요?
A. 두 손가락 (parallel-jaw 그리퍼) 으로 양쪽에서 집는 방식. GR-ConvNet 학습 가정. 우리 인형뽑기 크레인 = 3-finger 위에서 수직 하강 → mismatch 있음. 출력 4채널 중 (x, y, Q) 만 추출해 우회. 길쭉한 물체에서 정확도 약간 떨어질 수 있음.

### Q21. 무게중심을 잡지 않아도 돼요?
A. 모델은 "잡기 쉬워 보이는 위치" 학습 (좁은 평행면) — **무게중심과 다를 수 있음**. 인형 머리만 잡으면 들다 떨어짐. 진짜 정확도 = 인형뽑기 시연 데이터 (성공/실패) 로 fine-tune 후. 사전학습 가중치는 시작점.

### Q22. EC2 실측 안 했죠?
A. 안 함. 측정한 CPU 4-thread 는 H200 서버 CPU 를 4개로 제한한 시뮬레이션. EC2 vCPU 마이크로아키텍처 차이로 실제 latency 다를 수 있음. 50% 더 느려도 30~150ms 라 여유 큼. 후속 작업.

---

## 오픈소스 / 프레임워크 측면

### Q23. 인형뽑기인데 왜 "프레임워크" 라고 해요?
A. 슬로건 "**인형뽑기로 시작하는 집게 × AI 오픈소스 프레임워크**". 우리 차별화 = 인형뽑기 데모 단발 X, **다른 사람이 자기 모델 / 하드웨어 끼워서 쓸 수 있는 표준 인터페이스**. 인형뽑기는 first use case 일 뿐.

### Q24. 모델 swap 가능해요?
A. 표준 인터페이스 (`GraspAdapter` ABC, `RGBDInput`, `Grasp3DoF` 표준 타입) 설계 단계. 다음 트랙. 완성되면 다른 모델 (YOLOX, GG-CNN 등) 한 파일 어댑터로 swap 가능.

### Q25. 라이선스 왜 그렇게 신경 써요?
A. 오픈소스 프레임워크 = 다른 사람이 쓸 수 있어야 함. AGPL 모델 끼면 그 프레임워크 쓰는 모든 사람 코드도 AGPL 강제 → 사실상 못 쓰는 도구. BSD/Apache 만 채택해야 다양한 사용자 (스타트업/대기업 등) 가 끼워 쓸 수 있음.

---

## 다음 단계

### Q26. 다음 뭐 해요?
A. RealSense D405 재구매 → 인형 박스 RGB-D 수집 → 레이블링 → 두 AI fine-tune → 메인 레포 통합.

### Q27. RealSense 가 뭔데 그렇게 중요해요?
A. Intel 의 RGB-D 카메라. 일반 카메라(RGB) + depth (거리) 동시 출력. 인형뽑기 박스 위에서 인형까지 거리 (z 좌표) 파악 필수. 우리 모델 둘 다 RGB-D 입력. RealSense 없이 데이터 수집 불가능.

### Q28. fine-tune 이 정확히 뭐예요?
A. 남이 학습한 가중치 (Cornell/COCO) 를 시작점으로, 우리 인형뽑기 데이터로 추가 학습 = **도메인 적응**. from-scratch 학습보다 데이터 적게 들고 빠름. ~500~2000장 + 라벨 + 30 epoch 정도.

### Q29. 데이터 몇 장 필요해요?
A. fine-tune 기준:
- 50장 → 약간 적응 (위험: overfit)
- 200~500장 → 의미 있는 적응
- 1000+장 → 안정적
- 10000+장 → from-scratch 도 가능

목표 500장 + augmentation (회전/줌 등) → 실효 5000+장.

### Q30. 인형뽑기 PoC 언제 보여줄 수 있어요?
A. RealSense 도착 → 데이터 수집 (~1주) → 레이블링 (~3일) → fine-tune (~1일) → 통합 테스트 (~3일). **카메라 도착 후 ~2주** 추정.

---

## 자기 점검 — 막힌 질문 있으면

위 30개 중 **본인이 말로 못 답하는 게 있으면** 그게 본인 학습 갭. 그것만 깊게 보면 됨. 외우지 말고 이해.
