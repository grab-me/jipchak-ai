# Three-Jaw-Grasp 파이프라인 개발 워크플로우 히스토리 (Chronological Workflow)

본 문서는 인형뽑기 기계(3발 집게) 전용 추론 및 평가 파이프라인의 설계부터 모듈화, 리팩토링까지의 모든 과정을 시간 순서대로 기록한 개발 연대기입니다.

---

## 📅 페이즈 1: 기반 아키텍처 및 핵심 자료구조 확립 (Core Framework Design)
가장 먼저 어떤 외부 모델이 들어오더라도 파이프라인이 깨지지 않도록 "예측(Predict) -> 변환(Adapt) -> 평가(Evaluate)" 라는 절대적인 3단계 규약을 설계했습니다.

1. **규약서 작성 (`INTERFACE_CONTRACT.md`)**
   - 파이프라인이 모델과 어떻게 소통해야 하는지 규정하는 인터페이스 명세서 작성
2. **단일 자료구조 설계 (`three_jaw_grasp/candidate.py`)**
   - 수많은 외부 모델들의 결과 형태를 통일시킬 단일 객체 `GraspCandidate` (중심점 x,y,z / 폭 width / 각도 angle) 설계
3. **핵심 도메인 로직 설계 (`three_jaw_grasp/pipeline.py`)**
   - 모델 추론, 어댑터 변환, 평가 함수를 순서대로 호출하는 핵심 뼈대 `GraspPipeline` 클래스 구현

---

## 📅 페이즈 2: 설정 기반의 집게 평가 시스템 구축 (Configuration & Evaluator)
우리가 직접 3D 프린터로 뽑은 3발 집게의 물리적 특성을 코드에 반영하고 점수를 계산하는 로직을 구축했습니다.

4. **하드웨어 제원 파일 생성 (`config/gripper_spec.yaml`)**
   - 실제 3발 집게의 미터 단위 크기(최대 개방 폭, 집게 길이 등)를 변수로 분리하여 언제든 교체 가능하게 설정
5. **평가 알고리즘 구현 (`three_jaw_grasp/evaluator.py`)**
   - YAML 설정을 읽어서 들어온 후보 좌표 중 **안정성(Stability), 깊이(Depth), 넓이(Width), 원본 신뢰도 점수**를 기반으로 100점 만점의 가중치 합산(Score)을 계산하는 `ThreeJawEvaluator` 및 `score_detail()` 구현

---

## 📅 페이즈 3: 외부 모델 변환 어댑터 및 시각화 초기 테스트 (Early Integration)
만들어진 코어 파이프라인이 정상 동작하는지 테스트용 딥러닝 모델들을 붙여보기 시작했습니다.

6. **기본 어댑터 설계 (`three_jaw_grasp/adapters.py`)**
   - 2D GR-ConvNet 사각형(Rectangle)을 3발 집게 데이터로 파싱하는 `BaseGraspAdapter`, `RectGraspAdapter` 구현
7. **GR-ConvNet 연결 및 파편화 테스트 스크립트 작성 (`examples/test_detection.py`, `visualize_grasp.py`)**
   - Cornell 데이터셋과 실제 GR-ConvNet 가중치를 로드하여 테스트. 이때 최초로 Matplotlib을 이용해 3개의 선과 1개의 원형으로 표현된 **[3발 집게 시각화]** 로직이 탄생함

---

## 📅 페이즈 4: GraspNet(3D 모델) 도입 시도 및 블로커(Blocker) 해결
인형뽑기는 3D 환경이므로 3D 범용 파지 알고리즘인 GraspNet을 고려 및 시도했습니다.

8. **GraspNet 오픈소스 클론 및 3D 프로젝션 변환 시도 (`examples/test_graspnet.py`)**
   - `graspnetAPI` 및 `graspnet-baseline` 오픈소스 연동
   - 기존의 2D 픽셀과 달리 카메라 파라미터(Intrinsics)를 기반으로 3D 미터 좌표를 2D 이미지로 쏘아보내는 수학적 프로젝션 계층 설계
9. **C++ 핵심 컴파일 에러 봉착 및 우회 방안 모색**
   - Windows 환경에서 PointNet++ 커스텀 CUDA C++ 모듈 빌드 에러 이슈 감지 및 하드웨어적 한계 봉착
   - AI 서버 없이 빠르게 진행할 수 있는 **옵션 C (YOLO Mock)** 로의 선회 결정

---

## 📅 페이즈 5: 최종 정답(GT) 기반의 YOLO 시뮬레이션 평가 (Option C)
어차피 최종 운영은 YOLOv8 등 2D 모델을 활용할 예정이었으므로, Ground Truth(정답값)를 YOLO의 출력이라 가정하고 시스템 무결성을 테스트했습니다.

10. **YOLO 전용 어댑터 추가 및 모사 테스트 (`examples/test_yolo_mock.py`)**
    - Cornell 데이터셋 원본의 `cpos.txt` 바운딩 박스 4점을 읽고 중심과 각도를 파싱하는 `YoloGraspAdapter` 작성
    - **가장 무거운 딥러닝 모듈 없이 오직 평가 파이프라인의 안전성과 시각화 기능이 모두 완벽하게 100% 정상 작동**하는 것을 검증 성공
11. **불필요한 리소스 통제 (`.gitignore`)**
    - 용량이 큰 `dataset`, `.pth` 등 추후 깃허브 오염 방지를 위해 차단

---

## 📅 페이즈 6: 대규모 단일화 모듈 및 리팩토링의 달성 (Refactoring & Unification)
파편화되어 있던 시각화와 실행 코드들을 **"입출력이 다를 뿐 시각화 로직은 같다"**는 개념하에 하나의 완전체로 조립했습니다.

12. **통합 시각화 모듈 설립 (`three_jaw_grasp/visualizer.py`)**
    - 모든 개별 스크립트에 흩어진 그리기 코드를 모아, 2D 입력과 3D 입력(is_3d) 모두를 소화하는 마스터 함수 생성
13. **껍데기(Wrappers) 객체 격리 (`examples/wrappers.py`)**
    - GR-ConvNet, GraspNet Mock, Yolo Mock 이라는 모델 로딩 로직을 별도로 빼냄으로써 메인 실행 파일을 깨끗하게 정리
14. **최종 통합 터미널 어플리케이션 구축 (`examples/run_pipeline.py`)**
    - 기존의 4개 코드를 모두 삭제 폐기하고 1개의 `run_pipeline.py`로 대체
    - `--model yolo`, `--model grconvnet` 등 명령어 하나만으로도 완벽한 통합 실행을 가능하도록 완성

---

## 📅 페이즈 7: (향후 과제) 진정한 오픈소스 패턴으로의 진화 (Plugin/Registry)
향후 `GKNet`이나 다양한 사용자 고유 무거운 모델들의 **GPL 라이선스 오염을 피하고 독립적인 유연성 보장**을 위해 예정된 넥스트 스텝입니다.

15. **If 문 제거 및 동적 로딩 (Dynamic Import)** (예정)
    - `run_pipeline.py` 원본 코드 안의 모델 의존성을 0%로 만들고, 오직 파이썬 내부 `importlib`을 활용한 **자료구조 기반 동적 팩토리 패턴(Factory Pattern)** 적용 대기. 
    - 사용자가 밖에서 작성한 파일 경로만 파이프라인에 주면 USB처럼 바로 꽂히는 궁극적 플랫폼으로 고도화될 예정입니다.
