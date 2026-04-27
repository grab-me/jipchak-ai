# Walkthrough - Three-Jaw Grasp Evaluation Pipeline

## 프로젝트 맥락

본 파이프라인은 **3발 집게를 장착한 인형뽑기 기계**에서 AI가 최적의 파지 위치를 실시간으로 가이드해주는 시스템의 핵심 모듈입니다.

- 기계 내부: 실시간 카메라(또는 스테레오 카메라 2대)로 모니터링
- 사용자: 조이스틱으로 X/Y 이동, AI가 최적 위치 가이드
- 집게: 3D 프린터 제작 3발 집게, Z축 하강/상승으로 물체 파지
- 확장 목표: 산업 크레인, 소각장 크레인 등으로 적용 범위 확대

---

## 구현된 파일 구조

```
three-jaw-grasp/
├── three_jaw_grasp/
│   ├── __init__.py          # 공개 API 정의
│   ├── candidate.py         # GraspCandidate 표준 포맷
│   ├── adapters.py          # 모델 포맷 변환기 (3종)
│   ├── feature_extractor.py # 평가용 특징 추출 (15차원)
│   ├── evaluator.py         # Rule-based / MLP 평가기
│   └── pipeline.py          # 전체 흐름 제어
├── train/
│   ├── collect_data.py      # 학습 데이터 수집
│   ├── dataset.py           # PyTorch 데이터셋
│   └── train.py             # MLP 학습 스크립트
├── config/
│   └── gripper_spec.yaml    # 집게 스펙 & 평가 가중치
├── examples/
│   └── graspgroup_format.py # 동작 예시
├── INTERFACE_CONTRACT.md    # 인터페이스 규약 (핵심 문서)
├── walkthrough.md           # 이 파일
└── requirements.txt
```

---

## 주요 구현 사항

### 1. 핵심 모듈 (`three_jaw_grasp/`)

- **`candidate.py`**: 어떤 외부 모델의 출력도 수용하는 표준 `GraspCandidate` 데이터 포맷.
- **`adapters.py`**: 모델별 고유 출력 포맷을 `GraspCandidate`로 변환하는 3가지 기본 어댑터.
  - `GraspGroupAdapter`: GraspNet 계열
  - `RectGraspAdapter`: GR-ConvNet 계열
  - `PoseArrayAdapter`: AnyGrasp 계열
- **`feature_extractor.py`**: 파지 후보의 기하학적 정보 + Depth 패치 정보로 15차원 특징 벡터 추출.  
  학습과 추론에서 동일하게 사용되어 입력 불일치 문제를 방지.
- **`evaluator.py`**: 두 가지 평가 모드 지원.
  - **Rule-based**: 파지 폭, 신뢰도, 높이, 수직 안정성, 3발 120° 대칭 적합성 가중합
  - **MLP-based**: 학습된 신경망으로 파지 성공 확률 예측
- **`pipeline.py`**: 외부 모델 추론 → 어댑터 변환 → 평가 → 최적 파지 반환의 전체 흐름 조율.

### 2. 학습 파이프라인 (`train/`)

- **`dataset.py`**: PyTorch `Dataset` 기반 학습 데이터 로더.
- **`train.py`**: Binary Cross-Entropy 기반 MLP 학습, 가중치 `.pth` 저장.
- **`collect_data.py`**: 실험/시뮬레이션 결과를 특징 벡터로 변환하여 `npz` 저장.

### 3. 설정 시스템 (`config/`)

- **`gripper_spec.yaml`**: 집게 물리 스펙(min/max/ideal 폭, 충돌 임계값)과 평가 가중치를 코드 수정 없이 조정 가능.

---

## 동작 테스트 결과

`examples/graspgroup_format.py` 실행 결과:

```bash
--- Best Grasp for Three-Jaw Gripper ---
Center: (100, 100, 0.05)   ← 집게 이동 목표 좌표
Width:  0.04               ← 집게 벌림 폭 (4cm)
Angle:  0.0                ← 집게 회전 각도 (라디안)
Original Score: 0.9        ← 원본 탐지 모델 신뢰도
```

Rule-based 평가기가 바닥 충돌 위험(center_z < 0.02m)이 있는 후보를 올바르게 제외하고,  
신뢰도와 집게 폭 적합성이 높은 후보를 선택하는 것을 확인.

---

## 인터페이스 규약 검토 결과

| 항목 | 상태 | 비고 |
|---|---|---|
| 사용자 모델 인터페이스 규약 | ✅ 완료 | `predict(rgb, depth)` 단일 규약 |
| 어댑터 설명 및 예시 | ✅ 완료 | 3종 기본 + 커스텀 가이드 |
| GraspCandidate 좌표계 주의사항 | ✅ 완료 (수정됨) | 어댑터 구현 시 단위 명시 권장 추가 |
| 파생 파이프라인 코드 오류 | ✅ 수정됨 | `_get_candidates` 미구현 메서드 제거 |
| 산업현장 확장 예시 | ✅ 추가됨 | 크레인 파이프라인 예시 추가 |
| 스테레오 비전(일반 카메라 2대) | ✅ 추가됨 | OpenCV 기반 구현 절차 및 가이드 |
| 프로젝트 맥락(인형뽑기) | ✅ 추가됨 | Section 1-1에 명시 |

---

## 향후 할 일

### 오늘 (베이스 파이프라인 마무리)
- [ ] `evaluator.py`의 `_stability` 메서드 실제 수직 접근 각도 로직 구현
- [ ] `adapters.py` 각 어댑터 클래스에 docstring으로 좌표계/단위 명시
- [ ] 유닛 테스트 코드 작성 (Adapter → Extractor → Evaluator 데이터 흐름 검증)

### 다음 단계
- [ ] 스테레오 카메라 캘리브레이션 코드 작성 (`utils/stereo_calibration.py`)
- [ ] 실시간 카메라 스트림 연동 (`utils/camera_stream.py`)
- [ ] 실제 집게 스펙(폭, 높이 임계값)을 `config/gripper_spec.yaml`에 측정 후 입력
- [ ] 인형뽑기 기계 내부에서 테스트 이미지 수집 → MLP 학습용 데이터 구축
