# 모델 선정 결정문 (잠정)

**Last updated**: 2026-04-27
**Branch**: `test/GR-convnet`
**Status**: 잠정 채택 — 인형뽑기 도메인 검증 후 확정

## 결론

**Cornell GR-ConvNet ch16 (`epoch_30_iou_0.97`) 잠정 채택.**

근거: 라이선스/CPU 동작/latency/시각화 품질 4가지를 동시에 통과하는 유일한 후보. 동일 입력에 ch32와 사실상 동일한 grasp 위치를 짚으면서 CPU 4-thread 환경에서 2.7배 빠름.

## 1. 후보 비교표

| 모델 | 라이선스 | GPU total | CPU 4t total | DOF | CPU 동작 | 도메인 적합 |
|---|---|---|---|---|---|---|
| **GraspNet baseline** | CC BY-NC-SA | 157.5ms | **불가** | 6 (과잉) | ❌ pointnet2 CUDA-only | 일반 사물 |
| GR-ConvNet jacquard rgbd | BSD-3 | (not measured) | (not measured) | 4 | ✅ | ❌ 시뮬 학습, 실세계 일반화 약함 |
| GR-ConvNet jacquard d | BSD-3 | (not measured) | (not measured) | 4 | ✅ | ❌ 시뮬 학습, 박스 크기 비현실적 |
| GR-ConvNet cornell ch32 | BSD-3 | 3.17ms | 55.0ms | 4 | ✅ | ✅ 실 카메라 학습 |
| **GR-ConvNet cornell ch16** | **BSD-3** | **3.16ms** | **20.7ms** | **4** | **✅** | **✅ 실 카메라 학습** |

(모든 측정: H200 NVL 서버, GraspNet baseline 의 `doc/example_data/color.png` + `depth.png` 동일 입력)

## 2. 탈락 사유

### GraspNet baseline (6근거)
1. 라이선스 CC BY-NC-SA — 상업화 불가
2. Latency 157.5ms (H200) — 느림
3. 6 DOF 출력, 우리는 3 DOF 만 필요 → 과잉
4. NUM_POINT 2k~40k 스윕해도 collision detection 80ms 병목 (구조적)
5. Rigid object 강점, soft toy/곡선 약함 (도메인 갭)
6. **CPU 추론 구조적 불가** — `pointnet2.furthest_point_sampling` 등이 CUDA-only C++ 확장. EC2(GPU 없음) 배포 자체 불가능

### GR-ConvNet Jacquard pretrained
- Jacquard 데이터셋 = 시뮬레이션 렌더링 학습. 실 카메라 분포(Cornell, RealSense)와 분포 어긋남
- GraspNet `example_data` 입력 시:
  - Q맵 hot spot 위치가 cornell 과 다름 (컵 외곽/엉뚱한 영역)
  - 박스 크기 비현실적으로 작음 (width 학습 분포 차이)
- 우리 환경 = RealSense D405 실 카메라 → cornell pretrained 가 sim-to-real 갭 없음

## 3. 채택 사유 (cornell ch16)

### 정량
- BSD-3 → 상업화 OK
- 순수 PyTorch, CUDA 확장 0개 → CPU 동작 검증됨
- CPU 4t latency 20.7ms → 48 FPS (200ms 목표 대비 10배 마진)
- Params 479K (1.9MB FP32) → EC2 16GB 메모리 무시 가능
- ch32 대비 CPU 2.7배 빠름, 메모리 4배 작음

### 정성
- example_data 시각화에서 둥근 컵 영역에 Q맵 hot spot 정확히 짚음
- ch16 vs ch32 동일 입력에 동일 grasp center (2px 차이, 1.5cm depth 차이) → **두 사이즈 모두 같은 패턴 학습** 확인
- z 좌표 (RealSense D405 depth 사용 가정) 의미 있는 값 추출됨 (컵 위 0.41m, 둘레 0.45-0.48m)

## 4. 단서 / 한계 (확정 아닌 이유)

### 검증 부족
- **n=1**: GraspNet `example_data` 1장만 검증. 일반화 보장 X. → cornell dataset 일부 또는 인형 도메인 RGB-D 로 추가 검증 필요
- **인형 도메인 미검증**: 봉제 인형 RGB-D 0장 테스트. 분포 갭 가능성 — RealSense 셋업 후 별도 검증
- **다양한 scene 미검증**: cluttered 환경(여러 물체+박스 벽+그림자) 일반화 정확도 미측정

### 그리퍼 mismatch
- GR-ConvNet 학습 가정: **parallel-jaw 2-finger antipodal**
- 우리 시스템: **3-finger 수직 하강 (3 DOF)**
- 우회: 출력 4채널 (pos, cos, sin, width) 중 (x, y, Q) 만 추출, angle/width 무시 → "잡기 좋은 영역 heatmap" 으로 재해석
- 이 우회가 **antipodal hot spot ≠ 3-finger center** 인 케이스(예: 길쭉한 물체)에서 정확도 저하 가능성. 도메인에 따라 영향 다름
- 진짜 정확도 = **인형뽑기 시연 데이터로 fine-tune** 후. 사전학습 가중치는 시작점.

### EC2 실측 안 함
- 측정한 CPU 4-thread 는 H200 서버의 96-thread CPU 를 4개로 제한한 시뮬레이션
- SSAFY EC2 4vCPU 의 마이크로아키텍처 차이로 실 latency 다를 수 있음 (50% 더 느려도 30ms 이내라 여유는 큼)

## 5. 후속 작업

| 우선 | 작업 | 비고 |
|---|---|---|
| 1 | RealSense D405 셋업 + 인형 RGB-D 데이터 수집 | 도메인 검증 + fine-tune 데이터 |
| 2 | EC2 실측 (cornell ch16) | 마이크로아키텍처 차이 확인 |
| 3 | 3-finger center grasp 후처리 모듈 (`utils/grasp_to_3dof.py` 류) | (x, y, Q) → (x, y, z, score) 변환, NMS, workspace 필터 |
| 4 | Fine-tune 파이프라인 점검 (`gr-convnet/train_network.py` 분석) | 인형 데이터 학습 가능성 검증 |
| 5 | 다양한 cornell/일반 scene 5~10장 추가 검증 (선택) | n=1 → n=여러 |

## 6. 측정 로그 (재현 가능)

서버: SSAFY GPU jupyter08, conda env `jipchak`, PyTorch 2.3.0+cu121
입력: `graspnet-baseline/doc/example_data/{color,depth}.png` (1280×720 RealSense)

```bash
# Latency 측정 명령
cd ~/jipchak-ai-chaemok/gr-convnet

CKPT16=trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch16/epoch_30_iou_0.97
CKPT32=trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98

# GPU
python benchmark_grconv.py --network $CKPT16 --device cuda:0
python benchmark_grconv.py --network $CKPT32 --device cuda:0

# CPU 4-thread (EC2 시뮬)
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python benchmark_grconv.py --network $CKPT16 --device cpu --n_warmup 3 --n_runs 20
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python benchmark_grconv.py --network $CKPT32 --device cpu --n_warmup 3 --n_runs 20

# 시각화 (4 모델 비교)
python visualize_grid.py
# 시각화 (3-finger 관점, ch16 vs ch32)
python visualize_3finger.py
```

결과 파일: `gr-convnet/benchmark_grconv_*.txt`, `gr-convnet/results/grid_4models.png`, `gr-convnet/results/3finger_cornell_compare.png`
