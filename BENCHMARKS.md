# 벤치마크 결과 모음

모델 후보 비교용 latency / 메모리 측정. 결정 사유는 [MODEL_DECISION.md](MODEL_DECISION.md) 참고.

## 측정 환경

| 항목 | 값 |
|---|---|
| 서버 | SSAFY GPU jupyter08 |
| GPU | NVIDIA H200 NVL (143GB VRAM) |
| CPU | 96 threads (4 thread 제한 시 EC2 4vCPU 시뮬) |
| Conda env | `jipchak` |
| PyTorch | 2.3.0+cu121 |
| CUDA build | 12.1 |

## 입력 데이터 (모든 측정 공통)

`graspnet-baseline/doc/example_data/` — GraspNet baseline 데모용 cluttered scene
- `color.png` 1280×720 RGB (RealSense)
- `depth.png` 1280×720 16-bit depth (factor_depth=1000 → m 단위)
- `meta.mat` intrinsics + factor_depth

## 1. GraspNet baseline

### 1-1. GPU (H200) total

기준 설정: `NUM_POINT=20000`, `NUM_VIEW=300`, `COLLISION_THRESH=0.01`, warmup 3 + runs 20.

| Stage | mean (ms) | std | min | max |
|---|---|---|---|---|
| forward | 66.80 | 0.10 | 66.62 | 67.10 |
| decode | 1.03 | 0.02 | 1.01 | 1.09 |
| collision | 89.29 | 1.75 | 88.58 | 96.86 |
| nms_sort | 0.38 | 0.01 | 0.37 | 0.42 |
| **total** | **157.50** | 1.74 | 156.66 | 165.02 |

- Throughput full pipeline: 6.3 FPS
- Throughput forward only: 15.0 FPS
- GPU peak memory: 0.72 GB allocated / 1.08 GB reserved
- Params: 1,025,964 (4.1 MB FP32)

→ **collision detection (89ms) 이 forward (67ms) 보다 더 큰 병목**. NUM_POINT 줄여도 collision 부분은 유지됨 (아래 sweep 참고).

raw: [graspnet-baseline/benchmark_gpu.txt](graspnet-baseline/benchmark_gpu.txt) / [benchmark_result.txt](graspnet-baseline/benchmark_result.txt)

### 1-2. GPU NUM_POINT sweep (구조적 병목 확인)

warmup 3 + runs 10 / setting

| NUM_POINT | fwd+dec (ms) | post (ms) | total (ms) | FPS | n_grasps |
|---|---|---|---|---|---|
| 2,000 | 25.19 | 60.81 | 86.00 | 11.6 | 86 |
| 5,000 | 31.30 | 78.57 | 109.87 | 9.1 | 113 |
| 10,000 | 42.72 | 82.54 | 125.26 | 8.0 | 114 |
| 20,000 | 67.14 | 83.42 | 150.57 | 6.6 | 114 |
| 40,000 | 110.03 | 83.50 | 193.53 | 5.2 | 110 |

→ post (collision + NMS, 주로 CPU) **80ms 부근에 고정**. forward 만 줄여도 전체 latency 한계 명확.

raw: [graspnet-baseline/sweep_result.txt](graspnet-baseline/sweep_result.txt)

### 1-3. CPU 실행 시도 → 불가

```
RuntimeError: CPU not supported
File ".../pointnet2/pointnet2_utils.py", line 71, in forward
    return _ext.furthest_point_sampling(xyz, npoint)
```

`pointnet2.furthest_point_sampling` 등 backbone op 들이 **CUDA-only C++ 확장**. ONNX/TorchScript 우회 불가, 재구현 외 방법 없음.

→ **GraspNet baseline 은 EC2(GPU 없음, 4vCPU/16GB) 에서 실행 자체 불가능.**

raw: [graspnet-baseline/benchmark_cpu.txt](graspnet-baseline/benchmark_cpu.txt)

## 2. GR-ConvNet (cornell pretrained)

기준 설정: `output_size=224` (cornell native), `use_rgb=1`, `use_depth=1`, top-K=10

### 2-1. ch32 (`epoch_19_iou_0.98`, params 1,900,900 / 7.6 MB)

| 환경 | forward (ms) | post (ms) | total (ms) | FPS |
|---|---|---|---|---|
| GPU H200 | 1.34 | 1.83 | 3.17 | 315 |
| CPU 96 thread | 19.22 | 2.17 | 21.39 | 47 |
| **CPU 4 thread (EC2 sim)** | **53.29** | **1.71** | **55.00** | **18** |

GPU peak memory: 0.03 GB allocated / 0.05 GB reserved

raw:
- [gr-convnet/benchmark_grconv.gpu.txt](gr-convnet/benchmark_grconv.gpu.txt)
- [gr-convnet/benchmark_grconv_cpu.txt](gr-convnet/benchmark_grconv_cpu.txt)
- [gr-convnet/benchmark_grconv_cpu_4t.txt](gr-convnet/benchmark_grconv_cpu_4t.txt)

### 2-2. ch16 (`epoch_30_iou_0.97`, params 479,156 / 1.9 MB)

| 환경 | forward (ms) | post (ms) | total (ms) | FPS |
|---|---|---|---|---|
| GPU H200 | 1.32 | 1.84 | 3.16 | 316 |
| CPU 96 thread | 9.43 | 1.90 | 11.33 | 88 |
| **CPU 4 thread (EC2 sim)** | **18.92** | **1.79** | **20.71** | **48** |

GPU peak memory: 0.01 GB allocated / 0.03 GB reserved

raw:
- [gr-convnet/benchmark_grconv_ch16_gpu.txt](gr-convnet/benchmark_grconv_ch16_gpu.txt)
- [gr-convnet/benchmark_grconv_ch16_cpu.txt](gr-convnet/benchmark_grconv_ch16_cpu.txt)
- [gr-convnet/benchmark_grconv_ch16_cpu_4t.txt](gr-convnet/benchmark_grconv_ch16_cpu_4t.txt)

### 2-3. ch16 vs ch32 차이

| | params | GPU total | CPU 96t total | CPU 4t total |
|---|---|---|---|---|
| ch16 | 479K | 3.16ms | 11.3ms | 20.7ms |
| ch32 | 1.9M | 3.17ms | 21.4ms | 55.0ms |
| ratio | 4× | ~동일 | 1.9× | **2.7×** |

→ GPU 에서는 H200 너무 빨라 차이 무시 가능. **EC2 시뮬에선 ch16 이 2.7배 우세**. 시각화 품질은 사실상 동등 (`results/3finger_cornell_compare.png` 참고).

## 3. 최종 비교 (전체 후보)

| 모델 | 라이선스 | GPU H200 | CPU 4t | DOF | CPU 동작 |
|---|---|---|---|---|---|
| GraspNet baseline | CC BY-NC-SA | 157.5ms | 불가 | 6 | ❌ |
| GR-ConvNet jacquard rgbd | BSD-3 | (생략) | (생략) | 4 | ✅ but 도메인 부적합 |
| GR-ConvNet jacquard d | BSD-3 | (생략) | (생략) | 4 | ✅ but 도메인 부적합 |
| GR-ConvNet cornell ch32 | BSD-3 | 3.17ms | 55.0ms | 4 | ✅ |
| **GR-ConvNet cornell ch16** | **BSD-3** | **3.16ms** | **20.7ms** | **4** | **✅** |

200ms 목표 대비 마진:
- GraspNet GPU: 0.78× (못 미침, 더구나 EC2 GPU 없음)
- GR-ConvNet ch16 CPU 4t: **9.7× 마진** (20.7 / 200)

## 4. 정성 결과 시각화

`gr-convnet/results/` 디렉토리 (서버):
- `combined_cornell_ch32.png` — RGB + Q맵 + grasp 박스 합성 (cornell ch32 단일)
- `grid_4models.png` — 4 pretrained 동시 비교 2×2 (cornell ch16/ch32 + jacquard d/rgbd)
- `3finger_cornell_compare.png` — 3-finger 관점 (antipodal 박스 무시, 점만), cornell ch16 vs ch32

핵심 정성 결론:
- **cornell ch16/ch32 동일 grasp 위치 짚음** — pixel 좌표 (634, 283) vs (635, 286), z 0.413m vs 0.428m. 사실상 동일.
- **jacquard 두 모델 부적합** — Q맵 hot spot 위치 cornell 과 다름, antipodal 박스 비현실적으로 작음. 시뮬-실세계 갭.
- ch16 후보 다양 (Q 0.43~0.86, n=5), ch32 더 신중 (Q 0.55~0.80, n=4).

## 5. 재현 명령

```bash
cd ~/jipchak-ai-chaemok/gr-convnet
conda activate jipchak

CKPT16=trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch16/epoch_30_iou_0.97
CKPT32=trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98

# === GR-ConvNet latency ===
python benchmark_grconv.py --network $CKPT16 --device cuda:0
python benchmark_grconv.py --network $CKPT32 --device cuda:0

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python benchmark_grconv.py --network $CKPT16 --device cpu --n_warmup 3 --n_runs 20
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python benchmark_grconv.py --network $CKPT32 --device cpu --n_warmup 3 --n_runs 20

# === 시각화 ===
python visualize_grid.py
python visualize_3finger.py

# === GraspNet (참고용) ===
cd ../graspnet-baseline
python benchmark.py --device cuda:0          # 정상 동작 (157ms)
python benchmark.py --device cpu             # RuntimeError 확인용
python benchmark_sweep.py                     # NUM_POINT 스윕
```
