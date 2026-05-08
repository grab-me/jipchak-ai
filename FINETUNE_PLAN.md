# Fine-tune 계획 (GR-ConvNet → 인형뽑기 도메인)

**전제**: cornell ch16 잠정 채택. 사전학습 가중치를 시작점으로 인형 데이터로 fine-tune 해 도메인/3-finger 갭을 메운다.

## 1. GR-ConvNet 학습 구조 분석 (`train_network.py`)

### 입력 / 출력
- 입력: `(B, C, 224, 224)`, C = `1*use_depth + 3*use_rgb` (default 4 = RGB-D)
- 출력 4채널: `pos / cos(2θ) / sin(2θ) / width` 모두 224×224 맵

### Loss (`inference/models/grasp_model.py::compute_loss`)
```python
loss = smooth_l1(pos) + smooth_l1(cos) + smooth_l1(sin) + smooth_l1(width)
```
4채널 모두 동일 가중치. **우리는 pos(Q) 만 사용** → 가중치 재조정 또는 cos/sin/width loss 제거 필요.

### 학습 루프
- Optimizer: Adam (default LR 1e-3, 별도 hparam X)
- Batch size 8, epochs 50, **batches_per_epoch 1000** — 작은 데이터셋도 1000 step 강제
- Augmentation: random rotate (0°/90°/180°/270°), random zoom (0.5~1.0)
- Dropout 0.1
- Validation: IoU > 0.25 매칭률 측정 → `epoch_XX_iou_0.XX` 파일명으로 저장 (best 갱신 시 + 매 10 epoch)

### 미지원 기능
- **사전학습 가중치 로드 X** — 모델 from-scratch 만. fine-tune 하려면 코드 patch 필요 (5줄):
  ```python
  if args.pretrained:
      net = torch.load(args.pretrained, map_location=device)
  else:
      net = network(input_channels=..., dropout=..., ...)
  ```
- 학습률 스케줄러 X
- 조기 종료 X

## 2. Cornell 데이터 포맷 (`utils/data/cornell_data.py`)

각 샘플 3개 파일:
| 파일 | 내용 |
|---|---|
| `pcd*cpos.txt` | grasp 레이블 (4점 사각형, 8숫자 4줄, antipodal box) |
| `pcd*d.tiff` | depth 16-bit |
| `pcd*r.png` | RGB |

**`cpos.txt` 한 grasp = 4줄 4점**:
```
x1 y1
x2 y2
x3 y3
x4 y4
(빈 줄)
다음 grasp...
```

학습 시 `GraspRectangles.draw()` 가 4점 박스 → pos/cos/sin/width 4 맵으로 렌더링. **레이블이 antipodal 박스**라는 가정.

## 3. 인형뽑기 fine-tune 로드맵

### 단계 1 — 데이터 수집 (RealSense 셋업 후)
- 박스 위에서 봉제 인형 RGB-D 캡처 (~500~2000장 목표)
- 다양한 인형 / 위치 / 조명 / 박스 안 적재 상태
- factor_depth 등 RealSense 캡처 정보 메타로 함께 저장
- 자동 캡처 스크립트 (`tools/capture_realsense.py` 같은 거 — RealSense 셋업 후 작성)

### 단계 2 — 레이블링
**옵션 A: Cornell 포맷에 맞춰 가짜 antipodal 박스 (권장, 가벼움)**
- 사람이 인형 중심 점 (x, y) 만 클릭
- 자동으로 표준 박스 생성: 중심 (x, y), width=60px, length=30px, angle=0
- `cpos.txt` 4점 좌표로 저장
- 장점: 기존 학습 코드 그대로 (CornellDataset 재사용)
- 단점: cos/sin/width 채널은 무의미한 신호 학습 (모두 동일) — 그러나 우리는 안 씀

**옵션 B: 새 데이터셋 클래스 + Q맵만 학습 (정석, 무거움)**
- `JipchakDataset` 작성 — `(rgb, depth, q_target_map)` 만 반환
- compute_loss override — `loss = p_loss` 만
- 장점: 깔끔, 의미 있는 학습
- 단점: 학습 코드 1~2일 작업

→ **옵션 A 로 시작**. PoC 끝나면 필요 시 B 로 마이그레이션.

### 단계 3 — 학습 코드 patch (~10줄)
1. `train_network.py` 에 `--pretrained` 옵션 추가:
   ```python
   parser.add_argument('--pretrained', type=str, default='',
                       help='Path to pretrained network for fine-tuning')
   ...
   if args.pretrained:
       net = torch.load(args.pretrained, map_location=device)
       logging.info(f'Loaded pretrained from {args.pretrained}')
   else:
       net = network(input_channels=..., dropout=..., prob=..., channel_size=...)
   ```
2. 선택: Loss 가중치 조정 (pos 강조)
   - `compute_loss()` 를 상속한 `GraspModel` 변종 또는 train.py 안에서 직접 weighting

### 단계 4 — 학습 실행
```bash
python train_network.py \
  --dataset cornell \
  --dataset-path /path/to/jipchak-data \
  --pretrained trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch16/epoch_30_iou_0.97 \
  --channel-size 16 \
  --description jipchak-ch16 \
  --epochs 30 \
  --batch-size 8 \
  --use-depth 1 --use-rgb 1
```
- `channel-size 16` = ch16 사이즈 (사전학습 모델과 일치 필수)
- 50 → 30 epochs (fine-tune 이라 적게)
- LR 별도 안 낮춰도 Adam 이 적응할 가능성. 필요하면 patch.

### 단계 5 — 평가
- IoU metric 은 가짜 박스라 의미 약함. 새 metric:
  - Q맵 peak 위치 vs 사람 클릭 (x, y) 거리 (px)
  - 거리 < threshold (예: 20px) 비율 = "정확도"
- `validate()` 함수도 patch 또는 별도 평가 스크립트

## 4. 데이터 양 가이드 (경험 기반)

| 데이터 양 | 기대 |
|---|---|
| ~50장 | 사전학습 거의 그대로 + 약간 도메인 적응. 위험: overfit |
| 200~500장 | 의미 있는 적응 시작. 인형뽑기 박스 환경 인식 |
| 1000+장 | 안정적 fine-tune. 다양한 인형/조명 일반화 |
| 10000+장 | from-scratch 도 가능 |

**현실적 목표**: 500장 + augmentation (rotate/zoom 기본) → 실효 5000+장 = OK

## 5. 위험 / 단서

- **사전학습 가중치 채널 사이즈 일치 필수**: ch16 pretrained → channel_size 16 으로만 fine-tune. ch32 와 호환 X.
- **학습률 너무 크면 사전학습 망가짐**: 처음엔 LR 1e-4 정도로 낮춰 시도 권장 (Adam default 1e-3 의 1/10)
- **Cornell 가짜 박스 옵션의 한계**: cos/sin/width 채널이 무의미한 학습 → 노이즈 입력으로 작용 가능. p_loss 가중치 10× 같이 조정 필요할 수 있음
- **3-finger center vs antipodal**: 데이터 모을 때 사람이 클릭하는 점이 진짜로 "잡기 좋은 위치"인지 검증 필요. 인형뽑기 시연 (실제 잡아보고 성공/실패) 으로 검증 데이터 보강 가능

## 6. 코드 변경 예상 작업량

| 작업 | 예상 시간 |
|---|---|
| `train_network.py` `--pretrained` patch | 10분 |
| 데이터 캡처 스크립트 (RealSense) | 1~2시간 (셋업 후) |
| 레이블링 도구 (클릭 → cpos.txt) | 1~2시간 |
| Loss 가중치 조정 / Q-only loss | 30분~1시간 |
| 평가 metric (peak 거리) | 30분 |
| 문서화 / 학습 실행 / 결과 분석 | 반나절 |
| **합계 (RealSense 셋업 제외)** | **1~2일** |

**RealSense 셋업 + 데이터 수집 + 레이블링 시간이 코드 작업의 5배 이상.** Fine-tune 은 코드보다 **데이터 작업이 본질**.

## 7. 다음 자연스러운 액션

지금 시점에선 코드 patch 까지만 미리 해둘 가치 X (RealSense 셋업 전엔 검증 불가). 우선순위:

1. **RealSense D405 셋업** (하드웨어) — fine-tune 의 enabling factor
2. 셋업되면 → 캡처 스크립트 → 데이터 수집 → 레이블링 → fine-tune 코드 patch → 학습 실행
3. 그 사이에 EC2 실측 / 다른 scene 검증 같은 평행 작업 가능
