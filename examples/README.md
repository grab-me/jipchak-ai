# examples

표준 input 샘플과 사용 예제 모음.

## 샘플 데이터

현재는 GraspNet baseline 예제 데이터(`graspnet-baseline/doc/example_data/`)를 공용 샘플로 사용한다. 1280×720 RGB-D + workspace_mask + camera intrinsics(meta.mat).

추후 `examples/data/` 디렉토리에 인형뽑기 도메인 샘플(봉제 인형 RGB-D)을 추가한다.

## 사용 예제 (예정)

- `quickstart.py` — 어댑터 1개 로드해서 단일 이미지 추론
- `compare_adapters.py` — 동일 입력에 어댑터 N개 돌려 latency/출력 비교
- `to_3finger.py` — antipodal 출력에서 3-finger center grasp 후처리
