import argparse
import sys
import os

# 터미널 출력 시 한글 깨짐 방지
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from three_jaw_grasp.factory import TrainerFactory

# 핵심 파이프라인 학습기 로드
import train.train_mlp


def main():
    parser = argparse.ArgumentParser(
        description="Three-Jaw Grasp 통합 학습 엔트리포인트"
    )
    parser.add_argument(
        "--model", type=str, default="mlp", help="학습할 모델 종류 (mlp, yolo 등)"
    )
    parser.add_argument("--data_path", type=str, required=True, help="학습 데이터 경로")
    parser.add_argument(
        "--output_path", type=str, required=True, help="가중치 저장 경로"
    )
    parser.add_argument("--epochs", type=int, default=50, help="에폭 수")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 사이즈")
    parser.add_argument("--lr", type=float, default=0.001, help="학습률")

    args = parser.parse_args()

    try:
        # Factory에서 문자열 이름으로 학습 로직(함수)을 가져옵니다.
        train_func = TrainerFactory.get(args.model)
    except KeyError as e:
        print(f"[오류] {e}")
        return

    # 가져온 학습 함수에 인자를 전달하여 실행합니다.
    train_func(
        data_path=args.data_path,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
