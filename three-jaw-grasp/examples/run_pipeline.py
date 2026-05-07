"""
Unified inference/visualization runner for three-jaw grasp pipeline.

Examples:
    python examples/run_pipeline.py --model yolo --max 3 --save examples/output
    python examples/run_pipeline.py --model grconvnet --max 3
    python examples/run_pipeline.py --model graspnet --max 3
"""

import argparse
import datetime
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PIPELINE_ROOT)

from three_jaw_grasp import GraspPipeline, ThreeJawEvaluator
from three_jaw_grasp.adapters import YoloBoxGraspAdapter
from three_jaw_grasp.detector import YoloMockDetector
from three_jaw_grasp.evaluator_chick import ChickEvaluator
from three_jaw_grasp.factory import AdapterFactory, ModelFactory
from three_jaw_grasp.visualizer import draw_n_jaw_grasp

# Optional external model repo path for GR-ConvNet.
GRCONVNET_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(PIPELINE_ROOT)),
    "robotic-grasping-cornell",
)
if os.path.exists(GRCONVNET_ROOT):
    sys.path.insert(0, GRCONVNET_ROOT)

# Keep local wrappers import for yolo mock support.
from examples.wrappers import YoloMockModel

# Register plugin models.
import external_models.graspnet.plugin  # noqa: F401
import external_models.grconvnet.plugin  # noqa: F401


def _create_model_and_adapter(model_name: str, dataset_dir: str):
    if model_name == "yolo":
        return YoloMockModel(data_dir=dataset_dir), YoloBoxGraspAdapter()

    if model_name == "grconvnet":
        ckpt_path = os.path.join(GRCONVNET_ROOT, "models", "model_49.ckpt")
        model = ModelFactory.create(
            model_name,
            checkpoint_path=ckpt_path if os.path.exists(ckpt_path) else None,
        )
        return model, AdapterFactory.create(model_name)

    model = ModelFactory.create(model_name)
    return model, AdapterFactory.create(model_name)


def main():
    parser = argparse.ArgumentParser(description="Three-Jaw Grasp Pipeline Runner")
    parser.add_argument("--model", type=str, default="grconvnet", help="Model name")
    parser.add_argument("--dataset", type=str, default="01", help="Dataset folder name")
    parser.add_argument("--max", type=int, default=5, help="Number of images to process")
    parser.add_argument("--save", type=str, default=None, help="Output directory")
    parser.add_argument("--jaw-count", type=int, default=3, help="Number of jaws (2 or 3)")
    parser.add_argument("--evaluator", type=str, default="default", choices=["default", "chick"], help="Evaluator type")
    args = parser.parse_args()

    dataset_dir = os.path.join(PIPELINE_ROOT, "dataset", args.dataset)
    if not os.path.exists(dataset_dir):
        print(f"[ERROR] dataset directory not found: {dataset_dir}")
        return

    try:
        model, adapter = _create_model_and_adapter(args.model, dataset_dir)
    except KeyError as exc:
        print(f"[ERROR] {exc}")
        return

    is_3d = getattr(model, "is_3d", args.model == "graspnet")
    intrinsics = getattr(model, "intrinsics", None)

    if args.evaluator == "chick":
        evaluator = ChickEvaluator(
            config_path=os.path.join(PIPELINE_ROOT, "config", "gripper_spec.yaml"),
            jaw_count=args.jaw_count,
        )
    else:
        evaluator = ThreeJawEvaluator(
            config_path=os.path.join(PIPELINE_ROOT, "config", "gripper_spec.yaml"),
            jaw_count=args.jaw_count,
        )
    detector = None if args.model == "yolo" else YoloMockDetector(data_dir=dataset_dir)
    pipeline = GraspPipeline(
        model=model,
        adapter=adapter,
        evaluator=evaluator,
        detector=detector,
        is_3d=is_3d,
    )

    if args.save:
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        base_dir = args.save
        existing_dirs = glob.glob(os.path.join(base_dir, f"{args.model}_{date_str}_*"))
        max_num = 0
        for d in existing_dirs:
            try:
                max_num = max(max_num, int(os.path.basename(d).split("_")[-1]))
            except ValueError:
                continue

        save_dir = os.path.join(base_dir, f"{args.model}_{date_str}_{max_num + 1}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"[SAVE] output directory: {save_dir}")
        args.save = save_dir

    image_paths = sorted(glob.glob(os.path.join(dataset_dir, "*r.png")))
    print(f"\n[{args.model.upper()}] processing {min(args.max, len(image_paths))} images\n{'=' * 50}")

    for i, img_path in enumerate(image_paths[: args.max], start=1):
        fname = os.path.basename(img_path)
        prefix = fname.split("r.png")[0]
        rgb = np.array(Image.open(img_path).convert("RGB"))
        h, w = rgb.shape[:2]
        depth = np.full((h, w), 0.30, dtype=np.float32)

        try:
            best = pipeline.predict_best(rgb, depth, filename_prefix=prefix)
        except ValueError as exc:
            print(f"[{i}] {fname} -> [SKIP] {exc}")
            continue

        detail = evaluator.score_detail(best)
        print(
            f"[{i}] {fname} -> Total: {detail['total']:.3f} | "
            f"Center: (X={best.center_x:.2f}, Y={best.center_y:.2f})"
        )

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        draw_n_jaw_grasp(
            ax,
            rgb,
            best,
            detail,
            model_name=args.model,
            jaw_count=args.jaw_count,
            is_3d=is_3d,
            intrinsics=intrinsics,
        )
        plt.tight_layout()

        if args.save:
            out_path = os.path.join(args.save, f"{args.model}_{fname}")
            plt.savefig(out_path, dpi=100, bbox_inches="tight")

        plt.show()
        plt.close()


if __name__ == "__main__":
    main()
