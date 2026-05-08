"""
GR-ConvNet latency benchmark.
Mirrors graspnet-baseline/benchmark.py for fair comparison.

Uses GraspNet example data (../graspnet-baseline/doc/example_data) by default
so RGB+depth input is identical across both models.

Usage:
    python benchmark_grconv.py --device cuda:0
    python benchmark_grconv.py --device cpu --n_runs 20
"""
import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

from inference.post_process import post_process_output
from utils.data.camera_data import CameraData

logging.basicConfig(level=logging.WARNING)

DEFAULT_NETWORK = 'trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'
DEFAULT_RGB = '../graspnet-baseline/doc/example_data/color.png'
DEFAULT_DEPTH = '../graspnet-baseline/doc/example_data/depth.png'


def time_block(fn, device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return out, (t1 - t0) * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda:0 or cpu')
    parser.add_argument('--network', type=str, default=DEFAULT_NETWORK)
    parser.add_argument('--rgb_path', type=str, default=DEFAULT_RGB)
    parser.add_argument('--depth_path', type=str, default=DEFAULT_DEPTH)
    parser.add_argument('--use_depth', type=int, default=1)
    parser.add_argument('--use_rgb', type=int, default=1)
    parser.add_argument('--output_size', type=int, default=224,
                        help='224 for cornell models, 300 for jacquard')
    parser.add_argument('--n_warmup', type=int, default=5)
    parser.add_argument('--n_runs', type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 70)
    print("GR-ConvNet - Latency Benchmark")
    print("=" * 70)
    if device.type == 'cuda':
        print(f"Device: {torch.cuda.get_device_name(device)}  (torch device: {device})")
    else:
        print(f"Device: CPU  (threads: {torch.get_num_threads()})")
    print(f"PyTorch: {torch.__version__}, CUDA build: {torch.version.cuda}")
    print(f"Network:  {args.network}")
    print(f"RGB:      {args.rgb_path}")
    print(f"Depth:    {args.depth_path}")
    print(f"Input:    {args.output_size}x{args.output_size}, "
          f"use_depth={args.use_depth}, use_rgb={args.use_rgb}")
    print(f"Warmup: {args.n_warmup}, Runs: {args.n_runs}")
    print("=" * 70)

    # 1. Load + preprocess data
    print("\n[1/4] Loading + preprocessing data...")
    rgb = np.array(Image.open(args.rgb_path))
    depth = np.expand_dims(np.array(Image.open(args.depth_path)), axis=2)
    H, W = rgb.shape[:2]
    print(f"      RGB: {rgb.shape}, depth: {depth.shape}")

    img_data = CameraData(
        width=W, height=H, output_size=args.output_size,
        include_depth=bool(args.use_depth),
        include_rgb=bool(args.use_rgb),
    )
    x, depth_img, rgb_img = img_data.get_data(rgb=rgb, depth=depth)
    print(f"      Network input tensor: {tuple(x.shape)}")

    # 2. Load model
    print("\n[2/4] Loading model...")
    t0 = time.perf_counter()
    net = torch.load(args.network, map_location=device)
    net.eval()
    print(f"      Model loaded in {(time.perf_counter() - t0) * 1000:.1f} ms")
    n_params = sum(p.numel() for p in net.parameters())
    print(f"      Params: {n_params:,} ({n_params * 4 / 1e6:.1f} MB FP32)")

    xc = x.to(device)

    # 3. Warmup
    print(f"\n[3/4] Warmup x {args.n_warmup}...")
    for _ in range(args.n_warmup):
        with torch.no_grad():
            pred = net.predict(xc)
            _ = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
    if device.type == 'cuda':
        torch.cuda.synchronize()
    print("      Warmup done.")

    # 4. Benchmark
    print(f"\n[4/4] Benchmark x {args.n_runs}...")
    fwd_times, post_times = [], []
    for i in range(args.n_runs):
        with torch.no_grad():
            pred, t_fwd = time_block(lambda: net.predict(xc), device)
            (_q, _ang, _w), t_post = time_block(
                lambda: post_process_output(
                    pred['pos'], pred['cos'], pred['sin'], pred['width']
                ),
                device,
            )
        fwd_times.append(t_fwd)
        post_times.append(t_post)
        if i == 0:
            print(f"      Run 1: fwd={t_fwd:.2f}ms, post={t_post:.2f}ms")

    # Stats
    fwd = np.array(fwd_times)
    post = np.array(post_times)
    total = fwd + post

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'Stage':<20} {'mean (ms)':>12} {'std':>10} {'min':>10} {'max':>10}")
    print("-" * 70)
    for k, arr in [('forward_ms', fwd), ('post_ms', post), ('total_ms', total)]:
        print(f"{k:<20} {arr.mean():>12.2f} {arr.std():>10.2f} "
              f"{arr.min():>10.2f} {arr.max():>10.2f}")
    print("-" * 70)
    print(f"\nThroughput (full pipeline): {1000 / total.mean():.1f} FPS")
    print(f"Throughput (forward only):  {1000 / fwd.mean():.1f} FPS")
    if device.type == 'cuda':
        mem_alloc = torch.cuda.max_memory_allocated(device) / 1e9
        mem_reserved = torch.cuda.max_memory_reserved(device) / 1e9
        print(f"\nGPU peak memory: {mem_alloc:.2f} GB allocated, "
              f"{mem_reserved:.2f} GB reserved")
    print("=" * 70)


if __name__ == '__main__':
    main()
