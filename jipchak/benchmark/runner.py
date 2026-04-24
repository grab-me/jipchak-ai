"""어댑터 무관 벤치마크 러너.

같은 RGBDInput을 어떤 GraspAdapter에든 돌려 latency 통계를 수집한다.
graspnet-baseline/benchmark.py, gr-convnet/benchmark_grconv.py 의
공통 패턴을 한 곳에 모은 것.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from jipchak.core.adapter import GraspAdapter
from jipchak.core.types import RGBDInput


@dataclass
class BenchmarkResult:
    adapter_name: str
    device: str
    n_runs: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    fps: float
    n_grasps_mean: float
    gpu_peak_mb: float | None = None


def benchmark_adapter(
    adapter: GraspAdapter,
    sample: RGBDInput,
    device: str = "cuda:0",
    n_warmup: int = 3,
    n_runs: int = 20,
    top_k: int = 10,
) -> BenchmarkResult:
    """단일 입력에 대한 latency 측정. 입력 1장 반복 호출 패턴."""
    torch_device = torch.device(device)
    is_cuda = torch_device.type == "cuda"

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(torch_device)

    for _ in range(n_warmup):
        adapter.predict(sample, top_k=top_k)
    if is_cuda:
        torch.cuda.synchronize(torch_device)

    times_ms: list[float] = []
    n_grasps: list[int] = []
    for _ in range(n_runs):
        if is_cuda:
            torch.cuda.synchronize(torch_device)
        t0 = time.perf_counter()
        out = adapter.predict(sample, top_k=top_k)
        if is_cuda:
            torch.cuda.synchronize(torch_device)
        times_ms.append((time.perf_counter() - t0) * 1000)
        n_grasps.append(len(out))

    arr = np.array(times_ms)
    gpu_peak = (
        torch.cuda.max_memory_allocated(torch_device) / 1e6 if is_cuda else None
    )

    return BenchmarkResult(
        adapter_name=adapter.name,
        device=device,
        n_runs=n_runs,
        mean_ms=float(arr.mean()),
        std_ms=float(arr.std()),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
        fps=float(1000 / arr.mean()),
        n_grasps_mean=float(np.mean(n_grasps)),
        gpu_peak_mb=gpu_peak,
    )


def print_result(r: BenchmarkResult) -> None:
    print("=" * 60)
    print(f"Adapter: {r.adapter_name}  Device: {r.device}  Runs: {r.n_runs}")
    print("-" * 60)
    print(f"  mean: {r.mean_ms:7.2f} ms   std: {r.std_ms:6.2f}  "
          f"min: {r.min_ms:6.2f}  max: {r.max_ms:6.2f}")
    print(f"  fps:  {r.fps:7.1f}        grasps/run: {r.n_grasps_mean:.1f}")
    if r.gpu_peak_mb is not None:
        print(f"  GPU peak: {r.gpu_peak_mb:.1f} MB")
    print("=" * 60)
