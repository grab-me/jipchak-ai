"""
GraspNet baseline latency benchmark on H200.
Modified from demo.py - removes visualization, adds timing.
"""
import os
import sys
import numpy as np
import torch
import time
import scipy.io as scio
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

from graspnet import GraspNet, pred_decode
from graspnetAPI import GraspGroup
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

CHECKPOINT_PATH = 'logs/log_rs/checkpoint-rs.tar'
DATA_DIR = 'doc/example_data'
NUM_POINT = 20000
NUM_VIEW = 300
COLLISION_THRESH = 0.01
VOXEL_SIZE = 0.01
N_WARMUP = 3
N_RUNS = 20


def get_net():
    net = GraspNet(input_feature_dim=0, num_view=NUM_VIEW, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04],
                   is_training=False)
    device = torch.device("cuda:0")
    net.to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net, device


def get_and_process_data(data_dir):
    color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
    meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']

    camera = CameraInfo(1280.0, 720.0,
                        intrinsic[0][0], intrinsic[1][1],
                        intrinsic[0][2], intrinsic[1][2],
                        factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    if len(cloud_masked) >= NUM_POINT:
        idxs = np.random.choice(len(cloud_masked), NUM_POINT, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), NUM_POINT - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud_masked


def time_block(label, fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return out, (t1 - t0) * 1000


def run_inference(net, device, end_points, cloud_masked, measure=True):
    timings = {}

    # 1. Forward
    end_points['point_clouds'] = end_points['point_clouds'].to(device)
    def _forward():
        with torch.no_grad():
            return net(end_points)
    fwd_out, t_fwd = time_block('forward', _forward)
    timings['forward_ms'] = t_fwd

    # 2. Decode (GPU)
    def _decode():
        return pred_decode(fwd_out)
    grasp_preds, t_dec = time_block('decode', _decode)
    timings['decode_ms'] = t_dec

    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)

    # 3. Collision detection (CPU heavy)
    t0 = time.perf_counter()
    if COLLISION_THRESH > 0:
        mfcdetector = ModelFreeCollisionDetector(cloud_masked, voxel_size=VOXEL_SIZE)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05,
                                            collision_thresh=COLLISION_THRESH)
        gg = gg[~collision_mask]
    t1 = time.perf_counter()
    timings['collision_ms'] = (t1 - t0) * 1000

    # 4. NMS + sort (CPU)
    t0 = time.perf_counter()
    gg.nms()
    gg.sort_by_score()
    t1 = time.perf_counter()
    timings['nms_sort_ms'] = (t1 - t0) * 1000

    timings['total_ms'] = sum(timings.values())
    timings['n_grasps'] = len(gg)
    return gg, timings


def main():
    print("=" * 70)
    print("GraspNet Baseline - Latency Benchmark")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Data: {DATA_DIR}")
    print(f"NUM_POINT: {NUM_POINT}, NUM_VIEW: {NUM_VIEW}")
    print(f"Warmup: {N_WARMUP}, Runs: {N_RUNS}")
    print("=" * 70)

    print("\n[1/4] Loading model...")
    t0 = time.perf_counter()
    net, device = get_net()
    print(f"      Model loaded in {(time.perf_counter()-t0)*1000:.1f} ms")
    n_params = sum(p.numel() for p in net.parameters())
    print(f"      Params: {n_params:,} ({n_params*4/1e6:.1f} MB FP32)")

    print("\n[2/4] Loading + processing data...")
    end_points, cloud_masked = get_and_process_data(DATA_DIR)
    print(f"      Point cloud: {end_points['point_clouds'].shape}")
    print(f"      Workspace cloud: {len(cloud_masked):,} points")

    print(f"\n[3/4] Warmup x {N_WARMUP}...")
    for i in range(N_WARMUP):
        ep_copy = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in end_points.items()}
        _, _ = run_inference(net, device, ep_copy, cloud_masked)
    torch.cuda.synchronize()
    print("      Warmup done.")

    print(f"\n[4/4] Benchmark x {N_RUNS}...")
    all_t = {'forward_ms': [], 'decode_ms': [], 'collision_ms': [], 'nms_sort_ms': [], 'total_ms': []}
    n_grasps_list = []
    for i in range(N_RUNS):
        ep_copy = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in end_points.items()}
        gg, t = run_inference(net, device, ep_copy, cloud_masked)
        for k in all_t:
            all_t[k].append(t[k])
        n_grasps_list.append(t['n_grasps'])
        if i == 0:
            print(f"      Run 1: total={t['total_ms']:.2f}ms, n_grasps={t['n_grasps']}")

    # GPU memory
    mem_alloc = torch.cuda.max_memory_allocated() / 1e9
    mem_reserved = torch.cuda.max_memory_reserved() / 1e9

    # Stats
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'Stage':<20} {'mean (ms)':>12} {'std':>10} {'min':>10} {'max':>10}")
    print("-" * 70)
    for k in ['forward_ms', 'decode_ms', 'collision_ms', 'nms_sort_ms', 'total_ms']:
        arr = np.array(all_t[k])
        print(f"{k:<20} {arr.mean():>12.2f} {arr.std():>10.2f} {arr.min():>10.2f} {arr.max():>10.2f}")
    print("-" * 70)

    total = np.array(all_t['total_ms'])
    fwd = np.array(all_t['forward_ms'])
    print(f"\nThroughput (full pipeline): {1000/total.mean():.1f} FPS")
    print(f"Throughput (forward only):  {1000/fwd.mean():.1f} FPS")
    print(f"Avg n_grasps after collision+NMS: {np.mean(n_grasps_list):.0f}")
    print(f"\nGPU peak memory: {mem_alloc:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
    print("=" * 70)


if __name__ == '__main__':
    main()
