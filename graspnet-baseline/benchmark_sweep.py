"""
GraspNet latency sweep over NUM_POINT.
Measures how point cloud size affects inference speed.
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
NUM_VIEW = 300
COLLISION_THRESH = 0.01
VOXEL_SIZE = 0.01
N_WARMUP = 3
N_RUNS = 10

# Sweep these point counts
POINT_COUNTS = [2000, 5000, 10000, 20000, 40000]


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


def load_raw_data(data_dir):
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
    return cloud[mask], color[mask]


def sample_points(cloud_masked, color_masked, num_point):
    if len(cloud_masked) >= num_point:
        idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    return cloud_masked[idxs], color_masked[idxs]


def run_one(net, device, cloud_sampled, cloud_masked):
    end_points = {'point_clouds': torch.from_numpy(
        cloud_sampled[np.newaxis].astype(np.float32)).to(device)}

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = net(end_points)
    grasp_preds = pred_decode(out)
    torch.cuda.synchronize()
    t_fwd = (time.perf_counter() - t0) * 1000

    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)

    t0 = time.perf_counter()
    if COLLISION_THRESH > 0:
        mfcd = ModelFreeCollisionDetector(cloud_masked, voxel_size=VOXEL_SIZE)
        mask = mfcd.detect(gg, approach_dist=0.05, collision_thresh=COLLISION_THRESH)
        gg = gg[~mask]
    gg.nms()
    gg.sort_by_score()
    t_post = (time.perf_counter() - t0) * 1000

    return t_fwd, t_post, len(gg)


def main():
    print("=" * 80)
    print("GraspNet NUM_POINT Sweep (H200)")
    print("=" * 80)

    net, device = get_net()
    cloud_masked, color_masked = load_raw_data(DATA_DIR)
    print(f"Workspace cloud: {len(cloud_masked):,} points")
    print(f"Sweep: {POINT_COUNTS}")
    print(f"Warmup: {N_WARMUP}, Runs per setting: {N_RUNS}")
    print()

    print(f"{'NUM_POINT':>10} | {'fwd+dec (ms)':>14} {'std':>8} | "
          f"{'post (ms)':>12} {'std':>8} | {'total (ms)':>12} | "
          f"{'FPS':>6} | {'n_grasps':>9}")
    print("-" * 110)

    for n in POINT_COUNTS:
        # Warmup
        for _ in range(N_WARMUP):
            cs, _ = sample_points(cloud_masked, color_masked, n)
            run_one(net, device, cs, cloud_masked)

        # Measure
        fwd_list, post_list, n_grasps_list = [], [], []
        for _ in range(N_RUNS):
            cs, _ = sample_points(cloud_masked, color_masked, n)
            t_fwd, t_post, ng = run_one(net, device, cs, cloud_masked)
            fwd_list.append(t_fwd)
            post_list.append(t_post)
            n_grasps_list.append(ng)

        fwd = np.array(fwd_list)
        post = np.array(post_list)
        total = fwd + post
        ng_mean = np.mean(n_grasps_list)

        print(f"{n:>10,} | {fwd.mean():>14.2f} {fwd.std():>8.2f} | "
              f"{post.mean():>12.2f} {post.std():>8.2f} | "
              f"{total.mean():>12.2f} | {1000/total.mean():>6.1f} | {ng_mean:>9.0f}")

    print("-" * 110)
    print("\n* fwd+dec = forward pass + decode (GPU)")
    print("* post = collision detection + NMS + sort (mostly CPU)")


if __name__ == '__main__':
    main()
