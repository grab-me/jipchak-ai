"""
Headless visualization of GraspNet predictions.
Saves PNG images instead of trying to open Open3D GUI.
"""
import os
import sys
import numpy as np
import torch
import scipy.io as scio
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

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
OUTPUT_DIR = 'visualizations'
NUM_POINT = 20000
NUM_VIEW = 300
COLLISION_THRESH = 0.01
VOXEL_SIZE = 0.01
TOP_K = 50  # how many top grasps to visualize


def get_net(device):
    net = GraspNet(input_feature_dim=0, num_view=NUM_VIEW, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04],
                   is_training=False)
    net.to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net


def get_data(data_dir):
    color_img = np.array(Image.open(os.path.join(data_dir, 'color.png')))
    color = color_img.astype(np.float32) / 255.0
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

    return color_img, depth, cloud_masked, color_masked, cloud_sampled, intrinsic


def project_3d_to_2d(points_3d, intrinsic):
    """Camera intrinsic projection."""
    fx, fy = intrinsic[0][0], intrinsic[1][1]
    cx, cy = intrinsic[0][2], intrinsic[1][2]
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    z = np.where(z == 0, 1e-6, z)
    u = (x * fx / z + cx)
    v = (y * fy / z + cy)
    return np.stack([u, v], axis=1)


def plot_grasps_on_rgb(color_img, gg, intrinsic, save_path, top_k=TOP_K):
    """Project grasp centers + approach direction onto the RGB image."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(color_img)

    # Top K grasps by score
    n = min(top_k, len(gg))
    top = gg[:n]

    # Translations (grasp centers in 3D)
    translations = top.translations  # (N, 3)
    # Rotation matrices: rotation[:, :, 0] is the approach direction (x-axis of grasp frame)
    rotations = top.rotation_matrices  # (N, 3, 3)
    scores = top.scores  # (N,)

    # Project centers to 2D
    centers_2d = project_3d_to_2d(translations, intrinsic)

    # Project approach endpoints (centers + 5cm along approach direction)
    approach_dirs = rotations[:, :, 0]  # (N, 3)
    endpoints_3d = translations + approach_dirs * 0.05
    endpoints_2d = project_3d_to_2d(endpoints_3d, intrinsic)

    # Color by score (higher = greener)
    norm = plt.Normalize(vmin=scores.min(), vmax=scores.max())
    cmap = plt.cm.RdYlGn

    for i in range(n):
        c = cmap(norm(scores[i]))
        # Grasp center
        ax.scatter(centers_2d[i, 0], centers_2d[i, 1],
                   c=[c], s=80, edgecolors='black', linewidths=0.5, zorder=3)
        # Approach line
        ax.plot([centers_2d[i, 0], endpoints_2d[i, 0]],
                [centers_2d[i, 1], endpoints_2d[i, 1]],
                color=c, linewidth=1.5, alpha=0.7, zorder=2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label('Grasp score (red=low, green=high)', fontsize=10)

    ax.set_title(f'GraspNet predictions (top {n} of {len(gg)} after collision+NMS)\n'
                 f'Centers + approach direction projected to RGB',
                 fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  saved: {save_path}")


def plot_3d_scene(cloud_masked, color_masked, gg, save_path, top_k=TOP_K):
    """3D scatter of point cloud + grasp positions."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Subsample point cloud for plotting (too slow if all)
    n_plot = min(8000, len(cloud_masked))
    idxs = np.random.choice(len(cloud_masked), n_plot, replace=False)
    pc = cloud_masked[idxs]
    cc = color_masked[idxs]

    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2],
               c=cc, s=1, alpha=0.5)

    # Top K grasps
    n = min(top_k, len(gg))
    top = gg[:n]
    translations = top.translations
    rotations = top.rotation_matrices
    scores = top.scores

    norm = plt.Normalize(vmin=scores.min(), vmax=scores.max())
    cmap = plt.cm.RdYlGn

    for i in range(n):
        c = cmap(norm(scores[i]))
        t = translations[i]
        approach = rotations[i, :, 0] * 0.04
        # Grasp position
        ax.scatter(t[0], t[1], t[2], c=[c], s=60,
                   edgecolors='black', linewidths=0.5)
        # Approach arrow
        ax.plot([t[0], t[0] + approach[0]],
                [t[1], t[1] + approach[1]],
                [t[2], t[2] + approach[2]],
                color=c, linewidth=1.5, alpha=0.8)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'3D point cloud + top {n} grasps')
    # View from above-side (similar to camera)
    ax.view_init(elev=-60, azim=-90)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  saved: {save_path}")


def plot_score_distribution(gg, save_path):
    """Histogram of grasp scores."""
    fig, ax = plt.subplots(figsize=(10, 5))
    scores = gg.scores
    ax.hist(scores, bins=30, color='steelblue', edgecolor='black', alpha=0.8)
    ax.axvline(scores.mean(), color='red', linestyle='--',
               label=f'mean = {scores.mean():.3f}')
    ax.axvline(np.median(scores), color='orange', linestyle='--',
               label=f'median = {np.median(scores):.3f}')
    ax.set_xlabel('Grasp score')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of {len(gg)} grasp scores (after collision+NMS)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  saved: {save_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("GraspNet Visualization (headless, saves PNG)")
    print("=" * 70)

    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    print("\n[1/4] Loading model + data...")
    net = get_net(device)
    color_img, depth, cloud_masked, color_masked, cloud_sampled, intrinsic = get_data(DATA_DIR)
    print(f"  RGB: {color_img.shape}, Depth: {depth.shape}")
    print(f"  Workspace cloud: {len(cloud_masked):,} points")
    print(f"  Sampled to: {cloud_sampled.shape}")

    print("\n[2/4] Running inference...")
    end_points = {'point_clouds': torch.from_numpy(
        cloud_sampled[np.newaxis].astype(np.float32)).to(device)}
    with torch.no_grad():
        out = net(end_points)
    grasp_preds = pred_decode(out)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    print(f"  Raw grasps: {len(gg)}")

    print("\n[3/4] Collision detection + NMS...")
    if COLLISION_THRESH > 0:
        mfcd = ModelFreeCollisionDetector(cloud_masked, voxel_size=VOXEL_SIZE)
        mask = mfcd.detect(gg, approach_dist=0.05, collision_thresh=COLLISION_THRESH)
        gg = gg[~mask]
        print(f"  After collision filter: {len(gg)}")
    gg.nms()
    gg.sort_by_score()
    print(f"  After NMS + sort: {len(gg)}")
    print(f"  Top score: {gg.scores[0]:.3f}, Bottom: {gg.scores[-1]:.3f}")

    print("\n[4/4] Generating visualizations...")
    plot_grasps_on_rgb(color_img, gg, intrinsic,
                       os.path.join(OUTPUT_DIR, 'grasps_on_rgb.png'))
    plot_3d_scene(cloud_masked, color_masked, gg,
                  os.path.join(OUTPUT_DIR, 'grasps_3d.png'))
    plot_score_distribution(gg,
                            os.path.join(OUTPUT_DIR, 'score_distribution.png'))

    # Also save top-5 individual grasps as text
    print("\n[5/5] Top 5 grasps detail:")
    for i in range(min(5, len(gg))):
        g = gg[i]
        print(f"  #{i+1}: score={g.score:.3f}, "
              f"width={g.width*1000:.1f}mm, "
              f"depth={g.depth*1000:.1f}mm, "
              f"position=({g.translation[0]:.3f}, {g.translation[1]:.3f}, {g.translation[2]:.3f})")

    print(f"\nDone. Check {OUTPUT_DIR}/ for PNG files.")


if __name__ == '__main__':
    main()
