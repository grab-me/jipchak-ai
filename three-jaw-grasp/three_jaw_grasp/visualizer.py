import math

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from three_jaw_grasp.candidate import GraspCandidate


def _safe_weight(detail: dict, key: str) -> float:
    v = detail.get(key, None)
    if isinstance(v, dict):
        return float(v.get("weighted", 0.0))
    return 0.0


def draw_n_jaw_grasp(
    ax,
    rgb,
    grasp: GraspCandidate,
    detail: dict,
    model_name: str,
    jaw_count: int = 3,
    is_3d: bool = False,
    intrinsics=None,
):
    jaw_count = max(2, int(jaw_count))

    ax.imshow(rgb)
    ax.set_axis_off()

    if is_3d and intrinsics is not None:
        X, Y, Z = grasp.center_x, grasp.center_y, grasp.center_z
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        u = (X * fx / Z) + cx
        v = (Y * fy / Z) + cy
        radius = max((grasp.width / 2.0 * fx) / Z, 20.0)
    else:
        u = grasp.center_x
        v = grasp.center_y
        Z = grasp.center_z
        radius = max(grasp.width * 5.2, 100.0)

    angle = grasp.angle
    base_angle = angle - (math.pi / 2.0)

    base_colors = ["#FF4444", "#44FF44", "#4488FF", "#FFB347", "#66B2FF", "#DDA0DD"]
    jaw_colors = [base_colors[i % len(base_colors)] for i in range(jaw_count)]
    labels = [f"Jaw {i+1}" for i in range(jaw_count)]
    if is_3d:
        labels = [f"{l} (Proj)" for l in labels]

    for i, color in enumerate(jaw_colors):
        a = base_angle + i * (2.0 * math.pi / float(jaw_count))
        fx_line = u + radius * math.cos(a)
        fy_line = v + radius * math.sin(a)
        ax.plot([u, fx_line], [v, fy_line], color=color, lw=6.0, solid_capstyle="round", zorder=4)
        ax.scatter(fx_line, fy_line, color=color, s=150, zorder=5)

    circle = plt.Circle((u, v), radius, color="yellow", fill=False, lw=1.5, linestyle="--", alpha=0.7, zorder=3)
    ax.add_patch(circle)
    ax.scatter(u, v, color="yellow", s=300, marker="*", zorder=6)

    score_text = (
        f"[{model_name.upper()}] Total: {detail.get('total', 0.0):.3f}\n"
        f"width: {_safe_weight(detail, 'width'):.2f}  score: {_safe_weight(detail, 'score'):.2f}\n"
        f"height: {_safe_weight(detail, 'height'):.2f}  stab: {_safe_weight(detail, 'stability'):.2f}\n"
        f"angle: {math.degrees(angle):.1f} deg  Z: {Z:.3f}m"
    )
    if is_3d:
        score_text += f"\nProj (u,v): ({u:.1f}, {v:.1f})"

    ax.text(
        5,
        15,
        score_text,
        color="white",
        fontsize=8,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6),
    )

    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(jaw_colors, labels)]
    ax.legend(handles=patches, loc="lower right", fontsize=8, facecolor="black", labelcolor="white", framealpha=0.6)


def draw_three_jaw_grasp(
    ax,
    rgb,
    grasp: GraspCandidate,
    detail: dict,
    model_name: str,
    is_3d: bool = False,
    intrinsics=None,
):
    draw_n_jaw_grasp(
        ax=ax,
        rgb=rgb,
        grasp=grasp,
        detail=detail,
        model_name=model_name,
        jaw_count=3,
        is_3d=is_3d,
        intrinsics=intrinsics,
    )
