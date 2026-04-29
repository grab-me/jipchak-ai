"""
3발 집게 (Three-Jaw Gripper) 전용 통합 시각화 모듈
모든 모델(GR-ConvNet, GraspNet, YOLO 등)의 추론 결과를 이미지 위에 시각화합니다.
"""
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from three_jaw_grasp.candidate import GraspCandidate

def draw_three_jaw_grasp(ax, rgb, grasp: GraspCandidate, detail: dict, 
                         model_name: str, is_3d: bool = False, intrinsics = None):
    """
    파지 후보(grasp)를 기반으로 3발 집게를 화면에 렌더링합니다.
    
    Args:
        ax: matplotlib 축 (Axis)
        rgb: 배경이 될 RGB 원본 이미지
        grasp: 그릴 파지 후보 객체
        detail: 세부 점수(width, score, height 등)가 담긴 딕셔너리
        model_name: 시각화 타이틀에 사용할 모델명 (예: Yolo, GraspNet 등)
        is_3d: True일 경우 3D 미터 좌표(X,Y,Z)를 intrinsics 기반으로 투영하여 그림
        intrinsics: 3x3 카메라 파라미터 행렬 (is_3d=True일 때 필수)
    """
    ax.imshow(rgb)
    ax.set_axis_off()

    # 1. 중심점(u, v)과 반지름 구하기
    if is_3d and intrinsics is not None:
        X, Y, Z = grasp.center_x, grasp.center_y, grasp.center_z
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        # 3D -> 2D 투영
        u = (X * fx / Z) + cx
        v = (Y * fy / Z) + cy
        
        radius_m = grasp.width / 2.0
        radius = (radius_m * fx) / Z
        radius = max(radius, 20.0) # 눈에 보이게 최소 픽셀 보장
    else:
        u = grasp.center_x
        v = grasp.center_y
        Z = grasp.center_z
        radius = max(grasp.width, 20.0)

    angle = grasp.angle

    # 2. 3발 다리 그리기
    base_angle = angle - (math.pi / 2)
    jaw_colors = ['#FF4444', '#44FF44', '#4488FF']
    labels     = ['Jaw 1', 'Jaw 2', 'Jaw 3']
    
    if is_3d:
        labels = [f'{l} (Proj)' for l in labels]
        
    for i, color in enumerate(jaw_colors):
        a = base_angle + i * (2 * math.pi / 3)
        fx_line = u + radius * math.cos(a)
        fy_line = v + radius * math.sin(a)
        ax.plot([u, fx_line], [v, fy_line], color=color, lw=2.5, solid_capstyle='round', zorder=4, label=labels[i])
        ax.scatter(fx_line, fy_line, color=color, s=50, zorder=5)

    # 3. 중심점 및 반지름 렌더링
    circle = plt.Circle((u, v), radius, color='yellow', fill=False, lw=1.5, linestyle='--', alpha=0.7, zorder=3)
    ax.add_patch(circle)
    ax.scatter(u, v, color='yellow', s=100, marker='*', zorder=6)

    # 4. 점수표 (스코어 텍스트) 생성
    score_text = (
        f"[{model_name.upper()}] Total: {detail['total']:.3f}\n"
        f"width: {detail['width']['weighted']:.2f}  "
        f"score: {detail['score']['weighted']:.2f}\n"
        f"height: {detail['height']['weighted']:.2f}  "
        f"stab: {detail['stability']['weighted']:.2f}\n"
        f"angle: {math.degrees(angle):.1f}°  Z: {Z:.3f}m"
    )
    
    if is_3d:
        score_text += f"\nProj (u,v): ({u:.1f}, {v:.1f})"

    ax.text(5, 15, score_text, color='white', fontsize=8, va='top', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))
    
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(jaw_colors, labels)]
    ax.legend(handles=patches, loc='lower right', fontsize=8, facecolor='black', labelcolor='white', framealpha=0.6)
