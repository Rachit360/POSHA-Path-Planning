"""
=============================================================================
POSHA ROBOTICS — 3D WORKSPACE VISUALIZER
Visualizes robot arm paths for Macro and Micro dispense tasks
=============================================================================
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from path_planning_macro import (
    Pose, Waypoint, PiperKinematics, MacroDispensePlanner,
    MACRO_CONTAINERS, PAN_CENTERS, ROBOT_BASE, MM2M
)
from path_planning_micro import MicroDispensePlanner, SPICE_PODS


def draw_cylinder(ax, center, radius, height, color, alpha=0.3, label=None):
    """Draw a cylinder in 3D."""
    theta = np.linspace(0, 2*np.pi, 30)
    z = np.array([center[2], center[2] + height])
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = center[0] + radius * np.cos(theta_grid)
    y_grid = center[1] + radius * np.sin(theta_grid)
    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha)
    if label:
        ax.text(center[0], center[1], center[2] + height + 0.02,
                label, fontsize=7, ha='center', color=color)


def draw_box(ax, center, size, color, alpha=0.3, label=None):
    """Draw a box (rectangular prism) in 3D."""
    x, y, z = center
    dx, dy, dz = size
    vertices = np.array([
        [x-dx/2, y-dy/2, z], [x+dx/2, y-dy/2, z],
        [x+dx/2, y+dy/2, z], [x-dx/2, y+dy/2, z],
        [x-dx/2, y-dy/2, z+dz], [x+dx/2, y-dy/2, z+dz],
        [x+dx/2, y+dy/2, z+dz], [x-dx/2, y+dy/2, z+dz],
    ])
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
    ]
    poly = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='gray', linewidth=0.3)
    ax.add_collection3d(poly)
    if label:
        ax.text(x, y, z + dz + 0.02, label, fontsize=7, ha='center', color='black')


def plot_arm_at_waypoint(ax, joints, color='blue', alpha=0.7):
    """Draw robot arm links at a given joint configuration."""
    # Compute positions of each link
    T = np.eye(4)
    positions = [T[:3, 3].copy()]
    dh = PiperKinematics.DH_PARAMS

    for i, (a, alpha_dh, d, theta_offset) in enumerate(dh):
        theta = joints[i] + theta_offset
        ct, st = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(alpha_dh), math.sin(alpha_dh)
        Ti = np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0.0, sa,     ca,    d   ],
            [0.0, 0.0,    0.0,   1.0 ]
        ])
        T = T @ Ti
        positions.append(T[:3, 3].copy())

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]

    ax.plot(xs, ys, zs, '-o', color=color, alpha=alpha,
            linewidth=2, markersize=4)
    return positions[-1]  # return EE position


def visualize_macro_task():
    """Create 3D visualization of Task 1: Macro dispense paths."""
    planner = MacroDispensePlanner()
    path_c5_p2 = planner.plan_dispense(container_id=5, pan_id=2)
    path_c1_p1 = planner.plan_dispense(container_id=1, pan_id=1)

    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor('#1a1a2e')

    # ── Plot 1: Full 3D workspace ─────────────────────────────────────────
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_facecolor('#16213e')
    ax1.set_title('Task 1: Macro Dispense — 3D Workspace',
                  color='white', fontsize=11, fontweight='bold')

    # Draw macro containers
    for cid, cpose in MACRO_CONTAINERS.items():
        draw_box(ax1,
                 center=[cpose.x, cpose.y, cpose.z],
                 size=[0.08, 0.06, 0.12],
                 color='#ffa500',
                 alpha=0.5,
                 label=f'C{cid}')

    # Draw pans
    for pid, ppose in PAN_CENTERS.items():
        draw_cylinder(ax1,
                      center=[ppose.x, ppose.y, ppose.z],
                      radius=0.13,
                      height=0.06,
                      color='#4fc3f7',
                      alpha=0.5,
                      label=f'Pan{pid}')

    # Draw robot base
    ax1.scatter([ROBOT_BASE.x], [ROBOT_BASE.y], [ROBOT_BASE.z],
                s=100, c='lime', marker='^', zorder=5)
    ax1.text(ROBOT_BASE.x, ROBOT_BASE.y, ROBOT_BASE.z + 0.03,
             'Robot\nBase', color='lime', fontsize=7, ha='center')

    # Draw path: Container 5 → Pan 2
    path_pts_c5 = np.array([[wp.pose.x, wp.pose.y, wp.pose.z]
                             for wp in path_c5_p2])
    ax1.plot(path_pts_c5[:, 0], path_pts_c5[:, 1], path_pts_c5[:, 2],
             'r-o', linewidth=1.5, markersize=4, alpha=0.8, label='C5→Pan2')

    # Draw path: Container 1 → Pan 1
    path_pts_c1 = np.array([[wp.pose.x, wp.pose.y, wp.pose.z]
                             for wp in path_c1_p1])
    ax1.plot(path_pts_c1[:, 0], path_pts_c1[:, 1], path_pts_c1[:, 2],
             'y-o', linewidth=1.5, markersize=4, alpha=0.8, label='C1→Pan1')

    # Highlight key waypoints
    for wp in path_c5_p2:
        if 'STEP' in wp.name:
            ax1.scatter([wp.pose.x], [wp.pose.y], [wp.pose.z],
                        s=30, c='red', alpha=0.9)

    _style_3d_ax(ax1)
    ax1.legend(loc='upper left', fontsize=8, framealpha=0.3,
               labelcolor='white', facecolor='#1a1a2e')

    # ── Plot 2: Top-down view ─────────────────────────────────────────────
    ax2 = fig.add_subplot(122)
    ax2.set_facecolor('#16213e')
    ax2.set_title('Task 1: Top-Down View (XY Plane)',
                  color='white', fontsize=11, fontweight='bold')

    # Containers
    for cid, cp in MACRO_CONTAINERS.items():
        rect = plt.Rectangle((cp.x - 0.04, cp.y - 0.03), 0.08, 0.06,
                               fill=True, facecolor='#ffa500', alpha=0.6,
                               edgecolor='white', linewidth=0.5)
        ax2.add_patch(rect)
        ax2.text(cp.x, cp.y, f'C{cid}', ha='center', va='center',
                 fontsize=7, color='black', fontweight='bold')

    # Pans
    for pid, pp in PAN_CENTERS.items():
        circ = plt.Circle((pp.x, pp.y), 0.13, fill=True,
                           facecolor='#4fc3f7', alpha=0.4,
                           edgecolor='white', linewidth=1)
        ax2.add_patch(circ)
        ax2.text(pp.x, pp.y, f'Pan{pid}', ha='center', va='center',
                 fontsize=8, color='black', fontweight='bold')

    # Paths
    ax2.plot(path_pts_c5[:, 0], path_pts_c5[:, 1],
             'r-o', lw=1.5, ms=4, alpha=0.9, label='C5→Pan2')
    ax2.plot(path_pts_c1[:, 0], path_pts_c1[:, 1],
             'y-o', lw=1.5, ms=4, alpha=0.9, label='C1→Pan1')

    # Robot base
    ax2.plot(ROBOT_BASE.x, ROBOT_BASE.y, '^', ms=12,
             color='lime', label='Robot Base', zorder=10)

    ax2.set_xlabel('X (m)', color='white')
    ax2.set_ylabel('Y (m)', color='white')
    ax2.tick_params(colors='white')
    ax2.legend(loc='upper right', fontsize=8, facecolor='#1a1a2e',
               labelcolor='white', framealpha=0.7)
    ax2.grid(True, color='gray', alpha=0.2)
    for spine in ax2.spines.values():
        spine.set_edgecolor('gray')

    plt.tight_layout()
    plt.savefig('task1_macro_path.png', dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e')
    plt.close()
    print("✅ Saved: task1_macro_path.png")


def visualize_micro_task():
    """Create 3D visualization of Task 2: Micro dispense paths."""
    planner = MicroDispensePlanner()
    path_p1 = planner.plan_pod_dispense(pod_id=1, pan_id=2)
    path_p19 = planner.plan_pod_dispense(pod_id=19, pan_id=1)

    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor('#1a1a2e')

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_facecolor('#16213e')
    ax1.set_title('Task 2: Micro Dispense — 3D Workspace',
                  color='white', fontsize=11, fontweight='bold')

    # Draw all spice pods
    from path_planning_micro import _COL_A_X, _COL_B_X, _POD_Y, MM2M as mm2m
    for pid, ppose in SPICE_PODS.items():
        col = '#e91e63' if pid <= 10 else '#9c27b0'
        draw_cylinder(ax1,
                      center=[ppose.x, ppose.y, ppose.z],
                      radius=0.018, height=0.06,
                      color=col, alpha=0.6)
        ax1.text(ppose.x, ppose.y, ppose.z + 0.07,
                 str(pid), fontsize=5, ha='center', color='white')

    # Draw pans
    for pid, ppose in PAN_CENTERS.items():
        draw_cylinder(ax1,
                      center=[ppose.x, ppose.y, ppose.z],
                      radius=0.13, height=0.06,
                      color='#4fc3f7', alpha=0.4, label=f'Pan{pid}')

    # Draw paths
    pts_p1 = np.array([[w.pose.x, w.pose.y, w.pose.z] for w in path_p1])
    pts_p19 = np.array([[w.pose.x, w.pose.y, w.pose.z] for w in path_p19])

    ax1.plot(pts_p1[:, 0], pts_p1[:, 1], pts_p1[:, 2],
             'r-o', lw=1.5, ms=4, alpha=0.9, label='Pod1→Pan2')
    ax1.plot(pts_p19[:, 0], pts_p19[:, 1], pts_p19[:, 2],
             'y-o', lw=1.5, ms=4, alpha=0.9, label='Pod19→Pan1')

    ax1.scatter([ROBOT_BASE.x], [ROBOT_BASE.y], [ROBOT_BASE.z],
                s=100, c='lime', marker='^')

    _style_3d_ax(ax1)
    ax1.legend(loc='upper left', fontsize=8, framealpha=0.3,
               labelcolor='white', facecolor='#1a1a2e')

    # ── Side view ─────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(122)
    ax2.set_facecolor('#16213e')
    ax2.set_title('Task 2: Side View (XZ Plane — Pod Rack)',
                  color='white', fontsize=11, fontweight='bold')

    # Pod rack silhouette
    for pid, ppose in SPICE_PODS.items():
        col = '#e91e63' if pid <= 10 else '#9c27b0'
        circ = plt.Circle((ppose.x, ppose.z), 0.018,
                           color=col, alpha=0.7)
        ax2.add_patch(circ)
        ax2.text(ppose.x, ppose.z + 0.025, str(pid),
                 ha='center', fontsize=5, color='white')

    ax2.plot(pts_p1[:, 0], pts_p1[:, 2],
             'r-o', lw=1.5, ms=4, label='Pod1→Pan2')
    ax2.plot(pts_p19[:, 0], pts_p19[:, 2],
             'y-o', lw=1.5, ms=4, label='Pod19→Pan1')

    ax2.set_xlabel('X (m)', color='white')
    ax2.set_ylabel('Z (m)', color='white')
    ax2.tick_params(colors='white')
    ax2.legend(fontsize=8, facecolor='#1a1a2e',
               labelcolor='white', framealpha=0.7)
    ax2.grid(True, color='gray', alpha=0.2)
    for sp in ax2.spines.values():
        sp.set_edgecolor('gray')

    plt.tight_layout()
    plt.savefig('task2_micro_path.png', dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e')
    plt.close()
    print("✅ Saved: task2_micro_path.png")


def visualize_robot_arm_configurations():
    """Show arm configurations at key waypoints."""
    planner = MacroDispensePlanner()
    path = planner.plan_dispense(container_id=5, pan_id=2)

    key_wps = [wp for wp in path if 'STEP' in wp.name and wp.joint_angles][:4]

    fig = plt.figure(figsize=(16, 6))
    fig.patch.set_facecolor('#1a1a2e')

    for idx, wp in enumerate(key_wps):
        ax = fig.add_subplot(1, 4, idx+1, projection='3d')
        ax.set_facecolor('#16213e')
        ax.set_title(wp.name.replace('_', '\n'),
                     color='white', fontsize=7)
        plot_arm_at_waypoint(ax, wp.joint_angles, color='#00e5ff')
        _style_3d_ax(ax)

    plt.suptitle('Piper Arm Configurations — Container 5 → Pan 2',
                 color='white', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('arm_configurations.png', dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e')
    plt.close()
    print("✅ Saved: arm_configurations.png")


def _style_3d_ax(ax):
    ax.set_xlabel('X (m)', color='white', fontsize=8)
    ax.set_ylabel('Y (m)', color='white', fontsize=8)
    ax.set_zlabel('Z (m)', color='white', fontsize=8)
    ax.tick_params(colors='white', labelsize=6)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.grid(True, color='gray', alpha=0.15)


if __name__ == "__main__":
    print("Generating visualizations...")
    visualize_macro_task()
    visualize_micro_task()
    visualize_robot_arm_configurations()
    print("\n✅ All visualizations saved!")
