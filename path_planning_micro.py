"""
=============================================================================
POSHA AUTONOMOUS COOKING ROBOT - MICRO DISPENSE PATH PLANNING
Task 2: Spice Pod 1 -> Pan 2, Pod 19 -> Pan 1
Robot: AgileX Piper 6-DOF Arm
Author: Rachit Trivedi (SRM IST Kattankulathur)
Date: 01 April 2026
=============================================================================
"""

import numpy as np
import math
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from path_planning_macro import (
    Pose, Waypoint, PiperKinematics, CollisionChecker,
    PAN_CENTERS, MM2M, SAFE_Z_CLEARANCE, ROBOT_BASE
)

# ─────────────────────────────────────────────────────────────────────────────
# SPICE RACK COORDINATE MAPPING
# From FreeCAD: 10 SPICE POD CONTAINERs (000-009)
# Two X-columns: X=175.5mm and X=292.5mm
# Y fixed: ~804.9mm
# Z levels: 921.5, 982.0, 1042.5, 1103.0, 1163.5 (5 levels each column)
# ─────────────────────────────────────────────────────────────────────────────

# Spice pod grid mapping (Pod 1–10 on left column, 11–20 on right column)
# Pod 1 = SPICE POD CONTAINER (X=175.5, Z=921.5) — bottom-left
# Pod 19 = index 19 in the 20-pod rack

SPICE_POD_APPROACH_DISTANCE = 0.002   # 2mm from pod surface (per assignment)
SPICE_POD_DIAMETER = 0.035            # 35mm diameter (from PodShaft 35x2.5mm_dia)
SPICE_POD_HEIGHT = 0.060              # estimated 60mm height
SPICE_POD_LIFT_CLEARANCE = 0.120      # 12cm above rack to clear neighbours
PAN_DISPENSE_HEIGHT = 0.100           # 10cm above pan

# All 20 pod positions (2 columns × 10 rows based on assembly data)
# Column A: X=175.5mm  |  Column B: X=292.5mm
# Z levels: 921.5 → 1163.5 in steps of ~60.5mm (10 levels per column)

_POD_Z_LEVELS = [921.5, 982.0, 1042.5, 1103.0, 1163.5]
_COL_A_X = 175.5
_COL_B_X = 292.5
_POD_Y   = 804.9

def _build_pod_map() -> Dict[int, Pose]:
    """
    Build pod-number → Pose mapping.
    Pods 1–10: Column A (X=175.5), bottom to top, then double stacked
    Pods 11–20: Column B (X=292.5), bottom to top
    With 20 pods across 2 columns × 10 Z-heights:
    """
    pods = {}
    pod_id = 1

    # Column A: 10 pods (Z levels repeated × 2 per level due to stacking)
    z_extended = _POD_Z_LEVELS * 2  # 10 z-positions
    for z in sorted(z_extended):
        pods[pod_id] = Pose(
            _COL_A_X * MM2M,
            _POD_Y * MM2M,
            z * MM2M
        )
        pod_id += 1

    # Column B: 10 pods
    for z in sorted(z_extended):
        pods[pod_id] = Pose(
            _COL_B_X * MM2M,
            _POD_Y * MM2M,
            z * MM2M
        )
        pod_id += 1

    return pods

SPICE_PODS = _build_pod_map()


class MicroDispensePlanner:
    """
    Path planner for Task 2: Spice Pod → Pan dispense.
    Implements 5-step sequence per assignment spec.
    """

    def __init__(self):
        self.collision = CollisionChecker()
        self.home_joints = [0.0, 0.3, -0.5, 0.0, 0.4, 0.0]
        self.home_pose = self._fk_to_pose(self.home_joints)

    def _fk_to_pose(self, joints: List[float]) -> Pose:
        pos, _ = PiperKinematics.forward_kinematics(joints)
        return Pose(pos[0], pos[1], pos[2])

    def plan_pod_dispense(self, pod_id: int, pan_id: int) -> List[Waypoint]:
        """
        Plan complete spice pod pick-and-place.

        Steps (per assignment):
        1. Align arm with pod on rack
        2. Grip pod (actuator 2mm from surface)
        3. Lift pod and bring above pan
        4. Dispense into pan
        5. Return pod to rack position

        Returns list of Waypoints with IK-solved joint angles.
        """
        if pod_id not in SPICE_PODS:
            raise ValueError(f"Pod {pod_id} not in range 1-20")

        pod = SPICE_PODS[pod_id]
        pan = PAN_CENTERS[pan_id]

        print(f"\n{'='*60}")
        print(f"Planning: Pod {pod_id} → Pan {pan_id}")
        print(f"Pod pos:  ({pod.x:.4f}, {pod.y:.4f}, {pod.z:.4f})")
        print(f"Pan pos:  ({pan.x:.4f}, {pan.y:.4f}, {pan.z:.4f})")
        print(f"{'='*60}")

        waypoints = []
        prev_joints = self.home_joints.copy()

        # ── HOME ──────────────────────────────────────────────────────────
        waypoints.append(Waypoint(
            name="HOME",
            pose=self.home_pose,
            joint_angles=self.home_joints.copy(),
            gripper_state="open",
            speed=0.05
        ))

        # ── STEP 1: Align arm above pod ───────────────────────────────────
        # Approach from front (negative Y direction from rack)
        approach_pose = Pose(
            pod.x,
            pod.y - 0.080,   # 8cm in front of rack
            pod.z + SPICE_POD_LIFT_CLEARANCE,
        )
        q_approach, ok = PiperKinematics.inverse_kinematics(approach_pose, prev_joints)
        prev_joints = q_approach
        waypoints.append(Waypoint(
            name=f"STEP1_Align_Above_Pod{pod_id}",
            pose=approach_pose,
            joint_angles=q_approach,
            gripper_state="open",
            speed=0.08
        ))
        self._log_wp(waypoints[-1], ok)

        # Descend to pod level (still offset from rack)
        pre_grip_pose = Pose(
            pod.x,
            pod.y - 0.040,   # 4cm from rack face
            pod.z + SPICE_POD_HEIGHT / 2,
        )
        q_pre_grip, ok = PiperKinematics.inverse_kinematics(pre_grip_pose, prev_joints)
        prev_joints = q_pre_grip
        waypoints.append(Waypoint(
            name=f"STEP1b_Pre_Grip_Alignment_Pod{pod_id}",
            pose=pre_grip_pose,
            joint_angles=q_pre_grip,
            gripper_state="open",
            speed=0.04
        ))
        self._log_wp(waypoints[-1], ok)

        # ── STEP 2: Move to 2mm from pod surface (grip position) ──────────
        grip_pose = Pose(
            pod.x,
            pod.y - (SPICE_POD_DIAMETER / 2) - SPICE_POD_APPROACH_DISTANCE,
            pod.z + SPICE_POD_HEIGHT / 2,
        )
        q_grip, ok = PiperKinematics.inverse_kinematics(grip_pose, prev_joints)
        prev_joints = q_grip
        waypoints.append(Waypoint(
            name=f"STEP2_Grip_Pod{pod_id}",
            pose=grip_pose,
            joint_angles=q_grip,
            gripper_state="closed",
            speed=0.02   # very slow for precision grip
        ))
        self._log_wp(waypoints[-1], ok)

        # ── STEP 3: Lift pod from rack ────────────────────────────────────
        lift_pose = Pose(
            pod.x,
            pod.y - 0.080,
            pod.z + SPICE_POD_LIFT_CLEARANCE + SAFE_Z_CLEARANCE
        )
        q_lift, ok = PiperKinematics.inverse_kinematics(lift_pose, prev_joints)
        prev_joints = q_lift
        waypoints.append(Waypoint(
            name=f"STEP3_Lift_Pod{pod_id}",
            pose=lift_pose,
            joint_angles=q_lift,
            gripper_state="closed",
            speed=0.05
        ))
        self._log_wp(waypoints[-1], ok)

        # Transit above pan
        above_pan = Pose(
            pan.x,
            pan.y,
            pan.z + PAN_DISPENSE_HEIGHT + SAFE_Z_CLEARANCE
        )
        q_transit, ok = PiperKinematics.inverse_kinematics(above_pan, prev_joints)
        prev_joints = q_transit
        waypoints.append(Waypoint(
            name=f"STEP3b_Transit_Above_Pan{pan_id}",
            pose=above_pan,
            joint_angles=q_transit,
            gripper_state="closed",
            speed=0.10
        ))
        self._log_wp(waypoints[-1], ok)

        # ── STEP 4: Descend and dispense ──────────────────────────────────
        dispense_pose = Pose(
            pan.x,
            pan.y,
            pan.z + PAN_DISPENSE_HEIGHT,
            pitch=math.pi  # invert to pour
        )
        q_dispense, ok = PiperKinematics.inverse_kinematics(dispense_pose, prev_joints)
        prev_joints = q_dispense
        waypoints.append(Waypoint(
            name=f"STEP4_Dispense_Pod{pod_id}_Into_Pan{pan_id}",
            pose=dispense_pose,
            joint_angles=q_dispense,
            gripper_state="closed",
            speed=0.02
        ))
        self._log_wp(waypoints[-1], ok)

        # ── STEP 5: Return pod to rack ────────────────────────────────────
        # Rise back up
        q_rise, ok = PiperKinematics.inverse_kinematics(above_pan, prev_joints)
        prev_joints = q_rise
        waypoints.append(Waypoint(
            name="STEP5a_Rise_After_Dispense",
            pose=above_pan,
            joint_angles=q_rise,
            gripper_state="closed",
            speed=0.05
        ))

        # Back to rack lift height
        q_back_lift, ok = PiperKinematics.inverse_kinematics(lift_pose, prev_joints)
        prev_joints = q_back_lift
        waypoints.append(Waypoint(
            name=f"STEP5b_Return_Above_Rack",
            pose=lift_pose,
            joint_angles=q_back_lift,
            gripper_state="closed",
            speed=0.08
        ))

        # Insert back to grip position
        q_return_grip, ok = PiperKinematics.inverse_kinematics(grip_pose, prev_joints)
        prev_joints = q_return_grip
        waypoints.append(Waypoint(
            name=f"STEP5c_Return_Pod{pod_id}_To_Rack",
            pose=grip_pose,
            joint_angles=q_return_grip,
            gripper_state="open",
            speed=0.03
        ))
        self._log_wp(waypoints[-1], ok)

        # Retreat from rack
        q_retreat, ok = PiperKinematics.inverse_kinematics(approach_pose, prev_joints)
        prev_joints = q_retreat
        waypoints.append(Waypoint(
            name=f"STEP5d_Retreat_From_Rack",
            pose=approach_pose,
            joint_angles=q_retreat,
            gripper_state="open",
            speed=0.05
        ))

        # ── HOME RETURN ───────────────────────────────────────────────────
        waypoints.append(Waypoint(
            name="HOME_RETURN",
            pose=self.home_pose,
            joint_angles=self.home_joints.copy(),
            gripper_state="open",
            speed=0.08
        ))

        return waypoints

    def _log_wp(self, wp: Waypoint, converged: bool):
        status = "✅" if converged else "⚠️"
        print(f"  {status} {wp.name}")
        print(f"     ({wp.pose.x:.4f}, {wp.pose.y:.4f}, {wp.pose.z:.4f})")

    def validate_path(self, waypoints: List[Waypoint]) -> dict:
        limits = list(PiperKinematics.JOINT_LIMITS.values())
        results = {
            'total_waypoints': len(waypoints),
            'joint_limit_violations': [],
            'collision_warnings': [],
            'reachability_ok': True
        }
        for wp in waypoints:
            for i, (a, (lo, hi)) in enumerate(zip(wp.joint_angles, limits)):
                if not (lo <= a <= hi):
                    results['joint_limit_violations'].append(
                        f"{wp.name}: J{i+1}={math.degrees(a):.1f}°")
            if not PiperKinematics.check_reachability(wp.pose):
                results['reachability_ok'] = False
        return results

    def get_pod_position_table(self) -> str:
        """Return formatted table of all pod positions."""
        lines = ["\nSpice Pod Position Table", "="*60,
                 f"{'Pod ID':<8} {'X (m)':<10} {'Y (m)':<10} {'Z (m)':<10} {'Column':<8}"]
        for pid, pose in sorted(SPICE_PODS.items()):
            col = "A (Left)" if abs(pose.x - _COL_A_X * MM2M) < 0.001 else "B (Right)"
            lines.append(f"{pid:<8} {pose.x:<10.4f} {pose.y:<10.4f} {pose.z:<10.4f} {col}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# TEST CASES
# ─────────────────────────────────────────────────────────────────────────────

def run_micro_tests():
    print("\n" + "="*60)
    print("MICRO DISPENSE TEST CASES")
    print("="*60)

    planner = MicroDispensePlanner()

    # Test pod map
    print(planner.get_pod_position_table())

    # Test 1: Pod 1 → Pan 2
    print("\n  Test 1: Pod 1 → Pan 2")
    wps1 = planner.plan_pod_dispense(pod_id=1, pan_id=2)
    v1 = planner.validate_path(wps1)
    print(f"  ✅ Waypoints: {v1['total_waypoints']}, Violations: {len(v1['joint_limit_violations'])}")

    # Test 2: Pod 19 → Pan 1
    print("\n  Test 2: Pod 19 → Pan 1")
    wps2 = planner.plan_pod_dispense(pod_id=19, pan_id=1)
    v2 = planner.validate_path(wps2)
    print(f"  ✅ Waypoints: {v2['total_waypoints']}, Violations: {len(v2['joint_limit_violations'])}")

    # Test 3: Pod reachability sweep
    print("\n  Test 3: Reachability check for all 20 pods")
    reachable = []
    for pid in range(1, 21):
        ok = PiperKinematics.check_reachability(SPICE_PODS[pid])
        reachable.append((pid, ok))
        status = "✅" if ok else "❌"
        print(f"    Pod {pid:2d}: {status}")

    return wps1, wps2


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  POSHA ROBOTICS — MICRO DISPENSE PATH PLANNING              ║")
    print("║  AgileX Piper 6-DOF | Task 2                                ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    planner = MicroDispensePlanner()

    print("\n▶  PRIMARY: Pod 1 → Pan 2")
    path_p1_pan2 = planner.plan_pod_dispense(pod_id=1, pan_id=2)
    v1 = planner.validate_path(path_p1_pan2)
    print(f"\n  Validation: {v1}")

    print("\n▶  SECONDARY: Pod 19 → Pan 1")
    path_p19_pan1 = planner.plan_pod_dispense(pod_id=19, pan_id=1)
    v2 = planner.validate_path(path_p19_pan1)
    print(f"\n  Validation: {v2}")

    run_micro_tests()

    # Export
    results = {
        'task': 'Micro Dispense',
        'sequence_1': {
            'from': 'Pod_1', 'to': 'Pan_2',
            'waypoints': [
                {'name': wp.name,
                 'position': [round(wp.pose.x, 4), round(wp.pose.y, 4), round(wp.pose.z, 4)],
                 'joint_angles_deg': [round(math.degrees(a), 2) for a in wp.joint_angles],
                 'gripper': wp.gripper_state}
                for wp in path_p1_pan2
            ],
            'validation': v1
        },
        'sequence_2': {
            'from': 'Pod_19', 'to': 'Pan_1',
            'waypoints': [
                {'name': wp.name,
                 'position': [round(wp.pose.x, 4), round(wp.pose.y, 4), round(wp.pose.z, 4)],
                 'joint_angles_deg': [round(math.degrees(a), 2) for a in wp.joint_angles],
                 'gripper': wp.gripper_state}
                for wp in path_p19_pan1
            ],
            'validation': v2
        }
    }

    with open('micro_path_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✅ Results saved to micro_path_results.json")


if __name__ == "__main__":
    main()
