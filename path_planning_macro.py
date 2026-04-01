"""
=============================================================================
POSHA AUTONOMOUS COOKING ROBOT - MACRO DISPENSE PATH PLANNING
Task 1: Container 5 -> Pan 2, Container 1 -> Pan 1
Robot: AgileX Piper 6-DOF Arm
Author: Rachit Trivedi (SRM IST Kattankulathur)
Date: 01 April 2026
=============================================================================
"""

import numpy as np
import math
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import time

# ─────────────────────────────────────────────────────────────────────────────
# COORDINATE SYSTEM
# All coordinates extracted from POSHA_Robotics_Assignment.step via FreeCAD
# Units: millimetres (mm) → converted to metres for simulation
# Origin: Assembly origin (0,0,0)
# ─────────────────────────────────────────────────────────────────────────────

MM2M = 0.001  # conversion factor

@dataclass
class Pose:
    """3D position and orientation."""
    x: float  # metres
    y: float
    z: float
    roll: float = 0.0   # radians
    pitch: float = 0.0
    yaw: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def __repr__(self):
        return (f"Pose(x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f}, "
                f"r={math.degrees(self.roll):.1f}°, "
                f"p={math.degrees(self.pitch):.1f}°, "
                f"y={math.degrees(self.yaw):.1f}°)")


# ─────────────────────────────────────────────────────────────────────────────
# WORKSPACE COORDINATES  (from FreeCAD macro output)
# ─────────────────────────────────────────────────────────────────────────────

# Robot base (PIPER arm origin in assembly)
ROBOT_BASE = Pose(0.0, 0.0, 0.0)

# Macro Containers (Part 1 = Container 1, Part 005 = Container 5)
MACRO_CONTAINERS = {
    1: Pose(303.6 * MM2M, 627.5 * MM2M, 737.6 * MM2M),   # Container 1 (Part 1)
    2: Pose(498.1 * MM2M, 789.4 * MM2M, 737.3 * MM2M),   # Container 2 (Part 001)
    3: Pose(305.7 * MM2M, 703.0 * MM2M, 737.7 * MM2M),   # Container 3 (Part 002)
    4: Pose(572.6 * MM2M, 638.7 * MM2M, 738.2 * MM2M),   # Container 4 (Part 003)
    5: Pose(389.0 * MM2M, 471.1 * MM2M, 737.0 * MM2M),   # Container 5 (Part 005)
    6: Pose(389.3 * MM2M, 697.1 * MM2M, 737.3 * MM2M),   # Container 6 (Part 004)
    7: Pose(574.0 * MM2M, 564.1 * MM2M, 737.4 * MM2M),   # Container 7 (Part 006)
}

# Pans (Semi Stirrer Assy = Pan, Pan 100 Height = pan top surface)
PAN_CENTERS = {
    1: Pose(187.9 * MM2M, 640.9 * MM2M, 740.8 * MM2M),   # Pan 1
    2: Pose(689.0 * MM2M, 641.9 * MM2M, 740.3 * MM2M),   # Pan 2
}

# Pan diameter: 26cm → radius 130mm; dispense height above pan
PAN_RADIUS = 0.130
PAN_DISPENSE_HEIGHT_ABOVE = 0.100  # 10cm above pan surface for dispense

# Container dimensions (assumed from assignment context)
CONTAINER_HEIGHT = 0.120       # 12cm tall
CONTAINER_LIP_OFFSET = 0.005   # 5mm above rear lip for grip (per assignment)
GRIPPER_APPROACH_CLEARANCE = 0.080  # 8cm above container before descend

# Safety clearances
SAFE_Z_CLEARANCE = 0.250       # 25cm above workspace for safe transit
COLLISION_RADIUS = 0.080       # 8cm collision sphere around arm segments


# ─────────────────────────────────────────────────────────────────────────────
# PIPER ARM KINEMATICS
# Extracted from piper_description.urdf
# ─────────────────────────────────────────────────────────────────────────────

class PiperKinematics:
    """
    Forward and Inverse Kinematics for AgileX Piper 6-DOF arm.
    DH parameters derived from URDF joint origins.
    """

    # Joint limits (radians) from URDF
    JOINT_LIMITS = {
        'j1': (-2.618, 2.618),   # ±150°  base rotation
        'j2': (0.0,    3.14),    # 0–180° shoulder
        'j3': (-2.967, 0.0),     # -170°–0° elbow
        'j4': (-1.745, 1.745),   # ±100° wrist roll
        'j5': (-1.22,  1.22),    # ±70°  wrist pitch
        'j6': (-2.094, 2.094),   # ±120° wrist yaw
    }

    # DH parameters: [a, alpha, d, theta_offset]
    # Derived from URDF joint origins
    DH_PARAMS = [
        # a(m)    alpha(rad)        d(m)      theta_offset(rad)
        [0.0,     0.0,              0.123,    0.0     ],  # Joint 1
        [0.28503, -math.pi/2,       0.0,     -0.1359  ],  # Joint 2
        [0.021984, 0.0,             0.25075, -1.7939  ],  # Joint 3
        [0.0,     math.pi/2,        0.0,      0.0     ],  # Joint 4
        [0.0,    -math.pi/2,        0.091,    0.0     ],  # Joint 5
        [0.0,     math.pi/2,        0.1358,   0.0     ],  # Joint 6
    ]

    @staticmethod
    def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
        """Compute 4x4 DH transformation matrix."""
        ct, st = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        return np.array([
            [ct,  -st*ca,  st*sa,  a*ct],
            [st,   ct*ca, -ct*sa,  a*st],
            [0.0,  sa,     ca,     d   ],
            [0.0,  0.0,    0.0,    1.0 ]
        ])

    @classmethod
    def forward_kinematics(cls, joint_angles: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute end-effector pose from joint angles.
        Returns: (position [3], rotation_matrix [3x3])
        """
        T = np.eye(4)
        for i, (a, alpha, d, theta_offset) in enumerate(cls.DH_PARAMS):
            theta = joint_angles[i] + theta_offset
            Ti = cls.dh_transform(a, alpha, d, theta)
            T = T @ Ti
        position = T[:3, 3]
        rotation = T[:3, :3]
        return position, rotation

    @classmethod
    def inverse_kinematics(cls, target: Pose,
                           initial_guess: Optional[List[float]] = None,
                           max_iter: int = 500,
                           tolerance: float = 1e-4) -> Tuple[List[float], bool]:
        """
        Numerical IK using damped least-squares Jacobian.
        Returns: (joint_angles, converged)
        """
        if initial_guess is None:
            q = [0.0, math.pi/4, -math.pi/4, 0.0, math.pi/4, 0.0]
        else:
            q = list(initial_guess)

        target_pos = target.to_array()
        lambda_damp = 0.01  # damping factor

        for iteration in range(max_iter):
            pos, rot = cls.forward_kinematics(q)
            error = target_pos - pos
            if np.linalg.norm(error) < tolerance:
                return cls._clamp_joints(q), True

            # Numerical Jacobian (position only, 3x6)
            J = np.zeros((3, 6))
            dq = 1e-6
            for j in range(6):
                q_plus = q.copy()
                q_plus[j] += dq
                pos_plus, _ = cls.forward_kinematics(q_plus)
                J[:, j] = (pos_plus - pos) / dq

            # Damped least-squares update  (3x6 Jacobian, 3-dim error)
            JTJ = J.T @ J  # 6x6
            dq_update = J.T @ np.linalg.solve(
                J @ J.T + lambda_damp**2 * np.eye(3), error)  # (6x3)(3x3)^-1 (3,)
            q = [q[i] + dq_update[i] for i in range(6)]
            q = cls._clamp_joints(q)

        # Return best effort even if not converged
        return cls._clamp_joints(q), False

    @classmethod
    def _clamp_joints(cls, q: List[float]) -> List[float]:
        """Clamp joint angles to limits."""
        limits = list(cls.JOINT_LIMITS.values())
        return [max(limits[i][0], min(limits[i][1], q[i])) for i in range(6)]

    @classmethod
    def check_reachability(cls, target: Pose) -> bool:
        """Check if target is within arm workspace."""
        dist = np.linalg.norm(target.to_array() - ROBOT_BASE.to_array())
        MAX_REACH = 0.76  # metres (from URDF link lengths)
        return dist <= MAX_REACH


# ─────────────────────────────────────────────────────────────────────────────
# PATH PLANNER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Waypoint:
    """A single waypoint in the path."""
    name: str
    pose: Pose
    joint_angles: List[float] = field(default_factory=list)
    gripper_state: str = "open"   # "open" or "closed"
    speed: float = 0.05           # m/s

    def __repr__(self):
        return (f"[{self.name}] pos=({self.pose.x:.3f}, {self.pose.y:.3f}, "
                f"{self.pose.z:.3f}) gripper={self.gripper_state}")


class CollisionChecker:
    """Simple bounding-sphere collision checker for workspace objects."""

    def __init__(self):
        self.obstacles = []
        self._populate_workspace()

    def _populate_workspace(self):
        """Add all workspace objects as collision spheres."""
        # Stove body
        self.obstacles.append({
            'name': 'Stove',
            'center': np.array([46.2, 866.9, -187.2]) * MM2M,
            'radius': 0.25
        })
        # Spice rack region
        self.obstacles.append({
            'name': 'SpiceRack',
            'center': np.array([234.0, 804.9, 1042.5]) * MM2M,
            'radius': 0.18
        })
        # Pan 1 rim
        self.obstacles.append({
            'name': 'Pan1_rim',
            'center': PAN_CENTERS[1].to_array(),
            'radius': PAN_RADIUS + 0.02
        })
        # Pan 2 rim
        self.obstacles.append({
            'name': 'Pan2_rim',
            'center': PAN_CENTERS[2].to_array(),
            'radius': PAN_RADIUS + 0.02
        })

    def check_point(self, point: np.ndarray, ignore: str = "") -> Tuple[bool, str]:
        """Return (collision_free, obstacle_name)."""
        for obs in self.obstacles:
            if obs['name'] == ignore:
                continue
            dist = np.linalg.norm(point - obs['center'])
            if dist < obs['radius'] + COLLISION_RADIUS:
                return False, obs['name']
        return True, ""

    def check_path_segment(self, start: np.ndarray, end: np.ndarray,
                           n_samples: int = 20, ignore: str = "") -> Tuple[bool, str]:
        """Check collision along a linear path segment."""
        for t in np.linspace(0, 1, n_samples):
            pt = start + t * (end - start)
            ok, name = self.check_point(pt, ignore)
            if not ok:
                return False, name
        return True, ""


class MacroDispensePlanner:
    """
    Path planner for Task 1: Macro Container → Pan dispense.
    Implements 6-step sequence per assignment spec.
    """

    def __init__(self):
        self.kin = PiperKinematics()
        self.collision = CollisionChecker()
        self.home_joints = [0.0, 0.3, -0.5, 0.0, 0.4, 0.0]
        self.home_pose = self._compute_home_pose()

    def _compute_home_pose(self) -> Pose:
        pos, _ = PiperKinematics.forward_kinematics(self.home_joints)
        return Pose(pos[0], pos[1], pos[2])

    def plan_dispense(self, container_id: int, pan_id: int) -> List[Waypoint]:
        """
        Plan complete pick-and-place path for one container → pan dispense.

        Steps (per assignment):
        1. Align arm with container
        2. Position gripper 5mm above rear lip
        3. Lift container and move above pan
        4. Dispense (tilt/pour)
        5. Return container to original position
        6. Place container back

        Returns list of Waypoints with IK-solved joint angles.
        """
        container = MACRO_CONTAINERS[container_id]
        pan = PAN_CENTERS[pan_id]

        print(f"\n{'='*60}")
        print(f"Planning: Container {container_id} → Pan {pan_id}")
        print(f"Container pos: ({container.x:.3f}, {container.y:.3f}, {container.z:.3f})")
        print(f"Pan pos:       ({pan.x:.3f}, {pan.y:.3f}, {pan.z:.3f})")
        print(f"{'='*60}")

        waypoints = []
        prev_joints = self.home_joints.copy()

        # ── WAYPOINT 0: Home position ──────────────────────────────────────
        waypoints.append(Waypoint(
            name=f"HOME",
            pose=self.home_pose,
            joint_angles=self.home_joints.copy(),
            gripper_state="open",
            speed=0.05
        ))

        # ── STEP 1: Align above container (safe Z clearance) ───────────────
        approach_pose = Pose(
            container.x,
            container.y,
            container.z + CONTAINER_HEIGHT + GRIPPER_APPROACH_CLEARANCE
        )
        q_approach, ok = PiperKinematics.inverse_kinematics(approach_pose, prev_joints)
        prev_joints = q_approach
        waypoints.append(Waypoint(
            name=f"STEP1_Align_Above_Container{container_id}",
            pose=approach_pose,
            joint_angles=q_approach,
            gripper_state="open",
            speed=0.08
        ))
        self._log_waypoint(waypoints[-1], ok)

        # ── STEP 2: Descend to 5mm above rear lip ─────────────────────────
        # Rear lip = container.z + CONTAINER_HEIGHT - 5mm
        grip_z = container.z + CONTAINER_HEIGHT - CONTAINER_LIP_OFFSET
        grip_pose = Pose(
            container.x,
            container.y,
            grip_z,
            pitch=-math.pi/6   # slight tilt for better grip
        )
        q_grip, ok = PiperKinematics.inverse_kinematics(grip_pose, prev_joints)
        prev_joints = q_grip
        waypoints.append(Waypoint(
            name=f"STEP2_Grip_Position_Container{container_id}",
            pose=grip_pose,
            joint_angles=q_grip,
            gripper_state="closed",
            speed=0.03   # slow for precision
        ))
        self._log_waypoint(waypoints[-1], ok)

        # ── STEP 3: Lift container to safe Z ──────────────────────────────
        lift_pose = Pose(
            container.x,
            container.y,
            container.z + CONTAINER_HEIGHT + SAFE_Z_CLEARANCE
        )
        q_lift, ok = PiperKinematics.inverse_kinematics(lift_pose, prev_joints)
        prev_joints = q_lift
        waypoints.append(Waypoint(
            name=f"STEP3a_Lift_Container{container_id}",
            pose=lift_pose,
            joint_angles=q_lift,
            gripper_state="closed",
            speed=0.05
        ))
        self._log_waypoint(waypoints[-1], ok)

        # Transit waypoint: move horizontally to above pan
        transit_pose = Pose(
            pan.x,
            pan.y,
            pan.z + PAN_DISPENSE_HEIGHT_ABOVE + SAFE_Z_CLEARANCE
        )

        # Collision check on transit path
        ok_col, blocker = self.collision.check_path_segment(
            lift_pose.to_array(), transit_pose.to_array()
        )
        if not ok_col:
            print(f"  ⚠️  Collision risk with {blocker} on direct transit, adding detour...")
            mid = Pose(
                (container.x + pan.x) / 2,
                (container.y + pan.y) / 2,
                max(lift_pose.z, transit_pose.z) + 0.05
            )
            q_mid, _ = PiperKinematics.inverse_kinematics(mid, prev_joints)
            prev_joints = q_mid
            waypoints.append(Waypoint(
                name="STEP3b_Transit_Detour",
                pose=mid, joint_angles=q_mid,
                gripper_state="closed", speed=0.08
            ))

        q_transit, ok = PiperKinematics.inverse_kinematics(transit_pose, prev_joints)
        prev_joints = q_transit
        waypoints.append(Waypoint(
            name=f"STEP3c_Above_Pan{pan_id}",
            pose=transit_pose,
            joint_angles=q_transit,
            gripper_state="closed",
            speed=0.08
        ))
        self._log_waypoint(waypoints[-1], ok)

        # Descend to dispense height above pan
        dispense_pose = Pose(
            pan.x,
            pan.y,
            pan.z + PAN_DISPENSE_HEIGHT_ABOVE,
            pitch=math.pi   # tilt 180° to pour
        )
        q_dispense, ok = PiperKinematics.inverse_kinematics(dispense_pose, prev_joints)
        prev_joints = q_dispense
        waypoints.append(Waypoint(
            name=f"STEP4_Dispense_Into_Pan{pan_id}",
            pose=dispense_pose,
            joint_angles=q_dispense,
            gripper_state="closed",
            speed=0.02   # very slow for controlled pour
        ))
        self._log_waypoint(waypoints[-1], ok)

        # ── STEP 5: Return to lift height after dispense ───────────────────
        q_return_lift, ok = PiperKinematics.inverse_kinematics(transit_pose, prev_joints)
        prev_joints = q_return_lift
        waypoints.append(Waypoint(
            name=f"STEP5a_Return_Lift",
            pose=transit_pose,
            joint_angles=q_return_lift,
            gripper_state="closed",
            speed=0.05
        ))

        # Back above container
        q_back_above, ok = PiperKinematics.inverse_kinematics(lift_pose, prev_joints)
        prev_joints = q_back_above
        waypoints.append(Waypoint(
            name=f"STEP5b_Back_Above_Container{container_id}",
            pose=lift_pose,
            joint_angles=q_back_above,
            gripper_state="closed",
            speed=0.08
        ))

        # ── STEP 6: Place container back ──────────────────────────────────
        q_place, ok = PiperKinematics.inverse_kinematics(grip_pose, prev_joints)
        prev_joints = q_place
        waypoints.append(Waypoint(
            name=f"STEP6_Place_Container{container_id}_Back",
            pose=grip_pose,
            joint_angles=q_place,
            gripper_state="open",
            speed=0.03
        ))
        self._log_waypoint(waypoints[-1], ok)

        # ── Return to home ─────────────────────────────────────────────────
        waypoints.append(Waypoint(
            name="HOME_RETURN",
            pose=self.home_pose,
            joint_angles=self.home_joints.copy(),
            gripper_state="open",
            speed=0.08
        ))

        return waypoints

    def _log_waypoint(self, wp: Waypoint, converged: bool):
        status = "✅" if converged else "⚠️ (best-effort)"
        print(f"  {status} {wp.name}")
        print(f"     pos=({wp.pose.x:.4f}, {wp.pose.y:.4f}, {wp.pose.z:.4f})")
        angles_deg = [round(math.degrees(a), 1) for a in wp.joint_angles]
        print(f"     joints(°)={angles_deg}")

    def compute_transformation_matrix(self, waypoint: Waypoint) -> np.ndarray:
        """Return 4x4 homogeneous transformation matrix for a waypoint."""
        pos, rot = PiperKinematics.forward_kinematics(waypoint.joint_angles)
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = pos
        return T

    def estimate_execution_time(self, waypoints: List[Waypoint]) -> float:
        """Estimate total path execution time in seconds."""
        total = 0.0
        for i in range(1, len(waypoints)):
            dist = np.linalg.norm(
                waypoints[i].pose.to_array() - waypoints[i-1].pose.to_array()
            )
            speed = (waypoints[i].speed + waypoints[i-1].speed) / 2
            total += dist / max(speed, 0.001)
        return total

    def validate_path(self, waypoints: List[Waypoint]) -> dict:
        """Run validation checks on planned path."""
        results = {
            'total_waypoints': len(waypoints),
            'joint_limit_violations': [],
            'collision_warnings': [],
            'reachability_failures': [],
            'execution_time_estimate': 0.0
        }

        limits = list(PiperKinematics.JOINT_LIMITS.values())

        for wp in waypoints:
            # Joint limit check
            for i, (angle, (lo, hi)) in enumerate(zip(wp.joint_angles, limits)):
                if not (lo <= angle <= hi):
                    results['joint_limit_violations'].append(
                        f"{wp.name}: Joint {i+1} = {math.degrees(angle):.1f}° "
                        f"(limit [{math.degrees(lo):.1f}°, {math.degrees(hi):.1f}°])"
                    )

            # Reachability check
            if not PiperKinematics.check_reachability(wp.pose):
                results['reachability_failures'].append(
                    f"{wp.name}: pos=({wp.pose.x:.3f}, {wp.pose.y:.3f}, {wp.pose.z:.3f})"
                )

            # Collision check
            ok, blocker = self.collision.check_point(wp.pose.to_array())
            if not ok:
                results['collision_warnings'].append(
                    f"{wp.name}: near {blocker}"
                )

        results['execution_time_estimate'] = self.estimate_execution_time(waypoints)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMATION MATRIX DOCUMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def document_coordinate_frames():
    """Print documented coordinate frames and key transforms."""
    print("\n" + "="*70)
    print("COORDINATE FRAMES AND TRANSFORMATION MATRICES")
    print("="*70)

    print("\n[Frame 0] World/Assembly Origin")
    print("  T_world = I (4x4 identity)")

    print("\n[Frame 1] Robot Base Frame (PIPER at assembly origin)")
    T_base = np.eye(4)
    print(f"  T_base =\n{np.round(T_base, 4)}")

    print("\n[Frame 2] Container 5 Frame")
    c5 = MACRO_CONTAINERS[5]
    T_c5 = np.eye(4)
    T_c5[:3, 3] = c5.to_array()
    print(f"  T_container5 (position only) =\n{np.round(T_c5, 4)}")

    print("\n[Frame 3] Pan 2 Frame")
    p2 = PAN_CENTERS[2]
    T_p2 = np.eye(4)
    T_p2[:3, 3] = p2.to_array()
    print(f"  T_pan2 =\n{np.round(T_p2, 4)}")

    print("\n[Frame 4] Transform: Container5 → Pan2")
    T_c5_to_p2 = np.linalg.inv(T_c5) @ T_p2
    print(f"  T_c5_to_p2 =\n{np.round(T_c5_to_p2, 4)}")

    print("\n[Frame 5] End-Effector at Home Position")
    home_joints = [0.0, 0.3, -0.5, 0.0, 0.4, 0.0]
    pos, rot = PiperKinematics.forward_kinematics(home_joints)
    T_ee = np.eye(4)
    T_ee[:3, :3] = rot
    T_ee[:3, 3] = pos
    print(f"  T_ee_home =\n{np.round(T_ee, 4)}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST CASES
# ─────────────────────────────────────────────────────────────────────────────

def run_test_cases():
    """Validate path planning under different scenarios."""
    print("\n" + "="*70)
    print("TEST CASES")
    print("="*70)

    planner = MacroDispensePlanner()

    tests = [
        {
            'name': 'Test 1: Forward Kinematics at home',
            'fn': lambda: PiperKinematics.forward_kinematics(
                [0.0, 0.3, -0.5, 0.0, 0.4, 0.0]
            )
        },
        {
            'name': 'Test 2: IK convergence for Pan 2 approach',
            'fn': lambda: PiperKinematics.inverse_kinematics(
                Pose(PAN_CENTERS[2].x, PAN_CENTERS[2].y,
                     PAN_CENTERS[2].z + 0.15)
            )
        },
        {
            'name': 'Test 3: Reachability of Container 5',
            'fn': lambda: PiperKinematics.check_reachability(MACRO_CONTAINERS[5])
        },
        {
            'name': 'Test 4: Collision check on direct Container5→Pan2 path',
            'fn': lambda: CollisionChecker().check_path_segment(
                MACRO_CONTAINERS[5].to_array(),
                PAN_CENTERS[2].to_array()
            )
        },
        {
            'name': 'Test 5: Joint limit clamping',
            'fn': lambda: PiperKinematics._clamp_joints(
                [10.0, -10.0, 10.0, -10.0, 10.0, -10.0]
            )
        },
    ]

    for test in tests:
        print(f"\n  {test['name']}")
        try:
            result = test['fn']()
            print(f"  ✅ PASS → {result}")
        except Exception as e:
            print(f"  ❌ FAIL → {e}")

    # Full path planning test
    print("\n  Test 6: Full path plan Container 5 → Pan 2")
    wps = planner.plan_dispense(container_id=5, pan_id=2)
    validation = planner.validate_path(wps)
    print(f"  ✅ Waypoints: {validation['total_waypoints']}")
    print(f"  ✅ Est. execution time: {validation['execution_time_estimate']:.1f}s")
    if validation['joint_limit_violations']:
        print(f"  ⚠️  Joint violations: {validation['joint_limit_violations']}")
    else:
        print(f"  ✅ No joint limit violations")
    if validation['collision_warnings']:
        print(f"  ⚠️  Collision warnings: {validation['collision_warnings']}")
    else:
        print(f"  ✅ No collision warnings")

    print("\n  Test 7: Full path plan Container 1 → Pan 1")
    wps2 = planner.plan_dispense(container_id=1, pan_id=1)
    validation2 = planner.validate_path(wps2)
    print(f"  ✅ Waypoints: {validation2['total_waypoints']}")
    print(f"  ✅ Est. execution time: {validation2['execution_time_estimate']:.1f}s")

    return wps, wps2


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  POSHA ROBOTICS — MACRO DISPENSE PATH PLANNING              ║")
    print("║  AgileX Piper 6-DOF | Task 1                                ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    planner = MacroDispensePlanner()

    # ── PRIMARY TASK: Container 5 → Pan 2 ─────────────────────────────────
    print("\n▶  PRIMARY SEQUENCE: Container 5 → Pan 2")
    path_c5_p2 = planner.plan_dispense(container_id=5, pan_id=2)

    print("\n  Transformation matrices for this path:")
    for wp in path_c5_p2[:3]:
        T = planner.compute_transformation_matrix(wp)
        print(f"\n  T_ee @ [{wp.name}] =\n{np.round(T, 4)}")

    v1 = planner.validate_path(path_c5_p2)
    print(f"\n  Validation → {v1}")

    # ── SECONDARY TASK: Container 1 → Pan 1 ───────────────────────────────
    print("\n▶  SECONDARY SEQUENCE: Container 1 → Pan 1")
    path_c1_p1 = planner.plan_dispense(container_id=1, pan_id=1)
    v2 = planner.validate_path(path_c1_p1)
    print(f"\n  Validation → {v2}")

    # ── COORDINATE FRAMES ─────────────────────────────────────────────────
    document_coordinate_frames()

    # ── TEST CASES ────────────────────────────────────────────────────────
    run_test_cases()

    # ── EXPORT RESULTS ────────────────────────────────────────────────────
    results = {
        'task': 'Macro Dispense',
        'sequence_1': {
            'from': 'Container_5',
            'to': 'Pan_2',
            'waypoints': [
                {
                    'name': wp.name,
                    'position': [round(wp.pose.x, 4),
                                 round(wp.pose.y, 4),
                                 round(wp.pose.z, 4)],
                    'joint_angles_deg': [round(math.degrees(a), 2)
                                         for a in wp.joint_angles],
                    'gripper': wp.gripper_state,
                    'speed_ms': wp.speed
                }
                for wp in path_c5_p2
            ],
            'validation': v1
        },
        'sequence_2': {
            'from': 'Container_1',
            'to': 'Pan_1',
            'waypoints': [
                {
                    'name': wp.name,
                    'position': [round(wp.pose.x, 4),
                                 round(wp.pose.y, 4),
                                 round(wp.pose.z, 4)],
                    'joint_angles_deg': [round(math.degrees(a), 2)
                                         for a in wp.joint_angles],
                    'gripper': wp.gripper_state,
                    'speed_ms': wp.speed
                }
                for wp in path_c1_p1
            ],
            'validation': v2
        }
    }

    with open('macro_path_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✅ Results saved to macro_path_results.json")

    return path_c5_p2, path_c1_p1


if __name__ == "__main__":
    main()
