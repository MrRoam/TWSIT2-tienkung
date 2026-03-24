"""
Side-by-Side Visualization: BVH Skeleton vs Retargeted Robot (Reproduction Version)

Displays BVH skeleton (balls and sticks) next to the robot model
to visually compare the retargeting quality.

Usage:
    python pyroki/visualize_comparison.py \
        --bvh pyroki/data/raw_data/lafan1/walk1_subject1.bvh \
        --pkl pyroki/outputs_test/test_2.pkl \
        --urdf assets/Tienkung/urdf/walker_tienkung_ei.urdf

    # Windows absolute path example:
    python pyroki/visualize_comparison.py \
        --bvh D:/Desktop/TWIST2/pyroki/data/raw_data/lafan1/walk1_subject1.bvh \
        --pkl D:/Desktop/TWIST2/pyroki/outputs_test/test_2.pkl \
        --urdf D:/Desktop/TWIST2/assets/Tienkung/urdf/walker_tienkung_ei.urdf
"""

import sys
import os
import io
import argparse
import pickle
import time
import numpy as np
from typing import Dict, List

# Set UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.bvh_loader import BVHLoader
from utils.bvh_fk import compute_bvh_fk
from utils.coordinate_transform import CoordinateTransform


def get_default_tienkung_urdf(script_dir: str) -> str:
    repo_root = os.path.dirname(script_dir)
    return os.path.join(repo_root, "assets", "Tienkung", "urdf", "walker_tienkung_ei.urdf")

# Try to import yourdfpy for joint names extraction
try:
    import yourdfpy
except ImportError:
    print("Warning: yourdfpy not found. Assuming URDF joint order matches PKL.")
    yourdfpy = None

# BVH skeleton structure (LaFan1)
BVH_SKELETON_CONNECTIONS = [
    # Spine
    ('Hips', 'Spine'),
    ('Spine', 'Spine1'),
    ('Spine1', 'Spine2'),
    ('Spine2', 'Neck'),
    ('Neck', 'Head'),
    
    # Left leg
    ('Hips', 'LeftUpLeg'),
    ('LeftUpLeg', 'LeftLeg'),
    ('LeftLeg', 'LeftFoot'),
    ('LeftFoot', 'LeftToe'),
    
    # Right leg
    ('Hips', 'RightUpLeg'),
    ('RightUpLeg', 'RightLeg'),
    ('RightLeg', 'RightFoot'),
    ('RightFoot', 'RightToe'),
    
    # Left arm
    ('Spine2', 'LeftShoulder'),
    ('LeftShoulder', 'LeftArm'),
    ('LeftArm', 'LeftForeArm'),
    ('LeftForeArm', 'LeftHand'),
    
    # Right arm
    ('Spine2', 'RightShoulder'),
    ('RightShoulder', 'RightArm'),
    ('RightArm', 'RightForeArm'),
    ('RightForeArm', 'RightHand'),
]


def load_pkl_data(pkl_path: str) -> dict:
    """Load PKL motion data."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


class BVHSkeletonVisualizer:
    """Visualizes BVH skeleton using PyBullet debug lines and spheres."""
    
    def __init__(self, pybullet_client, offset: np.ndarray = np.array([0, 0, 0]), 
                 yaw_rotation: float = 0.0):
        """
        Args:
            pybullet_client: PyBullet physics client
            offset: Offset to apply to all positions (for side-by-side display)
            yaw_rotation: Yaw angle in radians to rotate BVH skeleton (for orientation alignment)
        """
        self.p = pybullet_client
        self.offset = offset
        self.yaw_rotation = yaw_rotation
        self.debug_point_ids = []  # Debug point IDs
        self.bone_lines = []  # List of line IDs
        self.frame_count = 0
        self.coord_trans = CoordinateTransform()
        
        # Precompute rotation matrix for yaw
        cos_yaw = np.cos(yaw_rotation)
        sin_yaw = np.sin(yaw_rotation)
        self.yaw_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
    
    def _transform_position(self, pos: np.ndarray, hip_pos = None) -> np.ndarray:
        """Transform BVH position to world position with rotation and offset."""
        # Transform to robot frame using the utility class
        pos_robot = self.coord_trans.transform_position(pos.reshape(1, 3))[0]
        
        # Apply yaw rotation around hip position
        if hip_pos is not None and self.yaw_rotation != 0:
            hip_robot = self.coord_trans.transform_position(hip_pos.reshape(1, 3))[0]
            # Rotate around hip XY position
            relative = pos_robot - np.array([hip_robot[0], hip_robot[1], 0])
            rotated = self.yaw_matrix @ relative
            pos_robot = rotated + np.array([hip_robot[0], hip_robot[1], 0])
        
        # Apply offset
        return pos_robot + self.offset
    
    def draw_frame(self, joint_positions: Dict[str, np.ndarray], 
                   connections: List[tuple] = BVH_SKELETON_CONNECTIONS):
        """
        Draw BVH skeleton for one frame using debug lines with replaceItemUniqueId.
        """
        # Check connection first
        if not self.p.isConnected():
            return
        
        # Get hip position for rotation center (in BVH frame)
        hip_pos_bvh = joint_positions.get('Hips', None)
        
        # Draw joints as small spheres using debug points
        point_idx = 0
        for joint_name, pos in joint_positions.items():
            if not self.p.isConnected():
                return
            
            # Transform to world position (with rotation and offset)
            pos_world = self._transform_position(pos, hip_pos_bvh)
            
            # Color joints by body part
            if 'Left' in joint_name:
                color = [1, 0, 0]  # Red
            elif 'Right' in joint_name:
                color = [0, 1, 0]  # Green
            else:
                color = [0, 0.5, 1]  # Light blue
            
            # Draw a small cross at the joint position (3 lines per joint)
            cross_size = 0.04
            cross_lines = [
                ([pos_world[0] - cross_size, pos_world[1], pos_world[2]],
                 [pos_world[0] + cross_size, pos_world[1], pos_world[2]]),
                ([pos_world[0], pos_world[1] - cross_size, pos_world[2]],
                 [pos_world[0], pos_world[1] + cross_size, pos_world[2]]),
                ([pos_world[0], pos_world[1], pos_world[2] - cross_size],
                 [pos_world[0], pos_world[1], pos_world[2] + cross_size]),
            ]
            
            for start, end in cross_lines:
                try:
                    if point_idx < len(self.debug_point_ids):
                        # Update existing line (no flicker)
                        self.debug_point_ids[point_idx] = self.p.addUserDebugLine(
                            start, end,
                            lineColorRGB=color,
                            lineWidth=3,
                            lifeTime=0,
                            replaceItemUniqueId=self.debug_point_ids[point_idx]
                        )
                    else:
                        # Create new line (first frame only)
                        line_id = self.p.addUserDebugLine(
                            start, end,
                            lineColorRGB=color,
                            lineWidth=3,
                            lifeTime=0
                        )
                        self.debug_point_ids.append(line_id)
                    point_idx += 1
                except:
                    pass
        
        # Draw bones as lines
        bone_idx = 0
        for joint1, joint2 in connections:
            if not self.p.isConnected():
                return
            if joint1 in joint_positions and joint2 in joint_positions:
                pos1 = self._transform_position(joint_positions[joint1], hip_pos_bvh)
                pos2 = self._transform_position(joint_positions[joint2], hip_pos_bvh)
                
                try:
                    if bone_idx < len(self.bone_lines):
                        # Update existing line (no flicker)
                        self.bone_lines[bone_idx] = self.p.addUserDebugLine(
                            pos1.tolist(),
                            pos2.tolist(),
                            lineColorRGB=[1, 1, 0],  # Yellow for bones
                            lineWidth=2,
                            lifeTime=0,
                            replaceItemUniqueId=self.bone_lines[bone_idx]
                        )
                    else:
                        # Create new line (first frame only)
                        line_id = self.p.addUserDebugLine(
                            pos1.tolist(),
                            pos2.tolist(),
                            lineColorRGB=[1, 1, 0],  # Yellow for bones
                            lineWidth=2,
                            lifeTime=0
                        )
                        self.bone_lines.append(line_id)
                    bone_idx += 1
                except:
                    pass
        
        self.frame_count += 1
    
    def clear(self):
        """Clear all visual elements."""
        for point_id in self.debug_point_ids:
            try:
                self.p.removeUserDebugItem(point_id)
            except:
                pass
        self.debug_point_ids.clear()
        
        for line_id in self.bone_lines:
            try:
                self.p.removeUserDebugItem(line_id)
            except:
                pass
        self.bone_lines.clear()


class RobotLinkVisualizer:
    """Draw the robot link skeleton with debug lines, independent of STL rendering."""

    def __init__(self, pybullet_client, robot_id: int):
        self.p = pybullet_client
        self.robot_id = robot_id
        self.cross_ids = []
        self.bone_ids = []
        self.parent_map = {}
        self.link_names = {-1: "pelvis"}

        for i in range(self.p.getNumJoints(robot_id)):
            joint_info = self.p.getJointInfo(robot_id, i)
            self.parent_map[i] = joint_info[16]
            self.link_names[i] = joint_info[12].decode("utf-8")

    def _get_link_world_position(self, link_index: int):
        if link_index == -1:
            pos, _ = self.p.getBasePositionAndOrientation(self.robot_id)
            return np.array(pos)
        state = self.p.getLinkState(self.robot_id, link_index, computeForwardKinematics=True)
        return np.array(state[4])

    def _get_link_color(self, link_name: str):
        if "_l_" in link_name:
            return [1, 0.2, 0.2]
        if "_r_" in link_name:
            return [0.2, 1, 0.2]
        return [0.2, 0.8, 1]

    def draw(self):
        if not self.p.isConnected():
            return

        cross_idx = 0
        bone_idx = 0
        cross_size = 0.035

        for link_index in range(-1, self.p.getNumJoints(self.robot_id)):
            link_name = self.link_names[link_index]
            pos = self._get_link_world_position(link_index)
            color = self._get_link_color(link_name)

            cross_lines = [
                ([pos[0] - cross_size, pos[1], pos[2]], [pos[0] + cross_size, pos[1], pos[2]]),
                ([pos[0], pos[1] - cross_size, pos[2]], [pos[0], pos[1] + cross_size, pos[2]]),
                ([pos[0], pos[1], pos[2] - cross_size], [pos[0], pos[1], pos[2] + cross_size]),
            ]
            for start, end in cross_lines:
                if cross_idx < len(self.cross_ids):
                    self.cross_ids[cross_idx] = self.p.addUserDebugLine(
                        start,
                        end,
                        lineColorRGB=color,
                        lineWidth=2,
                        lifeTime=0,
                        replaceItemUniqueId=self.cross_ids[cross_idx],
                    )
                else:
                    self.cross_ids.append(
                        self.p.addUserDebugLine(start, end, lineColorRGB=color, lineWidth=2, lifeTime=0)
                    )
                cross_idx += 1

            if link_index != -1:
                parent_index = self.parent_map[link_index]
                parent_pos = self._get_link_world_position(parent_index)
                if bone_idx < len(self.bone_ids):
                    self.bone_ids[bone_idx] = self.p.addUserDebugLine(
                        parent_pos.tolist(),
                        pos.tolist(),
                        lineColorRGB=[1, 0.7, 0.1],
                        lineWidth=1.5,
                        lifeTime=0,
                        replaceItemUniqueId=self.bone_ids[bone_idx],
                    )
                else:
                    self.bone_ids.append(
                        self.p.addUserDebugLine(
                            parent_pos.tolist(),
                            pos.tolist(),
                            lineColorRGB=[1, 0.7, 0.1],
                            lineWidth=1.5,
                            lifeTime=0,
                        )
                    )
                bone_idx += 1

    def clear(self):
        for item_id in self.cross_ids + self.bone_ids:
            try:
                self.p.removeUserDebugItem(item_id)
            except Exception:
                pass
        self.cross_ids.clear()
        self.bone_ids.clear()


def visualize_comparison(bvh_path: str, 
                        pkl_path: str, 
                        urdf_path: str,
                        speed: float = 1.0,
                        start_frame: int = 0,
                        end_frame: int = -1,
                        loop: bool = True,
                        bvh_follow_robot_yaw: bool = False,
                        fix_orientation: bool = False,
                        bvh_yaw_offset: float = 0.0,
                        show_robot_links: bool = False):
    """
    Visualize BVH skeleton and robot side-by-side.
    """
    import pybullet as p
    import pybullet_data
    from scipy.spatial.transform import Rotation as R
    
    # Load data
    print("Loading data...")
    bvh_loader = BVHLoader(bvh_path)
    pkl_data = load_pkl_data(pkl_path)
    coord_trans = CoordinateTransform()
    
    # Calculate yaw offset in radians
    # This converts the offset from degrees (human readable) to radians (for math)
    # Example: 90 degrees -> 1.57 radians
    yaw_offset_rad = np.radians(bvh_yaw_offset)
    
    n_frames_bvh = bvh_loader.frames
    n_frames_pkl = pkl_data['root_pos'].shape[0]
    n_frames = min(n_frames_bvh, n_frames_pkl)
    
    if end_frame < 0 or end_frame > n_frames:
        end_frame = n_frames
    
    print(f"BVH: {n_frames_bvh} frames @ {1.0/bvh_loader.frame_time:.1f} FPS")
    print(f"PKL: {n_frames_pkl} frames @ {pkl_data['fps']:.1f} FPS")
    print(f"Playing frames {start_frame} to {end_frame}")
    print()

    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")
    
    # Connect to PyBullet
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Configure GUI
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    
    # Enable shadows for better depth perception (and use bright materials to avoid darkness)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    
    # Use a clean white background? No, PyBullet doesn't support changing background color easily.
    # But we can load a better ground plane or grid.
    
    # Load ground plane
    # plane_id = p.loadURDF("plane.urdf")
    
    # Load a grid floor instead of the infinite checkerboard if desired?
    # For now, let's stick to plane.urdf but maybe change its color?
    plane_id = p.loadURDF("plane.urdf")
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.82, 0.84, 0.88, 1])
    
    # Compute initial BVH hip position in robot frame
    bvh_fk_frame0 = compute_bvh_fk(bvh_loader, start_frame)
    bvh_hip_pos_frame0 = bvh_fk_frame0.get('Hips', np.zeros(3))
    initial_bvh_pos = coord_trans.transform_position(bvh_hip_pos_frame0.reshape(1,3))[0]
    
    # Position offsets to place models side by side
    # Reduce the gap between them (was 2.0, now 1.0)
    bvh_offset = np.array([0, -1.0, 0]) - np.array([initial_bvh_pos[0], initial_bvh_pos[1], 0])
    robot_offset = np.array([0, 1.0, 0])
    
    def get_robot_visualization_quaternion(quat_xyzw):
        """Use full root orientation so torso lean remains visible."""
        if fix_orientation:
            return np.array([0.0, 0.0, 0.0, 1.0])
        return quat_xyzw
    
    # Load robot with upright orientation
    print(f"Loading robot URDF: {urdf_path}")
    init_z = 1.0 # Default height
    init_pos = [robot_offset[0], robot_offset[1], init_z]
    init_orn = get_robot_visualization_quaternion(pkl_data['root_rot'][start_frame])
    
    robot_id = p.loadURDF(urdf_path, init_pos, init_orn.tolist(), useFixedBase=False)
    
    # Use a darker robot color so the full body remains visible against the floor
    for j in range(-1, p.getNumJoints(robot_id)):
        p.changeVisualShape(robot_id, j, rgbaColor=[0.62, 0.68, 0.78, 1])

    print(f"Robot visual meshes loaded: {len(p.getVisualShapeData(robot_id))}")
    if show_robot_links:
        print("Robot link debug markers enabled.")
    
    # Get joint indices and names from URDF
    num_joints = p.getNumJoints(robot_id)
    urdf_joint_map = {}  # name -> index
    urdf_actuated_joints = []
    
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode('utf-8')
        joint_type = joint_info[2]
        if joint_type != p.JOINT_FIXED:
            urdf_joint_map[joint_name] = i
            urdf_actuated_joints.append(joint_name)
    
    print(f"Robot: {len(urdf_actuated_joints)} actuated joints in URDF")
    
    # Determine Joint Order
    # If yourdfpy is available, we use it to get the exact order the PKL was likely generated with
    # Otherwise, we assume PKL follows the URDF order found by PyBullet (risky but often true)

    config_joint_order = []
    if 'dof_names' in pkl_data:
        config_joint_order = pkl_data['dof_names']
        print(f"Using joint names from PKL: {len(config_joint_order)} joints")
    elif yourdfpy:
        print("Using yourdfpy to determine joint order...")
        urdf_model = yourdfpy.URDF.load(urdf_path)
        # IMPORTANT: Must match PyRoki's logic - only actuated, non-mimic joints
        # mimic joints are controlled by their referenced joint, not independently
        for j in urdf_model.robot.joints:
             if j.type != 'fixed' and (not hasattr(j, 'mimic') or j.mimic is None):
                 config_joint_order.append(j.name)
        print(f"Resolved {len(config_joint_order)} joints from yourdfpy")
    else:
        print("Using PyBullet joint order...")
        config_joint_order = urdf_actuated_joints

    # Build mapping: config index -> urdf index
    config_to_urdf_map = {}
    
    for cfg_idx, cfg_joint_name in enumerate(config_joint_order):
        if cfg_joint_name in urdf_joint_map:
            config_to_urdf_map[cfg_idx] = urdf_joint_map[cfg_joint_name]
        else:
            print(f"  Warning: Joint '{cfg_joint_name}' not found in PyBullet URDF")
    
    print(f"Mapping: {len(config_to_urdf_map)}/{len(config_joint_order)} joints mapped")
    
    # Check DOF match
    pkl_dof = pkl_data['dof_pos'].shape[1]
    if pkl_dof != len(config_joint_order):
        print(f"Warning: PKL has {pkl_dof} DOFs but we found {len(config_joint_order)} joints.")
    
    # Optionally align BVH yaw to robot yaw
    init_yaw = 0.0
    if bvh_follow_robot_yaw:
        r_init = R.from_quat(pkl_data['root_rot'][start_frame])
        init_yaw = r_init.as_euler('zyx', degrees=False)[0]
    
    # Apply manual offset
    init_yaw += yaw_offset_rad
    
    # Create BVH skeleton visualizer
    bvh_viz = BVHSkeletonVisualizer(p, offset=bvh_offset, yaw_rotation=init_yaw)
    robot_link_viz = RobotLinkVisualizer(p, robot_id) if show_robot_links else None
    
    # Add text labels
    p.addUserDebugText("BVH Skeleton", [0, -1.0, 2.2], textColorRGB=[1, 1, 1], textSize=1.5, lifeTime=0)
    p.addUserDebugText("Retargeted Robot", [0, 1.0, 2.2], textColorRGB=[1, 1, 1], textSize=1.5, lifeTime=0)
    
    # Use an oblique camera so torso and arms are not edge-on.
    p.resetDebugVisualizerCamera(cameraDistance=2.8, cameraYaw=55, cameraPitch=-18, cameraTargetPosition=[0, 0, 1.15])
    
    # Animation loop
    fps = pkl_data['fps']
    frame_time = 1.0 / (fps * speed)
    frame_idx = start_frame
    
    print("\nStarting visualization...")
    
    try:
        bvh_fk_start = compute_bvh_fk(bvh_loader, start_frame)
        bvh_start_hip = bvh_fk_start.get('Hips', np.zeros(3))
        
        while True:
            if not p.isConnected():
                break
            
            loop_start = time.time()
            
            # BVH
            bvh_joint_positions = compute_bvh_fk(bvh_loader, frame_idx)
            
            # Fix BVH in place (relative to start hip)
            current_hip = bvh_joint_positions.get('Hips', np.zeros(3))
            hip_delta = current_hip - bvh_start_hip
            
            bvh_fixed_positions = {}
            for name, pos in bvh_joint_positions.items():
                bvh_fixed_positions[name] = pos - hip_delta
            
            # Remove dynamic yaw update to prevent double rotation
            # if bvh_follow_robot_yaw:
            #     r_current = R.from_quat(pkl_data['root_rot'][frame_idx])
            #     current_yaw = r_current.as_euler('zyx', degrees=False)[0]
            #     
            #     # Apply manual offset
            #     current_yaw += yaw_offset_rad
            #     
            #     bvh_viz.yaw_rotation = current_yaw
            #     c, s = np.cos(current_yaw), np.sin(current_yaw)
            #     bvh_viz.yaw_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            
            if p.isConnected():
                bvh_viz.draw_frame(bvh_fixed_positions)
            else:
                break
            
            # Robot
            robot_dof = pkl_data['dof_pos'][frame_idx]
            robot_orn = get_robot_visualization_quaternion(pkl_data['root_rot'][frame_idx])
            robot_z = pkl_data['root_pos'][frame_idx][2]
            
            # Use raw Z from PKL? Or relative?
            # PKL Z is usually absolute world Z.
            # But we want to offset X/Y but keep Z (or adjust Z if needed).
            robot_pos = [robot_offset[0], robot_offset[1], robot_z]
            
            if p.isConnected():
                p.resetBasePositionAndOrientation(robot_id, robot_pos, robot_orn.tolist())
                
                for cfg_idx, urdf_idx in config_to_urdf_map.items():
                    if cfg_idx < len(robot_dof):
                        p.resetJointState(robot_id, urdf_idx, robot_dof[cfg_idx])

                if robot_link_viz is not None:
                    robot_link_viz.draw()
                
                p.stepSimulation()
            else:
                break
            
            # Progress
            if frame_idx % 50 == 0:
                print(f"Frame {frame_idx}/{end_frame}")
            
            frame_idx += 1
            if frame_idx >= end_frame:
                if loop:
                    frame_idx = start_frame
                    print("\n--- Looping ---\n")
                else:
                    break
            
            # Sleep
            elapsed = time.time() - loop_start
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        bvh_viz.clear()
        if robot_link_viz is not None:
            robot_link_viz.clear()
        if p.isConnected():
            p.disconnect()

def main():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Defaults
    default_bvh = os.path.join(script_dir, "data", "raw_data", "lafan1", "walk1_subject1.bvh")
    default_pkl = os.path.join(script_dir, "output.pkl")
    default_urdf = get_default_tienkung_urdf(script_dir)
    
    # -------------------------------------------------------------------------
    # BVH Orientation Offset (in degrees)
    # Modify this value to fix orientation mismatch (e.g., 90, -90, 180)
    DEFAULT_BVH_YAW_OFFSET = 90.0  
    # -------------------------------------------------------------------------

    parser = argparse.ArgumentParser(description="Side-by-side visualization")
    parser.add_argument("--bvh", type=str, default=default_bvh, help="Path to BVH file")
    parser.add_argument("--pkl", type=str, default=default_pkl, help="Path to PKL file")
    parser.add_argument("--urdf", type=str, default=default_urdf, help="Path to URDF file")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--bvh-follow-robot-yaw", action="store_true")
    parser.add_argument("--fix-orientation", action="store_true")
    parser.add_argument("--bvh-yaw-offset", type=float, default=DEFAULT_BVH_YAW_OFFSET, help="Manual yaw offset for BVH skeleton in degrees")
    parser.add_argument("--show-robot-links", action="store_true", help="Draw robot link markers and parent-child bones, independent of STL visibility")
    
    args = parser.parse_args()
    
    visualize_comparison(
        bvh_path=args.bvh,
        pkl_path=args.pkl,
        urdf_path=args.urdf,
        speed=args.speed,
        start_frame=args.start,
        end_frame=args.end,
        loop=not args.no_loop,
        bvh_follow_robot_yaw=args.bvh_follow_robot_yaw,
        fix_orientation=args.fix_orientation,
        bvh_yaw_offset=args.bvh_yaw_offset,
        show_robot_links=args.show_robot_links,
    )

if __name__ == "__main__":
    main()
