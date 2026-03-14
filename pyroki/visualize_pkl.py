
"""
Visualization: Retargeted Robot Motion (PKL only)

Displays the robot model executing motion from a PKL file.
No BVH file required.

Usage:
    python pyroki/visualize_pkl.py \
        --pkl pyroki/outputs/aiming1_subject1.pkl \
        --urdf assets/Tienkung/urdf/humanoid_simple.urdf
"""

import sys
import os
import io
import argparse
import pickle
import time
import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R

# Set UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import yourdfpy for joint names extraction
try:
    import yourdfpy
except ImportError:
    print("Warning: yourdfpy not found. Assuming URDF joint order matches PKL.")
    yourdfpy = None

def load_pkl_data(pkl_path: str) -> dict:
    """Load PKL motion data."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def visualize_pkl(pkl_path: str, 
                  urdf_path: str,
                  speed: float = 1.0,
                  start_frame: int = 0,
                  end_frame: int = -1,
                  loop: bool = True,
                  fix_orientation: bool = False):
    """
    Visualize robot motion from PKL.
    """
    
    # Load data
    print(f"Loading PKL data: {pkl_path}")
    pkl_data = load_pkl_data(pkl_path)
    
    n_frames_pkl = pkl_data['root_pos'].shape[0]
    
    if end_frame < 0 or end_frame > n_frames_pkl:
        end_frame = n_frames_pkl
    
    fps = pkl_data.get('fps', 30.0) # Default to 30 if not present
    print(f"PKL: {n_frames_pkl} frames @ {fps:.1f} FPS")
    print(f"Playing frames {start_frame} to {end_frame}")
    print()
    
    # Connect to PyBullet
    try:
        physics_client = p.connect(p.GUI)
    except:
        print("Failed to connect to PyBullet GUI. Is another instance running?")
        return

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Configure GUI
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    
    # Load ground plane
    plane_id = p.loadURDF("plane.urdf")
    
    def extract_yaw_quaternion(quat_xyzw):
        """Extract only yaw (Z-axis) rotation from a quaternion."""
        if fix_orientation:
            return np.array([0.0, 0.0, 0.0, 1.0])
        # If rotation is identity (or close to it), just return it
        if np.allclose(quat_xyzw, [0, 0, 0, 1]):
             return quat_xyzw
             
        try:
            r = R.from_quat(quat_xyzw)
            euler = r.as_euler('zyx', degrees=False)  # Get euler angles
            # euler is [z, y, x]
            # We usually want to keep the full orientation for the robot base
            # unless we specifically want to lock pitch/roll.
            # The original script extracted ONLY yaw. Let's stick to that if that's the intention,
            # BUT for general visualization, we usually want the full rotation provided in the PKL.
            # However, the original script said "Extract only yaw". 
            # Let's check if the PKL contains full rotation or just yaw.
            # Usually humanoid controllers handle balance, so input might be just yaw.
            # But here we are visualizing the RESULT of retargeting or recording.
            # Let's use the FULL rotation by default, as that's what's likely in the PKL.
            
            # Update: The original script explicitly extracted yaw. 
            # "Extract only yaw (Z-axis) rotation from a quaternion."
            # This suggests the robot base in simulation might be kept upright by a controller,
            # but the PKL might contain noisy pitch/roll? 
            # Or maybe the PKL *is* the reference motion which includes pitch/roll?
            # Let's try using the FULL rotation first. If it looks bad (falling over), we can restrict it.
            # Actually, let's follow the original script logic to be safe: extract yaw only.
            # Wait, if I'm visualizing a "fall" or "get up" motion, pitch/roll is crucial.
            # "aiming" might also involve leaning.
            # Let's try to use the full rotation.
            return quat_xyzw
        except Exception as e:
            print(f"Error converting quaternion: {e}")
            return np.array([0.0, 0.0, 0.0, 1.0])

    # Load robot
    print(f"Loading robot URDF: {urdf_path}")
    init_pos = pkl_data['root_pos'][start_frame]
    init_orn = pkl_data['root_rot'][start_frame]
    
    # Ensure init_pos is reasonable (sometimes Z is 0 in local frame?)
    # If Z is very low, we might need to offset it.
    # But usually PKL has world coordinates.
    
    robot_id = p.loadURDF(urdf_path, init_pos, init_orn, useFixedBase=False)
    
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
    
    # Determine Joint Order from PKL (assuming standard order or matching names)
    # The PKL usually corresponds to a specific list of joints.
    # We need to map PKL data columns to URDF joints.
    # If 'dof_names' is in PKL, use it.
    # If not, we have to guess or use the logic from the original script.
    
    config_joint_order = []
    if 'dof_names' in pkl_data:
        config_joint_order = pkl_data['dof_names']
        print(f"Using joint names from PKL: {len(config_joint_order)} joints")
    elif yourdfpy:
        print("Using yourdfpy to determine joint order (fallback)...")
        # Attempt to load URDF to guess order
        try:
            urdf_model = yourdfpy.URDF.load(urdf_path)
            for j in urdf_model.robot.joints:
                 if j.type != 'fixed' and (not hasattr(j, 'mimic') or j.mimic is None):
                     config_joint_order.append(j.name)
        except Exception as e:
            print(f"yourdfpy failed: {e}")
            config_joint_order = urdf_actuated_joints
    else:
        print("Using PyBullet joint order (fallback)...")
        config_joint_order = urdf_actuated_joints

    # Build mapping: config index -> urdf index
    config_to_urdf_map = {}
    
    for cfg_idx, cfg_joint_name in enumerate(config_joint_order):
        if cfg_joint_name in urdf_joint_map:
            config_to_urdf_map[cfg_idx] = urdf_joint_map[cfg_joint_name]
        else:
            # print(f"  Warning: Joint '{cfg_joint_name}' not found in PyBullet URDF")
            pass
            
    print(f"Mapping: {len(config_to_urdf_map)} joints mapped")
    
    # Check DOF match
    if 'dof_pos' in pkl_data:
        pkl_dof = pkl_data['dof_pos'].shape[1]
        if pkl_dof != len(config_joint_order):
            print(f"Warning: PKL has {pkl_dof} DOFs but we identified {len(config_joint_order)} joints.")
    
    # Set camera
    p.resetDebugVisualizerCamera(cameraDistance=3.0, cameraYaw=90, cameraPitch=-15, cameraTargetPosition=[0, 0, 1.0])
    
    # Animation loop
    frame_time = 1.0 / (fps * speed)
    frame_idx = start_frame
    
    print("\nStarting visualization...")
    
    try:
        while True:
            if not p.isConnected():
                break
            
            loop_start = time.time()
            
            # Robot Update
            robot_dof = pkl_data['dof_pos'][frame_idx]
            robot_pos = pkl_data['root_pos'][frame_idx]
            robot_orn = pkl_data['root_rot'][frame_idx]
            
            if p.isConnected():
                p.resetBasePositionAndOrientation(robot_id, robot_pos, robot_orn)
                
                for cfg_idx, urdf_idx in config_to_urdf_map.items():
                    if cfg_idx < len(robot_dof):
                        p.resetJointState(robot_id, urdf_idx, robot_dof[cfg_idx])
                
                p.stepSimulation()
            else:
                break
            
            # Progress
            if frame_idx % 50 == 0:
                print(f"Frame {frame_idx}/{end_frame}", end='\r')
            
            frame_idx += 1
            if frame_idx >= end_frame:
                if loop:
                    frame_idx = start_frame
                    # print("\n--- Looping ---\n")
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
        if p.isConnected():
            p.disconnect()

def main():
    parser = argparse.ArgumentParser(description="Visualize PKL motion")
    parser.add_argument("--pkl", type=str, required=True, help="Path to PKL file")
    parser.add_argument("--urdf", type=str, required=True, help="Path to URDF file")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--no-loop", action="store_true")
    
    args = parser.parse_args()
    
    visualize_pkl(
        pkl_path=args.pkl,
        urdf_path=args.urdf,
        speed=args.speed,
        start_frame=args.start,
        end_frame=args.end,
        loop=not args.no_loop
    )

if __name__ == "__main__":
    main()
