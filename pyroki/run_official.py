"""
Official-style Retargeting Reproduction Script
Uses PyRoki (JAX-based) for optimization-based IK.

This script follows the standard retargeting pipeline:
1. Load Robot (URDF)
2. Load Motion (BVH)
3. Coordinate Alignment (BVH -> Robot)
4. IK Optimization (Frame-by-frame or Batch)
5. Export to TWIST2 format
"""

import os
import sys
import numpy as np
import argparse
from typing import List, Dict, Tuple
import time

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import jax
    import jax.numpy as jnp
    import jaxlie
    import pyroki as pk
    import yourdfpy
    print(f"JAX version: {jax.__version__}")
except ImportError as e:
    print(f"Error: PyRoki/JAX/yourdfpy not found. Please install them first. {e}")
    sys.exit(1)

from utils.bvh_loader import BVHLoader
from utils.coordinate_transform import CoordinateTransform
from utils.exporter import TWIST2Exporter
from scipy.spatial.transform import Rotation as R_scipy

def main():
    parser = argparse.ArgumentParser(description="PyRoki Retargeting Repro")
    # Use local path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_bvh = os.path.join(script_dir, "data", "raw_data", "lafan1", "walk1_subject1.bvh")
    parser.add_argument("--bvh", type=str, default=default_bvh, help="Input BVH file")
    
    default_urdf = os.path.join(script_dir, "assets", "urdf", "humanoid_simple.urdf")
    parser.add_argument("--urdf", type=str, default=default_urdf, help="Robot URDF file")
    parser.add_argument("--output", type=str, default="output.pkl", help="Output PKL file")
    args = parser.parse_args()

    print(f"Loading Robot from: {args.urdf}")
    # Load robot
    urdf_model = yourdfpy.URDF.load(args.urdf)
    robot = pk.Robot.from_urdf(urdf_model)
    dof = robot.joints.num_actuated_joints
    print(f"Robot loaded. DOF: {dof}")
    
    print(f"Loading BVH from: {args.bvh}")
    loader = BVHLoader(args.bvh)
    print(f"Motion loaded. Frames: {loader.frames}, Time: {loader.frame_time}")

    # Coordinate Transform
    coord_trans = CoordinateTransform()

    # Define Mappings (Hardcoded for Tiangong - Extended for 26 DOF)
    # BVH Joint -> Robot Link
    mappings = {
        "Hips": "pelvis",
        
        # Left Leg
        "LeftUpLeg": "hip_pitch_l_link",
        "LeftLeg": "knee_pitch_l_link",
        "LeftFoot": "ankle_pitch_l_link",
        "LeftToe": "ankle_roll_l_link",
        
        # Right Leg
        "RightUpLeg": "hip_pitch_r_link",
        "RightLeg": "knee_pitch_r_link",
        "RightFoot": "ankle_pitch_r_link",
        "RightToe": "ankle_roll_r_link",
        
        # Waist
        "Spine": "waist_link",
        "Spine1": "waist_link",
        "Spine2": "waist_link",
        
        # Left Arm
        "LeftShoulder": "shoulder_roll_l_link",
        "LeftArm": "left_link2",
        "LeftForeArm": "elbow_l_link",
        "LeftHand": "L_hand_base_link",
        
        # Right Arm
        "RightShoulder": "shoulder_roll_r_link",
        "RightArm": "right_link2",
        "RightForeArm": "elbow_r_link",
        "RightHand": "R_hand_base_link",
        
        # Head
        "Neck": "waist_link",
        "Head": "waist_link"
    }
    
    # Verify links exist
    valid_mappings = {}
    for bvh_name, robot_link in mappings.items():
        if robot_link in robot.links.names:
            valid_mappings[bvh_name] = robot_link
        else:
            print(f"Warning: Link '{robot_link}' not found in robot. Skipping {bvh_name}.")
            # Fallback for hands if mapped to tcp
            if "Hand" in bvh_name:
                  # Try finding similar
                  candidates = [l for l in robot.links.names if "hand" in l.lower() and "base" in l.lower()]
                  if candidates:
                      print(f"  -> Found candidate for {bvh_name}: {candidates[0]}")
                      valid_mappings[bvh_name] = candidates[0]
 
    print(f"Active Mappings: {valid_mappings}")
    print("\nMapping details:")
    for bvh, link in valid_mappings.items():
        print(f"  {bvh} -> {link}")
    print()

    # --- Pre-computation: Extract Targets ---
    print("Extracting targets...")
    from utils.bvh_fk import compute_bvh_fk_batch
    
    print("Computing BVH FK...")
    bvh_pos_dict = compute_bvh_fk_batch(loader) # {joint: [N, 3]}
    
    n_frames = loader.frames
    
    # 1. Compute Root Trajectory (Pos & Rot) from BVH Hips
    print("Computing Root Trajectory...")
    hips_pos_bvh = bvh_pos_dict["Hips"] # [N, 3]
    
    # Transform to Robot Frame (World)
    root_pos_robot = coord_trans.transform_position(hips_pos_bvh)
    
    # Adjust Height (Retargeting often needs height offset)
    # Tiangong pelvis height vs BVH Hips height
    # Let's align the first frame height or use a fixed offset?
    # For now, keep as is, but maybe offset z so min z >= 0 if needed.
    # Actually, usually we match the lowest point (feet) to ground.
    # I'll leave it raw for now.
    
    # Compute Root Rotation (SVD/Yaw Method)
    # Find Hip Joints
    def find_joint(candidates):
        for c in candidates:
            for name in bvh_pos_dict.keys():
                if c.lower() in name.lower():
                    return name
        return None
        
    l_hip_name = find_joint(['LeftUpLeg', 'LeftHip', 'LHip'])
    r_hip_name = find_joint(['RightUpLeg', 'RightHip', 'RHip'])
    
    print(f"Using hips: {l_hip_name}, {r_hip_name}")
    
    l_hip_pos_robot = coord_trans.transform_position(bvh_pos_dict[l_hip_name])
    r_hip_pos_robot = coord_trans.transform_position(bvh_pos_dict[r_hip_name])
    
    # Vector R->L (approx +Y in Robot Frame)
    vec_hips = l_hip_pos_robot - r_hip_pos_robot
    vec_hips /= (np.linalg.norm(vec_hips, axis=1, keepdims=True) + 1e-8)
    
    # Compute Yaw
    # We want to rotate Robot (facing +X) so its Y axis aligns with vec_hips
    # Angle of vec_hips in XY plane
    yaw_angles = np.arctan2(vec_hips[:, 1], vec_hips[:, 0]) - np.pi/2
    yaw_angles = np.unwrap(yaw_angles)
    
    # Smooth Yaw
    from scipy.signal import savgol_filter
    if n_frames > 7:
        yaw_angles = savgol_filter(yaw_angles, 7, 3)
    
    print(f"Yaw angles shape: {yaw_angles.shape}")
    
    # Create Root Rotations (Z-rotation only)
    # Ensure shape (N, 1) for from_euler
    root_rot_scipy = R_scipy.from_euler('z', yaw_angles.reshape(-1, 1) if yaw_angles.ndim == 1 else yaw_angles)
    root_rot_quat = root_rot_scipy.as_quat() # xyzw
    # Convert to wxyz for JAX (if needed) or keep as is for Export
    # JAXLie uses wxyz usually? jaxlie.SO3(wxyz)
    root_rot_wxyz = np.column_stack((root_rot_quat[:, 3], root_rot_quat[:, 0], root_rot_quat[:, 1], root_rot_quat[:, 2]))
    
    # 2. Prepare IK Targets
    ik_targets = {} # link_name -> [N, 3]
    
    for bvh_j, robot_l in valid_mappings.items():
        if bvh_j == "Hips": continue # Root handled separately
        
        pos_bvh = bvh_pos_dict[bvh_j]
        pos_robot = coord_trans.transform_position(pos_bvh)
        ik_targets[robot_l] = pos_robot
        
    # 3. IK Optimization Loop
    print("Starting IK Optimization...")
    
    # JAX Compilation setup
    # Function to compute loss for a single frame
    # q: [dof]
    # root_tf: SE3
    # target_map: dict of link_idx -> target_pos [3]
    
    # Pre-lookup link indices
    link_indices = {name: robot.links.names.index(name) for name in ik_targets.keys()}
    target_link_indices = jnp.array(list(link_indices.values()))
    
    # Prepare target array for JAX: [N, n_targets, 3]
    # Order matches target_link_indices
    target_array_list = []
    for name in link_indices.keys():
        target_array_list.append(ik_targets[name])
    target_array_global = jnp.stack(target_array_list, axis=1) # [N, n_targets, 3]
    
    # Limits
    q_min = jnp.array(robot.joints.lower_limits)
    q_max = jnp.array(robot.joints.upper_limits)
    
    @jax.jit
    def fk_fn(q):
        return robot.forward_kinematics(q)

    @jax.jit
    def loss_fn(q, root_tf_wxyz, root_pos, target_pos_list):
        # q: [dof]
        # root_tf_wxyz: [4]
        # root_pos: [3]
        # target_pos_list: [n_targets, 3]
        
        # Construct Root Transform
        # Assuming robot.forward_kinematics returns transforms RELATIVE TO WORLD? 
        # Usually FK assumes root at identity unless specified.
        # If we want to move root, we apply root transform to all FK results 
        # OR we treat root as a joint (floating base).
        # PyRoki's Robot likely assumes fixed base at 0.
        # So Global_Pose = Root_TF * Local_Pose
        
        local_transforms = fk_fn(q) # List of SE3
        
        root_rot = jaxlie.SO3(root_tf_wxyz)
        root_se3 = jaxlie.SE3.from_rotation_and_translation(root_rot, root_pos)
        
        total_error = 0.0
        
        # Calculate position error
        # We need to iterate over target indices. 
        # JAX loop or hardcoded unroll? List comprehension works for fixed graph.
        
        current_pos_list = []
        for idx in range(len(target_link_indices)):
            link_idx = target_link_indices[idx]
            
            # link_pose is [7] (wxyz, xyz)
            link_pose = local_transforms[link_idx]
            link_rot = jaxlie.SO3(link_pose[:4])
            link_trans = link_pose[4:7]
            link_se3 = jaxlie.SE3.from_rotation_and_translation(link_rot, link_trans)
            
            # Global = Root * Local
            global_tf = root_se3.multiply(link_se3)
            current_pos_list.append(global_tf.translation())
        
        pos_loss = jnp.sum((jnp.stack(current_pos_list) - target_pos_list)**2)
        
        # Regularization (stay close to zero/nominal)
        reg_loss = jnp.sum(q**2) * 0.01
        
        # Joint Limits (soft)
        lower_viol = jnp.maximum(q_min - q, 0)
        upper_viol = jnp.maximum(q - q_max, 0)
        limit_loss = jnp.sum(lower_viol**2 + upper_viol**2) * 10.0
        
        return pos_loss + reg_loss + limit_loss

    loss_grad_fn = jax.value_and_grad(loss_fn)
    
    # Solver Loop
    print("Running JAX Scan Optimization...")
    
    # Optimization parameters
    lr = 0.1
    iterations = 100 # Increased iterations for better convergence
    
    start_time = time.time()
    
    # Define scan function
    def solve_frame(carrier, inputs):
        q_init = carrier
        root_w, root_p, targets = inputs
        
        def step_fn(i, q):
            loss, grad = loss_grad_fn(q, root_w, root_p, targets)
            q_new = q - lr * grad
            return jnp.clip(q_new, q_min, q_max)
            
        q_final = jax.lax.fori_loop(0, iterations, step_fn, q_init)
        return q_final, q_final # carry (next init), output (current frame)

    # Prepare inputs for scan
    # Ensure all are jnp arrays
    scan_inputs = (
        jnp.array(root_rot_wxyz),
        jnp.array(root_pos_robot),
        jnp.array(target_array_global)
    )
    
    # Run scan
    # Initialize with default pose (zeros or robot default)
    q_init = jnp.zeros(dof)
    
    _, q_traj_jax = jax.lax.scan(
        solve_frame,
        q_init,
        scan_inputs
    )
    
    # Block until ready to measure time accurately
    q_traj_jax.block_until_ready()
    print(f"IK Done. Time: {time.time() - start_time:.2f}s")
    
    q_traj = np.array(q_traj_jax)
    
    # 4. Compute Local Body Positions (for Output)
    print("Computing local body positions...")
    # 26 body links for full robot tracking
    key_body_names = [
        # Root
        "pelvis",
        
        # Waist
        "waist_link",
        
        # Left Leg
        "hip_roll_l_link",
        "hip_yaw_l_link",
        "hip_pitch_l_link",
        "knee_pitch_l_link",
        "ankle_pitch_l_link",
        "ankle_roll_l_link",
        
        # Right Leg
        "hip_roll_r_link",
        "hip_yaw_r_link",
        "hip_pitch_r_link",
        "knee_pitch_r_link",
        "ankle_pitch_r_link",
        "ankle_roll_r_link",
        
        # Left Arm
        "left_link0",
        "shoulder_roll_l_link",
        "left_link2",
        "elbow_l_link",
        "left_link4",
        "L_hand_base_link",
        
        # Right Arm
        "right_link0",
        "shoulder_roll_r_link",
        "right_link2",
        "elbow_r_link",
        "right_link4",
        "R_hand_base_link"
    ]
    # Filter to valid ones
    key_body_names = [n for n in key_body_names if n in robot.links.names]
    key_body_indices = [robot.links.names.index(n) for n in key_body_names]
    
    @jax.jit
    def compute_local_pos_frame(q, r_w, r_p):
        local_tfs = fk_fn(q)
        root_se3 = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(r_w), r_p)
        
        # Compute root rotation matrix (for local coordinate transformation)
        root_rot_matrix = jnp.array([
            [jnp.cos(r_w[2]), -jnp.sin(r_w[2]), 0.0],
            [jnp.sin(r_w[2]), jnp.cos(r_w[2]), 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        positions = []
        for idx in key_body_indices:
            # Get global pos of link
            l_pose = local_tfs[idx]
            l_se3 = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(l_pose[:4]), l_pose[4:7])
            g_se3 = root_se3.multiply(l_se3)
            g_pos = g_se3.translation()
             
            # Local pos = Root_rotation^T * (Global_pos - Root_pos)
            global_offset = g_pos - r_p
            local_offset = jnp.matmul(root_rot_matrix.T, global_offset)
            positions.append(local_offset)
        return jnp.stack(positions)

    # Batch compute
    local_body_pos = jax.vmap(compute_local_pos_frame)(q_traj_jax, scan_inputs[0], scan_inputs[1])
    local_body_pos = np.array(local_body_pos) # [N, n_bodies, 3]

    # Export
    # Slice other arrays to match q_traj length (though scan does all frames now)
    exporter = TWIST2Exporter(dof, key_body_names)
    
    exporter.export(
        root_pos=root_pos_robot,
        root_rot=root_rot_wxyz, # Use wxyz format
        dof_pos=q_traj,
        local_body_pos=local_body_pos,
        fps=1.0/loader.frame_time,
        output_path=args.output
    )

if __name__ == "__main__":
    main()
