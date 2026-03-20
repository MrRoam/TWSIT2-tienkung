"""
Official-style retargeting script for Tienkung.
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np

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

from scipy.spatial.transform import Rotation as R_scipy

from utils.bvh_loader import BVHLoader
from utils.coordinate_transform import CoordinateTransform
from utils.exporter import TWIST2Exporter
from utils.bvh_fk import compute_bvh_fk_batch


def get_default_tienkung_urdf(script_dir: str) -> str:
    repo_root = os.path.dirname(script_dir)
    return os.path.join(repo_root, "assets", "Tienkung", "urdf", "walker_tienkung_ei.urdf")


def get_actuated_joint_names(urdf_model) -> List[str]:
    joint_names = []
    for joint in urdf_model.robot.joints:
        if joint.type == "fixed":
            continue
        if hasattr(joint, "mimic") and joint.mimic is not None:
            continue
        joint_names.append(joint.name)
    return joint_names


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, eps, None)


def compute_root_orientation_from_bvh(
    bvh_pos_dict: Dict[str, np.ndarray],
    coord_trans: CoordinateTransform,
    l_hip_name: str,
    r_hip_name: str,
    spine_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    l_hip_pos_robot = coord_trans.transform_position(bvh_pos_dict[l_hip_name])
    r_hip_pos_robot = coord_trans.transform_position(bvh_pos_dict[r_hip_name])
    spine_pos_robot = coord_trans.transform_position(bvh_pos_dict[spine_name])

    hip_center = 0.5 * (l_hip_pos_robot + r_hip_pos_robot)
    lateral = normalize_vectors(l_hip_pos_robot - r_hip_pos_robot)
    up_guess = normalize_vectors(spine_pos_robot - hip_center)

    forward = np.cross(lateral, up_guess)
    degenerate = np.linalg.norm(forward, axis=1) < 1e-6
    if np.any(degenerate):
        forward[degenerate] = np.array([1.0, 0.0, 0.0])
    forward = normalize_vectors(forward)

    up = normalize_vectors(np.cross(forward, lateral))
    lateral = normalize_vectors(np.cross(up, forward))

    rot_mats = np.stack([forward, lateral, up], axis=-1)
    root_rot_xyzw = R_scipy.from_matrix(rot_mats).as_quat()
    root_rot_wxyz = np.column_stack(
        (
            root_rot_xyzw[:, 3],
            root_rot_xyzw[:, 0],
            root_rot_xyzw[:, 1],
            root_rot_xyzw[:, 2],
        )
    )
    return root_rot_xyzw, root_rot_wxyz


def find_joint_name(bvh_pos_dict: Dict[str, np.ndarray], candidates: List[str]) -> Optional[str]:
    for candidate in candidates:
        candidate_lower = candidate.lower()
        for name in bvh_pos_dict.keys():
            if candidate_lower in name.lower():
                return name
    return None


def main():
    parser = argparse.ArgumentParser(description="PyRoki retargeting for Tienkung")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_bvh = os.path.join(script_dir, "data", "raw_data", "lafan1", "walk1_subject1.bvh")
    default_urdf = get_default_tienkung_urdf(script_dir)

    parser.add_argument("--bvh", type=str, default=default_bvh, help="Input BVH file")
    parser.add_argument("--urdf", type=str, default=default_urdf, help="Robot URDF file")
    parser.add_argument("--output", type=str, default="output.pkl", help="Output PKL file")
    args = parser.parse_args()

    print(f"Loading robot from: {args.urdf}")
    urdf_model = yourdfpy.URDF.load(args.urdf)
    robot = pk.Robot.from_urdf(urdf_model)
    actuated_joint_names = get_actuated_joint_names(urdf_model)
    dof = robot.joints.num_actuated_joints
    print(f"Robot loaded. DOF: {dof}")
    print(f"Actuated joints: {actuated_joint_names}")

    print(f"Loading BVH from: {args.bvh}")
    loader = BVHLoader(args.bvh)
    print(f"Motion loaded. Frames: {loader.frames}, Time: {loader.frame_time}")

    coord_trans = CoordinateTransform()

    mapping_specs = [
        ("LeftUpLeg", "hip_yaw_l_link"),
        ("LeftLeg", "knee_pitch_l_link"),
        ("LeftFoot", "ankle_pitch_l_link"),
        ("LeftToe", "ankle_roll_l_link"),
        ("RightUpLeg", "hip_yaw_r_link"),
        ("RightLeg", "knee_pitch_r_link"),
        ("RightFoot", "ankle_pitch_r_link"),
        ("RightToe", "ankle_roll_r_link"),
        ("Spine2", "waist_yaw_link"),
        ("LeftShoulder", "shoulder_pitch_l_link"),
        ("LeftArm", "shoulder_yaw_l_link"),
        ("LeftForeArm", "elbow_yaw_l_link"),
        ("LeftHand", "wrist_roll_l_link"),
        ("RightShoulder", "shoulder_pitch_r_link"),
        ("RightArm", "shoulder_yaw_r_link"),
        ("RightForeArm", "elbow_yaw_r_link"),
        ("RightHand", "wrist_roll_r_link"),
    ]

    valid_mappings: List[Tuple[str, str]] = []
    used_links = set()
    for bvh_name, robot_link in mapping_specs:
        if robot_link not in robot.links.names:
            print(f"Warning: link '{robot_link}' not found in robot. Skipping {bvh_name}.")
            continue
        if robot_link in used_links:
            print(f"Warning: duplicate robot link target '{robot_link}'. Skipping {bvh_name}.")
            continue
        valid_mappings.append((bvh_name, robot_link))
        used_links.add(robot_link)

    print(f"Active mappings: {valid_mappings}")

    print("Extracting targets...")
    bvh_pos_dict = compute_bvh_fk_batch(loader)
    n_frames = loader.frames

    print("Computing root trajectory...")
    hips_pos_bvh = bvh_pos_dict["Hips"]
    root_pos_robot = coord_trans.transform_position(hips_pos_bvh)

    l_hip_name = find_joint_name(bvh_pos_dict, ["LeftUpLeg", "LeftHip", "LHip"])
    r_hip_name = find_joint_name(bvh_pos_dict, ["RightUpLeg", "RightHip", "RHip"])
    spine_name = find_joint_name(bvh_pos_dict, ["Spine2", "Spine1", "Spine", "Chest"])
    if l_hip_name is None or r_hip_name is None or spine_name is None:
        raise ValueError(
            f"Failed to find root orientation joints: l_hip={l_hip_name}, r_hip={r_hip_name}, spine={spine_name}"
        )

    print(f"Using root orientation joints: {l_hip_name}, {r_hip_name}, {spine_name}")
    root_rot_xyzw, root_rot_wxyz = compute_root_orientation_from_bvh(
        bvh_pos_dict=bvh_pos_dict,
        coord_trans=coord_trans,
        l_hip_name=l_hip_name,
        r_hip_name=r_hip_name,
        spine_name=spine_name,
    )

    ik_targets = {}
    for bvh_joint, robot_link in valid_mappings:
        ik_targets[robot_link] = coord_trans.transform_position(bvh_pos_dict[bvh_joint])

    print("Starting IK optimization...")
    link_indices = {name: robot.links.names.index(name) for name in ik_targets.keys()}
    target_link_indices = jnp.array(list(link_indices.values()))
    target_array_global = jnp.stack([ik_targets[name] for name in link_indices.keys()], axis=1)

    q_min = jnp.array(robot.joints.lower_limits)
    q_max = jnp.array(robot.joints.upper_limits)

    @jax.jit
    def fk_fn(q):
        return robot.forward_kinematics(q)

    @jax.jit
    def loss_fn(q, root_tf_wxyz, root_pos, target_pos_list):
        local_transforms = fk_fn(q)
        root_rot = jaxlie.SO3(root_tf_wxyz)
        root_se3 = jaxlie.SE3.from_rotation_and_translation(root_rot, root_pos)

        current_pos_list = []
        for idx in range(len(target_link_indices)):
            link_idx = target_link_indices[idx]
            link_pose = local_transforms[link_idx]
            link_rot = jaxlie.SO3(link_pose[:4])
            link_trans = link_pose[4:7]
            link_se3 = jaxlie.SE3.from_rotation_and_translation(link_rot, link_trans)
            global_tf = root_se3.multiply(link_se3)
            current_pos_list.append(global_tf.translation())

        pos_loss = jnp.sum((jnp.stack(current_pos_list) - target_pos_list) ** 2)
        reg_loss = jnp.sum(q ** 2) * 0.01
        lower_viol = jnp.maximum(q_min - q, 0)
        upper_viol = jnp.maximum(q - q_max, 0)
        limit_loss = jnp.sum(lower_viol ** 2 + upper_viol ** 2) * 10.0
        return pos_loss + reg_loss + limit_loss

    loss_grad_fn = jax.value_and_grad(loss_fn)

    print("Running JAX scan optimization...")
    lr = 0.1
    iterations = 100
    start_time = time.time()

    def solve_frame(carrier, inputs):
        q_init = carrier
        root_w, root_p, targets = inputs

        def step_fn(_, q):
            loss, grad = loss_grad_fn(q, root_w, root_p, targets)
            q_new = q - lr * grad
            return jnp.clip(q_new, q_min, q_max)

        q_final = jax.lax.fori_loop(0, iterations, step_fn, q_init)
        return q_final, q_final

    scan_inputs = (
        jnp.array(root_rot_wxyz),
        jnp.array(root_pos_robot),
        jnp.array(target_array_global),
    )

    q_init = jnp.zeros(dof)
    _, q_traj_jax = jax.lax.scan(solve_frame, q_init, scan_inputs)
    q_traj_jax.block_until_ready()
    print(f"IK done. Time: {time.time() - start_time:.2f}s")

    q_traj = np.array(q_traj_jax)

    print("Computing local body positions...")
    key_body_names = [
        "pelvis",
        "waist_yaw_link",
        "hip_roll_l_link",
        "hip_pitch_l_link",
        "hip_yaw_l_link",
        "knee_pitch_l_link",
        "ankle_pitch_l_link",
        "ankle_roll_l_link",
        "hip_roll_r_link",
        "hip_pitch_r_link",
        "hip_yaw_r_link",
        "knee_pitch_r_link",
        "ankle_pitch_r_link",
        "ankle_roll_r_link",
        "shoulder_pitch_l_link",
        "shoulder_roll_l_link",
        "shoulder_yaw_l_link",
        "elbow_pitch_l_link",
        "elbow_yaw_l_link",
        "wrist_roll_l_link",
        "shoulder_pitch_r_link",
        "shoulder_roll_r_link",
        "shoulder_yaw_r_link",
        "elbow_pitch_r_link",
        "elbow_yaw_r_link",
        "wrist_roll_r_link",
    ]
    key_body_indices = [robot.links.names.index(name) for name in key_body_names if name in robot.links.names]
    key_body_names = [name for name in key_body_names if name in robot.links.names]

    @jax.jit
    def compute_local_pos_frame(q, r_w, r_p):
        local_tfs = fk_fn(q)
        root_so3 = jaxlie.SO3(r_w)
        root_se3 = jaxlie.SE3.from_rotation_and_translation(root_so3, r_p)
        root_rot_matrix = root_so3.as_matrix()

        positions = []
        for idx in key_body_indices:
            l_pose = local_tfs[idx]
            l_se3 = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(l_pose[:4]), l_pose[4:7])
            g_se3 = root_se3.multiply(l_se3)
            g_pos = g_se3.translation()
            local_offset = jnp.matmul(root_rot_matrix.T, g_pos - r_p)
            positions.append(local_offset)
        return jnp.stack(positions)

    local_body_pos = jax.vmap(compute_local_pos_frame)(q_traj_jax, scan_inputs[0], scan_inputs[1])
    local_body_pos = np.array(local_body_pos)

    exporter = TWIST2Exporter(dof, key_body_names)
    exporter.export(
        root_pos=root_pos_robot,
        root_rot=root_rot_xyzw,
        dof_pos=q_traj,
        local_body_pos=local_body_pos,
        dof_names=actuated_joint_names,
        fps=1.0 / loader.frame_time,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
