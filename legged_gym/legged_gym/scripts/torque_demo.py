#!/usr/bin/env python3

import argparse
import math
import os

import isaacgym
import torch
from isaacgym import gymapi, gymtorch, gymutil


TIENKUNG_STAND_JOINT_ANGLES = {
    "hip_roll_l_joint": 0.0,
    "hip_pitch_l_joint": -0.2,
    "hip_yaw_l_joint": 0.0,
    "knee_pitch_l_joint": 0.4,
    "ankle_pitch_l_joint": -0.2,
    "ankle_roll_l_joint": 0.0,
    "hip_roll_r_joint": 0.0,
    "hip_pitch_r_joint": -0.2,
    "hip_yaw_r_joint": 0.0,
    "knee_pitch_r_joint": 0.4,
    "ankle_pitch_r_joint": -0.2,
    "ankle_roll_r_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "head_yaw_joint": 0.0,
    "head_pitch_joint": 0.0,
    "head_roll_joint": 0.0,
    "shoulder_pitch_l_joint": 0.0,
    "shoulder_roll_l_joint": 0.4,
    "shoulder_yaw_l_joint": 0.0,
    "elbow_pitch_l_joint": -1.2,
    "elbow_yaw_l_joint": 0.0,
    "wrist_pitch_l_joint": 0.0,
    "wrist_roll_l_joint": 0.0,
    "shoulder_pitch_r_joint": 0.0,
    "shoulder_roll_r_joint": -0.4,
    "shoulder_yaw_r_joint": 0.0,
    "elbow_pitch_r_joint": -1.2,
    "elbow_yaw_r_joint": 0.0,
    "wrist_pitch_r_joint": 0.0,
    "wrist_roll_r_joint": 0.0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal Isaac Gym torque demo: apply torque to one DOF and watch the robot move."
    )
    parser.add_argument("--asset", type=str, required=True, help="Absolute path to a URDF or MJCF asset.")
    parser.add_argument("--joint", type=str, default=None, help="Joint/DOF name to actuate.")
    parser.add_argument("--joint-index", type=int, default=None, help="DOF index to actuate if joint name is not used.")
    parser.add_argument("--list-dofs", action="store_true", help="Print DOF names and exit.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Simulation device, for example cpu or cuda:0.")
    parser.add_argument("--headless", action="store_true", help="Disable the viewer.")
    parser.add_argument("--fix-base", action="store_true", help="Fix the base link to isolate the actuated joint motion.")
    parser.add_argument("--torque", type=float, default=20.0, help="Torque amplitude.")
    parser.add_argument("--mode", type=str, choices=["constant", "sine"], default="sine", help="Torque signal type.")
    parser.add_argument("--frequency", type=float, default=0.5, help="Frequency used by sine mode.")
    parser.add_argument("--duration", type=float, default=10.0, help="Demo duration in seconds.")
    parser.add_argument("--dt", type=float, default=0.005, help="Physics timestep.")
    parser.add_argument("--start-height", type=float, default=1.0, help="Initial actor base height.")
    parser.add_argument(
        "--init",
        type=str,
        choices=["zero", "tienkung_stand"],
        default="zero",
        help="Initial joint pose used before torque is applied.",
    )
    parser.add_argument(
        "--hold-pose",
        action="store_true",
        help="Apply PD holding torques to all non-target joints so the robot keeps its initial pose.",
    )
    parser.add_argument("--hold-kp", type=float, default=80.0, help="PD proportional gain used by --hold-pose.")
    parser.add_argument("--hold-kd", type=float, default=4.0, help="PD derivative gain used by --hold-pose.")
    parser.add_argument("--passive-damping", type=float, default=0.0, help="Passive joint damping used in effort mode.")
    parser.add_argument("--print-every", type=int, default=30, help="Print joint state every N simulation steps.")
    return parser.parse_args()


def build_sim(gym, args):
    sim_device_type, sim_device_id = gymutil.parse_device_str(args.device)
    use_gpu = sim_device_type == "cuda"
    graphics_device_id = -1 if args.headless else sim_device_id

    sim_params = gymapi.SimParams()
    sim_params.dt = args.dt
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = use_gpu
    sim_params.physx.use_gpu = use_gpu
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 1

    sim = gym.create_sim(sim_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym simulation.")

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane_params.static_friction = 1.0
    plane_params.dynamic_friction = 1.0
    gym.add_ground(sim, plane_params)

    viewer = None
    if not args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise RuntimeError("Failed to create Isaac Gym viewer.")

    torch_device = torch.device(args.device if use_gpu else "cpu")
    return sim, viewer, sim_device_id, torch_device


def load_asset(gym, sim, args):
    asset_path = os.path.abspath(args.asset)
    asset_root = os.path.dirname(asset_path)
    asset_file = os.path.basename(asset_path)

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = args.fix_base
    asset_options.collapse_fixed_joints = False
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT)

    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    if asset is None:
        raise RuntimeError(f"Failed to load asset: {asset_path}")

    dof_names = list(gym.get_asset_dof_names(asset))
    body_names = list(gym.get_asset_rigid_body_names(asset))
    return asset, dof_names, body_names


def resolve_joint_index(dof_names, args):
    if args.joint is not None:
        if args.joint not in dof_names:
            raise ValueError(f"Joint '{args.joint}' not found. Use --list-dofs to inspect available names.")
        return dof_names.index(args.joint)

    if args.joint_index is not None:
        if args.joint_index < 0 or args.joint_index >= len(dof_names):
            raise ValueError(f"Joint index {args.joint_index} is out of range [0, {len(dof_names) - 1}].")
        return args.joint_index

    return 0


def compute_torque(args, t):
    if args.mode == "constant":
        return args.torque
    return args.torque * math.sin(2.0 * math.pi * args.frequency * t)


def build_initial_dof_positions(dof_names, args):
    if args.init == "zero":
        return [0.0] * len(dof_names)

    if args.init == "tienkung_stand":
        return [TIENKUNG_STAND_JOINT_ANGLES.get(name, 0.0) for name in dof_names]

    raise ValueError(f"Unsupported init mode: {args.init}")


def main():
    args = parse_args()
    gym = gymapi.acquire_gym()
    sim = None
    viewer = None

    try:
        sim, viewer, sim_device_id, torch_device = build_sim(gym, args)
        asset, dof_names, body_names = load_asset(gym, sim, args)

        print(f"Loaded asset: {os.path.abspath(args.asset)}")
        print(f"Rigid bodies: {len(body_names)}")
        print(f"DOFs: {len(dof_names)}")

        if args.list_dofs:
            for i, name in enumerate(dof_names):
                print(f"{i:02d}: {name}")
            return

        joint_idx = resolve_joint_index(dof_names, args)
        joint_name = dof_names[joint_idx]
        print(f"Actuated DOF: {joint_idx} ({joint_name})")
        print(f"Torque mode: {args.mode}, amplitude={args.torque}, frequency={args.frequency}")
        print(f"Initial pose: {args.init}")
        if args.hold_pose:
            print(f"Hold pose: enabled (kp={args.hold_kp}, kd={args.hold_kd}, excluding target joint)")
        else:
            print("Hold pose: disabled")

        env = gym.create_env(sim, gymapi.Vec3(-2.0, -2.0, 0.0), gymapi.Vec3(2.0, 2.0, 2.0), 1)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, args.start_height)
        actor = gym.create_actor(env, asset, pose, "torque_demo_robot", 0, 1)

        dof_props = gym.get_actor_dof_properties(env, actor)
        dof_props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
        dof_props["stiffness"].fill(0.0)
        dof_props["damping"].fill(args.passive_damping)
        gym.set_actor_dof_properties(env, actor, dof_props)

        actor_dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
        initial_positions = build_initial_dof_positions(dof_names, args)
        actor_dof_states["pos"][:] = initial_positions
        actor_dof_states["vel"].fill(0.0)
        gym.set_actor_dof_states(env, actor, actor_dof_states, gymapi.STATE_ALL)

        if viewer is not None:
            cam_pos = gymapi.Vec3(2.5, 2.5, 1.6)
            cam_target = gymapi.Vec3(0.0, 0.0, 1.0)
            gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)

        # Tensor APIs are only safe after the simulation has been fully prepared.
        gym.prepare_sim(sim)
        dof_state_tensor = gym.acquire_dof_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(1, len(dof_names), 2)
        actuation = torch.zeros((1, len(dof_names)), dtype=torch.float32, device=torch_device)
        hold_targets = torch.tensor(initial_positions, dtype=torch.float32, device=torch_device).view(1, len(dof_names))
        hold_mask = torch.ones((1, len(dof_names)), dtype=torch.float32, device=torch_device)
        hold_mask[0, joint_idx] = 0.0
        effort_limits = torch.tensor(dof_props["effort"], dtype=torch.float32, device=torch_device).view(1, len(dof_names))

        num_steps = max(1, int(args.duration / args.dt))
        for step in range(num_steps):
            if viewer is not None and gym.query_viewer_has_closed(viewer):
                break

            t = step * args.dt
            actuation.zero_()
            if args.hold_pose:
                hold_torque = args.hold_kp * (hold_targets - dof_state[..., 0]) - args.hold_kd * dof_state[..., 1]
                actuation += hold_torque * hold_mask

            actuation[0, joint_idx] += compute_torque(args, t)
            actuation.copy_(torch.maximum(torch.minimum(actuation, effort_limits), -effort_limits))

            gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(actuation))
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.refresh_dof_state_tensor(sim)

            if viewer is not None:
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)
                gym.sync_frame_time(sim)

            if step % args.print_every == 0 or step == num_steps - 1:
                joint_pos = dof_state[0, joint_idx, 0].item()
                joint_vel = dof_state[0, joint_idx, 1].item()
                joint_tau = actuation[0, joint_idx].item()
                print(
                    f"step={step:05d} time={t:6.3f}s "
                    f"tau={joint_tau:8.3f} pos={joint_pos:8.4f} vel={joint_vel:8.4f}"
                )

    finally:
        if viewer is not None:
            gym.destroy_viewer(viewer)
        if sim is not None:
            gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
