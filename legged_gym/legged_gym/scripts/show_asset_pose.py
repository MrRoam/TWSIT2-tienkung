#!/usr/bin/env python3

import argparse
import os

import isaacgym
from isaacgym import gymapi, gymutil


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
        description="Load an asset into Isaac Gym and keep it completely still for visual inspection."
    )
    parser.add_argument(
        "--asset",
        type=str,
        default="/data/shared_folder/GMR/assets/tienkung_ei/mjcf/tienkung_ei_v1.xml",
        help="Absolute path to a URDF or MJCF asset.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Simulation device, for example cpu or cuda:0.")
    parser.add_argument("--headless", action="store_true", help="Disable the viewer.")
    parser.add_argument("--list-dofs", action="store_true", help="Print DOF names and exit.")
    parser.add_argument("--dt", type=float, default=0.005, help="Physics timestep.")
    parser.add_argument("--start-height", type=float, default=1.0, help="Initial actor base height.")
    parser.add_argument(
        "--init",
        type=str,
        choices=["zero", "tienkung_stand"],
        default="tienkung_stand",
        help="Initial joint pose used for display.",
    )
    parser.add_argument(
        "--gravity",
        type=str,
        choices=["on", "off"],
        default="off",
        help="Use gravity or keep the asset floating in place.",
    )
    parser.add_argument(
        "--fix-base",
        dest="fix_base",
        action="store_true",
        help="Fix the base link in place. Enabled by default for static display.",
    )
    parser.add_argument(
        "--free-base",
        dest="fix_base",
        action="store_false",
        help="Allow the base link to move freely.",
    )
    parser.set_defaults(fix_base=True)
    return parser.parse_args()


def build_initial_dof_positions(dof_names, init_mode):
    if init_mode == "zero":
        return [0.0] * len(dof_names)
    if init_mode == "tienkung_stand":
        return [TIENKUNG_STAND_JOINT_ANGLES.get(name, 0.0) for name in dof_names]
    raise ValueError(f"Unsupported init mode: {init_mode}")


def main():
    args = parse_args()
    gym = gymapi.acquire_gym()
    sim = None
    viewer = None

    try:
        sim_device_type, sim_device_id = gymutil.parse_device_str(args.device)
        use_gpu = sim_device_type == "cuda"
        graphics_device_id = -1 if args.headless else sim_device_id

        sim_params = gymapi.SimParams()
        sim_params.dt = args.dt
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81 if args.gravity == "on" else 0.0)
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
        gym.add_ground(sim, plane_params)

        if not args.headless:
            viewer = gym.create_viewer(sim, gymapi.CameraProperties())
            if viewer is None:
                raise RuntimeError("Failed to create Isaac Gym viewer.")

        asset_path = os.path.abspath(args.asset)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = args.fix_base
        asset_options.collapse_fixed_joints = False
        asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
        if asset is None:
            raise RuntimeError(f"Failed to load asset: {asset_path}")

        dof_names = list(gym.get_asset_dof_names(asset))
        body_names = list(gym.get_asset_rigid_body_names(asset))
        print(f"Loaded asset: {asset_path}")
        print(f"Rigid bodies: {len(body_names)}")
        print(f"DOFs: {len(dof_names)}")
        print(f"Initial pose: {args.init}")
        print(f"Gravity: {args.gravity}")
        print(f"Fix base: {args.fix_base}")

        if args.list_dofs:
            for i, name in enumerate(dof_names):
                print(f"{i:02d}: {name}")
            return

        env = gym.create_env(sim, gymapi.Vec3(-2.0, -2.0, 0.0), gymapi.Vec3(2.0, 2.0, 2.0), 1)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, args.start_height)
        actor = gym.create_actor(env, asset, pose, "static_pose_robot", 0, 1)

        dof_props = gym.get_actor_dof_properties(env, actor)
        dof_props["driveMode"].fill(gymapi.DOF_MODE_NONE)
        dof_props["stiffness"].fill(0.0)
        dof_props["damping"].fill(0.0)
        gym.set_actor_dof_properties(env, actor, dof_props)

        actor_dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
        actor_dof_states["pos"][:] = build_initial_dof_positions(dof_names, args.init)
        actor_dof_states["vel"].fill(0.0)
        gym.set_actor_dof_states(env, actor, actor_dof_states, gymapi.STATE_ALL)

        if viewer is not None:
            cam_pos = gymapi.Vec3(2.5, 2.5, 1.6)
            cam_target = gymapi.Vec3(0.0, 0.0, 1.0)
            gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)

        gym.prepare_sim(sim)
        while viewer is not None and not gym.query_viewer_has_closed(viewer):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

        if viewer is None:
            gym.simulate(sim)
            gym.fetch_results(sim, True)

    finally:
        if viewer is not None:
            gym.destroy_viewer(viewer)
        if sim is not None:
            gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
