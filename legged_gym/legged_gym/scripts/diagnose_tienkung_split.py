#!/usr/bin/env python3

import argparse
import copy
import json
import pickle
from types import SimpleNamespace

import isaacgym
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_rotate_inverse

import torch
import yaml
import numpy as np
import sys
import types

from legged_gym.envs import *  # noqa: F401,F403 - registers tasks
from legged_gym.gym_utils.task_registry import task_registry
from legged_gym.gym_utils.helpers import class_to_dict, get_load_path, parse_sim_params, set_seed
from legged_gym.envs.base.legged_robot import euler_from_quaternion


LEFT_ANKLE = "ankle_roll_l_link"
RIGHT_ANKLE = "ankle_roll_r_link"
LEFT_KNEE = "knee_pitch_l_link"
RIGHT_KNEE = "knee_pitch_r_link"

LEG_ACTIVE_NAMES = [
    "hip_roll_l_joint",
    "hip_pitch_l_joint",
    "hip_yaw_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "ankle_roll_l_joint",
    "hip_roll_r_joint",
    "hip_pitch_r_joint",
    "hip_yaw_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_r_joint",
]

TRAIN30_START10_DEFAULTS = {
    "hip_roll_l_joint": -0.0882,
    "hip_pitch_l_joint": -0.1274,
    "hip_yaw_l_joint": 0.0790,
    "knee_pitch_l_joint": 0.3272,
    "ankle_pitch_l_joint": -0.2256,
    "ankle_roll_l_joint": 0.0496,
    "hip_roll_r_joint": -0.0110,
    "hip_pitch_r_joint": -0.1781,
    "hip_yaw_r_joint": -0.0907,
    "knee_pitch_r_joint": 0.3165,
    "ankle_pitch_r_joint": -0.1545,
    "ankle_roll_r_joint": 0.0530,
    "waist_yaw_joint": -0.0061,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose why TianGong legs splay at motion start.")
    parser.add_argument(
        "--task",
        type=str,
        default="tienkung_stu_future_cc_stage1",
        help="Registered task used to build the env.",
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        default="/home/qsh/workspace_twist2/TWIST2/legged_gym/motion_data_configs/tienkung_ei_walk1.yaml",
        help="Single-motion yaml or pkl to diagnose.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=12, help="Analyze the first N source motion frames.")
    parser.add_argument("--lift-z", type=float, default=0.25, help="Extra height used for the contact-free lifted tests.")
    parser.add_argument("--policy-steps", type=int, default=1, help="Number of policy control steps for PD scenarios.")
    parser.add_argument("--policy-proj-name", type=str, default=None, help="Optional trained policy project name.")
    parser.add_argument("--policy-exptid", type=str, default=None, help="Optional trained policy run/exptid.")
    parser.add_argument("--policy-checkpoint", type=int, default=-1, help="Checkpoint id to load when policy is enabled.")
    parser.add_argument(
        "--default-legs-source",
        type=str,
        default="current",
        choices=["current", "train30_start10", "motion_frame0", "motion_start10"],
        help="Override the locomotion lower-body default pose before diagnosis.",
    )
    parser.add_argument("--pretty", action="store_true", help="Print full per-frame json.")
    return parser.parse_args()


def build_env(task_name, motion_file, sim_device, seed):
    env_cfg, _ = task_registry.get_cfgs(task_name)
    env_cfg = copy.deepcopy(env_cfg)

    env_cfg.seed = seed
    env_cfg.env.num_envs = 1
    env_cfg.motion.motion_file = motion_file
    env_cfg.env.rand_reset = False
    env_cfg.env.randomize_start_pos = False
    env_cfg.env.randomize_start_yaw = False
    env_cfg.noise.add_noise = False
    env_cfg.motion.motion_curriculum = False
    env_cfg.domain_rand.domain_rand_general = False
    env_cfg.domain_rand.randomize_gravity = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_end_effector = False
    env_cfg.domain_rand.randomize_motor = False
    env_cfg.domain_rand.action_delay = False

    use_gpu = sim_device.startswith("cuda")
    args = SimpleNamespace(
        physics_engine=gymapi.SIM_PHYSX,
        use_gpu=use_gpu,
        subscenes=0,
        use_gpu_pipeline=use_gpu,
        num_threads=0,
        device=sim_device,
    )

    set_seed(seed)
    sim_params = parse_sim_params(args, {"sim": class_to_dict(env_cfg.sim)})
    task_class = task_registry.get_task_class(task_name)
    env = task_class(
        cfg=env_cfg,
        sim_params=sim_params,
        physics_engine=gymapi.SIM_PHYSX,
        sim_device=sim_device,
        headless=True,
    )
    return env


def build_policy_runner(env, task_name, proj_name, exptid, checkpoint, rl_device):
    _, train_cfg = task_registry.get_cfgs(task_name)
    train_cfg = copy.deepcopy(train_cfg)
    train_cfg.runner.resume = False

    runner_args = SimpleNamespace(
        seed=None,
        max_iterations=None,
        resume=False,
        experiment_name=None,
        run_name=None,
        load_run=None,
        checkpoint=None,
        fix_action_std=False,
        no_rand=False,
        teacher_exptid=None,
        teacher_checkpoint=-1,
        eval_student=False,
        config_overrides={},
        rl_device=rl_device,
        resumeid=None,
        proj_name=proj_name,
        num_envs=None,
        rows=None,
        cols=None,
        record_video=False,
        teleop_mode=False,
    )
    runner, _ = task_registry.make_alg_runner(log_root=None, env=env, name=task_name, args=runner_args, train_cfg=train_cfg)
    run_root = f"/home/qsh/workspace_twist2/TWIST2/legged_gym/logs/{proj_name}/{exptid}"
    load_path = get_load_path(run_root, checkpoint=checkpoint)
    runner.load(load_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    normalizer = None
    if env.cfg.env.normalize_obs:
        try:
            normalizer = runner.get_normalizer(device=env.device)
        except Exception:
            normalizer = None
    return policy, normalizer, load_path


def load_motion_dof_array(motion_file):
    sys.modules.setdefault("numpy._core", types.ModuleType("numpy._core"))
    sys.modules["numpy._core"].multiarray = np.core.multiarray
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)

    motion_path = motion_file
    if motion_file.endswith((".yaml", ".yml")):
        cfg = yaml.safe_load(open(motion_file, "r"))
        root = cfg["root_path"]
        rel = cfg["motions"][0]["file"]
        motion_path = f"{root}/{rel}"

    data = pickle.load(open(motion_path, "rb"))
    return np.asarray(data["dof_pos"], dtype=np.float64)


def maybe_override_leg_defaults(env, motion_file, source):
    if source == "current":
        return

    if source == "train30_start10":
        override = TRAIN30_START10_DEFAULTS
    else:
        arr = load_motion_dof_array(motion_file)
        if source == "motion_frame0":
            src = arr[:1]
        elif source == "motion_start10":
            src = arr[: min(10, len(arr))]
        else:
            raise ValueError(f"Unexpected default source: {source}")

        active_name_to_local_idx = {name: i for i, name in enumerate(env.cfg.asset.active_dof_names)}
        override = {
            name: float(src[:, active_name_to_local_idx[name]].mean())
            for name in LEG_ACTIVE_NAMES + ["waist_yaw_joint"]
            if name in active_name_to_local_idx
        }

    active_name_to_local_idx = {name: i for i, name in enumerate(env.cfg.asset.active_dof_names)}
    full_name_to_idx = {name: i for i, name in enumerate(env.dof_names)}
    for name, value in override.items():
        if name in active_name_to_local_idx:
            active_idx = env._active_dof_indices[active_name_to_local_idx[name]]
            env.default_dof_pos_all[:, active_idx] = value
        if name in full_name_to_idx:
            env.default_dof_pos_all[:, full_name_to_idx[name]] = value
            env.cfg.init_state.default_joint_angles[name] = value


def refresh_sim_tensors(env):
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_force_sensor_tensor(env.sim)
    env.base_quat[:] = env.root_states[:, 3:7]
    env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
    env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])
    env.roll, env.pitch, env.yaw = euler_from_quaternion(env.base_quat)


def sync_root_and_dofs(env, full_dof_pos, full_dof_vel, root_pos, root_rot, root_vel, root_ang_vel):
    env.dof_pos[0] = full_dof_pos[0]
    env.dof_vel[0] = full_dof_vel[0]
    env.root_states[0, 0:3] = root_pos[0]
    env.root_states[0, 3:7] = root_rot[0]
    env.root_states[0, 7:10] = root_vel[0]
    env.root_states[0, 10:13] = root_ang_vel[0]

    env_ids_int32 = torch.tensor([0], device=env.device, dtype=torch.int32)
    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_state),
        gymtorch.unwrap_tensor(env_ids_int32),
        1,
    )
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env_ids_int32),
        1,
    )


def simulate_no_torque(env, num_substeps=1):
    env.torques.zero_()
    env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(env.torques))
    for _ in range(num_substeps):
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
        env.gym.refresh_dof_state_tensor(env.sim)
    refresh_sim_tensors(env)


def simulate_policy_steps(env, actions, num_policy_steps):
    for _ in range(num_policy_steps):
        clipped = torch.clip(actions, -env.cfg.normalization.clip_actions / env.cfg.control.action_scale,
                             env.cfg.normalization.clip_actions / env.cfg.control.action_scale)
        env.actions[:] = clipped
        for _ in range(env.cfg.control.decimation):
            env.torques = env._compute_torques(env.actions).view(env.torques.shape)
            env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(env.torques))
            env.gym.simulate(env.sim)
            env.gym.fetch_results(env.sim, True)
            env.gym.refresh_dof_state_tensor(env.sim)
        refresh_sim_tensors(env)


def body_name_to_idx(names):
    return {name: i for i, name in enumerate(names)}


def root_local_positions(root_pos, root_rot, body_global):
    rel = body_global - root_pos[:, None, :]
    flat_rel = rel.reshape(-1, 3)
    flat_rot = root_rot[:, None, :].expand(rel.shape[0], rel.shape[1], 4).reshape(-1, 4)
    local = quat_rotate_inverse(flat_rot, flat_rel)
    return local.reshape(rel.shape[0], rel.shape[1], 3)


def lateral_metrics_from_local(local_positions, idx_map):
    la = local_positions[0, idx_map[LEFT_ANKLE]]
    ra = local_positions[0, idx_map[RIGHT_ANKLE]]
    lk = local_positions[0, idx_map[LEFT_KNEE]]
    rk = local_positions[0, idx_map[RIGHT_KNEE]]
    return {
        "ankle_gap_y": float(abs(la[1] - ra[1]).item()),
        "knee_gap_y": float(abs(lk[1] - rk[1]).item()),
        "ankle_x_sep": float(abs(la[0] - ra[0]).item()),
        "knee_x_sep": float(abs(lk[0] - rk[0]).item()),
        "left_ankle": [float(x) for x in la.tolist()],
        "right_ankle": [float(x) for x in ra.tolist()],
        "left_knee": [float(x) for x in lk.tolist()],
        "right_knee": [float(x) for x in rk.tolist()],
    }


def motion_local_positions_subset(body_pos, motion_idx_map):
    wanted = [LEFT_ANKLE, RIGHT_ANKLE, LEFT_KNEE, RIGHT_KNEE]
    subset = torch.stack([body_pos[0, motion_idx_map[name]] for name in wanted], dim=0).unsqueeze(0)
    subset_idx = {name: i for i, name in enumerate(wanted)}
    return subset, subset_idx


def compute_mapping_error(env, motion_body_local, root_pos, root_rot):
    ref_body_pos = torch.zeros_like(env._ref_body_pos)
    env._assign_ref_body_pos_from_motion(ref_body_pos, root_pos, root_rot, motion_body_local)
    ref_local = root_local_positions(root_pos, root_rot, ref_body_pos)
    errs = []
    worst = ("", 0.0)
    for motion_idx, env_idx in env._motion_to_env_body:
        body_name = env.body_names[env_idx]
        motion_local = motion_body_local[0, motion_idx]
        mapped_local = ref_local[0, env_idx]
        err = float(torch.norm(mapped_local - motion_local).item())
        errs.append(err)
        if err > worst[1]:
            worst = (body_name, err)
    mae = float(sum(errs) / max(len(errs), 1))
    max_err = float(max(errs) if errs else 0.0)
    return mae, max_err, worst


def compute_asset_error(env, motion_body_local, root_pos, root_rot):
    actual_local = root_local_positions(env.root_states[:, 0:3], env.root_states[:, 3:7], env.rigid_body_states[:, :, 0:3])
    errs = []
    worst = ("", 0.0)
    for motion_idx, env_idx in env._motion_to_env_body:
        body_name = env.body_names[env_idx]
        motion_local = motion_body_local[0, motion_idx]
        asset_local = actual_local[0, env_idx]
        err = float(torch.norm(asset_local - motion_local).item())
        errs.append(err)
        if err > worst[1]:
            worst = (body_name, err)
    mae = float(sum(errs) / max(len(errs), 1))
    max_err = float(max(errs) if errs else 0.0)
    return mae, max_err, worst, actual_local


def active_leg_snapshot(env, full_dof_pos):
    active = full_dof_pos[0, env._active_dof_indices]
    active_name_to_local_idx = {name: i for i, name in enumerate(env.cfg.asset.active_dof_names)}
    return {
        name: float(active[active_name_to_local_idx[name]].item())
        for name in LEG_ACTIVE_NAMES
    }


def active_leg_default_snapshot(env):
    active_default = env.default_dof_pos_all[0, env._active_dof_indices]
    active_name_to_local_idx = {name: i for i, name in enumerate(env.cfg.asset.active_dof_names)}
    return {
        name: float(active_default[active_name_to_local_idx[name]].item())
        for name in LEG_ACTIVE_NAMES
    }


def active_leg_ref_delta(env, full_dof_pos):
    active = full_dof_pos[0, env._active_dof_indices]
    active_default = env.default_dof_pos_all[0, env._active_dof_indices]
    active_name_to_local_idx = {name: i for i, name in enumerate(env.cfg.asset.active_dof_names)}
    return {
        name: float((active[active_name_to_local_idx[name]] - active_default[active_name_to_local_idx[name]]).item())
        for name in LEG_ACTIVE_NAMES
    }


def compute_ref_action(env, full_dof_pos):
    active_target = full_dof_pos[:, env._active_dof_indices]
    active_default = env.default_dof_pos_all[:, env._active_dof_indices]
    return (active_target - active_default) / env.cfg.control.action_scale


def leg_ref_action_snapshot(env, ref_action):
    active_name_to_local_idx = {name: i for i, name in enumerate(env.cfg.asset.active_dof_names)}
    return {
        name: float(ref_action[0, active_name_to_local_idx[name]].item())
        for name in LEG_ACTIVE_NAMES
    }


def get_policy_actions(env, obs, policy, normalizer):
    if normalizer is not None:
        obs = normalizer.normalize(obs.detach())
    else:
        obs = obs.detach()
    return policy(obs, hist_encoding=True)


def scenario_metrics(env, motion_local, motion_idx_map, ref_root_pos, ref_root_rot):
    subset, subset_idx = motion_local_positions_subset(motion_local, motion_idx_map)
    motion_metrics = lateral_metrics_from_local(subset, subset_idx)
    mapping_mae, mapping_max, mapping_worst = compute_mapping_error(env, motion_local, ref_root_pos, ref_root_rot)
    return motion_metrics, mapping_mae, mapping_max, mapping_worst


def main():
    args = parse_args()
    env = build_env(args.task, args.motion_file, args.device, args.seed)
    maybe_override_leg_defaults(env, args.motion_file, args.default_legs_source)
    policy = None
    normalizer = None
    loaded_policy_path = None
    if args.policy_proj_name and args.policy_exptid:
        policy, normalizer, loaded_policy_path = build_policy_runner(
            env=env,
            task_name=args.task,
            proj_name=args.policy_proj_name,
            exptid=args.policy_exptid,
            checkpoint=args.policy_checkpoint,
            rl_device=args.device,
        )

    try:
        env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
        env.reset_idx(env_ids, motion_ids=torch.tensor([0], device=env.device, dtype=torch.long))
        refresh_sim_tensors(env)

        motion_idx_map = body_name_to_idx(env._motion_lib._body_link_list)
        env_body_idx = body_name_to_idx(env.body_names)

        fps = float(env._motion_lib._motion_fps[0].item())
        frame_count = min(args.max_frames, int(env._motion_lib._motion_num_frames[0].item()))
        results = []

        for frame_idx in range(frame_count):
            t = frame_idx / fps
            motion_ids = torch.tensor([0], device=env.device, dtype=torch.long)
            motion_times = torch.tensor([t], device=env.device, dtype=torch.float)
            root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos, *_ = env._motion_lib.calc_motion_frame(
                motion_ids, motion_times
            )
            root_pos[:, 2] += env.cfg.motion.height_offset
            full_dof_pos, full_dof_vel = env._expand_motion_dofs_to_full(dof_pos, dof_vel, env_ids=env_ids)

            motion_metrics, mapping_mae, mapping_max, mapping_worst = scenario_metrics(
                env, body_pos, motion_idx_map, root_pos, root_rot
            )

            # Scenario A: lifted kinematic injection, zero torque. This isolates mapping + asset geometry.
            lifted_root_pos = root_pos.clone()
            lifted_root_pos[:, 2] += args.lift_z
            sync_root_and_dofs(
                env,
                full_dof_pos=full_dof_pos,
                full_dof_vel=torch.zeros_like(full_dof_vel),
                root_pos=lifted_root_pos,
                root_rot=root_rot,
                root_vel=torch.zeros_like(root_vel),
                root_ang_vel=torch.zeros_like(root_ang_vel),
            )
            simulate_no_torque(env, num_substeps=1)
            asset_mae_lift, asset_max_lift, asset_worst_lift, actual_local_lift = compute_asset_error(
                env, body_pos, lifted_root_pos, root_rot
            )
            lifted_metrics = lateral_metrics_from_local(actual_local_lift[:, [env_body_idx[LEFT_ANKLE], env_body_idx[RIGHT_ANKLE], env_body_idx[LEFT_KNEE], env_body_idx[RIGHT_KNEE]], :], {
                LEFT_ANKLE: 0, RIGHT_ANKLE: 1, LEFT_KNEE: 2, RIGHT_KNEE: 3
            })

            # Scenario B: training-style RSI reset, zero torque, one physics step.
            grounded_root_pos = root_pos.clone()
            grounded_root_pos[:, 2] += 0.05
            sync_root_and_dofs(
                env,
                full_dof_pos=full_dof_pos,
                full_dof_vel=full_dof_vel * env._reset_ref_vel_factor,
                root_pos=grounded_root_pos,
                root_rot=root_rot,
                root_vel=root_vel * env._reset_ref_vel_factor,
                root_ang_vel=root_ang_vel * env._reset_ref_vel_factor,
            )
            simulate_no_torque(env, num_substeps=1)
            asset_mae_ground, asset_max_ground, asset_worst_ground, actual_local_ground = compute_asset_error(
                env, body_pos, grounded_root_pos, root_rot
            )
            grounded_metrics = lateral_metrics_from_local(actual_local_ground[:, [env_body_idx[LEFT_ANKLE], env_body_idx[RIGHT_ANKLE], env_body_idx[LEFT_KNEE], env_body_idx[RIGHT_KNEE]], :], {
                LEFT_ANKLE: 0, RIGHT_ANKLE: 1, LEFT_KNEE: 2, RIGHT_KNEE: 3
            })
            grounded_contact = {
                "left_foot_fz": float(env.contact_forces[0, env.feet_indices[0], 2].item()),
                "right_foot_fz": float(env.contact_forces[0, env.feet_indices[1], 2].item()),
            }

            # Scenario C: training-style reset then one control step with zero action.
            sync_root_and_dofs(
                env,
                full_dof_pos=full_dof_pos,
                full_dof_vel=full_dof_vel * env._reset_ref_vel_factor,
                root_pos=grounded_root_pos,
                root_rot=root_rot,
                root_vel=root_vel * env._reset_ref_vel_factor,
                root_ang_vel=root_ang_vel * env._reset_ref_vel_factor,
            )
            simulate_no_torque(env, num_substeps=1)
            zero_action = torch.zeros((1, env.num_actions), device=env.device, dtype=torch.float)
            simulate_policy_steps(env, zero_action, args.policy_steps)
            _, _, _, zero_local = compute_asset_error(env, body_pos, env.root_states[:, 0:3], env.root_states[:, 3:7])
            zero_metrics = lateral_metrics_from_local(zero_local[:, [env_body_idx[LEFT_ANKLE], env_body_idx[RIGHT_ANKLE], env_body_idx[LEFT_KNEE], env_body_idx[RIGHT_KNEE]], :], {
                LEFT_ANKLE: 0, RIGHT_ANKLE: 1, LEFT_KNEE: 2, RIGHT_KNEE: 3
            })
            zero_torque = {
                "hip_roll_l": float(env.torques[0, env._active_dof_indices[0]].item()),
                "hip_yaw_l": float(env.torques[0, env._active_dof_indices[2]].item()),
                "hip_roll_r": float(env.torques[0, env._active_dof_indices[6]].item()),
                "hip_yaw_r": float(env.torques[0, env._active_dof_indices[8]].item()),
            }

            # Scenario D: training-style reset then one control step that explicitly holds the reference.
            sync_root_and_dofs(
                env,
                full_dof_pos=full_dof_pos,
                full_dof_vel=full_dof_vel * env._reset_ref_vel_factor,
                root_pos=grounded_root_pos,
                root_rot=root_rot,
                root_vel=root_vel * env._reset_ref_vel_factor,
                root_ang_vel=root_ang_vel * env._reset_ref_vel_factor,
            )
            simulate_no_torque(env, num_substeps=1)
            ref_action = compute_ref_action(env, full_dof_pos)
            simulate_policy_steps(env, ref_action, args.policy_steps)
            _, _, _, hold_local = compute_asset_error(env, body_pos, env.root_states[:, 0:3], env.root_states[:, 3:7])
            hold_metrics = lateral_metrics_from_local(hold_local[:, [env_body_idx[LEFT_ANKLE], env_body_idx[RIGHT_ANKLE], env_body_idx[LEFT_KNEE], env_body_idx[RIGHT_KNEE]], :], {
                LEFT_ANKLE: 0, RIGHT_ANKLE: 1, LEFT_KNEE: 2, RIGHT_KNEE: 3
            })
            hold_torque = {
                "hip_roll_l": float(env.torques[0, env._active_dof_indices[0]].item()),
                "hip_yaw_l": float(env.torques[0, env._active_dof_indices[2]].item()),
                "hip_roll_r": float(env.torques[0, env._active_dof_indices[6]].item()),
                "hip_yaw_r": float(env.torques[0, env._active_dof_indices[8]].item()),
            }

            policy_metrics = None
            policy_torque = None
            policy_action_snapshot = None
            if policy is not None:
                sync_root_and_dofs(
                    env,
                    full_dof_pos=full_dof_pos,
                    full_dof_vel=full_dof_vel * env._reset_ref_vel_factor,
                    root_pos=grounded_root_pos,
                    root_rot=root_rot,
                    root_vel=root_vel * env._reset_ref_vel_factor,
                    root_ang_vel=root_ang_vel * env._reset_ref_vel_factor,
                )
                simulate_no_torque(env, num_substeps=1)
                obs = env.get_observations()
                policy_actions = get_policy_actions(env, obs, policy, normalizer)
                policy_action_snapshot = leg_ref_action_snapshot(env, policy_actions)
                simulate_policy_steps(env, policy_actions, args.policy_steps)
                _, _, _, policy_local = compute_asset_error(env, body_pos, env.root_states[:, 0:3], env.root_states[:, 3:7])
                policy_metrics = lateral_metrics_from_local(policy_local[:, [env_body_idx[LEFT_ANKLE], env_body_idx[RIGHT_ANKLE], env_body_idx[LEFT_KNEE], env_body_idx[RIGHT_KNEE]], :], {
                    LEFT_ANKLE: 0, RIGHT_ANKLE: 1, LEFT_KNEE: 2, RIGHT_KNEE: 3
                })
                policy_torque = {
                    "hip_roll_l": float(env.torques[0, env._active_dof_indices[0]].item()),
                    "hip_yaw_l": float(env.torques[0, env._active_dof_indices[2]].item()),
                    "hip_roll_r": float(env.torques[0, env._active_dof_indices[6]].item()),
                    "hip_yaw_r": float(env.torques[0, env._active_dof_indices[8]].item()),
                }

            results.append(
                {
                    "frame": frame_idx,
                    "time_s": round(t, 4),
                    "leg_defaults": active_leg_default_snapshot(env),
                    "leg_joints": active_leg_snapshot(env, full_dof_pos),
                    "leg_ref_minus_default": active_leg_ref_delta(env, full_dof_pos),
                    "leg_ref_action": leg_ref_action_snapshot(env, ref_action),
                    "motion": motion_metrics,
                    "mapping": {
                        "body_mae": mapping_mae,
                        "body_max": mapping_max,
                        "worst_body": mapping_worst[0],
                        "worst_err": mapping_worst[1],
                    },
                    "asset_lifted_zero_torque": {
                        **lifted_metrics,
                        "body_mae": asset_mae_lift,
                        "body_max": asset_max_lift,
                        "worst_body": asset_worst_lift[0],
                        "worst_err": asset_worst_lift[1],
                    },
                    "training_reset_zero_torque": {
                        **grounded_metrics,
                        "body_mae": asset_mae_ground,
                        "body_max": asset_max_ground,
                        "worst_body": asset_worst_ground[0],
                        "worst_err": asset_worst_ground[1],
                        **grounded_contact,
                    },
                    "training_reset_zero_action": {
                        **zero_metrics,
                        **zero_torque,
                    },
                    "training_reset_hold_ref_action": {
                        **hold_metrics,
                        **hold_torque,
                    },
                    "training_reset_policy_action": (
                        {
                            **policy_metrics,
                            **policy_torque,
                            "action": policy_action_snapshot,
                        }
                        if policy_metrics is not None
                        else None
                    ),
                }
            )

        summary = {
            "task": args.task,
            "motion_file": args.motion_file,
            "default_legs_source": args.default_legs_source,
            "policy_path": loaded_policy_path,
            "fps": fps,
            "frames_analyzed": frame_count,
            "results": results,
        }

        if args.pretty:
            print(json.dumps(summary, indent=2))
        else:
            for row in results:
                print(
                    f"frame={row['frame']:02d} t={row['time_s']:.3f} "
                    f"motion ankle={row['motion']['ankle_gap_y']:.3f} knee={row['motion']['knee_gap_y']:.3f} | "
                    f"map_mae={row['mapping']['body_mae']:.4f} max={row['mapping']['body_max']:.4f} | "
                    f"lift ankle={row['asset_lifted_zero_torque']['ankle_gap_y']:.3f} knee={row['asset_lifted_zero_torque']['knee_gap_y']:.3f} mae={row['asset_lifted_zero_torque']['body_mae']:.4f} | "
                    f"reset ankle={row['training_reset_zero_torque']['ankle_gap_y']:.3f} knee={row['training_reset_zero_torque']['knee_gap_y']:.3f} fz=({row['training_reset_zero_torque']['left_foot_fz']:.1f},{row['training_reset_zero_torque']['right_foot_fz']:.1f}) | "
                    f"zero_act ankle={row['training_reset_zero_action']['ankle_gap_y']:.3f} knee={row['training_reset_zero_action']['knee_gap_y']:.3f} | "
                    f"hold_ref ankle={row['training_reset_hold_ref_action']['ankle_gap_y']:.3f} knee={row['training_reset_hold_ref_action']['knee_gap_y']:.3f}"
                )

        # High-level quick read so the command line immediately points at the likely layer.
        motion_ankle = sum(r["motion"]["ankle_gap_y"] for r in results) / frame_count
        lift_ankle = sum(r["asset_lifted_zero_torque"]["ankle_gap_y"] for r in results) / frame_count
        reset_ankle = sum(r["training_reset_zero_torque"]["ankle_gap_y"] for r in results) / frame_count
        zero_ankle = sum(r["training_reset_zero_action"]["ankle_gap_y"] for r in results) / frame_count
        hold_ankle = sum(r["training_reset_hold_ref_action"]["ankle_gap_y"] for r in results) / frame_count
        if policy is not None:
            policy_ankle = sum(r["training_reset_policy_action"]["ankle_gap_y"] for r in results) / frame_count
        else:
            policy_ankle = None
        mapping_mae = sum(r["mapping"]["body_mae"] for r in results) / frame_count
        lift_mae = sum(r["asset_lifted_zero_torque"]["body_mae"] for r in results) / frame_count
        reset_mae = sum(r["training_reset_zero_torque"]["body_mae"] for r in results) / frame_count
        summary_line = (
            "\\nSUMMARY "
            f"motion_ankle_gap={motion_ankle:.3f} "
            f"lifted_ankle_gap={lift_ankle:.3f} "
            f"reset_ankle_gap={reset_ankle:.3f} "
            f"zero_action_ankle_gap={zero_ankle:.3f} "
            f"hold_ref_ankle_gap={hold_ankle:.3f} "
        )
        if policy_ankle is not None:
            summary_line += f"policy_action_ankle_gap={policy_ankle:.3f} "
        summary_line += f"mapping_mae={mapping_mae:.4f} lifted_body_mae={lift_mae:.4f} reset_body_mae={reset_mae:.4f}"
        print(summary_line)
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
