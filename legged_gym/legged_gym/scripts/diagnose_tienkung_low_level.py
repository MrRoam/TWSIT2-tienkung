#!/usr/bin/env python3

import argparse
import copy
import json
import os
from datetime import datetime
from types import SimpleNamespace

import isaacgym
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_rotate_inverse

import torch

from legged_gym.envs import *  # noqa: F401,F403 - task registration side effect
from legged_gym.envs.base.legged_robot import euler_from_quaternion
from legged_gym.gym_utils.helpers import class_to_dict, parse_sim_params, set_seed
from legged_gym.gym_utils.task_registry import task_registry


DEFAULT_TASK = "tienkung_stu_future_cc_stage1"
DEFAULT_SINGLE_JOINTS = [
    "hip_pitch_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "waist_yaw_joint",
    "shoulder_pitch_l_joint",
]
DEFAULT_STEP_AMPLITUDES = [0.30, 0.50, 0.80]
DEFAULT_STANCE_DELTAS = {
    "hip_roll_l_joint": 0.04,
    "hip_roll_r_joint": -0.04,
    "waist_yaw_joint": 0.03,
    "shoulder_pitch_l_joint": 0.05,
    "shoulder_pitch_r_joint": -0.05,
}
NON_REFERENCE_MODES = {"single_joint_step", "pose_hold", "stance_contact"}


def parse_args():
    parser = argparse.ArgumentParser(description="Low-level TianKung control diagnostics without policy inference.")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "single_joint_step", "pose_hold", "stance_contact", "reference_tracking"],
    )
    parser.add_argument("--headless", action="store_true", help="Run without viewer.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--motion-file",
        type=str,
        default=None,
        help="Optional motion yaml/pkl override used by the task config.",
    )

    parser.add_argument("--use-current-phys", action="store_true", help="Use current task physics instead of baseline physics.")
    parser.add_argument("--compare-phys", action="store_true", help="Run baseline and current physics back-to-back.")
    parser.add_argument("--static-friction", type=float, default=1.0, help="Baseline ground static friction.")
    parser.add_argument("--dynamic-friction", type=float, default=1.0, help="Baseline ground dynamic friction.")
    parser.add_argument("--disable-rand", action="store_true", help="Disable domain randomization even for current physics.")
    parser.add_argument(
        "--pd-profile",
        type=str,
        default="current",
        choices=["current", "g1_like"],
        help="Optional runtime PD override profile for diagnostics.",
    )

    parser.add_argument("--joint-name", type=str, default=None, help="Single joint name for single_joint_step mode.")
    parser.add_argument("--joint-group", type=str, default="default", choices=["default", "legs", "upper"])
    parser.add_argument("--step-amplitudes", type=str, default="0.05,0.10,0.20")
    parser.add_argument("--lift-z", type=float, default=0.25, help="Extra base height for in-air tests.")

    parser.add_argument("--pre-steps", type=int, default=None)
    parser.add_argument("--pre-seconds", type=float, default=None, help="Optional pre-hold duration in seconds before switching targets.")
    parser.add_argument("--hold-steps", type=int, default=None)
    parser.add_argument("--hold-seconds", type=float, default=None, help="Optional hold/release duration in seconds for the active phase.")
    parser.add_argument("--reference-steps", type=int, default=120)
    parser.add_argument("--contact-threshold", type=float, default=5.0)
    parser.add_argument("--slip-threshold", type=float, default=0.15)

    parser.add_argument("--save-json", type=str, default=None, help="Optional output path for structured results.")
    return parser.parse_args()


def parse_float_list(text):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def resolve_pre_steps(env, args, default_steps=20, default_seconds=None):
    if args.pre_steps is not None:
        return max(0, int(args.pre_steps))
    if args.pre_seconds is not None:
        return max(0, int(round(args.pre_seconds / env.dt)))
    if default_seconds is not None:
        return max(0, int(round(default_seconds / env.dt)))
    return max(0, int(default_steps))


def resolve_hold_steps(env, args, default_steps=100, default_seconds=None):
    if args.hold_steps is not None:
        return max(0, int(args.hold_steps))
    if args.hold_seconds is not None:
        return max(0, int(round(args.hold_seconds / env.dt)))
    if default_seconds is not None:
        return max(0, int(round(default_seconds / env.dt)))
    return max(0, int(default_steps))


def zero_out_domain_rand(env_cfg):
    env_cfg.domain_rand.domain_rand_general = False
    env_cfg.domain_rand.randomize_gravity = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_end_effector = False
    env_cfg.domain_rand.randomize_motor = False
    env_cfg.domain_rand.action_delay = False


class StubMotionLib:
    """Minimal motion source for diagnostics that do not need real trajectory data."""

    def __init__(self, env):
        self._env = env
        self.device = env.device
        self._num_motions = 1
        self._motion_names = ["__stub_stationary__"]
        self._motion_length = 10.0
        self._body_link_list = list(env.body_names)
        self._num_dof = env.num_dof
        self._num_bodies = len(self._body_link_list)

    def num_motions(self):
        return self._num_motions

    def get_motion_length(self, motion_ids):
        if torch.is_tensor(motion_ids):
            return torch.full(
                motion_ids.shape,
                float(self._motion_length),
                device=motion_ids.device,
                dtype=torch.float,
            )
        return torch.tensor(float(self._motion_length), device=self.device, dtype=torch.float)

    def get_motion_names(self):
        return list(self._motion_names)

    def get_key_body_idx(self, key_body_names):
        name_to_idx = {name: idx for idx, name in enumerate(self._body_link_list)}
        return torch.tensor(
            [name_to_idx.get(name, 0) for name in key_body_names],
            device=self.device,
            dtype=torch.long,
        )

    def sample_motions(self, n, **kwargs):
        return torch.zeros(n, device=self.device, dtype=torch.long)

    def sample_time(self, motion_ids):
        return torch.zeros_like(motion_ids, dtype=torch.float, device=motion_ids.device)

    def calc_motion_frame(self, motion_ids, motion_times):
        if not torch.is_tensor(motion_ids):
            motion_ids = torch.tensor([motion_ids], device=self.device, dtype=torch.long)
        n = int(motion_ids.shape[0])
        base_dof_pos = self._env.default_dof_pos_all[0].detach().clone()
        root_pos = torch.zeros((n, 3), device=self.device, dtype=torch.float)
        root_rot = torch.zeros((n, 4), device=self.device, dtype=torch.float)
        root_rot[:, 3] = 1.0
        root_vel = torch.zeros((n, 3), device=self.device, dtype=torch.float)
        root_ang_vel = torch.zeros((n, 3), device=self.device, dtype=torch.float)
        dof_pos = base_dof_pos.unsqueeze(0).repeat(n, 1)
        dof_vel = torch.zeros((n, self._num_dof), device=self.device, dtype=torch.float)
        body_pos = torch.zeros((n, self._num_bodies, 3), device=self.device, dtype=torch.float)
        root_pos_delta_local = torch.zeros((n, 3), device=self.device, dtype=torch.float)
        root_rot_delta_local = torch.zeros((n, 3), device=self.device, dtype=torch.float)
        return (
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            dof_vel,
            body_pos,
            root_pos_delta_local,
            root_rot_delta_local,
        )


def build_env(args, phys_label, load_motion=True):
    env_cfg, _ = task_registry.get_cfgs(args.task)
    env_cfg = copy.deepcopy(env_cfg)

    env_cfg.seed = args.seed
    env_cfg.env.num_envs = args.num_envs
    env_cfg.env.rand_reset = False
    env_cfg.env.randomize_start_pos = False
    env_cfg.env.randomize_start_yaw = False
    env_cfg.noise.add_noise = False
    env_cfg.terrain.curriculum = False
    env_cfg.motion.motion_curriculum = False
    if args.motion_file is not None and hasattr(env_cfg, "motion"):
        env_cfg.motion.motion_file = args.motion_file

    if phys_label == "baseline_phys":
        env_cfg.terrain.static_friction = args.static_friction
        env_cfg.terrain.dynamic_friction = args.dynamic_friction
        zero_out_domain_rand(env_cfg)
    elif args.disable_rand:
        zero_out_domain_rand(env_cfg)

    use_gpu = args.device.startswith("cuda")
    sim_args = SimpleNamespace(
        physics_engine=gymapi.SIM_PHYSX,
        use_gpu=use_gpu,
        subscenes=0,
        use_gpu_pipeline=use_gpu,
        num_threads=0,
        device=args.device,
    )
    set_seed(args.seed)
    sim_params = parse_sim_params(sim_args, {"sim": class_to_dict(env_cfg.sim)})
    task_class = task_registry.get_task_class(args.task)
    had_local_load_motions = "_load_motions" in task_class.__dict__
    original_load_motions = task_class.__dict__.get("_load_motions")

    if not load_motion:
        def _load_stub_motions(self):
            self._motion_lib = StubMotionLib(self)

        task_class._load_motions = _load_stub_motions

    try:
        env = task_class(
            cfg=env_cfg,
            sim_params=sim_params,
            physics_engine=gymapi.SIM_PHYSX,
            sim_device=args.device,
            headless=args.headless,
        )
    finally:
        if not load_motion:
            if had_local_load_motions:
                task_class._load_motions = original_load_motions
            else:
                delattr(task_class, "_load_motions")

    env.reset()
    apply_pd_profile(env, args.pd_profile)
    return env


def apply_pd_profile(env, profile_name):
    if profile_name == "current":
        return

    if profile_name != "g1_like":
        raise ValueError(f"Unknown pd profile: {profile_name}")

    def gains_for_name(name):
        lname = name.lower()
        if "hip_yaw" in lname:
            return 100.0, 2.0
        if "hip_roll" in lname:
            return 100.0, 2.0
        if "hip_pitch" in lname:
            return 100.0, 2.0
        if "knee" in lname:
            return 150.0, 4.0
        if "ankle" in lname:
            return 40.0, 2.0
        if "waist" in lname:
            return 150.0, 4.0
        if "shoulder" in lname:
            return 40.0, 5.0
        if "elbow" in lname:
            return 40.0, 5.0
        if "wrist" in lname:
            return 40.0, 5.0
        if "head" in lname:
            return 40.0, 5.0
        return float(env.p_gains[env.dof_names.index(name)].item()), float(env.d_gains[env.dof_names.index(name)].item())

    for i, name in enumerate(env.dof_names):
        kp, kd = gains_for_name(name)
        env.p_gains[i] = kp
        env.d_gains[i] = kd


def destroy_env(env):
    try:
        if getattr(env, "viewer", None) is not None:
            env.gym.destroy_viewer(env.viewer)
    except Exception:
        pass
    try:
        env.gym.destroy_sim(env.sim)
    except Exception:
        pass


def refresh_env_state(env):
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_force_sensor_tensor(env.sim)
    env.base_quat[:] = env.root_states[:, 3:7]
    env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
    env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])
    env.roll, env.pitch, env.yaw = euler_from_quaternion(env.base_quat)


def all_env_ids(env):
    return torch.arange(env.num_envs, device=env.device, dtype=torch.int32)


def set_full_state(env, full_dof_pos, full_dof_vel, root_pos, root_rot, root_vel, root_ang_vel):
    env_ids = all_env_ids(env)
    env.dof_pos[:] = full_dof_pos
    env.dof_vel[:] = full_dof_vel
    env.root_states[:, 0:3] = root_pos
    env.root_states[:, 3:7] = root_rot
    env.root_states[:, 7:10] = root_vel
    env.root_states[:, 10:13] = root_ang_vel

    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_state),
        gymtorch.unwrap_tensor(env_ids),
        len(env_ids),
    )
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env_ids),
        len(env_ids),
    )
    env.episode_length_buf[:] = 0
    env.reset_buf[:] = 0
    env.last_actions[:] = 0.0
    env.last_dof_vel[:] = 0.0
    env.last_torques[:] = 0.0
    env.action_history_buf[:] = 0.0
    refresh_env_state(env)


def enforce_locked_state(
    env,
    frozen_dof_pos=None,
    frozen_dof_vel=None,
    frozen_dof_indices=None,
    locked_root_state=None,
):
    env_ids = all_env_ids(env)

    if frozen_dof_indices is not None and frozen_dof_indices.numel() > 0:
        env.dof_pos[:, frozen_dof_indices] = frozen_dof_pos[:, frozen_dof_indices]
        env.dof_vel[:, frozen_dof_indices] = frozen_dof_vel[:, frozen_dof_indices]
        env.gym.set_dof_state_tensor_indexed(
            env.sim,
            gymtorch.unwrap_tensor(env.dof_state),
            gymtorch.unwrap_tensor(env_ids),
            len(env_ids),
        )

    if locked_root_state is not None:
        env.root_states[:, 0:3] = locked_root_state["root_pos"]
        env.root_states[:, 3:7] = locked_root_state["root_rot"]
        env.root_states[:, 7:10] = locked_root_state["root_vel"]
        env.root_states[:, 10:13] = locked_root_state["root_ang_vel"]
        env.gym.set_actor_root_state_tensor_indexed(
            env.sim,
            gymtorch.unwrap_tensor(env.root_states),
            gymtorch.unwrap_tensor(env_ids),
            len(env_ids),
        )


def make_identity_root_rot(env):
    quat = torch.zeros((env.num_envs, 4), device=env.device, dtype=env.root_states.dtype)
    quat[:, 3] = 1.0
    return quat


def make_standing_state(env, airborne=False, lift_z=0.25):
    full_dof_pos = env.default_dof_pos_all.clone()
    full_dof_vel = torch.zeros_like(env.dof_vel)
    root_pos = env.root_states[:, 0:3].clone()
    root_pos[:, 0:2] = env.env_origins[:, 0:2]
    root_pos[:, 2] = float(env.base_init_state[2].item())
    if airborne:
        root_pos[:, 2] += lift_z
    root_rot = make_identity_root_rot(env)
    root_vel = torch.zeros_like(env.root_states[:, 7:10])
    root_ang_vel = torch.zeros_like(env.root_states[:, 10:13])
    return full_dof_pos, full_dof_vel, root_pos, root_rot, root_vel, root_ang_vel


def get_controlled_joint_names(env):
    if hasattr(env.cfg.asset, "active_dof_names"):
        return list(env.cfg.asset.active_dof_names)
    return list(env.dof_names)


def active_name_to_full_index(env):
    if hasattr(env.cfg.asset, "active_dof_names") and hasattr(env, "_active_dof_indices"):
        return {name: int(env._active_dof_indices[i].item()) for i, name in enumerate(env.cfg.asset.active_dof_names)}
    return {name: idx for idx, name in enumerate(env.dof_names)}


def resolve_single_joint_names(args):
    if args.joint_name:
        return [args.joint_name]
    if args.joint_group == "legs":
        return ["hip_pitch_l_joint", "knee_pitch_l_joint", "ankle_pitch_l_joint"]
    if args.joint_group == "upper":
        return ["waist_yaw_joint", "shoulder_pitch_l_joint"]
    return DEFAULT_SINGLE_JOINTS


def apply_joint_deltas(env, base_pose, deltas):
    pose = base_pose.clone()
    name_to_idx = active_name_to_full_index(env)
    for name, delta in deltas.items():
        pose[:, name_to_idx[name]] += delta
    return pose


def get_tracked_indices(env, tracked_joint_names):
    name_to_idx = active_name_to_full_index(env)
    return [name_to_idx[name] for name in tracked_joint_names]


def compute_tau(env, q_target_full, qd_target_full):
    pos_err = q_target_full - env.dof_pos
    vel_err = qd_target_full - env.dof_vel
    if env.cfg.domain_rand.randomize_motor:
        tau_cmd = env.motor_strength[0] * env.p_gains * pos_err + env.motor_strength[1] * env.d_gains * vel_err
    else:
        tau_cmd = env.p_gains * pos_err + env.d_gains * vel_err
    tau_applied = torch.clip(tau_cmd, -env.torque_limits, env.torque_limits)
    return tau_cmd, tau_applied


def simulate_control_step(
    env,
    q_target_full,
    qd_target_full,
    force_zero_torque=False,
    frozen_dof_pos=None,
    frozen_dof_vel=None,
    frozen_dof_indices=None,
    locked_root_state=None,
):
    last_tau_cmd = None
    last_tau_applied = None
    max_ratio = None
    for _ in range(env.cfg.control.decimation):
        tau_cmd, nominal_tau = compute_tau(env, q_target_full, qd_target_full)
        tau_applied = torch.zeros_like(nominal_tau) if force_zero_torque else nominal_tau
        ratio = torch.abs(tau_applied) / torch.clamp(env.torque_limits, min=1e-6)
        max_ratio = ratio if max_ratio is None else torch.maximum(max_ratio, ratio)
        last_tau_cmd = tau_cmd
        last_tau_applied = tau_applied
        env.torques[:] = tau_applied
        env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(env.torques))
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_actor_root_state_tensor(env.sim)
        enforce_locked_state(
            env,
            frozen_dof_pos=frozen_dof_pos,
            frozen_dof_vel=frozen_dof_vel,
            frozen_dof_indices=frozen_dof_indices,
            locked_root_state=locked_root_state,
        )

    env.episode_length_buf += 1
    refresh_env_state(env)
    return last_tau_cmd, last_tau_applied, max_ratio


def feet_metrics(env, contact_threshold):
    feet_forces = env.contact_forces[:, env.feet_indices, :]
    feet_contacts = feet_forces[:, :, 2] > contact_threshold
    feet_tangent_speed = torch.norm(env.rigid_body_states[:, env.feet_indices, 7:9], dim=-1)
    return feet_contacts, feet_tangent_speed

def init_time_series(tracked_joint_names):
    return {
        "tracked_joint_names": tracked_joint_names,
        "time_s": [],
        "q_target": [],
        "q": [],
        "qdot": [],
        "tau_cmd": [],
        "tau_applied": [],
        "torque_limit": [],
        "torque_ratio": [],
        "feet_contact": [],
        "feet_tangent_speed": [],
        "torque_released": [],
        "root_pos": [],
        "root_rpy": [],
    }


def record_step(
    time_series,
    env,
    tracked_indices,
    q_target_full,
    tau_cmd,
    tau_applied,
    torque_ratio,
    time_s,
    contact_threshold,
    torque_released,
):
    feet_contacts, feet_tangent_speed = feet_metrics(env, contact_threshold)
    env0 = 0
    roll, pitch, yaw = euler_from_quaternion(env.root_states[:, 3:7])

    time_series["time_s"].append(float(time_s))
    time_series["q_target"].append(q_target_full[env0, tracked_indices].detach().cpu().tolist())
    time_series["q"].append(env.dof_pos[env0, tracked_indices].detach().cpu().tolist())
    time_series["qdot"].append(env.dof_vel[env0, tracked_indices].detach().cpu().tolist())
    time_series["tau_cmd"].append(tau_cmd[env0, tracked_indices].detach().cpu().tolist())
    time_series["tau_applied"].append(tau_applied[env0, tracked_indices].detach().cpu().tolist())
    time_series["torque_limit"].append(env.torque_limits[tracked_indices].detach().cpu().tolist())
    time_series["torque_ratio"].append(torque_ratio[env0, tracked_indices].detach().cpu().tolist())
    time_series["feet_contact"].append(feet_contacts[env0].detach().cpu().tolist())
    time_series["feet_tangent_speed"].append(feet_tangent_speed[env0].detach().cpu().tolist())
    time_series["torque_released"].append(bool(torque_released))
    time_series["root_pos"].append(env.root_states[env0, 0:3].detach().cpu().tolist())
    time_series["root_rpy"].append([float(roll[env0].item()), float(pitch[env0].item()), float(yaw[env0].item())])


def compute_summary(
    env,
    tracked_joint_names,
    tracked_indices,
    q_target_hist,
    q_hist,
    qdot_hist,
    tau_applied_hist,
    torque_ratio_hist,
    root_pos_hist,
    root_rpy_hist,
    feet_contact_hist,
    feet_speed_hist,
    active_start_step,
    slip_threshold,
):
    q_target_tensor = torch.stack(q_target_hist)
    q_tensor = torch.stack(q_hist)
    qdot_tensor = torch.stack(qdot_hist)
    tau_applied_tensor = torch.stack(tau_applied_hist)
    torque_ratio_tensor = torch.stack(torque_ratio_hist)
    root_pos_tensor = torch.stack(root_pos_hist)
    root_rpy_tensor = torch.stack(root_rpy_hist)
    feet_contact_tensor = torch.stack(feet_contact_hist)
    feet_speed_tensor = torch.stack(feet_speed_hist)

    tracked_err = q_tensor - q_target_tensor
    active_err = tracked_err[active_start_step:]
    tail_len = max(1, active_err.shape[0] // 5)
    steady_err = active_err[-tail_len:]

    max_abs_err = float(active_err.abs().max().item())
    mean_abs_err = float(active_err.abs().mean().item())
    steady_state_err = float(steady_err.abs().mean().item())

    desired_delta = (q_target_tensor[-1] - q_target_tensor[active_start_step]).detach()
    actual_delta = q_tensor[active_start_step:] - q_target_tensor[active_start_step].unsqueeze(0)
    desired_mag = desired_delta.abs()
    desired_sign = torch.sign(desired_delta)
    projected_response = actual_delta * desired_sign.unsqueeze(0)
    overshoot = torch.clamp(projected_response.max(dim=0).values - desired_mag, min=0.0)

    saturation_ratio = float((torque_ratio_tensor[active_start_step:] > 0.98).float().mean().item())
    max_torque_ratio = float(torque_ratio_tensor[active_start_step:].max().item())
    max_abs_qdot = float(qdot_tensor[active_start_step:].abs().max().item())
    max_abs_tau = float(tau_applied_tensor[active_start_step:].abs().max().item())

    root_pos_start = root_pos_tensor[0]
    root_xy_drift = torch.norm(root_pos_tensor[:, :, 0:2] - root_pos_start[:, 0:2].unsqueeze(0), dim=-1)
    root_z_drift = torch.abs(root_pos_tensor[:, :, 2] - root_pos_start[:, 2].unsqueeze(0))
    max_root_xy_drift = float(root_xy_drift.max().item())
    max_root_z_drift = float(root_z_drift.max().item())
    max_root_roll = float(root_rpy_tensor[:, :, 0].abs().max().item())
    max_root_pitch = float(root_rpy_tensor[:, :, 1].abs().max().item())

    contact_speed = feet_speed_tensor * feet_contact_tensor.float()
    max_contact_tangent_speed = float(contact_speed.max().item())
    slipped = bool(max_contact_tangent_speed > slip_threshold)

    summary = {
        "tracked_joint_names": tracked_joint_names,
        "max_abs_err_rad": max_abs_err,
        "mean_abs_err_rad": mean_abs_err,
        "steady_state_err_rad": steady_state_err,
        "overshoot_rad": float(overshoot.max().item()),
        "max_abs_qdot_rad_s": max_abs_qdot,
        "max_abs_tau_nm": max_abs_tau,
        "saturation_ratio": saturation_ratio,
        "max_torque_ratio": max_torque_ratio,
        "max_root_xy_drift_m": max_root_xy_drift,
        "max_root_z_drift_m": max_root_z_drift,
        "max_root_roll_rad": max_root_roll,
        "max_root_pitch_rad": max_root_pitch,
        "max_contact_tangent_speed_m_s": max_contact_tangent_speed,
        "slipped": slipped,
        "terrain_static_friction": float(env.cfg.terrain.static_friction),
        "terrain_dynamic_friction": float(env.cfg.terrain.dynamic_friction),
        "shape_friction_mean": float(env.friction_coeffs_tensor.mean().item()),
        "motor_strength_mean": float(env.motor_strength.mean().item()),
    }
    return summary


def run_closed_loop_scenario(
    env,
    scenario_name,
    tracked_joint_names,
    q_target_schedule,
    qd_target_schedule,
    total_steps,
    active_start_step,
    contact_threshold,
    slip_threshold,
    frozen_dof_pos=None,
    frozen_dof_vel=None,
    frozen_dof_indices=None,
    locked_root_state=None,
    force_zero_torque_schedule=None,
):
    tracked_indices = get_tracked_indices(env, tracked_joint_names)
    time_series = init_time_series(tracked_joint_names)

    q_target_hist = []
    q_hist = []
    qdot_hist = []
    tau_applied_hist = []
    torque_ratio_hist = []
    root_pos_hist = []
    root_rpy_hist = []
    feet_contact_hist = []
    feet_speed_hist = []

    for step in range(total_steps):
        q_target_full = q_target_schedule(step)
        qd_target_full = qd_target_schedule(step)
        torque_released = force_zero_torque_schedule(step) if force_zero_torque_schedule is not None else False
        tau_cmd, tau_applied, torque_ratio = simulate_control_step(
            env,
            q_target_full,
            qd_target_full,
            force_zero_torque=torque_released,
            frozen_dof_pos=frozen_dof_pos,
            frozen_dof_vel=frozen_dof_vel,
            frozen_dof_indices=frozen_dof_indices,
            locked_root_state=locked_root_state,
        )

        q_target_hist.append(q_target_full[:, tracked_indices].detach().clone())
        q_hist.append(env.dof_pos[:, tracked_indices].detach().clone())
        qdot_hist.append(env.dof_vel[:, tracked_indices].detach().clone())
        tau_applied_hist.append(tau_applied[:, tracked_indices].detach().clone())
        torque_ratio_hist.append(torque_ratio[:, tracked_indices].detach().clone())
        root_pos_hist.append(env.root_states[:, 0:3].detach().clone())
        root_rpy_hist.append(torch.stack((env.roll.detach(), env.pitch.detach(), env.yaw.detach()), dim=-1))
        feet_contacts, feet_tangent_speed = feet_metrics(env, contact_threshold)
        feet_contact_hist.append(feet_contacts.detach().clone())
        feet_speed_hist.append(feet_tangent_speed.detach().clone())

        record_step(
            time_series,
            env,
            tracked_indices,
            q_target_full,
            tau_cmd,
            tau_applied,
            torque_ratio,
            time_s=step * env.dt,
            contact_threshold=contact_threshold,
            torque_released=torque_released,
        )
        if not env.headless:
            env.render()

    summary = compute_summary(
        env=env,
        tracked_joint_names=tracked_joint_names,
        tracked_indices=tracked_indices,
        q_target_hist=q_target_hist,
        q_hist=q_hist,
        qdot_hist=qdot_hist,
        tau_applied_hist=tau_applied_hist,
        torque_ratio_hist=torque_ratio_hist,
        root_pos_hist=root_pos_hist,
        root_rpy_hist=root_rpy_hist,
        feet_contact_hist=feet_contact_hist,
        feet_speed_hist=feet_speed_hist,
        active_start_step=active_start_step,
        slip_threshold=slip_threshold,
    )
    return {"scenario": scenario_name, "summary": summary, "time_series": time_series}


def run_single_joint_step(env, args):
    results = []
    joint_names = resolve_single_joint_names(args)
    amplitudes = parse_float_list(args.step_amplitudes)
    pre_steps = resolve_pre_steps(env, args, default_steps=20)

    for joint_name in joint_names:
        for amplitude in amplitudes:
            full_dof_pos, full_dof_vel, root_pos, root_rot, root_vel, root_ang_vel = make_standing_state(
                env, airborne=True, lift_z=args.lift_z
            )
            set_full_state(env, full_dof_pos, full_dof_vel, root_pos, root_rot, root_vel, root_ang_vel)
            base_target = full_dof_pos.clone()
            step_target = full_dof_pos.clone()
            joint_full_idx = active_name_to_full_index(env)[joint_name]
            step_target[:, joint_full_idx] += amplitude
            zero_vel = torch.zeros_like(full_dof_vel)
            all_dof_indices = torch.arange(env.num_dof, device=env.device, dtype=torch.long)
            frozen_mask = all_dof_indices != joint_full_idx
            frozen_dof_indices = all_dof_indices[frozen_mask]
            locked_root_state = {
                "root_pos": root_pos.clone(),
                "root_rot": root_rot.clone(),
                "root_vel": root_vel.clone(),
                "root_ang_vel": root_ang_vel.clone(),
            }

            result = run_closed_loop_scenario(
                env=env,
                scenario_name=f"single_joint_isolated_step:{joint_name}:{amplitude:+.3f}",
                tracked_joint_names=[joint_name],
                q_target_schedule=lambda step, bt=base_target, st=step_target, ps=pre_steps: bt if step < ps else st,
                qd_target_schedule=lambda step, zv=zero_vel: zv,
                total_steps=pre_steps + args.hold_steps,
                active_start_step=pre_steps,
                contact_threshold=args.contact_threshold,
                slip_threshold=args.slip_threshold,
                frozen_dof_pos=full_dof_pos,
                frozen_dof_vel=zero_vel,
                frozen_dof_indices=frozen_dof_indices,
                locked_root_state=locked_root_state,
            )
            results.append(result)
    return results


def run_pose_hold(env, args):
    full_dof_pos, full_dof_vel, root_pos, root_rot, root_vel, root_ang_vel = make_standing_state(env, airborne=False)
    set_full_state(env, full_dof_pos, full_dof_vel, root_pos, root_rot, root_vel, root_ang_vel)
    base_target = full_dof_pos.clone()
    zero_vel = torch.zeros_like(full_dof_vel)
    tracked_joint_names = get_controlled_joint_names(env)
    stand_steps = resolve_pre_steps(env, args, default_seconds=10.0)
    release_steps = resolve_hold_steps(env, args, default_seconds=10.0)
    return [
        run_closed_loop_scenario(
            env=env,
            scenario_name="pose_hold:stand_then_release",
            tracked_joint_names=tracked_joint_names,
            q_target_schedule=lambda step, bt=base_target: bt,
            qd_target_schedule=lambda step, zv=zero_vel: zv,
            total_steps=stand_steps + release_steps,
            active_start_step=stand_steps,
            contact_threshold=args.contact_threshold,
            slip_threshold=args.slip_threshold,
            force_zero_torque_schedule=lambda step, ss=stand_steps: step >= ss,
        )
    ]


def run_stance_contact(env, args):
    full_dof_pos, full_dof_vel, root_pos, root_rot, root_vel, root_ang_vel = make_standing_state(env, airborne=False)
    set_full_state(env, full_dof_pos, full_dof_vel, root_pos, root_rot, root_vel, root_ang_vel)
    base_target = full_dof_pos.clone()
    stance_target = apply_joint_deltas(env, base_target, DEFAULT_STANCE_DELTAS)
    zero_vel = torch.zeros_like(full_dof_vel)
    tracked_joint_names = sorted(DEFAULT_STANCE_DELTAS.keys())
    pre_steps = resolve_pre_steps(env, args, default_steps=20)
    hold_steps = resolve_hold_steps(env, args, default_steps=100)
    return [
        run_closed_loop_scenario(
            env=env,
            scenario_name="stance_contact:shift_hold",
            tracked_joint_names=tracked_joint_names,
            q_target_schedule=lambda step, bt=base_target, st=stance_target, ps=pre_steps: bt if step < ps else st,
            qd_target_schedule=lambda step, zv=zero_vel: zv,
            total_steps=pre_steps + hold_steps,
            active_start_step=pre_steps,
            contact_threshold=args.contact_threshold,
            slip_threshold=args.slip_threshold,
        )
    ]


def run_reference_tracking(env, args):
    env.reset()
    refresh_env_state(env)
    tracked_joint_names = get_controlled_joint_names(env)

    def q_target_schedule(step):
        env._update_ref_motion()
        return env._ref_dof_pos.clone()

    def qd_target_schedule(step):
        return env._ref_dof_vel.clone()

    result = run_closed_loop_scenario(
        env=env,
        scenario_name="reference_tracking:motion_pd",
        tracked_joint_names=tracked_joint_names,
        q_target_schedule=q_target_schedule,
        qd_target_schedule=qd_target_schedule,
        total_steps=args.reference_steps,
        active_start_step=0,
        contact_threshold=args.contact_threshold,
        slip_threshold=args.slip_threshold,
    )

    ref_root_err = torch.norm(env.root_states[:, 0:3] - env._ref_root_pos, dim=-1)
    root_rot_err = torch.norm(env.root_states[:, 3:7] - env._ref_root_rot, dim=-1)
    result["summary"]["final_root_pos_err_m"] = float(ref_root_err.mean().item())
    result["summary"]["final_root_rot_err_l2"] = float(root_rot_err.mean().item())
    return [result]


def run_mode(env, args):
    if args.mode == "single_joint_step":
        return run_single_joint_step(env, args)
    if args.mode == "pose_hold":
        return run_pose_hold(env, args)
    if args.mode == "stance_contact":
        return run_stance_contact(env, args)
    if args.mode == "reference_tracking":
        return run_reference_tracking(env, args)

    results = []
    results.extend(run_single_joint_step(env, args))
    results.extend(run_pose_hold(env, args))
    results.extend(run_stance_contact(env, args))
    results.extend(run_reference_tracking(env, args))
    return results


def run_phys_suite(args, phys_label):
    results = []

    if args.mode in NON_REFERENCE_MODES:
        env = build_env(args, phys_label, load_motion=False)
        try:
            results = run_mode(env, args)
        finally:
            destroy_env(env)
        return results

    if args.mode == "reference_tracking":
        env = build_env(args, phys_label, load_motion=True)
        try:
            results = run_reference_tracking(env, args)
        finally:
            destroy_env(env)
        return results

    env = build_env(args, phys_label, load_motion=False)
    try:
        results.extend(run_single_joint_step(env, args))
        results.extend(run_pose_hold(env, args))
        results.extend(run_stance_contact(env, args))
    finally:
        destroy_env(env)

    env = build_env(args, phys_label, load_motion=True)
    try:
        results.extend(run_reference_tracking(env, args))
    finally:
        destroy_env(env)

    return results


def summarize_to_terminal(phys_label, results):
    print("=" * 100)
    print(f"[{phys_label}] completed {len(results)} diagnostic scenario(s)")
    for result in results:
        summary = result["summary"]
        print(
            f"{result['scenario']}: "
            f"max_err={summary['max_abs_err_rad']:.4f} rad, "
            f"steady_err={summary['steady_state_err_rad']:.4f} rad, "
            f"overshoot={summary['overshoot_rad']:.4f} rad, "
            f"sat_ratio={summary['saturation_ratio']:.3f}, "
            f"max_tau_ratio={summary['max_torque_ratio']:.3f}, "
            f"slip={summary['slipped']}, "
            f"max_slip_speed={summary['max_contact_tangent_speed_m_s']:.4f} m/s, "
            f"root_xy_drift={summary['max_root_xy_drift_m']:.4f} m"
        )


def default_save_path(args):
    stamp = datetime.now().strftime("%m%d_%H%M%S")
    run_name = f"low_level_diag_{args.task}_{stamp}.json"
    return os.path.join(
        "/home/qsh/workspace_twist2/TWIST2/legged_gym/logs/diagnostics",
        run_name,
    )


def main():
    args = parse_args()
    phys_labels = ["baseline_phys", "current_phys"] if args.compare_phys else (["current_phys"] if args.use_current_phys else ["baseline_phys"])

    all_results = {
        "task": args.task,
        "device": args.device,
        "mode": args.mode,
        "pd_profile": args.pd_profile,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "motion_file": args.motion_file,
        "baseline_static_friction": args.static_friction,
        "baseline_dynamic_friction": args.dynamic_friction,
        "disable_rand": args.disable_rand,
        "motion_loading_policy": {
            "single_joint_step": "stub",
            "pose_hold": "stub",
            "stance_contact": "stub",
            "reference_tracking": "real",
        },
        "results_by_phys": {},
    }

    for phys_label in phys_labels:
        results = run_phys_suite(args, phys_label)
        summarize_to_terminal(phys_label, results)
        all_results["results_by_phys"][phys_label] = results

    save_path = args.save_json or default_save_path(args)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print("=" * 100)
    print(f"Structured results saved to {save_path}")


if __name__ == "__main__":
    main()
