#!/usr/bin/env python3

import argparse
import copy
from types import SimpleNamespace

import isaacgym  # noqa: F401 - required before gym imports
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_rotate_inverse

import torch

from legged_gym.envs import *  # noqa: F401,F403 - registers tasks
from legged_gym.gym_utils.task_registry import task_registry
from legged_gym.gym_utils.helpers import class_to_dict, parse_sim_params, set_seed
from legged_gym.envs.base.legged_robot import euler_from_quaternion


DEFAULT_MODEL_PATH = (
    "/home/qsh/workspace_twist2/TWIST2/legged_gym/logs/"
    "tienkung_stu_future_cc_stage1/tk_cc_stage_train10000_0407_112740/model_2000.pt"
)
DEFAULT_MOTION_FILE = (
    "/home/qsh/workspace_twist2/TWIST2/legged_gym/motion_data_configs/tienkung_ei_walk1.yaml"
)
DEFAULT_TASK = "tienkung_stu_future_cc_stage1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kinematic test for a trained TianKung future policy."
    )
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--motion-file", type=str, default=DEFAULT_MOTION_FILE)
    parser.add_argument("--motion-id", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument(
        "--sim-substeps",
        type=int,
        default=1,
        help="Zero-torque sim steps after each kinematic overwrite so rigid bodies refresh.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--print-every", type=int, default=100)
    return parser.parse_args()


def build_env(task_name, motion_file, sim_device, seed, headless):
    env_cfg, _ = task_registry.get_cfgs(task_name)
    env_cfg = copy.deepcopy(env_cfg)

    env_cfg.seed = seed
    env_cfg.env.num_envs = 1
    env_cfg.motion.motion_file = motion_file
    env_cfg.env.rand_reset = False
    env_cfg.env.randomize_start_pos = False
    env_cfg.env.randomize_start_yaw = False
    env_cfg.env.record_video = False
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
        headless=headless,
    )
    return env


def build_policy_runner(env, task_name, model_path, rl_device):
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
        eval_student=True,
        config_overrides={},
        rl_device=rl_device,
        resumeid=None,
        proj_name=None,
        num_envs=None,
        rows=None,
        cols=None,
        record_video=False,
        teleop_mode=False,
    )
    runner, _ = task_registry.make_alg_runner(
        log_root=None,
        env=env,
        name=task_name,
        args=runner_args,
        train_cfg=train_cfg,
    )
    runner.load(model_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    normalizer = None
    if env.cfg.env.normalize_obs:
        try:
            normalizer = runner.get_normalizer(device=env.device)
        except Exception:
            normalizer = None
    return policy, normalizer


def refresh_sim_tensors(env):
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_force_sensor_tensor(env.sim)

    env.base_quat[:] = env.root_states[:, 3:7]
    env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
    env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])
    env.projected_gravity[:] = quat_rotate_inverse(env.base_quat, env.gravity_vec)
    env.roll, env.pitch, env.yaw = euler_from_quaternion(env.base_quat)


def set_kinematic_state(env, full_dof_pos, full_dof_vel, root_pos, root_rot, root_vel, root_ang_vel):
    env.dof_pos[:] = full_dof_pos
    env.dof_vel[:] = full_dof_vel
    env.root_states[:, 0:3] = root_pos
    env.root_states[:, 3:7] = root_rot
    env.root_states[:, 7:10] = root_vel
    env.root_states[:, 10:13] = root_ang_vel

    env_ids_int32 = torch.arange(env.num_envs, device=env.device, dtype=torch.int32)
    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_state),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32),
    )
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32),
    )


def simulate_zero_torque(env, num_substeps):
    env.torques.zero_()
    env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(env.torques))
    for _ in range(num_substeps):
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
    refresh_sim_tensors(env)


def get_policy_actions(env, policy, normalizer):
    clip_obs = env.cfg.normalization.clip_observations
    obs = torch.clip(env.get_observations(), -clip_obs, clip_obs)
    if normalizer is not None:
        obs = normalizer.normalize(obs.detach())
    else:
        obs = obs.detach()
    return policy(obs, hist_encoding=True)


def actions_to_full_targets(env, actions):
    full_targets = env.default_dof_pos_all.clone()
    full_targets[:, env._active_dof_indices] = (
        env.default_dof_pos_all[:, env._active_dof_indices]
        + actions * env.cfg.control.action_scale
    )
    return full_targets


def action_snapshot(env, actions):
    active_names = env.cfg.asset.active_dof_names
    useful = [
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
    idx_map = {name: i for i, name in enumerate(active_names)}
    return {name: float(actions[0, idx_map[name]].item()) for name in useful if name in idx_map}


def main():
    args = parse_args()

    env = build_env(
        task_name=args.task,
        motion_file=args.motion_file,
        sim_device=args.device,
        seed=args.seed,
        headless=args.headless,
    )
    policy, normalizer = build_policy_runner(
        env=env,
        task_name=args.task,
        model_path=args.model_path,
        rl_device=args.device,
    )

    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    motion_ids = torch.full_like(env_ids, args.motion_id)
    env.reset_idx(env_ids, motion_ids=motion_ids)
    refresh_sim_tensors(env)
    env.compute_observations()
    env.reset_buf.zero_()

    last_full_targets = env._ref_dof_pos.clone()
    policy_dt = env.dt

    print(f"[kinematic-test] task={args.task}")
    print(f"[kinematic-test] model={args.model_path}")
    print(f"[kinematic-test] motion_file={args.motion_file}")
    print(f"[kinematic-test] motion_id={args.motion_id}")
    print(f"[kinematic-test] device={args.device}, headless={args.headless}")

    try:
        for step in range(args.num_steps):
            env._update_ref_motion()
            env.compute_observations()

            actions = get_policy_actions(env, policy, normalizer)
            env.action_history_buf = torch.cat(
                [env.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()],
                dim=1,
            )
            env.actions[:] = actions

            full_targets = actions_to_full_targets(env, actions)
            full_target_vel = (full_targets - last_full_targets) / max(policy_dt, 1e-6)
            last_full_targets = full_targets.clone()

            set_kinematic_state(
                env=env,
                full_dof_pos=full_targets,
                full_dof_vel=full_target_vel,
                root_pos=env._ref_root_pos,
                root_rot=env._ref_root_rot,
                root_vel=env._ref_root_vel,
                root_ang_vel=env._ref_root_ang_vel,
            )
            simulate_zero_torque(env, args.sim_substeps)
            env.render()

            env.last_actions[:] = env.actions[:]
            env.last_dof_vel[:] = env.dof_vel[:]
            env.last_root_vel[:] = env.root_states[:, 7:13]
            env.last_root_pos[:] = env.root_states[:, 0:3]
            env.last_root_rot[:] = env.root_states[:, 3:7]

            active_target = full_targets[:, env._active_dof_indices]
            active_ref = env._ref_dof_pos[:, env._active_dof_indices]
            active_rmse = torch.sqrt(torch.mean((active_target - active_ref) ** 2)).item()

            if step % args.print_every == 0:
                print(
                    f"[kinematic-test] step={step} "
                    f"motion_t={float(env._get_motion_times()[0].item()):.3f}s "
                    f"action_norm={float(torch.norm(actions[0]).item()):.3f} "
                    f"active_target_ref_rmse={active_rmse:.4f}"
                )
                print(f"[kinematic-test] action_sample={action_snapshot(env, actions)}")

            env.episode_length_buf += 1
            env.common_step_counter += 1
            env.global_counter += 1
            env.total_env_steps_counter += 1
    finally:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
