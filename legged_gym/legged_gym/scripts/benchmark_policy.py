# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.gym_utils import get_args, task_registry
import torch
import faulthandler
from tqdm import tqdm
from termcolor import cprint
import numpy as np
import time


def parse_motion_indices(indices_text, total_num_motions):
    selected = []
    for part in indices_text.split(","):
        part = part.strip()
        if len(part) == 0:
            continue
        idx = int(part)
        if idx < 0 or idx >= total_num_motions:
            raise ValueError(f"motion index {idx} 越界，合法范围为 [0, {total_num_motions - 1}]")
        selected.append(idx)
    return selected


def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="jit"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint

def set_play_cfg(env_cfg):
    env_cfg.env.num_envs = 1#2 if not args.num_envs else args.num_envs
    env_cfg.env.episode_length_s = 60
    # env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True
    
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 5
    env_cfg.domain_rand.max_push_vel_xy = 2.5
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.action_delay = False
    
    if hasattr(env_cfg, "motion"):
        env_cfg.motion.motion_curriculum = False


def play(args):
    faulthandler.enable()
    display_available = bool(os.environ.get("DISPLAY"))
    if not args.headless and not display_available:
        cprint("DISPLAY 未设置，自动切换到 headless 模式", "yellow")
        args.headless = True
    
    exptid = args.exptid
    log_pth = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.proj_name, args.exptid)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    set_play_cfg(env_cfg)

    env_cfg.env.record_video = args.record_video
    if_normalize = env_cfg.env.normalize_obs
    cprint(f"if_normalize: {if_normalize}", "green")
    if env_cfg.env.record_video:
        env_cfg.env.episode_length_s = 10

    if args.task.startswith("g1_"):
        env_cfg.motion.motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/g1_mocap_origin_test.yaml"
    motion_file_override = os.environ.get("BENCHMARK_MOTION_FILE", "").strip()
    if motion_file_override and hasattr(env_cfg, "motion"):
        env_cfg.motion.motion_file = motion_file_override
        cprint(f"using motion file override: {motion_file_override}", "yellow")
    env_cfg.env.rand_reset = False
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)

    if args.use_jit:
        path = os.path.join(log_pth, "traced")
        model, checkpoint = get_load_path(root=path, checkpoint=args.checkpoint)
        path = os.path.join(path, model)
        print("Loading jit for policy: ", path)
        policy_jit = torch.jit.load(path, map_location=env.device)
        print("policy_jit: ", policy_jit)
    else:
        policy = ppo_runner.get_inference_policy(device=env.device)
        if if_normalize:
            try:
                normalizer = ppo_runner.get_normalizer(device=env.device)
            except:
                print("No normalizer found")
                normalizer = None
        print("policy: ", policy)

    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, requires_grad=False)

    mp4_writers = []
    video_enabled = args.record_video
    if video_enabled:
        import imageio
        env.enable_viewer_sync = not args.headless
        camera_handles = getattr(env, "_rendering_camera_handles", [])
        if len(camera_handles) != env.num_envs or any([int(h) < 0 for h in camera_handles]):
            cprint("当前环境无法创建有效相机，已自动关闭视频录制", "yellow")
            video_enabled = False
        # env.enable_viewer_sync = False
        if video_enabled:
            for i in range(env.num_envs):
                video_name = args.proj_name + "-" + args.exptid +".mp4"
                run_name = os.path.basename(log_pth.rstrip("/\\"))
                path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "videos_retarget", run_name)
                os.makedirs(path, exist_ok=True)
                video_name = os.path.join(path, video_name)
                try:
                    mp4_writer = imageio.get_writer(video_name, fps=max(1, int(1/env.dt)), format="FFMPEG")
                except Exception as e:
                    raise RuntimeError("无法写入 mp4，请先安装视频后端：pip install \"imageio[ffmpeg]\"") from e
                cprint(f"Recording video to {video_name}", "green")
                mp4_writers.append(mp4_writer)

    if args.record_log:
        import json
        run_name = log_pth.split("/")[-1]
        logs_dict = []
        dict_name = args.proj_name + "-" + args.exptid + ".json"
        path = f"../../logs/env_logs/{run_name}"
        if not os.path.exists(path):
            os.makedirs(path)
        dict_name = os.path.join(path, dict_name)
        

    env_id = env.lookat_id

    total_num_motions = env._motion_lib.num_motions()
    max_motions_override = int(os.environ.get("BENCHMARK_MAX_MOTIONS", "0"))
    max_steps_override = int(os.environ.get("BENCHMARK_MAX_STEPS_PER_MOTION", "0"))
    progress_interval = int(os.environ.get("BENCHMARK_PROGRESS_INTERVAL", "500"))
    motion_indices_override = os.environ.get("BENCHMARK_MOTION_INDICES", "").strip()
    if motion_indices_override:
        eval_motion_ids = parse_motion_indices(motion_indices_override, total_num_motions)
    else:
        eval_motion_ids = list(range(total_num_motions))
    if max_motions_override > 0:
        eval_motion_ids = eval_motion_ids[:max_motions_override]
    num_motions = len(eval_motion_ids)
    cprint(f"num_motions: {num_motions}", "green")
    motion_names = env._motion_lib.get_motion_names()
    motion_names = [str(name) for name in motion_names]
    os.makedirs("benchmark_results", exist_ok=True)
    motion_list_path = os.path.join("benchmark_results", f"{args.proj_name}-{args.exptid}-motions.txt")
    with open(motion_list_path, "w") as f:
        for idx, motion_name in enumerate(motion_names):
            f.write(f"{idx}\t{motion_name}\n")
    cprint(f"motion list saved to {motion_list_path}", "green")
    per_motion_report_path = f"benchmark_results/{args.proj_name}-{args.exptid}-{args.checkpoint}-per_motion.csv"
    with open(per_motion_report_path, "w") as f:
        f.write("motion_idx,motion_name,steps,avg_error_tracking_joint_dof,avg_error_tracking_joint_vel,avg_error_tracking_root_translation,avg_error_tracking_root_rotation,avg_error_tracking_root_vel,avg_error_tracking_keybody_pos,avg_error_feet_slip,avg_error_tracking_root_ang_vel\n")
    per_motion_rows = []
    
    # env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    error_tracking_joint_dof = []
    error_tracking_joint_vel = []
    error_tracking_root_translation = []
    error_tracking_root_rotation = []
    error_tracking_root_vel = []
    error_tracking_root_ang_vel = []
    error_tracking_keybody_pos = []
    error_feet_slip = []
    
    # error tracking keybody pos
    error_tracking_keybody_pos_hand = []
    error_tracking_keybody_pos_feet = []
    error_tracking_keybody_pos_knee = []
    error_tracking_keybody_pos_elbow = []
    error_tracking_keybody_pos_head = []
    
    for eval_i, motion_idx in enumerate(tqdm(eval_motion_ids)):
        motion_id = torch.tensor([motion_idx], device=env.device, dtype=torch.long)
        motion_name = motion_names[motion_idx] if motion_idx < len(motion_names) else f"motion_{motion_idx}"
        motion_time = torch.zeros((1,), device=env.device, dtype=torch.float)
        motion_length = env._motion_lib.get_motion_length(motion_id)
        # traj_length = int(env.max_episode_length) // 10
        # traj_length = int(env.max_episode_length) 
        traj_length = int(motion_length / env.dt) * 1
        if max_steps_override > 0:
            traj_length = min(traj_length, max_steps_override)
        cprint(f"evaluating motion {eval_i + 1}/{num_motions}: {motion_name} (steps={traj_length})", "cyan")
        reset_env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
        env.reset_idx(env_ids=reset_env_ids, motion_ids=motion_id)
        motion_start_time = time.time()
        motion_error_tracking_joint_dof = []
        motion_error_tracking_joint_vel = []
        motion_error_tracking_root_translation = []
        motion_error_tracking_root_rotation = []
        motion_error_tracking_root_vel = []
        motion_error_tracking_root_ang_vel = []
        motion_error_tracking_keybody_pos = []
        motion_error_feet_slip = []
        
        
        for t in range(traj_length):
            if t == 0 or (t + 1) % progress_interval == 0 or t == traj_length - 1:
                elapsed = max(time.time() - motion_start_time, 1e-6)
                fps = (t + 1) / elapsed
                print(f"motion {eval_i + 1}/{num_motions} step {t + 1}/{traj_length}")
                print(f"motion {eval_i + 1}/{num_motions} elapsed {elapsed:.1f}s fps {fps:.1f}")
            
            obs = env.get_observations()
        
            if args.use_jit:
                actions = policy_jit(obs.detach())
            else:
                if if_normalize and normalizer is not None:
                    normalized_obs = normalizer.normalize(obs.detach())
                else:
                    normalized_obs = obs.detach()
                actions = policy(normalized_obs, hist_encoding=True)
                
            if t == 0:
                print("step1 env.step start")
            obs, _, rews, dones, infos = env.step(actions.detach())
            if t == 0:
                print("step1 env.step done")
            
            error_tracking_joint_dof.append(env._error_tracking_joint_dof().item())
            error_tracking_joint_vel.append(env._error_tracking_joint_vel().item())
            error_tracking_root_translation.append(env._error_tracking_root_translation().item())
            error_tracking_root_rotation.append(env._error_tracking_root_rotation().item())
            error_tracking_root_vel.append(env._error_tracking_root_vel().item())
            error_tracking_root_ang_vel.append(env._error_tracking_root_ang_vel().item())
            motion_error_tracking_joint_dof.append(error_tracking_joint_dof[-1])
            motion_error_tracking_joint_vel.append(error_tracking_joint_vel[-1])
            motion_error_tracking_root_translation.append(error_tracking_root_translation[-1])
            motion_error_tracking_root_rotation.append(error_tracking_root_rotation[-1])
            motion_error_tracking_root_vel.append(error_tracking_root_vel[-1])
            motion_error_tracking_root_ang_vel.append(error_tracking_root_ang_vel[-1])
            keybody_error_output = env._error_tracking_keybody_pos()
            if isinstance(keybody_error_output, (tuple, list)) and len(keybody_error_output) == 2:
                error_tracking_keybody_pos_single, error_tracking_keybody_pos_diff = keybody_error_output
            else:
                error_tracking_keybody_pos_single = keybody_error_output
                error_tracking_keybody_pos_diff = None
            
            if error_tracking_keybody_pos_diff is not None and error_tracking_keybody_pos_diff.ndim >= 2 and error_tracking_keybody_pos_diff.shape[1] >= 9:
                keybody_hand_err = (error_tracking_keybody_pos_diff[0,0]+error_tracking_keybody_pos_diff[0,1])/2
                error_tracking_keybody_pos_hand.append(keybody_hand_err.item())
                keybody_feet_err = (error_tracking_keybody_pos_diff[0,2]+error_tracking_keybody_pos_diff[0,3])/2
                error_tracking_keybody_pos_feet.append(keybody_feet_err.item())
                keybody_knee_err = (error_tracking_keybody_pos_diff[0,4]+error_tracking_keybody_pos_diff[0,5])/2
                error_tracking_keybody_pos_knee.append(keybody_knee_err.item())
                keybody_elbow_err = (error_tracking_keybody_pos_diff[0,6]+error_tracking_keybody_pos_diff[0,7])/2
                error_tracking_keybody_pos_elbow.append(keybody_elbow_err.item())
                keybody_head_err = error_tracking_keybody_pos_diff[0,8]
                error_tracking_keybody_pos_head.append(keybody_head_err.item())
            else:
                keybody_err_scalar = error_tracking_keybody_pos_single.item()
                error_tracking_keybody_pos_hand.append(keybody_err_scalar)
                error_tracking_keybody_pos_feet.append(keybody_err_scalar)
                error_tracking_keybody_pos_knee.append(keybody_err_scalar)
                error_tracking_keybody_pos_elbow.append(keybody_err_scalar)
                error_tracking_keybody_pos_head.append(keybody_err_scalar)
            
            error_tracking_keybody_pos.append(error_tracking_keybody_pos_single.item())
            error_feet_slip.append(env._error_feet_slip().item())
            motion_error_tracking_keybody_pos.append(error_tracking_keybody_pos[-1])
            motion_error_feet_slip.append(error_feet_slip[-1])
            
            if video_enabled:
                imgs = env.render_record(mode='rgb_array')
                if imgs is not None and len(imgs) == env.num_envs:
                    for i in range(env.num_envs):
                        if imgs[i] is None or imgs[i].size == 0:
                            continue
                        mp4_writers[i].append_data(imgs[i])
                        
            if args.record_log:
                log_dict = env.get_episode_log()
                logs_dict.append(log_dict)
            
            # Interaction
            if env.button_pressed:
                print(f"env_id: {env.lookat_id:<{5}}")
        per_motion_row = {
            "motion_idx": motion_idx,
            "motion_name": motion_name,
            "steps": traj_length,
            "avg_error_tracking_joint_dof": np.mean(motion_error_tracking_joint_dof),
            "avg_error_tracking_joint_vel": np.mean(motion_error_tracking_joint_vel),
            "avg_error_tracking_root_translation": np.mean(motion_error_tracking_root_translation),
            "avg_error_tracking_root_rotation": np.mean(motion_error_tracking_root_rotation),
            "avg_error_tracking_root_vel": np.mean(motion_error_tracking_root_vel),
            "avg_error_tracking_keybody_pos": np.mean(motion_error_tracking_keybody_pos),
            "avg_error_feet_slip": np.mean(motion_error_feet_slip),
            "avg_error_tracking_root_ang_vel": np.mean(motion_error_tracking_root_ang_vel)
        }
        per_motion_rows.append(per_motion_row)
        with open(per_motion_report_path, "a") as f:
            f.write(
                f"{per_motion_row['motion_idx']},\"{per_motion_row['motion_name']}\",{per_motion_row['steps']},{per_motion_row['avg_error_tracking_joint_dof']:.6f},{per_motion_row['avg_error_tracking_joint_vel']:.6f},{per_motion_row['avg_error_tracking_root_translation']:.6f},{per_motion_row['avg_error_tracking_root_rotation']:.6f},{per_motion_row['avg_error_tracking_root_vel']:.6f},{per_motion_row['avg_error_tracking_keybody_pos']:.6f},{per_motion_row['avg_error_feet_slip']:.6f},{per_motion_row['avg_error_tracking_root_ang_vel']:.6f}\n"
            )
    
    total_error = np.mean(error_tracking_joint_dof) + np.mean(error_tracking_joint_vel) + np.mean(error_tracking_root_translation) + np.mean(error_tracking_root_rotation) + np.mean(error_tracking_root_vel) + np.mean(error_tracking_keybody_pos) + np.mean(error_feet_slip)
    total_error += np.mean(error_tracking_root_ang_vel)
    
    # print avg error
    cprint(f"Policy: {args.exptid}", "green")
    cprint(f"avg error_tracking_joint_dof: {np.mean(error_tracking_joint_dof):.4f}", "green")
    cprint(f"avg error_tracking_joint_vel: {np.mean(error_tracking_joint_vel):.4f}", "green")
    cprint(f"avg error_tracking_root_translation: {np.mean(error_tracking_root_translation):.4f}", "green")
    cprint(f"avg error_tracking_root_rotation: {np.mean(error_tracking_root_rotation):.4f}", "green")
    cprint(f"avg error_tracking_root_vel: {np.mean(error_tracking_root_vel):.4f}", "green")
    cprint(f"avg error_tracking_keybody_pos: {np.mean(error_tracking_keybody_pos):.4f}", "green")
    cprint(f"avg error_feet_slip: {np.mean(error_feet_slip):.4f}", "green")
    cprint(f"avg error_tracking_root_ang_vel: {np.mean(error_tracking_root_ang_vel):.4f}", "green")
    
    cprint(f"avg error_tracking_keybody_pos_hand: {np.mean(error_tracking_keybody_pos_hand):.4f}", "green")
    cprint(f"avg error_tracking_keybody_pos_feet: {np.mean(error_tracking_keybody_pos_feet):.4f}", "green")
    cprint(f"avg error_tracking_keybody_pos_knee: {np.mean(error_tracking_keybody_pos_knee):.4f}", "green")
    cprint(f"avg error_tracking_keybody_pos_elbow: {np.mean(error_tracking_keybody_pos_elbow):.4f}", "green")
    cprint(f"avg error_tracking_keybody_pos_head: {np.mean(error_tracking_keybody_pos_head):.4f}", "green")
    
    cprint(f"total_error: {total_error:.4f}", "green")
    
    # output as a txt file
    os.makedirs("benchmark_results", exist_ok=True)
    with open(f"benchmark_results/{args.proj_name}-{args.exptid}-{args.checkpoint}.txt", "w") as f:
        f.write(f"total_error: {total_error:.4f}\n")
        f.write(f"avg error_tracking_joint_dof: {np.mean(error_tracking_joint_dof):.4f}\n")
        f.write(f"avg error_tracking_joint_vel: {np.mean(error_tracking_joint_vel):.4f}\n")
        f.write(f"avg error_tracking_root_translation: {np.mean(error_tracking_root_translation):.4f}\n")
        f.write(f"avg error_tracking_root_rotation: {np.mean(error_tracking_root_rotation):.4f}\n")
        f.write(f"avg error_tracking_root_vel: {np.mean(error_tracking_root_vel):.4f}\n")
        f.write(f"avg error_tracking_keybody_pos: {np.mean(error_tracking_keybody_pos):.4f}\n")
        f.write(f"avg error_feet_slip: {np.mean(error_feet_slip):.4f}\n")
        f.write(f"avg error_tracking_root_ang_vel: {np.mean(error_tracking_root_ang_vel):.4f}\n")
        f.write(f"avg error_tracking_keybody_pos_hand: {np.mean(error_tracking_keybody_pos_hand):.4f}\n")
        f.write(f"avg error_tracking_keybody_pos_feet: {np.mean(error_tracking_keybody_pos_feet):.4f}\n")
        f.write(f"avg error_tracking_keybody_pos_knee: {np.mean(error_tracking_keybody_pos_knee):.4f}\n")
        f.write(f"avg error_tracking_keybody_pos_elbow: {np.mean(error_tracking_keybody_pos_elbow):.4f}\n")
        f.write(f"avg error_tracking_keybody_pos_head: {np.mean(error_tracking_keybody_pos_head):.4f}\n")
        print(f"output to benchmark_results/{args.proj_name}-{args.exptid}-{args.checkpoint}.txt")
    print(f"output to {per_motion_report_path}")
        
    if video_enabled:
        for mp4_writer in mp4_writers:
            mp4_writer.close()
            
    if args.record_log:
        with open(dict_name, 'w') as f:
            json.dump(logs_dict, f)
    

if __name__ == '__main__':
    args = get_args()
    play(args)
