from isaacgym.torch_utils import *

import os
import pickle

import torch
import yaml

from legged_gym.envs.base.humanoid_mimic import HumanoidMimic
from .tienkung_mimic_config import TienkungMimicCfg
from legged_gym.gym_utils.math import *
from pose.utils import torch_utils
from pose.utils.motion_lib_pkl import MotionLib
from legged_gym.envs.base.legged_robot import euler_from_quaternion
from legged_gym.envs.base.humanoid_char import convert_to_local_root_body_pos, convert_to_global_root_body_pos

'''
① 严格校验 motion 文件
通过 _validate_motion_file() 在加载前检查 pkl/yaml 中每条 motion 的字段和形状是否完整，坏数据直接报错。
② 建立 active dof 机制
通过 _ensure_active_dof_indices()、_expand_motion_dofs_to_full()、_compute_torques() 把“策略只控制部分关节”这件事和“仿真机器人有完整关节集合”衔接起来。
③ 从 MotionLib 取参考动作
通过 _reset_ref_motion() 和 _update_ref_motion() 在 reset 时和每个仿真步更新参考 root / dof / body 状态。
④ 构造模仿观测
通过 _get_mimic_obs() 把未来若干参考帧的信息拼起来，再在 compute_observations() 中和当前机器人 proprio、上一时刻动作、历史观测、privileged latent 组合成最终 obs。
⑤ 定义针对天工的额外奖励
主要额外约束了腰部和踝部的速度、加速度以及踝部动作幅度，让动作更稳、更平滑。
'''

class TienkungMimic(HumanoidMimic):
    def __init__(self, cfg: TienkungMimicCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._ensure_active_dof_indices()
        self.last_feet_z = 0.05
        self.episode_length = torch.zeros((self.num_envs), device=self.device)
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))

    def _resolve_motion_files(self, motion_file): #解析 yaml 或单个 pkl，返回实际要读的 motion 文件列表
        if motion_file.endswith(".yaml"):
            with open(motion_file, "r") as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)

            if not isinstance(motion_config, dict):
                raise ValueError(f"Motion yaml must contain a dict config: {motion_file}")

            motion_root_path = motion_config.get("root_path", None)
            motion_list = motion_config.get("motions", None)
            if not motion_root_path or not isinstance(motion_list, list) or len(motion_list) == 0:
                raise ValueError(f"Motion yaml must define non-empty root_path and motions: {motion_file}")

            resolved_files = []
            for motion_entry in motion_list:
                if not isinstance(motion_entry, dict) or "file" not in motion_entry:
                    raise ValueError(f"Invalid motion entry in yaml {motion_file}: {motion_entry}")
                resolved_files.append(os.path.join(motion_root_path, motion_entry["file"]))
            return resolved_files

        return [motion_file]

    def _validate_motion_file(self, motion_file): #严格校验 motion 字段，防止坏数据被静默兜底
        required_fields = ("fps", "root_pos", "root_rot", "dof_pos", "local_body_pos", "link_body_list")
        expected_active_dofs = len(self.cfg.asset.active_dof_names)
        expected_full_dofs = self.num_dof
        required_key_bodies = set(self.cfg.motion.key_bodies)

        for curr_file in self._resolve_motion_files(motion_file):
            if not os.path.exists(curr_file):
                raise FileNotFoundError(f"Motion file does not exist: {curr_file}")

            try:
                with open(curr_file, "rb") as f:
                    motion_data = pickle.load(f)
            except Exception as exc:
                raise ValueError(f"Failed to load motion pickle {curr_file}: {exc}") from exc

            if not isinstance(motion_data, dict):
                raise ValueError(f"Motion pickle must contain a dict: {curr_file}")

            missing_fields = [field for field in required_fields if field not in motion_data]
            if missing_fields:
                raise ValueError(f"Motion pickle missing required fields {missing_fields}: {curr_file}")

            fps = motion_data["fps"]
            root_pos = torch.as_tensor(motion_data["root_pos"])
            root_rot = torch.as_tensor(motion_data["root_rot"])
            dof_pos = torch.as_tensor(motion_data["dof_pos"])
            local_body_pos = torch.as_tensor(motion_data["local_body_pos"])
            link_body_list = list(motion_data["link_body_list"])

            if fps <= 0:
                raise ValueError(f"Motion fps must be positive: {curr_file}")
            if len(link_body_list) == 0:
                raise ValueError(f"Motion link_body_list cannot be empty: {curr_file}")
            if root_pos.ndim != 2 or root_pos.shape[1] != 3 or root_pos.shape[0] == 0:
                raise ValueError(f"Motion root_pos must have shape [T, 3]: {curr_file}")
            if root_rot.ndim != 2 or root_rot.shape != (root_pos.shape[0], 4):
                raise ValueError(f"Motion root_rot must have shape [T, 4] and align with root_pos: {curr_file}")
            if dof_pos.ndim != 2 or dof_pos.shape[0] != root_pos.shape[0]:
                raise ValueError(f"Motion dof_pos must have shape [T, D] and align with root_pos: {curr_file}")
            if dof_pos.shape[1] not in (expected_active_dofs, expected_full_dofs):
                raise ValueError(
                    f"Motion dof_pos width must be {expected_active_dofs} (active) or {expected_full_dofs} (full), "
                    f"got {dof_pos.shape[1]}: {curr_file}"
                )
            if local_body_pos.ndim != 3 or local_body_pos.shape[0] != root_pos.shape[0] or local_body_pos.shape[2] != 3:
                raise ValueError(f"Motion local_body_pos must have shape [T, B, 3] and align with root_pos: {curr_file}")
            if local_body_pos.shape[1] != len(link_body_list):
                raise ValueError(
                    f"Motion local_body_pos body count {local_body_pos.shape[1]} does not match link_body_list "
                    f"length {len(link_body_list)}: {curr_file}"
                )

            missing_key_bodies = sorted(required_key_bodies - set(link_body_list))
            if missing_key_bodies:
                raise ValueError(
                    f"Motion link_body_list is missing required key bodies {missing_key_bodies}: {curr_file}"
                )

    def _load_motions(self): #加载通过严格校验的 motion 文件
        motion_file = self.cfg.motion.motion_file
        self._validate_motion_file(motion_file)

        self._motion_lib = MotionLib(
            motion_file=motion_file,
            device=self.device,
            sample_ratio=self.cfg.motion.sample_ratio,
            motion_decompose=self.cfg.motion.motion_decompose,
            motion_smooth=self.cfg.motion.motion_smooth,
        )
        return
        
    def _ensure_active_dof_indices(self): #提取配置中指定的关节索引（因为灵巧手不需要重定向）
        if hasattr(self, "_active_dof_indices"):
            return

        dof_name_to_idx = {name: i for i, name in enumerate(self.dof_names)}
        missing = [name for name in self.cfg.asset.active_dof_names if name not in dof_name_to_idx]
        if missing:
            raise ValueError(f"Active DoF names missing in asset dof_names: {missing}")

        active_idx = [dof_name_to_idx[name] for name in self.cfg.asset.active_dof_names]
        self._active_dof_indices = torch.tensor(active_idx, dtype=torch.long, device=self.device)
        self._num_active_dofs = len(active_idx)
        active_mask = torch.zeros(self.num_dof, dtype=torch.bool, device=self.device)
        active_mask[self._active_dof_indices] = True
        self._inactive_dof_indices = torch.arange(self.num_dof, device=self.device, dtype=torch.long)[~active_mask]

    def _set_inactive_dofs_to_default(self, full_dof_pos, full_dof_vel, env_ids=None):
        self._ensure_active_dof_indices()
        if self._inactive_dof_indices.numel() == 0:
            return full_dof_pos, full_dof_vel

        if env_ids is None:
            default_dof_pos = self.default_dof_pos_all
        else:
            default_dof_pos = self.default_dof_pos_all[env_ids]

        full_dof_pos[:, self._inactive_dof_indices] = default_dof_pos[:, self._inactive_dof_indices]
        full_dof_vel[:, self._inactive_dof_indices] = 0.0
        return full_dof_pos, full_dof_vel

    def _expand_motion_dofs_to_full(self, dof_pos, dof_vel, env_ids=None): #补齐从motion中提取出来的数据，如果是全的就全部更新，如果只有激活关节，就将其余关节设置为仿真环境中的值，这里重定向后的应该是全的
        self._ensure_active_dof_indices()
        if dof_pos.shape[-1] == self.num_dof:
            full_dof_pos = dof_pos.clone()
            full_dof_vel = dof_vel.clone()
        elif dof_pos.shape[-1] == self._num_active_dofs:
            if env_ids is None:
                full_dof_pos = self.default_dof_pos_all.clone()
                full_dof_vel = torch.zeros_like(self.dof_vel)
            else:
                full_dof_pos = self.default_dof_pos_all[env_ids].clone()
                full_dof_vel = torch.zeros_like(self.dof_vel[env_ids])

            full_dof_pos[:, self._active_dof_indices] = dof_pos
            full_dof_vel[:, self._active_dof_indices] = dof_vel
        else:
            raise ValueError(
                f"Unexpected motion dof dim {dof_pos.shape[-1]}, "
                f"expected {self._num_active_dofs} or {self.num_dof}."
            )
        return self._set_inactive_dofs_to_default(full_dof_pos, full_dof_vel, env_ids=env_ids)

    def _assign_ref_body_pos_from_motion(self, ref_body_pos, root_pos, root_rot, body_pos): #将motion中对应body的数据加载到仿真环境中对应的body
        if not hasattr(self, "_motion_to_env_body"):
            env_body_name_to_idx = {name: i for i, name in enumerate(self.body_names)}
            motion_to_env = []
            for motion_idx, name in enumerate(self._motion_lib._body_link_list):
                if name in env_body_name_to_idx:
                    motion_to_env.append((motion_idx, env_body_name_to_idx[name]))
            self._motion_to_env_body = motion_to_env

        ref_body_pos[:] = 0.0
        body_pos_global = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=body_pos)
        for motion_idx, env_idx in self._motion_to_env_body:
            ref_body_pos[:, env_idx, :] = body_pos_global[:, motion_idx, :]

    def _reset_ref_motion(self, env_ids, motion_ids=None): #重置参考动作，episode 开始时，选一段新动作并设定起点（采样方法？）
        self._ensure_active_dof_indices()

        n = len(env_ids)
        if motion_ids is None:
            if (hasattr(self.cfg.motion, 'use_error_aware_sampling') and
                    self.cfg.motion.use_error_aware_sampling):
                motion_ids = self._motion_lib.sample_motions(
                    n,
                    motion_difficulty=self.motion_difficulty,
                    max_key_body_error=self.max_key_body_error,
                    use_error_aware_sampling=True,
                    error_sampling_power=self.cfg.motion.error_sampling_power,
                    error_sampling_threshold=self.cfg.motion.error_sampling_threshold,
                )
            else:
                motion_ids = self._motion_lib.sample_motions(n, motion_difficulty=self.motion_difficulty)

        if self._rand_reset:
            motion_times = self._motion_lib.sample_time(motion_ids)
        else:
            motion_times = torch.zeros(motion_ids.shape, device=self.device, dtype=torch.float)

        self._motion_ids[env_ids] = motion_ids
        self._motion_time_offsets[env_ids] = motion_times

        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos, _, _ = \
            self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        root_pos[:, 2] += self.cfg.motion.height_offset

        self._ref_root_pos[env_ids] = root_pos
        self._ref_root_rot[env_ids] = root_rot
        self._ref_root_vel[env_ids] = root_vel
        self._ref_root_ang_vel[env_ids] = root_ang_vel

        full_dof_pos, full_dof_vel = self._expand_motion_dofs_to_full(dof_pos, dof_vel, env_ids=env_ids)
        self._ref_dof_pos[env_ids] = full_dof_pos
        self._ref_dof_vel[env_ids] = full_dof_vel

        ref_body_pos = self._ref_body_pos[env_ids].clone()
        self._assign_ref_body_pos_from_motion(ref_body_pos, root_pos, root_rot, body_pos)
        self._ref_body_pos[env_ids] = ref_body_pos

    def _update_ref_motion(self): #更新参考动作，episode 运行过程中，沿着这段动作持续往后推进参考状态
        self._ensure_active_dof_indices()

        motion_ids = self._motion_ids
        motion_times = self._get_motion_times()
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos, _, _ = \
            self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        root_pos[:, 2] += self.cfg.motion.height_offset
        root_pos[:, :2] += self.episode_init_origin[:, :2]

        self._ref_root_pos[:] = root_pos
        self._ref_root_rot[:] = root_rot
        self._ref_root_vel[:] = root_vel
        self._ref_root_ang_vel[:] = root_ang_vel

        full_dof_pos, full_dof_vel = self._expand_motion_dofs_to_full(dof_pos, dof_vel)
        self._ref_dof_pos[:] = full_dof_pos
        self._ref_dof_vel[:] = full_dof_vel

        ref_body_pos = self._ref_body_pos.clone()
        self._assign_ref_body_pos_from_motion(ref_body_pos, root_pos, root_rot, body_pos)
        self._ref_body_pos[:] = ref_body_pos

    def _compute_torques(self, actions): #根据策略输出的动作计算力矩
        self._ensure_active_dof_indices()
        full_actions = torch.zeros(actions.shape[0], self.num_dof, dtype=actions.dtype, device=actions.device)
        full_actions[:, self._active_dof_indices] = actions
        return super()._compute_torques(full_actions)

    def _get_noise_scale_vec(self, cfg): #参考观测不加噪声，本体观测添加噪声
        noise_scale_vec = torch.zeros(1, self.cfg.env.n_proprio, device=self.device)
        if not self.cfg.noise.add_noise:
            return noise_scale_vec

        ang_vel_dim = 3
        imu_dim = 2
        noise_start_dim = self.cfg.env.n_mimic_obs * len(self._tar_motion_steps_priv)
        noise_scale_vec[:, noise_start_dim:noise_start_dim + ang_vel_dim] = self.cfg.noise.noise_scales.ang_vel
        noise_scale_vec[:, noise_start_dim + ang_vel_dim:noise_start_dim + ang_vel_dim + imu_dim] = self.cfg.noise.noise_scales.imu
        noise_scale_vec[:, noise_start_dim + (ang_vel_dim + imu_dim):noise_start_dim + (ang_vel_dim + imu_dim) + self.cfg.env.num_actions] = self.cfg.noise.noise_scales.dof_pos
        noise_scale_vec[:, noise_start_dim + (ang_vel_dim + imu_dim) + self.cfg.env.num_actions:noise_start_dim + (ang_vel_dim + imu_dim) + 2 * self.cfg.env.num_actions] = self.cfg.noise.noise_scales.dof_vel
        return noise_scale_vec

    def _get_body_indices(self): #获得body的索引，后续施加力矩可以使用Isaac Gym 自己的 DOF 驱动接口进行
        upper_arm_names = [s for s in self.body_names if self.cfg.asset.upper_arm_name in s]
        lower_arm_names = [s for s in self.body_names if self.cfg.asset.lower_arm_name in s]
        torso_name = [s for s in self.body_names if self.cfg.asset.torso_name in s]
        self.torso_indices = torch.zeros(len(torso_name), dtype=torch.long, device=self.device,
                                                 requires_grad=False)
        for j in range(len(torso_name)):
            self.torso_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                                  torso_name[j])
        self.upper_arm_indices = torch.zeros(len(upper_arm_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for j in range(len(upper_arm_names)):
            self.upper_arm_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                                upper_arm_names[j])
        self.lower_arm_indices = torch.zeros(len(lower_arm_names), dtype=torch.long, device=self.device,
                                                requires_grad=False)
        for j in range(len(lower_arm_names)):
            self.lower_arm_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                                lower_arm_names[j])
        knee_names = [s for s in self.body_names if self.cfg.asset.shank_name in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])

    def _get_mimic_obs(self): #获得未来几个时间步的参考序列的观测
        num_steps = self._tar_motion_steps_priv.shape[0]
        assert num_steps > 0, "Invalid number of target observation steps"
        motion_times = self._get_motion_times().unsqueeze(-1)
        obs_motion_times = self._tar_motion_steps_priv * self.dt + motion_times
        motion_ids_tiled = torch.broadcast_to(self._motion_ids.unsqueeze(-1), obs_motion_times.shape)
        motion_ids_tiled = motion_ids_tiled.flatten()
        obs_motion_times = obs_motion_times.flatten()
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos, root_pos_delta_local, root_rot_delta_local = self._motion_lib.calc_motion_frame(motion_ids_tiled, obs_motion_times)
        
        roll, pitch, yaw = euler_from_quaternion(root_rot)
        roll = roll.reshape(self.num_envs, num_steps, 1)
        pitch = pitch.reshape(self.num_envs, num_steps, 1)
        yaw = yaw.reshape(self.num_envs, num_steps, 1)
        if not self.global_obs:
            root_vel = quat_rotate_inverse(root_rot, root_vel)
            root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)
      
        whole_key_body_pos = body_pos[:, self._key_body_ids_motion, :]
        if self.global_obs:
            whole_key_body_pos = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=whole_key_body_pos)
        whole_key_body_pos = whole_key_body_pos.reshape(self.num_envs, num_steps, -1)
        
        root_pos = root_pos.reshape(self.num_envs, num_steps, root_pos.shape[-1])
        root_vel = root_vel.reshape(self.num_envs, num_steps, root_vel.shape[-1])
        root_rot = root_rot.reshape(self.num_envs, num_steps, root_rot.shape[-1])
        root_ang_vel = root_ang_vel.reshape(self.num_envs, num_steps, root_ang_vel.shape[-1])
        if dof_pos.shape[-1] == self.num_dof:
            dof_pos = dof_pos[:, self._active_dof_indices]
        dof_pos = dof_pos.reshape(self.num_envs, num_steps, dof_pos.shape[-1])
        
        # align mocap
        mimic_obs_buf = torch.cat((
            root_pos[..., 0:3], # 3 dims
            roll, pitch, yaw, # 3 dims
            root_vel, # 3 dims
            root_ang_vel[..., 0:3], # 3 dims
            dof_pos, # num_dof dims
        ), dim=-1)[:, :] # shape: (num_envs, 1, 3*4 + num_dof)
        
        return mimic_obs_buf.reshape(self.num_envs, -1)

    def _get_current_mimic_obs(self): #获得未来第一个时间步的参考序列的观测
        # for Joao evaluation
        num_steps = self._tar_motion_steps_priv.shape[0]
        assert num_steps > 0, "Invalid number of target observation steps"
        motion_times = self._get_motion_times().unsqueeze(-1)
        obs_motion_times = self._tar_motion_steps_priv * self.dt + motion_times
        motion_ids_tiled = torch.broadcast_to(self._motion_ids.unsqueeze(-1), obs_motion_times.shape)
        motion_ids_tiled = motion_ids_tiled.flatten()
        obs_motion_times = obs_motion_times.flatten()
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos, root_pos_delta_local, root_rot_delta_local = self._motion_lib.calc_motion_frame(motion_ids_tiled, obs_motion_times)
        
        roll, pitch, yaw = euler_from_quaternion(root_rot)
        roll = roll.reshape(self.num_envs, num_steps, 1)
        pitch = pitch.reshape(self.num_envs, num_steps, 1)
        yaw = yaw.reshape(self.num_envs, num_steps, 1)
        if not self.global_obs:
            root_vel = quat_rotate_inverse(root_rot, root_vel)
            root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)
      
        whole_key_body_pos = body_pos[:, self._key_body_ids_motion, :]
        if self.global_obs:
            whole_key_body_pos = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=whole_key_body_pos)
        whole_key_body_pos = whole_key_body_pos.reshape(self.num_envs, num_steps, -1)
        
        root_pos = root_pos.reshape(self.num_envs, num_steps, root_pos.shape[-1])
        root_vel = root_vel.reshape(self.num_envs, num_steps, root_vel.shape[-1])
        root_rot = root_rot.reshape(self.num_envs, num_steps, root_rot.shape[-1])
        root_ang_vel = root_ang_vel.reshape(self.num_envs, num_steps, root_ang_vel.shape[-1])
        if dof_pos.shape[-1] == self.num_dof:
            dof_pos = dof_pos[:, self._active_dof_indices]
        dof_pos = dof_pos.reshape(self.num_envs, num_steps, dof_pos.shape[-1])
        

        cur_dof_pos = dof_pos[:, 0, :]
        cur_root_pos = root_pos[:, 0, :]
        cur_root_rot = root_rot[:, 0, :]
        return cur_dof_pos, cur_root_pos, cur_root_rot
    
    def _get_current_dof(self): #获得机器人的本体观测
        # for Joao evaluation
        self._ensure_active_dof_indices()
        current_dof = self.dof_pos[:, self._active_dof_indices]
        current_root_pos = self.root_states[:, 0:3]
        current_root_rot = self.root_states[:, 3:7]
        return current_dof, current_root_pos, current_root_rot
    
    
    
    def compute_observations(self): #拼接最终观测
        self._ensure_active_dof_indices()
        # imu_obs = torch.stack((self.roll, self.pitch, self.yaw - self.init_yaw), dim=1)
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        active_dof_pos = self.dof_pos[:, self._active_dof_indices]
        active_default_dof_pos = self.default_dof_pos_all[:, self._active_dof_indices]
        active_dof_vel = self.dof_vel[:, self._active_dof_indices]
        
        self.base_yaw_quat = quat_from_euler_xyz(0*self.yaw, 0*self.yaw, self.yaw)
        
        mimic_obs = self._get_mimic_obs()
        obs_buf = torch.cat((
                            mimic_obs, # (9 + num_dof) * num_steps
                            self.base_ang_vel  * self.obs_scales.ang_vel,   # 3 dims
                            imu_obs,    # 2 dims
                            self.reindex((active_dof_pos - active_default_dof_pos) * self.obs_scales.dof_pos),
                            self.reindex(active_dof_vel * self.obs_scales.dof_vel),
                            self.reindex(self.action_history_buf[:, -1]),
                            ),dim=-1)
        if self.cfg.noise.add_noise and self.headless:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec * min(self.total_env_steps_counter / (self.cfg.noise.noise_increasing_steps * 24),  1.)
        elif self.cfg.noise.add_noise and not self.headless:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec
        else:
            obs_buf += 0.
        
        dof_vel_start_dim = mimic_obs.shape[1] + 5 + active_dof_pos.shape[1]
        obs_buf[:, [dof_vel_start_dim + 4, dof_vel_start_dim + 5, dof_vel_start_dim + 10, dof_vel_start_dim + 11]] = 0. #是左右踝关节相关速度吗？
        if self.cfg.domain_rand.domain_rand_general:
            priv_latent = torch.cat((
                self.mass_params_tensor,
                self.friction_coeffs_tensor,
                self.motor_strength[0][:, self._active_dof_indices] - 1,
                self.motor_strength[1][:, self._active_dof_indices] - 1,
                self.base_lin_vel,
            ), dim=-1)
        else:
            priv_latent = torch.zeros((self.num_envs, self.cfg.env.n_priv_latent), device=self.device)
            priv_latent = torch.cat((priv_latent, self.base_lin_vel), dim=-1)

       
        self.obs_buf = torch.cat([obs_buf, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)

        if self.cfg.env.history_len > 0:
            self.obs_history_buf = torch.where(
                (self.episode_length_buf <= 1)[:, None, None], 
                torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
                torch.cat([
                    self.obs_history_buf[:, 1:],
                    obs_buf.unsqueeze(1)
                ], dim=1)
            )
        


############################################################################################################
##################################### Extra Reward Functions################################################
############################################################################################################

    def _reward_tracking_joint_vel(self):
        # TianGong override for early locomotion: use weighted mean L1 velocity error so a few fast
        # joints do not drive the whole term into exponential saturation during the bring-up stage.
        vel_diff = self._ref_dof_vel - self.dof_vel
        weight_sum = torch.clamp(torch.sum(self._dof_err_w), min=1.0)
        vel_err = torch.sum(self._dof_err_w * torch.abs(vel_diff), dim=-1) / weight_sum
        vel_scale = self.cfg.rewards.tracking_joint_vel_err_scale
        return torch.exp(-vel_scale * vel_err)

    def _reward_tracking_root_vel(self):
        # TianGong override for early locomotion: keep linear velocity tracking strong, but reduce
        # angular-velocity dominance so turning/upper-body mismatch does not kill this reward entirely.
        if self.global_obs:
            root_vel_diff = self._ref_root_vel - self.root_states[:, 7:10]
            root_ang_vel_diff = self._ref_root_ang_vel - self.root_states[:, 10:13]
        else:
            local_ref_root_vel = quat_rotate_inverse(self._ref_root_rot, self._ref_root_vel)
            root_vel_diff = local_ref_root_vel - self.base_lin_vel
            local_ref_root_ang_vel = quat_rotate_inverse(self._ref_root_rot, self._ref_root_ang_vel)
            root_ang_vel_diff = local_ref_root_ang_vel - self.base_ang_vel

        root_vel_err = torch.mean(root_vel_diff * root_vel_diff, dim=-1)
        root_ang_vel_err = torch.mean(root_ang_vel_diff * root_ang_vel_diff, dim=-1)
        lin_scale = self.cfg.rewards.tracking_root_lin_vel_err_scale
        ang_scale = self.cfg.rewards.tracking_root_ang_vel_err_scale
        return torch.exp(-(lin_scale * root_vel_err + ang_scale * root_ang_vel_err))

    def _ensure_anti_split_body_indices(self):
        if hasattr(self, "_anti_split_feet_body_ids"):
            return

        # TianGong anti-split helper: track ankles and knees directly by link name so we can
        # penalize sideward opening without affecting normal forward step length. Safe to remove later.
        body_name_to_idx = {name: i for i, name in enumerate(self.body_names)}
        feet_body_names = ["ankle_roll_l_link", "ankle_roll_r_link"]
        knee_body_names = ["knee_pitch_l_link", "knee_pitch_r_link"]
        missing = [
            name for name in (feet_body_names + knee_body_names)
            if name not in body_name_to_idx
        ]
        if missing:
            raise ValueError(f"TianGong anti-split body names missing in asset bodies: {missing}")

        self._anti_split_feet_body_ids = torch.tensor(
            [body_name_to_idx[name] for name in feet_body_names],
            dtype=torch.long,
            device=self.device,
        )
        self._anti_split_knee_body_ids = torch.tensor(
            [body_name_to_idx[name] for name in knee_body_names],
            dtype=torch.long,
            device=self.device,
        )

    def _get_yaw_local_body_pos_pair(self, body_ids):
        body_pos = self.rigid_body_states[:, body_ids, 0:3] - self.root_states[:, 0:3].unsqueeze(1)
        base_yaw_quat = quat_from_euler_xyz(0 * self.yaw, 0 * self.yaw, self.yaw)
        body_pos = convert_to_local_root_body_pos(base_yaw_quat, body_pos)

        ref_body_pos = self._ref_body_pos[:, body_ids, :] - self._ref_root_pos.unsqueeze(1)
        _, _, ref_yaw = euler_from_quaternion(self._ref_root_rot)
        ref_yaw_quat = quat_from_euler_xyz(0 * ref_yaw, 0 * ref_yaw, ref_yaw)
        ref_body_pos = convert_to_local_root_body_pos(ref_yaw_quat, ref_body_pos)
        return body_pos, ref_body_pos

    def _reward_tracking_leg_lateral(self):
        self._ensure_anti_split_body_indices()

        # TianGong anti-split helper: use yaw-aligned local lateral spacing only, but clamp against
        # absolute soft limits instead of the reference clip. This keeps normal forward stride free
        # while penalizing the wide-base "split-leg to survive" local optimum.
        feet_body_pos, _ = self._get_yaw_local_body_pos_pair(self._anti_split_feet_body_ids)
        knee_body_pos, _ = self._get_yaw_local_body_pos_pair(self._anti_split_knee_body_ids)

        feet_lateral = torch.abs(feet_body_pos[:, 0, 1] - feet_body_pos[:, 1, 1])
        knee_lateral = torch.abs(knee_body_pos[:, 0, 1] - knee_body_pos[:, 1, 1])

        feet_excess = torch.clamp(
            feet_lateral - self.cfg.rewards.tracking_leg_lateral_feet_soft_max,
            min=0.0,
        )
        knee_excess = torch.clamp(
            knee_lateral - self.cfg.rewards.tracking_leg_lateral_knee_soft_max,
            min=0.0,
        )
        feet_err = torch.square(feet_excess)
        knee_err = torch.square(knee_excess)
        feet_rew = torch.exp(-self.cfg.rewards.tracking_leg_lateral_feet_err_scale * feet_err)
        knee_rew = torch.exp(-self.cfg.rewards.tracking_leg_lateral_knee_err_scale * knee_err)
        return 0.5 * (feet_rew + knee_rew)

    def _reward_waist_dof_acc(self): #腰部加速度惩罚
        self._ensure_active_dof_indices()
        waist_dof_idx = [12]
        last_vel = self.last_dof_vel[:, self._active_dof_indices]
        dof_vel = self.dof_vel[:, self._active_dof_indices]
        return torch.sum(torch.square((last_vel - dof_vel) / self.dt)[:, waist_dof_idx], dim=1)
    
    def _reward_waist_dof_vel(self): #腰部速度惩罚
        self._ensure_active_dof_indices()
        waist_dof_idx = [12]
        dof_vel = self.dof_vel[:, self._active_dof_indices]
        return torch.sum(torch.square(dof_vel[:, waist_dof_idx]), dim=1)
    
    def _reward_ankle_dof_acc(self): #踝关节加速度惩罚
        self._ensure_active_dof_indices()
        ankle_dof_idx = [4, 5, 10, 11]
        last_vel = self.last_dof_vel[:, self._active_dof_indices]
        dof_vel = self.dof_vel[:, self._active_dof_indices]
        return torch.sum(torch.square((last_vel - dof_vel) / self.dt)[:, ankle_dof_idx], dim=1)
     
    def _reward_ankle_dof_vel(self): #踝关节速度惩罚
        self._ensure_active_dof_indices()
        ankle_dof_idx = [4, 5, 10, 11]
        dof_vel = self.dof_vel[:, self._active_dof_indices]
        return torch.sum(torch.square(dof_vel[:, ankle_dof_idx]), dim=1)
    
    def _reward_ankle_action(self):
        return torch.norm(self.action_history_buf[:, -1, [4, 5, 10, 11]], dim=1)
