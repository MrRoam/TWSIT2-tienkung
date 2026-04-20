from isaacgym.torch_utils import *

import os
import pickle

import torch
import yaml

from legged_gym.envs.base.humanoid_mimic import HumanoidMimic
from .tienkung_mimic_distill_config import TienkungMimicPrivCfg
from pose.utils.motion_lib_pkl import MotionLib
from legged_gym.envs.base.legged_robot import euler_from_quaternion
from legged_gym.envs.base.humanoid_char import convert_to_local_root_body_pos, convert_to_global_root_body_pos

'''
① 复用 HumanoidMimic 的 teacher/student 蒸馏框架
通过 obs_type 在同一套环境逻辑里同时支持 privileged teacher 训练和带历史编码的 student 训练。
② 建立天工 30 个 active dof 与 XML 全量 dof 的桥接
通过 _ensure_active_dof_indices()、_expand_motion_dofs_to_full()、_slice_motion_dofs_to_active() 在动作、参考轨迹和观测之间切换 active/full 表示。
③ 对 motion 数据做严格校验并安全加载
通过 _resolve_motion_files()、_validate_motion_file()、_load_motions() 保证 yaml/pkl 中的字段、维度和关键 body 都完整可用。
④ 维护参考轨迹并构造 teacher/student 两套模仿观测
通过 _reset_ref_motion()、_update_ref_motion() 更新参考 root/dof/body 状态，再在 _get_mimic_obs() 和 compute_observations() 中拼出 privileged obs 与 student obs。
⑤ 在蒸馏路径上加上天工特有的平滑约束
额外奖励重点约束腰部与踝部的速度、加速度和踝部动作幅度，让蒸馏出来的动作更稳更顺。
'''


class TienkungMimicDistill(HumanoidMimic):
    def __init__(self, cfg: TienkungMimicPrivCfg, sim_params, physics_engine, sim_device, headless):  # 初始化蒸馏环境，并根据 obs_type 准备 active dof、历史缓存和 student 计数器。
        self.cfg = cfg
        self.obs_type = cfg.env.obs_type
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._ensure_active_dof_indices()
        self.last_feet_z = 0.05
        self.episode_length = torch.zeros((self.num_envs), device=self.device)
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        if self.obs_type == "student":
            self.total_env_steps_counter = 24 * 100000
            self.global_counter = 24 * 100000

    # TianGong adapter: reject malformed motion clips up front so distill runs do not silently
    # ingest partial PKLs that happen to "work" on G1's plain-mimic path.
    def _resolve_motion_files(self, motion_file):  # 解析 yaml 或单个 pkl，返回实际需要校验和加载的 motion 文件列表。
        if motion_file.endswith(".yaml"):
            with open(motion_file, "r") as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)

            if not isinstance(motion_config, dict):
                raise ValueError(f"Motion yaml must contain a dict config: {motion_file}")

            motion_root_path = motion_config.get("root_path")
            motion_list = motion_config.get("motions")
            if not motion_root_path or not isinstance(motion_list, list) or len(motion_list) == 0:
                raise ValueError(f"Motion yaml must define non-empty root_path and motions: {motion_file}")

            resolved_files = []
            for motion_entry in motion_list:
                if not isinstance(motion_entry, dict) or "file" not in motion_entry:
                    raise ValueError(f"Invalid motion entry in yaml {motion_file}: {motion_entry}")
                resolved_files.append(os.path.join(motion_root_path, motion_entry["file"]))
            return resolved_files

        return [motion_file]

    def _validate_motion_file(self, motion_file):  # 严格检查 motion 字段、维度和关键 body，避免坏数据混进蒸馏训练。
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

    def _load_motions(self):  # 在正式构建 MotionLib 前先做严格校验，保证后续参考轨迹查询稳定可用。
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

    # TianGong adapter: keep G1 distill observation/training path, but expand 30-action control
    # and motion clips into the full 54-dof asset state used by the XML.
    def _ensure_active_dof_indices(self):  # 建立 active dof 到全量 dof 的索引映射，作为所有适配逻辑的入口。
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

    def _set_inactive_dofs_to_default(self, full_dof_pos, full_dof_vel, env_ids=None):  # 把非 active dof 统一钳回默认姿态和零速度，避免手部等未控关节漂走。
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

    def _expand_motion_dofs_to_full(self, dof_pos, dof_vel, env_ids=None):  # 把 motion 中的 active dof 或 full dof 统一展开成仿真资产需要的全量关节状态。
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
                f"Unexpected motion dof dim {dof_pos.shape[-1]}, expected {self._num_active_dofs} or {self.num_dof}."
            )

        return self._set_inactive_dofs_to_default(full_dof_pos, full_dof_vel, env_ids=env_ids)

    def _slice_motion_dofs_to_active(self, dof_pos):  # 把 motion 或仿真里的全量 dof 切回 student/teacher 真正使用的 30 维 active dof。
        self._ensure_active_dof_indices()
        if dof_pos.shape[-1] == self.num_dof:
            return dof_pos[:, self._active_dof_indices]
        if dof_pos.shape[-1] == self._num_active_dofs:
            return dof_pos
        raise ValueError(
            f"Unexpected motion dof dim {dof_pos.shape[-1]}, expected {self._num_active_dofs} or {self.num_dof}."
        )

    def _assign_ref_body_pos_from_motion(self, ref_body_pos, root_pos, root_rot, body_pos):  # 把 motion body 名称映射到天工仿真 body，并写入参考刚体位置。
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

    def _reset_ref_motion(self, env_ids, motion_ids=None):  # 在 reset 时采样一段新 motion，并初始化参考 root/dof/body 轨迹。
        self._ensure_active_dof_indices()
        n = len(env_ids)
        if motion_ids is None:
            motion_ids = self._motion_lib.sample_motions(n, motion_difficulty=self.motion_difficulty)

        if self._rand_reset:
            motion_times = self._motion_lib.sample_time(motion_ids)
        else:
            motion_times = torch.zeros(motion_ids.shape, device=self.device, dtype=torch.float)

        self._motion_ids[env_ids] = motion_ids
        self._motion_time_offsets[env_ids] = motion_times

        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos, root_pos_delta_local, root_rot_delta_local = (
            self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        )
        root_pos[:, 2] += self.cfg.motion.height_offset

        self._ref_root_pos[env_ids] = root_pos
        self._ref_root_rot[env_ids] = root_rot
        self._ref_root_vel[env_ids] = root_vel
        self._ref_root_ang_vel[env_ids] = root_ang_vel

        full_dof_pos, full_dof_vel = self._expand_motion_dofs_to_full(dof_pos, dof_vel, env_ids=env_ids)
        self._ref_dof_pos[env_ids] = full_dof_pos
        self._ref_dof_vel[env_ids] = full_dof_vel
        self._ref_root_pos_delta_local[env_ids] = root_pos_delta_local
        self._ref_root_rot_delta_local[env_ids] = root_rot_delta_local

        ref_body_pos = self._ref_body_pos[env_ids].clone()
        self._assign_ref_body_pos_from_motion(ref_body_pos, root_pos, root_rot, body_pos)
        self._ref_body_pos[env_ids] = ref_body_pos

    def _update_ref_motion(self):  # 在 episode 推进过程中按当前 motion time 更新整套参考状态。
        self._ensure_active_dof_indices()
        motion_ids = self._motion_ids
        motion_times = self._get_motion_times()
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos, root_pos_delta_local, root_rot_delta_local = (
            self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        )
        root_pos[:, 2] += self.cfg.motion.height_offset
        root_pos[:, :2] += self.episode_init_origin[:, :2]

        self._ref_root_pos[:] = root_pos
        self._ref_root_rot[:] = root_rot
        self._ref_root_vel[:] = root_vel
        self._ref_root_ang_vel[:] = root_ang_vel

        full_dof_pos, full_dof_vel = self._expand_motion_dofs_to_full(dof_pos, dof_vel)
        self._ref_dof_pos[:] = full_dof_pos
        self._ref_dof_vel[:] = full_dof_vel
        self._ref_root_pos_delta_local[:] = root_pos_delta_local
        self._ref_root_rot_delta_local[:] = root_rot_delta_local

        ref_body_pos = self._ref_body_pos.clone()
        self._assign_ref_body_pos_from_motion(ref_body_pos, root_pos, root_rot, body_pos)
        self._ref_body_pos[:] = ref_body_pos

    def _get_body_indices(self):  # 缓存腰、臂、膝等关键 body 在 Isaac Gym 中的刚体索引，供奖励和观测复用。
        upper_arm_names = [s for s in self.body_names if self.cfg.asset.upper_arm_name in s]
        lower_arm_names = [s for s in self.body_names if self.cfg.asset.lower_arm_name in s]
        torso_name = [s for s in self.body_names if self.cfg.asset.torso_name in s]
        self.torso_indices = torch.zeros(len(torso_name), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(torso_name)):
            self.torso_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], torso_name[j])
        self.upper_arm_indices = torch.zeros(len(upper_arm_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(upper_arm_names)):
            self.upper_arm_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], upper_arm_names[j])
        self.lower_arm_indices = torch.zeros(len(lower_arm_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(lower_arm_names)):
            self.lower_arm_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], lower_arm_names[j])
        knee_names = [s for s in self.body_names if self.cfg.asset.shank_name in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])

    def _init_buffers(self):  # 额外初始化学生历史观测和 privileged 历史观测缓存。
        super()._init_buffers()
        self.obs_history_buf = torch.zeros((self.num_envs, self.cfg.env.history_len, self.cfg.env.n_obs_single), device=self.device)
        self.privileged_obs_history_buf = torch.zeros(
            (self.num_envs, self.cfg.env.history_len, self.cfg.env.n_priv_obs_single), device=self.device
        )

    def _compute_torques(self, actions):  # 把 30 维 student 动作扩回全量 dof 动作后，复用父类 PD 力矩计算。
        self._ensure_active_dof_indices()
        full_actions = torch.zeros(actions.shape[0], self.num_dof, dtype=actions.dtype, device=actions.device)
        full_actions[:, self._active_dof_indices] = actions
        return super()._compute_torques(full_actions)

    def _get_noise_scale_vec(self, cfg):  # 按天工 30 维 proprio 布局生成噪声缩放向量。
        noise_scale_vec = torch.zeros(1, self.cfg.env.n_proprio, device=self.device)
        if not self.cfg.noise.add_noise:
            return noise_scale_vec

        ang_vel_dim = 3
        imu_dim = 2
        noise_scale_vec[:, 0:ang_vel_dim] = self.cfg.noise.noise_scales.ang_vel
        noise_scale_vec[:, ang_vel_dim:ang_vel_dim + imu_dim] = self.cfg.noise.noise_scales.imu
        noise_scale_vec[:, ang_vel_dim + imu_dim:ang_vel_dim + imu_dim + self.cfg.env.num_actions] = self.cfg.noise.noise_scales.dof_pos
        noise_scale_vec[:, ang_vel_dim + imu_dim + self.cfg.env.num_actions:ang_vel_dim + imu_dim + 2 * self.cfg.env.num_actions] = self.cfg.noise.noise_scales.dof_vel
        return noise_scale_vec

    def _get_mimic_obs(self):  # 从未来若干参考帧提取 privileged mimic obs 和 student mimic obs 两套模仿特征。
        num_steps = self._tar_motion_steps_priv.shape[0]
        assert num_steps > 0, "Invalid number of target observation steps"
        motion_times = self._get_motion_times().unsqueeze(-1)
        obs_motion_times = self._tar_motion_steps_priv * self.dt + motion_times
        motion_ids_tiled = torch.broadcast_to(self._motion_ids.unsqueeze(-1), obs_motion_times.shape).flatten()
        obs_motion_times = obs_motion_times.flatten()
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos, root_pos_delta_local, root_rot_delta_local = (
            self._motion_lib.calc_motion_frame(motion_ids_tiled, obs_motion_times)
        )

        active_dof_pos = self._slice_motion_dofs_to_active(dof_pos)
        roll, pitch, yaw = euler_from_quaternion(root_rot)
        roll = roll.reshape(self.num_envs, num_steps, 1)
        pitch = pitch.reshape(self.num_envs, num_steps, 1)
        yaw = yaw.reshape(self.num_envs, num_steps, 1)

        root_vel_local = quat_rotate_inverse(root_rot, root_vel)
        root_ang_vel_local = quat_rotate_inverse(root_rot, root_ang_vel)

        whole_key_body_pos = body_pos[:, self._key_body_ids_motion, :]
        whole_key_body_pos_global = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=whole_key_body_pos)

        whole_key_body_pos = whole_key_body_pos.reshape(self.num_envs, num_steps, -1)
        whole_key_body_pos_global = whole_key_body_pos_global.reshape(self.num_envs, num_steps, -1)

        root_pos = root_pos.reshape(self.num_envs, num_steps, root_pos.shape[-1])
        active_dof_pos = active_dof_pos.reshape(self.num_envs, num_steps, active_dof_pos.shape[-1])
        root_vel_local = root_vel_local.reshape(self.num_envs, num_steps, root_vel_local.shape[-1])
        root_ang_vel_local = root_ang_vel_local.reshape(self.num_envs, num_steps, root_ang_vel_local.shape[-1])
        root_pos_delta_local = root_pos_delta_local.reshape(self.num_envs, num_steps, root_pos_delta_local.shape[-1])
        root_rot_delta_local = root_rot_delta_local.reshape(self.num_envs, num_steps, root_rot_delta_local.shape[-1])
        root_pos_distance_to_target = root_pos - self.root_states[:, 0:3].reshape(self.num_envs, 1, -1)

        priv_mimic_obs_buf = torch.cat(
            (
                root_pos,
                root_pos_distance_to_target,
                roll, pitch, yaw,
                root_vel_local,
                root_ang_vel_local,
                root_pos_delta_local,
                root_rot_delta_local,
                active_dof_pos,
                whole_key_body_pos if not self.global_obs else whole_key_body_pos_global,
            ),
            dim=-1,
        )

        mimic_obs_buf = torch.cat(
            (
                root_vel_local[..., :2],
                root_pos[..., 2:3],
                roll, pitch,
                root_ang_vel_local[..., 2:3],
                active_dof_pos,
            ),
            dim=-1,
        )[:, self._tar_motion_steps_idx_in_teacher, :]

        return priv_mimic_obs_buf.reshape(self.num_envs, -1), mimic_obs_buf.reshape(self.num_envs, -1)

    def compute_observations(self):  # 把 mimic、proprio、privileged info 和历史缓存拼成 teacher/student 最终输入。
        self._ensure_active_dof_indices()
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        self.base_yaw_quat = quat_from_euler_xyz(0 * self.yaw, 0 * self.yaw, self.yaw)
        priv_mimic_obs, mimic_obs = self._get_mimic_obs()

        active_dof_pos = self.dof_pos[:, self._active_dof_indices]
        active_default_dof_pos = self.default_dof_pos_all[:, self._active_dof_indices]
        active_dof_vel = self.dof_vel[:, self._active_dof_indices]

        proprio_obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                imu_obs,
                self.reindex((active_dof_pos - active_default_dof_pos) * self.obs_scales.dof_pos),
                self.reindex(active_dof_vel * self.obs_scales.dof_vel),
                self.reindex(self.action_history_buf[:, -1]),
            ),
            dim=-1,
        )

        if self.cfg.noise.add_noise and self.headless:
            proprio_obs_buf += (2 * torch.rand_like(proprio_obs_buf) - 1) * self.noise_scale_vec * min(
                self.total_env_steps_counter / (self.cfg.noise.noise_increasing_steps * 24),
                1.0,
            )
        elif self.cfg.noise.add_noise and not self.headless:
            proprio_obs_buf += (2 * torch.rand_like(proprio_obs_buf) - 1) * self.noise_scale_vec

        dof_vel_start_dim = 3 + 2 + active_dof_pos.shape[1]
        ankle_idx = [4, 5, 10, 11]
        proprio_obs_buf[:, [dof_vel_start_dim + i for i in ankle_idx]] = 0.0

        key_body_pos = self.rigid_body_states[:, self._key_body_ids, :3] - self.root_states[:, None, :3]
        if not self.global_obs:
            key_body_pos = convert_to_local_root_body_pos(self.root_states[:, 3:7], key_body_pos)
        key_body_pos = key_body_pos.reshape(self.num_envs, -1)

        active_motor_strength = torch.cat(
            (
                self.motor_strength[0][:, self._active_dof_indices] - 1,
                self.motor_strength[1][:, self._active_dof_indices] - 1,
            ),
            dim=-1,
        )
        priv_info = torch.cat(
            (
                self.base_lin_vel,
                self.root_states[:, 0:3],
                self.root_states[:, 3:7],
                key_body_pos,
                (self.contact_forces[:, self.feet_indices, 2] > 5.0).float(),
                self.mass_params_tensor,
                self.friction_coeffs_tensor,
                active_motor_strength,
            ),
            dim=-1,
        )

        obs_buf = torch.cat((mimic_obs, proprio_obs_buf), dim=-1)
        priv_obs_buf = torch.cat((priv_mimic_obs, proprio_obs_buf, priv_info), dim=-1)

        self.privileged_obs_buf = priv_obs_buf
        if self.obs_type == "priv":
            self.obs_buf = priv_obs_buf
        elif self.obs_type == "student":
            self.obs_buf = torch.cat([obs_buf, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)

        if self.cfg.env.history_len > 0:
            reset_mask = self.episode_length_buf <= 1
            if reset_mask.any():
                reset_indices = reset_mask.nonzero(as_tuple=False).squeeze(-1)
                self.privileged_obs_history_buf[reset_indices] = priv_obs_buf[reset_indices].unsqueeze(1).expand(
                    -1, self.cfg.env.history_len, -1
                )

            continue_mask = ~reset_mask
            if continue_mask.any():
                continue_indices = continue_mask.nonzero(as_tuple=False).squeeze(-1)
                self.privileged_obs_history_buf[continue_indices, :-1] = self.privileged_obs_history_buf[continue_indices, 1:]
                self.privileged_obs_history_buf[continue_indices, -1] = priv_obs_buf[continue_indices]

            if self.obs_type == "priv":
                self.obs_history_buf[:] = self.privileged_obs_history_buf[:]
            elif self.obs_type == "student":
                if reset_mask.any():
                    reset_indices = reset_mask.nonzero(as_tuple=False).squeeze(-1)
                    self.obs_history_buf[reset_indices] = obs_buf[reset_indices].unsqueeze(1).expand(
                        -1, self.cfg.env.history_len, -1
                    )
                if continue_mask.any():
                    continue_indices = continue_mask.nonzero(as_tuple=False).squeeze(-1)
                    self.obs_history_buf[continue_indices, :-1] = self.obs_history_buf[continue_indices, 1:]
                    self.obs_history_buf[continue_indices, -1] = obs_buf[continue_indices]

    def _reward_tracking_joint_vel(self):  # 天工蒸馏路径使用 L1+加权均值形式，避免早期速度误差指数项饱和到 0。
        vel_diff = self._ref_dof_vel - self.dof_vel
        weight_sum = torch.clamp(torch.sum(self._dof_err_w), min=1.0)
        vel_err = torch.sum(self._dof_err_w * torch.abs(vel_diff), dim=-1) / weight_sum
        vel_scale = getattr(self.cfg.rewards, "tracking_joint_vel_err_scale", 0.1)
        return torch.exp(-vel_scale * vel_err)

    def _reward_waist_dof_acc(self):  # 惩罚腰部关节加速度过大，抑制躯干抖动。
        self._ensure_active_dof_indices()
        waist_dof_idx = [12]
        dof_vel = self.dof_vel[:, self._active_dof_indices]
        last_dof_vel = self.last_dof_vel[:, self._active_dof_indices]
        return torch.sum(torch.square((last_dof_vel - dof_vel) / self.dt)[:, waist_dof_idx], dim=1)

    def _reward_waist_dof_vel(self):  # 惩罚腰部角速度过大，约束上身扭摆幅度。
        self._ensure_active_dof_indices()
        waist_dof_idx = [12]
        dof_vel = self.dof_vel[:, self._active_dof_indices]
        return torch.sum(torch.square(dof_vel[:, waist_dof_idx]), dim=1)

    def _reward_ankle_dof_acc(self):  # 惩罚踝关节加速度过大，让落地和摆腿更平滑。
        self._ensure_active_dof_indices()
        ankle_dof_idx = [4, 5, 10, 11]
        dof_vel = self.dof_vel[:, self._active_dof_indices]
        last_dof_vel = self.last_dof_vel[:, self._active_dof_indices]
        return torch.sum(torch.square((last_dof_vel - dof_vel) / self.dt)[:, ankle_dof_idx], dim=1)

    def _reward_ankle_dof_vel(self):  # 惩罚踝关节速度过大，避免脚踝高频摆动。
        self._ensure_active_dof_indices()
        ankle_dof_idx = [4, 5, 10, 11]
        dof_vel = self.dof_vel[:, self._active_dof_indices]
        return torch.sum(torch.square(dof_vel[:, ankle_dof_idx]), dim=1)

    def _reward_ankle_action(self):  # 惩罚踝关节动作幅度过大，减少策略对脚踝的暴力控制。
        return torch.norm(self.action_history_buf[:, -1, [4, 5, 10, 11]], dim=1)
