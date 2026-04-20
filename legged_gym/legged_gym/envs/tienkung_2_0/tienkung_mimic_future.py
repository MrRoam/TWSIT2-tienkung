from isaacgym.torch_utils import *

import torch

from .tienkung_mimic_distill import TienkungMimicDistill
from .tienkung_mimic_future_config import TienkungMimicStuFutureCfg
from legged_gym.envs.base.legged_robot import euler_from_quaternion
from legged_gym.envs.base.humanoid_char import convert_to_local_root_body_pos, convert_to_global_root_body_pos

'''
① 继承 TienkungMimicDistill，在不改主干动力学与奖励的前提下加入 future obs
它复用 distill 的 active/full dof 适配、motion 校验和参考轨迹维护，只额外扩展学生输入。
② 用一次 MotionLib 采样同时拿到 privileged 帧和 future 帧
_get_unified_motion_data() 把 teacher 帧和 future 帧合并查询，避免重复采样带来的开销和时序不一致。
③ 保留 teacher / student 双观测结构
_get_mimic_obs() 继续返回 priv_mimic_obs 与 mimic_obs，并在 student_future 模式下额外返回 future_obs。
④ future obs 只服务 student_future 策略
_build_future_obs_from_data() 把未来帧压成与当前 mimic_obs 同构的表示，让策略显式知道“接下来要去哪里”。
⑤ compute_observations() 最终把 current mimic、history、future 三类信息拼成学生输入
privileged_obs_buf 仍保留给 teacher / critic，而 obs_buf 会根据 obs_type 切换成 priv、student 或 student_future 版本。
'''


class TienkungMimicFuture(TienkungMimicDistill):
    def __init__(self, cfg: TienkungMimicStuFutureCfg, sim_params, physics_engine, sim_device, headless):  # 初始化 future 配置，并在 student_future 模式下登记未来参考帧索引。
        self.future_cfg = cfg.env
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        if self.obs_type == "student_future":
            self._tar_motion_steps_future = torch.tensor(
                getattr(cfg.env, "tar_motion_steps_future", [0]),
                device=self.device,
                dtype=torch.long,
            )
            print(f"Tienkung future motion enabled with steps: {self._tar_motion_steps_future.tolist()}")

    def _get_unified_motion_data(self):  # 用一次 motion 查询同时生成 privileged 帧和 future 帧所需的全部参考特征。
        if self.obs_type == "student_future" and hasattr(self, "_tar_motion_steps_future"):
            all_steps = torch.cat([self._tar_motion_steps_priv, self._tar_motion_steps_future])
            num_priv_steps = self._tar_motion_steps_priv.shape[0]
            num_future_steps = self._tar_motion_steps_future.shape[0]
        else:
            all_steps = self._tar_motion_steps_priv
            num_priv_steps = self._tar_motion_steps_priv.shape[0]
            num_future_steps = 0

        total_steps = all_steps.shape[0]
        motion_times = self._get_motion_times().unsqueeze(-1)
        obs_motion_times = all_steps * self.dt + motion_times
        motion_ids_tiled = torch.broadcast_to(self._motion_ids.unsqueeze(-1), obs_motion_times.shape).flatten()
        obs_motion_times = obs_motion_times.flatten()

        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos, root_pos_delta_local, root_rot_delta_local = \
            self._motion_lib.calc_motion_frame(motion_ids_tiled, obs_motion_times)

        # TianGong future should reuse the distill adapter's active/full-dof bridge
        # instead of depending on the old plain-mimic helper name.
        dof_pos = self._slice_motion_dofs_to_active(dof_pos)
        dof_vel = self._slice_motion_dofs_to_active(dof_vel)

        roll, pitch, yaw = euler_from_quaternion(root_rot)
        roll = roll.reshape(self.num_envs, total_steps, 1)
        pitch = pitch.reshape(self.num_envs, total_steps, 1)
        yaw = yaw.reshape(self.num_envs, total_steps, 1)

        root_vel_local = quat_rotate_inverse(root_rot, root_vel)
        root_ang_vel_local = quat_rotate_inverse(root_rot, root_ang_vel)

        whole_key_body_pos = body_pos[:, self._key_body_ids_motion, :]
        whole_key_body_pos_global = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=whole_key_body_pos)

        root_pos = root_pos.reshape(self.num_envs, total_steps, root_pos.shape[-1])
        root_vel = root_vel.reshape(self.num_envs, total_steps, root_vel.shape[-1])
        root_rot = root_rot.reshape(self.num_envs, total_steps, root_rot.shape[-1])
        root_ang_vel = root_ang_vel.reshape(self.num_envs, total_steps, root_ang_vel.shape[-1])
        dof_pos = dof_pos.reshape(self.num_envs, total_steps, dof_pos.shape[-1])
        dof_vel = dof_vel.reshape(self.num_envs, total_steps, dof_vel.shape[-1])
        root_vel_local = root_vel_local.reshape(self.num_envs, total_steps, root_vel_local.shape[-1])
        root_ang_vel_local = root_ang_vel_local.reshape(self.num_envs, total_steps, root_ang_vel_local.shape[-1])
        root_pos_delta_local = root_pos_delta_local.reshape(self.num_envs, total_steps, root_pos_delta_local.shape[-1])
        root_rot_delta_local = root_rot_delta_local.reshape(self.num_envs, total_steps, root_rot_delta_local.shape[-1])
        whole_key_body_pos = whole_key_body_pos.reshape(self.num_envs, total_steps, -1)
        whole_key_body_pos_global = whole_key_body_pos_global.reshape(self.num_envs, total_steps, -1)
        root_pos_distance_to_target = root_pos - self.root_states[:, 0:3].reshape(self.num_envs, 1, -1)

        return {
            "root_pos": root_pos,
            "root_vel": root_vel,
            "root_rot": root_rot,
            "root_ang_vel": root_ang_vel,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "root_pos_delta_local": root_pos_delta_local,
            "root_rot_delta_local": root_rot_delta_local,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "root_vel_local": root_vel_local,
            "root_ang_vel_local": root_ang_vel_local,
            "whole_key_body_pos": whole_key_body_pos,
            "whole_key_body_pos_global": whole_key_body_pos_global,
            "root_pos_distance_to_target": root_pos_distance_to_target,
            "num_priv_steps": num_priv_steps,
            "num_future_steps": num_future_steps,
        }

    def _build_future_obs_from_data(self, motion_data):  # 从统一采样结果里切出未来帧，并压成 student_future 专用的 future_obs。
        if self.obs_type != "student_future" or motion_data["num_future_steps"] == 0:
            return torch.zeros(self.num_envs, 0, device=self.device)

        num_priv_steps = motion_data["num_priv_steps"]
        root_pos = motion_data["root_pos"][:, num_priv_steps:]
        root_vel_local = motion_data["root_vel_local"][:, num_priv_steps:]
        root_ang_vel_local = motion_data["root_ang_vel_local"][:, num_priv_steps:]
        roll = motion_data["roll"][:, num_priv_steps:]
        pitch = motion_data["pitch"][:, num_priv_steps:]
        dof_pos = motion_data["dof_pos"][:, num_priv_steps:]

        future_obs = torch.cat(
            (
                root_vel_local[..., :2],
                root_pos[..., 2:3],
                roll,
                pitch,
                root_ang_vel_local[..., 2:3],
                dof_pos,
            ),
            dim=-1,
        )
        return future_obs

    def _get_mimic_obs(self):  # 在 distill 原有 mimic obs 基础上，按需额外返回 future obs。
        motion_data = self._get_unified_motion_data()
        num_steps = motion_data["num_priv_steps"]

        root_pos = motion_data["root_pos"][:, :num_steps]
        dof_pos = motion_data["dof_pos"][:, :num_steps]
        root_pos_delta_local = motion_data["root_pos_delta_local"][:, :num_steps]
        root_rot_delta_local = motion_data["root_rot_delta_local"][:, :num_steps]
        roll = motion_data["roll"][:, :num_steps]
        pitch = motion_data["pitch"][:, :num_steps]
        yaw = motion_data["yaw"][:, :num_steps]
        root_vel_local = motion_data["root_vel_local"][:, :num_steps]
        root_ang_vel_local = motion_data["root_ang_vel_local"][:, :num_steps]
        whole_key_body_pos = motion_data["whole_key_body_pos"][:, :num_steps]
        whole_key_body_pos_global = motion_data["whole_key_body_pos_global"][:, :num_steps]
        root_pos_distance_to_target = motion_data["root_pos_distance_to_target"][:, :num_steps]

        priv_mimic_obs_buf = torch.cat(
            (
                root_pos,
                root_pos_distance_to_target,
                roll,
                pitch,
                yaw,
                root_vel_local,
                root_ang_vel_local,
                root_pos_delta_local,
                root_rot_delta_local,
                dof_pos,
                whole_key_body_pos if not self.global_obs else whole_key_body_pos_global,
            ),
            dim=-1,
        )

        mimic_obs_buf = torch.cat(
            (
                root_vel_local[..., :2],
                root_pos[..., 2:3],
                roll,
                pitch,
                root_ang_vel_local[..., 2:3],
                dof_pos,
            ),
            dim=-1,
        )[:, self._tar_motion_steps_idx_in_teacher, :]

        priv_mimic_obs = priv_mimic_obs_buf.reshape(self.num_envs, -1)
        mimic_obs = mimic_obs_buf.reshape(self.num_envs, -1)

        if self.obs_type == "student_future":
            future_obs = self._build_future_obs_from_data(motion_data)
            return priv_mimic_obs, mimic_obs, future_obs

        return priv_mimic_obs, mimic_obs

    def compute_observations(self):  # 按 obs_type 组装 privileged、history 和 future 三种信息，形成最终策略输入。
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        self.base_yaw_quat = quat_from_euler_xyz(0 * self.yaw, 0 * self.yaw, self.yaw)

        if self.obs_type == "student_future":
            priv_mimic_obs, mimic_obs, future_obs = self._get_mimic_obs()
        else:
            priv_mimic_obs, mimic_obs = self._get_mimic_obs()
            future_obs = None

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
            noise_scale = min(self.total_env_steps_counter / (self.cfg.noise.noise_increasing_steps * 24), 1.0)
            proprio_obs_buf += (2 * torch.rand_like(proprio_obs_buf) - 1) * self.noise_scale_vec * noise_scale
        elif self.cfg.noise.add_noise and not self.headless:
            proprio_obs_buf += (2 * torch.rand_like(proprio_obs_buf) - 1) * self.noise_scale_vec

        dof_vel_start_dim = 3 + 2 + active_dof_pos.shape[1]
        ankle_idx = [4, 5, 10, 11]
        proprio_obs_buf[:, [dof_vel_start_dim + i for i in ankle_idx]] = 0.0

        key_body_pos = self.rigid_body_states[:, self._key_body_ids, :3]
        key_body_pos = key_body_pos - self.root_states[:, None, :3]
        if not self.global_obs:
            key_body_pos = convert_to_local_root_body_pos(self.root_states[:, 3:7], key_body_pos)
        key_body_pos = key_body_pos.reshape(self.num_envs, -1)

        priv_info = torch.cat(
            (
                self.base_lin_vel,
                self.root_states[:, 0:3],
                self.root_states[:, 3:7],
                key_body_pos,
                self.contact_forces[:, self.feet_indices, 2] > 5.0,
                self.mass_params_tensor,
                self.friction_coeffs_tensor,
                self.motor_strength[0][:, self._active_dof_indices] - 1.0,
                self.motor_strength[1][:, self._active_dof_indices] - 1.0,
            ),
            dim=-1,
        )

        obs_buf = torch.cat((mimic_obs, proprio_obs_buf), dim=-1)
        priv_obs_buf = torch.cat((priv_mimic_obs, proprio_obs_buf, priv_info), dim=-1)
        self.privileged_obs_buf = priv_obs_buf

        if self.obs_type == "priv":
            self.obs_buf = priv_obs_buf
        elif self.obs_type == "student_future":
            obs_components = [obs_buf, self.obs_history_buf.view(self.num_envs, -1)]
            if future_obs is not None:
                obs_components.append(future_obs.view(self.num_envs, -1))
            self.obs_buf = torch.cat(obs_components, dim=-1)
        else:
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
            else:
                if reset_mask.any():
                    reset_indices = reset_mask.nonzero(as_tuple=False).squeeze(-1)
                    self.obs_history_buf[reset_indices] = obs_buf[reset_indices].unsqueeze(1).expand(
                        -1, self.cfg.env.history_len, -1
                    )
                if continue_mask.any():
                    continue_indices = continue_mask.nonzero(as_tuple=False).squeeze(-1)
                    self.obs_history_buf[continue_indices, :-1] = self.obs_history_buf[continue_indices, 1:]
                    self.obs_history_buf[continue_indices, -1] = obs_buf[continue_indices]
