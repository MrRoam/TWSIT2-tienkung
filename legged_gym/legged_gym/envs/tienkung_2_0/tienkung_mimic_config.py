from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfg, HumanoidMimicCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR

TIENKUNG_MOTOR_ID_TO_JOINT = {
    # Head motors 1-3
    1: "head_yaw_joint",
    2: "head_pitch_joint",
    3: "head_roll_joint",
    # Left arm motors 11-17
    11: "shoulder_pitch_l_joint",
    12: "shoulder_roll_l_joint",
    13: "shoulder_yaw_l_joint",
    14: "elbow_pitch_l_joint",
    15: "elbow_yaw_l_joint",
    16: "wrist_pitch_l_joint",
    17: "wrist_roll_l_joint",
    # Right arm motors 21-27
    21: "shoulder_pitch_r_joint",
    22: "shoulder_roll_r_joint",
    23: "shoulder_yaw_r_joint",
    24: "elbow_pitch_r_joint",
    25: "elbow_yaw_r_joint",
    26: "wrist_pitch_r_joint",
    27: "wrist_roll_r_joint",
    # Waist motor 31
    31: "waist_yaw_joint",
    # Leg motors follow the XML joint order within each leg chain.
    51: "hip_roll_l_joint",
    52: "hip_pitch_l_joint",
    53: "hip_yaw_l_joint",
    54: "knee_pitch_l_joint",
    55: "ankle_pitch_l_joint",
    56: "ankle_roll_l_joint",
    61: "hip_roll_r_joint",
    62: "hip_pitch_r_joint",
    63: "hip_yaw_r_joint",
    64: "knee_pitch_r_joint",
    65: "ankle_pitch_r_joint",
    66: "ankle_roll_r_joint",
}

TIENKUNG_PD_GAINS_BY_MOTOR_ID = {
    1: (80.0, 5.0),
    2: (80.0, 5.0),
    3: (80.0, 5.0),
    11: (200.0, 15.0),
    12: (200.0, 15.0),
    13: (200.0, 15.0),
    14: (200.0, 15.0),
    15: (200.0, 15.0),
    16: (80.0, 5.0),
    17: (80.0, 5.0),
    21: (200.0, 15.0),
    22: (200.0, 15.0),
    23: (200.0, 15.0),
    24: (200.0, 15.0),
    25: (200.0, 15.0),
    26: (80.0, 5.0),
    27: (80.0, 5.0),
    # The provided table does not include motor 31, so keep the previous waist PD.
    31: (150.0, 4.0),
    # TianGong locomotion bring-up: reduce hip roll/yaw lateral authority to curb split-leg solutions.
    51: (1200.0, 45.0),
    52: (1000.0, 30.0),
    53: (1200.0, 45.0),
    54: (1000.0, 30.0),
    55: (50.0, 3.0),
    56: (50.0, 3.0),
    61: (1200.0, 45.0),
    62: (1000.0, 30.0),
    63: (1200.0, 45.0),
    64: (1000.0, 30.0),
    65: (50.0, 3.0),
    66: (50.0, 3.0),
}

TIENKUNG_JOINT_STIFFNESS = {
    TIENKUNG_MOTOR_ID_TO_JOINT[motor_id]: gains[0]
    for motor_id, gains in TIENKUNG_PD_GAINS_BY_MOTOR_ID.items()
}

TIENKUNG_JOINT_DAMPING = {
    TIENKUNG_MOTOR_ID_TO_JOINT[motor_id]: gains[1]
    for motor_id, gains in TIENKUNG_PD_GAINS_BY_MOTOR_ID.items()
}


class TienkungMimicCfg(HumanoidMimicCfg):
    class env(HumanoidMimicCfg.env): #环境配置
        tar_motion_steps_priv = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                                 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

        num_envs = 256
        num_actions = 30
        n_priv = 0
        n_mimic_obs = 3 * 4 + 30
        n_priv_mimic_obs = len(tar_motion_steps_priv) * n_mimic_obs
        n_proprio = len(tar_motion_steps_priv) * n_mimic_obs + 3 + 2 + 3 * num_actions
        n_priv_latent = 4 + 1 + 2 * num_actions
        extra_critic_obs = 3
        history_len = 10

        num_observations = n_proprio + n_priv_latent + history_len * n_proprio + n_priv + extra_critic_obs
        num_privileged_obs = num_observations

        env_spacing = 3.0
        send_timeouts = True
        episode_length_s = 10

        randomize_start_pos = True
        randomize_start_yaw = False

        history_encoding = True
        contact_buf_len = 10

        normalize_obs = True

        enable_early_termination = True
        pose_termination = True
        pose_termination_dist = 0.85  # Tuned on walk1 100-iter A/B; outperformed both 0.8 and 0.9.
        root_tracking_termination_dist = 0.9
        # Tuning knobs kept at current behavior by default so single-variable A/B runs can
        # override them from the CLI without editing source again. Safe to remove after tuning.
        reset_ref_vel_factor = 0.8
        contact_force_termination_threshold = 1.0
        rand_reset = True
        track_root = False
        # New XML has 54 DoFs (includes fingers). We keep 30-DoF control/tracking and
        # zero-weight finger joints in tracking rewards.
        dof_err_w = (
            [1.0, 1.0, 1.0, 1.0, 0.1, 0.1] +  # Left leg (6)
            [1.0, 1.0, 1.0, 1.0, 0.1, 0.1] +  # Right leg (6)
            [1.0, 1.0, 1.0, 1.0] +            # Waist + head (4)
            [1.0] * 7 +                       # Left arm (7)
            [0.0] * 12 +                      # Left fingers (12)
            [1.0] * 7 +                       # Right arm (7)
            [0.0] * 12                        # Right fingers (12)
        )  # total 54

        global_obs = False

    class terrain(HumanoidMimicCfg.terrain): #地形配置
        mesh_type = "trimesh"
        height = [0, 0.00]
        horizontal_scale = 0.1

    class init_state(HumanoidMimicCfg.init_state): #关节初始状态
        pos = [0, 0, 1.0]
        default_joint_angles = {
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

            "left_thumb_1_joint": 0.0,
            "left_thumb_2_joint": 0.0,
            "left_thumb_3_joint": 0.0,
            "left_thumb_4_joint": 0.0,
            "left_index_1_joint": 0.0,
            "left_index_2_joint": 0.0,
            "left_middle_1_joint": 0.0,
            "left_middle_2_joint": 0.0,
            "left_ring_1_joint": 0.0,
            "left_ring_2_joint": 0.0,
            "left_little_1_joint": 0.0,
            "left_little_2_joint": 0.0,

            "right_thumb_1_joint": 0.0,
            "right_thumb_2_joint": 0.0,
            "right_thumb_3_joint": 0.0,
            "right_thumb_4_joint": 0.0,
            "right_index_1_joint": 0.0,
            "right_index_2_joint": 0.0,
            "right_middle_1_joint": 0.0,
            "right_middle_2_joint": 0.0,
            "right_ring_1_joint": 0.0,
            "right_ring_2_joint": 0.0,
            "right_little_1_joint": 0.0,
            "right_little_2_joint": 0.0,
        }

    class control(HumanoidMimicCfg.control): #PD控制器参数
        stiffness = {
            **TIENKUNG_JOINT_STIFFNESS,
            "thumb": 0.0,
            "index": 0.0,
            "middle": 0.0,
            "ring": 0.0,
            "little": 0.0,
        }
        damping = {
            **TIENKUNG_JOINT_DAMPING,
            "thumb": 0.0,
            "index": 0.0,
            "middle": 0.0,
            "ring": 0.0,
            "little": 0.0,
        }

        # TianGong locomotion bring-up: shrink action magnitude so lateral joints stop over-correcting.
        action_scale = 0.38
        decimation = 10 #控制频率是：仿真频率：500 Hz 策略控制频率：50 Hz

    class sim(HumanoidMimicCfg.sim): #仿真时间步长
        dt = 0.002

    class normalization(HumanoidMimicCfg.normalization): #动作裁剪
        clip_actions = 5.0

    class asset(HumanoidMimicCfg.asset): #机器人结构
        file = "/data/shared_folder/GMR/assets/tienkung_ei/mjcf/tienkung_ei_v1.xml"

        torso_name: str = "pelvis"
        chest_name: str = "waist_yaw_link"

        thigh_name: str = "hip_pitch"
        shank_name: str = "knee_pitch"
        foot_name: str = "ankle_roll"
        waist_name: list = ["waist_yaw_link"]
        upper_arm_name: str = "shoulder_roll"
        lower_arm_name: str = "elbow"
        hand_name: list = ["wrist_roll_l_link", "wrist_roll_r_link"]

        feet_bodies = ["ankle_roll_l_link", "ankle_roll_r_link"]
        n_lower_body_dofs: int = 12
        active_dof_names = [
            "hip_roll_l_joint", "hip_pitch_l_joint", "hip_yaw_l_joint", "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
            "hip_roll_r_joint", "hip_pitch_r_joint", "hip_yaw_r_joint", "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
            "waist_yaw_joint", "head_yaw_joint", "head_pitch_joint", "head_roll_joint",
            "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint", "elbow_pitch_l_joint", "elbow_yaw_l_joint", "wrist_pitch_l_joint", "wrist_roll_l_joint",
            "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint", "elbow_pitch_r_joint", "elbow_yaw_r_joint", "wrist_pitch_r_joint", "wrist_roll_r_joint",
        ]

        penalize_contacts_on = ["shoulder", "elbow", "hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]

        # Use XML defaults for armature to avoid length mismatch with 54-DoF model.
        dof_armature = []

        collapse_fixed_joints = False

    class rewards(HumanoidMimicCfg.rewards): #训练奖惩
        regularization_names = [
            "feet_stumble",          # 正则项：惩罚摆腿时脚被绊到
            "feet_contact_forces",   # 正则项：惩罚足底接触冲击过大
            "lin_vel_z",             # 正则项：惩罚机身竖直方向速度
            "ang_vel_xy",            # 正则项：惩罚机身 roll/pitch 角速度
            "orientation",           # 正则项：惩罚机身倾斜
            "dof_pos_limits",        # 正则项：惩罚关节贴近或超过限位
            "dof_torque_limits",     # 正则项：惩罚关节扭矩接近上限
            "collision",             # 正则项：惩罚非足部身体碰撞
            "torque_penalty",        # 正则项：惩罚整体扭矩过大
            "thigh_torque_roll_yaw", # 正则项：惩罚大腿 roll/yaw 方向扭矩
            "thigh_roll_yaw_acc",    # 正则项：惩罚大腿 roll/yaw 方向加速度
            "dof_acc",               # 正则项：惩罚全身关节加速度过大
            "dof_vel",               # 正则项：惩罚全身关节速度过大
            "action_rate",           # 正则项：惩罚相邻策略动作变化过快
        ]
        regularization_scale = 1.0              # 正则项总缩放系数
        regularization_scale_range = [0.8, 2.0] # 正则项 curriculum 可变化范围
        regularization_scale_curriculum = False # 是否逐步增大正则项权重
        regularization_scale_gamma = 0.0001     # 正则项 curriculum 变化速度

        class scales:
            tracking_joint_dof = 1.0           # 跟踪参考关节角
            tracking_joint_vel = 0.2           # 跟踪参考关节速度
            tracking_root_translation = 1.0    # 跟踪 root/pelvis 的空间位置
            tracking_root_rotation = 0.8       # 跟踪 root/pelvis 的朝向
            tracking_root_vel = 1.0            # 跟踪 root 的线速度和角速度
            tracking_keybody_pos = 2.0         # 跟踪关键 body 点位，如头/肘/腕/膝/踝
            tracking_leg_lateral = 1.0         # TianGong 专项：限制双腿横向张开过大，抑制迈步时劈叉

            feet_slip = -0.1                  # 惩罚支撑脚滑移
            feet_contact_forces = -2.5e-4     # 惩罚落地/受力冲击过大
            feet_stumble = -1.25              # 惩罚脚尖或脚侧绊地

            dof_pos_limits = -5.0             # 惩罚关节越界或贴边
            dof_torque_limits = -1.0          # 惩罚扭矩接近极限

            dof_vel = -5e-5                   # 惩罚全身关节速度过大
            dof_acc = -2.5e-8                 # 惩罚全身关节加速度过大
            action_rate = -0.005              # 惩罚动作输出抖动

            feet_air_time = 5.0               # 鼓励形成正常迈步腾空时间

            ang_vel_xy = -0.005               # 惩罚机身 roll/pitch 角速度过大

            waist_dof_acc = 0.0               # 第一阶段关闭；基础步态稳定后再小权重开启
            waist_dof_vel = 0.0               # 第一阶段关闭；基础步态稳定后再小权重开启
            ankle_dof_acc = -5e-8             # 专门抑制踝关节高频加速度
            ankle_dof_vel = -1e-4             # 专门抑制踝关节速度过大
            ankle_action = 0.0                # 第一阶段关闭；若后面发现踝动作过猛再开启

        min_dist = 0.1                        # 双脚最小允许间距
        max_dist = 0.4                        # 双脚最大允许间距
        max_knee_dist = 0.4                   # 双膝最大允许间距
        target_feet_height = 0.07             # 期望脚部抬脚高度
        only_positive_rewards = False         # 是否把总 reward 截断为非负
        tracking_sigma = 0.2                  # 平移/关节类跟踪奖励的软化系数
        tracking_sigma_ang = 0.125            # 角度类跟踪奖励的软化系数
        # TianGong temporary tuning: soften velocity-tracking exponentials so early locomotion training
        # does not drive tracking_joint_vel / tracking_root_vel into near-zero reward saturation.
        tracking_joint_vel_err_scale = 0.1
        tracking_root_lin_vel_err_scale = 1.0
        tracking_root_ang_vel_err_scale = 0.02
        # TianGong anti-split helper: use absolute lateral width caps instead of following a wide
        # reference clip, so normal forward step length is preserved while split-leg solutions are not.
        tracking_leg_lateral_feet_err_scale = 80.0
        tracking_leg_lateral_knee_err_scale = 80.0
        tracking_leg_lateral_feet_soft_max = 0.24
        tracking_leg_lateral_knee_soft_max = 0.30
        max_contact_force = 350               # 超过该值开始惩罚足底接触力
        soft_torque_limit = 0.95              # 超过该比例开始惩罚扭矩
        torque_safety_limit = 0.9             # 扭矩安全阈值

        termination_roll = 1.5                # roll 超阈值则终止
        termination_pitch = 1.5               # pitch 超阈值则终止
        root_height_diff_threshold = 0.3      # root 高度偏离参考过大则终止

    class domain_rand: #仿真参数随机化
        domain_rand_general = True

        randomize_gravity = (True and domain_rand_general)
        gravity_rand_interval_s = 4
        gravity_range = (-0.05, 0.05)     # 先减小重力扰动，避免新 asset 早期被随机倾斜打散

        randomize_friction = (True and domain_rand_general)
        friction_range = [0.4, 1.5]       # 缩窄摩擦范围，先覆盖常见地面，减少极端滑/粘情况

        randomize_base_mass = (True and domain_rand_general)
        added_mass_range = [-2.0, 2.0]    # 缩窄躯干附加质量扰动，先保训练稳定性

        randomize_base_com = (True and domain_rand_general)
        added_com_range = [-0.03, 0.03]   # 缩窄质心偏移，降低上身前后左右乱晃

        push_robots = False
        push_interval_s = 4
        max_push_vel_xy = 1.0

        push_end_effector = False
        push_end_effector_interval_s = 2
        max_push_force_end_effector = 20.0

        randomize_motor = (True and domain_rand_general)
        motor_strength_range = [0.9, 1.1] # TianGong 先用更保守的电机强度扰动

        action_delay = False
        action_buf_len = 8

    class noise(HumanoidMimicCfg.noise): #噪声随机化
        add_noise = True
        noise_increasing_steps = 10000     # 放慢噪声爬升，让策略先学会基本模仿

        class noise_scales:
            dof_pos = 0.005               # 关节位置观测噪声，TianGong 先保守一点
            dof_vel = 0.05                # 关节速度观测噪声，避免速度跟踪一开始就被压死
            lin_vel = 0.05                # 当前 override 中未实际注入，但保持与整体噪声等级一致
            ang_vel = 0.05                # 基座角速度噪声，先减半
            gravity = 0.02                # 当前 override 中未实际注入，保守保留
            imu = 0.05                    # IMU 观测噪声，先减半

    class evaluations: #评估指标
        tracking_joint_dof = True
        tracking_joint_vel = True
        tracking_root_translation = True
        tracking_root_rotation = True
        tracking_root_vel = True
        tracking_root_ang_vel = True
        tracking_keybody_pos = True
        tracking_root_pose_delta_local = True
        tracking_root_rotation_delta_local = True

    class motion(HumanoidMimicCfg.motion): #参考动作库配置
        motion_curriculum = True
        motion_curriculum_gamma = 0.01
        key_bodies = [
            "ankle_roll_l_link",
            "ankle_roll_r_link",
            "knee_pitch_l_link",
            "knee_pitch_r_link",
            # TianGong locomotion bring-up: drop head from pose-termination key bodies so torso tilt
            # does not dominate pose_fail before lower-body walking stabilizes.
        ]

        motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/tienkung_ei_train30.yaml"

        reset_consec_frames = 30


class TienkungMimicCfgPPO(HumanoidMimicCfgPPO): #PPO训练器配置
    seed = 1

    class runner(HumanoidMimicCfgPPO.runner): 
        policy_class_name = "ActorCriticMimic"
        algorithm_class_name = "PPO"
        runner_class_name = "OnPolicyRunnerMimic"
        max_iterations = 30002

        save_interval = 500
        experiment_name = "tienkung_2_0_mimic"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

    class algorithm(HumanoidMimicCfgPPO.algorithm): #算法细节
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005

    class policy(HumanoidMimicCfgPPO.policy): #网络结构
        action_std = [0.7] * 12 + [0.4] * 4 + [0.5] * 14
        init_noise_std = 0.8
        obs_context_len = 11
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = "silu"
