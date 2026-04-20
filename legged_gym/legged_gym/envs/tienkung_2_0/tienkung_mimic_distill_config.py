from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfg, HumanoidMimicCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR

'''
① 这份文件定义天工 distill 系列任务的静态配置骨架
包括动作空间、关键 body、active dof、PD 参数、观测维度、奖励、随机化和 motion 数据源。
② TienkungMimicPrivCfg 是 teacher / privileged 训练的母配置
它决定 30 维控制、54 dof 资产映射，以及 privileged mimic obs 和 priv_info 的尺寸组织方式。
③ TienkungMimicStuCfg 与 TienkungMimicStuRLCfg 定义学生侧观测接口
它们把 teacher 的全量 privileged 观测收缩成 mimic_obs + proprio + history，供 student 模型使用。
④ TienkungMimicPrivCfgPPO、TienkungMimicStuCfgDAgger、TienkungMimicStuRLCfgDAgger 定义训练器
分别对应 teacher PPO、纯 student DAgger、以及带 RL 的 DaggerPPO 训练超参数与网络结构。
⑤ 所有维度和超参都围绕“30 个 active dof 控制天工全量资产”展开
所以 motor strength、action std、奖励权重和观测长度都只显式建模 active dof。
'''


TIENKUNG_DISTILL_NUM_ACTIONS = 30
TIENKUNG_DISTILL_KEY_BODIES = [
    "wrist_roll_l_link",
    "wrist_roll_r_link",
    "ankle_roll_l_link",
    "ankle_roll_r_link",
    "knee_pitch_l_link",
    "knee_pitch_r_link",
    "elbow_pitch_l_link",
    "elbow_pitch_r_link",
    "head_roll_link",
]
TIENKUNG_DISTILL_ACTIVE_DOF_NAMES = [
    "hip_roll_l_joint", "hip_pitch_l_joint", "hip_yaw_l_joint", "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
    "hip_roll_r_joint", "hip_pitch_r_joint", "hip_yaw_r_joint", "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
    "waist_yaw_joint", "head_yaw_joint", "head_pitch_joint", "head_roll_joint",
    "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint", "elbow_pitch_l_joint", "elbow_yaw_l_joint", "wrist_pitch_l_joint", "wrist_roll_l_joint",
    "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint", "elbow_pitch_r_joint", "elbow_yaw_r_joint", "wrist_pitch_r_joint", "wrist_roll_r_joint",
]
TIENKUNG_MOTOR_ID_TO_JOINT = {
    1: "head_yaw_joint",
    2: "head_pitch_joint",
    3: "head_roll_joint",
    11: "shoulder_pitch_l_joint",
    12: "shoulder_roll_l_joint",
    13: "shoulder_yaw_l_joint",
    14: "elbow_pitch_l_joint",
    15: "elbow_yaw_l_joint",
    16: "wrist_pitch_l_joint",
    17: "wrist_roll_l_joint",
    21: "shoulder_pitch_r_joint",
    22: "shoulder_roll_r_joint",
    23: "shoulder_yaw_r_joint",
    24: "elbow_pitch_r_joint",
    25: "elbow_yaw_r_joint",
    26: "wrist_pitch_r_joint",
    27: "wrist_roll_r_joint",
    31: "waist_yaw_joint",
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
    1: (40.0, 5.0),
    2: (40.0, 5.0),
    3: (40.0, 5.0),
    11: (40.0, 5.0),
    12: (40.0, 5.0),
    13: (40.0, 5.0),
    14: (40.0, 5.0),
    15: (40.0, 5.0),
    16: (40.0, 5.0),
    17: (40.0, 5.0),
    21: (40.0, 5.0),
    22: (40.0, 5.0),
    23: (40.0, 5.0),
    24: (40.0, 5.0),
    25: (40.0, 5.0),
    26: (40.0, 5.0),
    27: (40.0, 5.0),
    # Keep the waist a bit stiffer than the limbs so the torso does not go slack immediately.
    31: (150.0, 4.0),
    51: (100.0, 2.0),
    52: (100.0, 2.0),
    53: (100.0, 2.0),
    54: (150.0, 4.0),
    55: (40.0, 2.0),
    56: (40.0, 2.0),
    61: (100.0, 2.0),
    62: (100.0, 2.0),
    63: (100.0, 2.0),
    64: (150.0, 4.0),
    65: (40.0, 2.0),
    66: (40.0, 2.0),
}
TIENKUNG_JOINT_STIFFNESS = {
    TIENKUNG_MOTOR_ID_TO_JOINT[motor_id]: gains[0]
    for motor_id, gains in TIENKUNG_PD_GAINS_BY_MOTOR_ID.items()
}
TIENKUNG_JOINT_DAMPING = {
    TIENKUNG_MOTOR_ID_TO_JOINT[motor_id]: gains[1]
    for motor_id, gains in TIENKUNG_PD_GAINS_BY_MOTOR_ID.items()
}


class TienkungMimicPrivCfg(HumanoidMimicCfg):
    # 定义天工 teacher / privileged 模仿环境的主体配置。
    class env(HumanoidMimicCfg.env):
        tar_motion_steps_priv = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                                 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
        tar_motion_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                            50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

        num_envs = 4096
        num_actions = TIENKUNG_DISTILL_NUM_ACTIONS
        obs_type = "priv"
        n_priv_latent = 4 + 1 + 2 * num_actions
        extra_critic_obs = 3
        n_priv = 0

        n_proprio = 3 + 2 + 3 * num_actions
        n_priv_mimic_obs = len(tar_motion_steps_priv) * (21 + num_actions + 3 * len(TIENKUNG_DISTILL_KEY_BODIES))
        n_mimic_obs_single = 6 + num_actions
        n_mimic_obs = len(tar_motion_steps) * n_mimic_obs_single
        n_priv_info = 3 + 3 + 4 + 3 * len(TIENKUNG_DISTILL_KEY_BODIES) + 2 + 4 + 1 + 2 * num_actions
        history_len = 10

        n_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info
        n_priv_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info
        num_observations = n_priv_obs_single
        num_privileged_obs = n_priv_obs_single

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
        pose_termination_dist = 0.7
        rand_reset = True

        track_root = False
        root_tracking_termination_dist = 2.0

        dof_err_w = (
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] +
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] +
            [1.0, 1.0, 1.0, 1.0] +
            [1.0] * 7 +
            [0.0] * 12 +
            [1.0] * 7 +
            [0.0] * 12
        )

        global_obs = False

    class terrain(HumanoidMimicCfg.terrain):
        mesh_type = "plane"
        height = [0, 0.00]
        horizontal_scale = 0.1
        static_friction = 1.5
        dynamic_friction = 1.5

    class init_state(HumanoidMimicCfg.init_state):
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

    class control(HumanoidMimicCfg.control):
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

        action_scale = 0.5
        decimation = 10

    class sim(HumanoidMimicCfg.sim):
        dt = 0.002

    class normalization(HumanoidMimicCfg.normalization):
        clip_actions = 5.0

    class asset(HumanoidMimicCfg.asset):
        file = "/data/shared_folder/GMR/assets/tienkung_ei/mjcf/tienkung_ei_v1.xml"

        torso_name: str = "pelvis"
        chest_name: str = "waist_yaw_link"
        thigh_name: str = "hip_pitch"
        shank_name: str = "knee_pitch"
        foot_name: str = "ankle_roll"
        waist_name: list = ["waist_yaw_link"]
        upper_arm_name: str = "shoulder_roll"
        lower_arm_name: str = "elbow"
        hand_name: list = ["wrist_roll_r_link", "wrist_roll_l_link"]

        feet_bodies = ["ankle_roll_l_link", "ankle_roll_r_link"]
        n_lower_body_dofs: int = 12
        active_dof_names = TIENKUNG_DISTILL_ACTIVE_DOF_NAMES

        penalize_contacts_on = ["shoulder", "elbow", "hip", "knee"]
        terminate_after_contacts_on = []
        dof_armature = []
        collapse_fixed_joints = False

    class rewards(HumanoidMimicCfg.rewards):
        regularization_names = []
        regularization_scale = 1.0
        regularization_scale_range = [0.8, 2.0]
        regularization_scale_curriculum = False
        regularization_scale_gamma = 0.0001

        class scales:
            tracking_joint_dof = 2.0
            tracking_joint_vel = 0.2
            tracking_root_translation_z = 1.0
            tracking_root_rotation = 1.0
            tracking_root_linear_vel = 1.0
            tracking_root_angular_vel = 1.0
            tracking_keybody_pos = 2.0
            tracking_keybody_pos_global = 2.0
            alive = 0.5
            feet_slip = -0.1
            feet_contact_forces = -5e-4
            feet_stumble = -1.25
            dof_pos_limits = -5.0
            dof_torque_limits = -1.0
            dof_vel = -1e-4
            dof_acc = -5e-8
            action_rate = -0.01
            feet_air_time = 5.0
            ang_vel_xy = -0.01
            ankle_dof_acc = -1e-7
            ankle_dof_vel = -2e-4

        min_dist = 0.1
        max_dist = 0.4
        max_knee_dist = 0.4
        feet_height_target = 0.2
        feet_air_time_target = 0.5
        only_positive_rewards = False
        tracking_sigma = 0.2
        tracking_sigma_ang = 0.125
        max_contact_force = 500
        soft_torque_limit = 0.95
        torque_safety_limit = 0.9
        termination_roll = 4.0
        termination_pitch = 4.0
        root_height_diff_threshold = 0.3
        # Distill/future-specific velocity tracking shaping used by
        # TienkungMimicDistill._reward_tracking_joint_vel().
        tracking_joint_vel_err_scale = 0.1

    class evaluations:
        tracking_joint_dof = True
        tracking_joint_vel = True
        tracking_root_translation = True
        tracking_root_rotation = True
        tracking_root_vel = True
        tracking_root_ang_vel = True
        tracking_keybody_pos = True
        tracking_root_pose_delta_local = True
        tracking_root_rotation_delta_local = True

    class domain_rand:
        domain_rand_general = True
        randomize_gravity = (True and domain_rand_general)
        gravity_rand_interval_s = 4
        gravity_range = (-0.1, 0.1)
        randomize_friction = (True and domain_rand_general)
        friction_range = [0.1, 2.0]
        randomize_base_mass = (True and domain_rand_general)
        added_mass_range = [-3.0, 3.0]
        randomize_base_com = (True and domain_rand_general)
        added_com_range = [-0.05, 0.05]
        push_robots = (True and domain_rand_general)
        push_interval_s = 4
        max_push_vel_xy = 1.0
        push_end_effector = (False and domain_rand_general)
        push_end_effector_interval_s = 2
        max_push_force_end_effector = 10.0
        randomize_motor = (True and domain_rand_general)
        motor_strength_range = [0.8, 1.2]
        action_delay = (True and domain_rand_general)
        action_buf_len = 8

    class noise(HumanoidMimicCfg.noise):
        add_noise = True
        noise_increasing_steps = 50_000

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.1
            lin_vel = 0.1
            ang_vel = 0.1
            gravity = 0.05
            imu = 0.1

    class motion(HumanoidMimicCfg.motion):
        motion_curriculum = True
        motion_curriculum_gamma = 0.01
        reset_consec_frames = 30
        key_bodies = TIENKUNG_DISTILL_KEY_BODIES
        upper_key_bodies = [
            "wrist_roll_l_link",
            "wrist_roll_r_link",
            "elbow_pitch_l_link",
            "elbow_pitch_r_link",
            "head_roll_link",
        ]
        sample_ratio = 1.0
        motion_smooth = True
        motion_decompose = False
        motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/tienkung_ei_train_fullbody_no_object.yaml"


class TienkungMimicStuCfg(TienkungMimicPrivCfg):
    # 定义只看学生观测并依赖历史编码的 student 配置。
    class env(TienkungMimicPrivCfg.env):
        obs_type = "student"
        tar_motion_steps = [1]
        n_mimic_obs_single = TienkungMimicPrivCfg.env.n_mimic_obs_single
        n_mimic_obs = len(tar_motion_steps) * n_mimic_obs_single
        n_proprio = TienkungMimicPrivCfg.env.n_proprio
        n_obs_single = n_mimic_obs + n_proprio
        num_observations = n_obs_single * (TienkungMimicPrivCfg.env.history_len + 1)


class TienkungMimicStuRLCfg(TienkungMimicPrivCfg):
    # 定义 student 强化学习 / DAgger 路径复用的观测与任务接口配置。
    class env(TienkungMimicPrivCfg.env):
        obs_type = "student"
        tar_motion_steps = [1]
        n_mimic_obs_single = TienkungMimicPrivCfg.env.n_mimic_obs_single
        n_mimic_obs = len(tar_motion_steps) * n_mimic_obs_single
        n_proprio = TienkungMimicPrivCfg.env.n_proprio
        n_obs_single = n_mimic_obs + n_proprio
        num_observations = n_obs_single * (TienkungMimicPrivCfg.env.history_len + 1)


class TienkungMimicPrivCfgPPO(HumanoidMimicCfgPPO):
    # 定义 privileged teacher 的 PPO 训练超参数和网络结构。
    seed = 1

    class runner(HumanoidMimicCfgPPO.runner):
        policy_class_name = "ActorCriticMimic"
        algorithm_class_name = "PPO"
        runner_class_name = "OnPolicyRunnerMimic"
        max_iterations = 1_000_002
        save_interval = 500
        experiment_name = "tienkung_priv_mimic"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

    class algorithm(HumanoidMimicCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005

    class policy(HumanoidMimicCfgPPO.policy):
        action_std = [0.7] * 12 + [0.4] * 4 + [0.5] * 14
        init_noise_std = 1.0
        obs_context_len = 11
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = "silu"
        layer_norm = True
        motion_latent_dim = 128


class TienkungMimicStuCfgDAgger(TienkungMimicPrivCfgPPO):
    # 定义 imitation-only student 的 DAgger 训练配置。
    seed = 1

    class teachercfg(TienkungMimicPrivCfgPPO):
        pass

    class runner(TienkungMimicPrivCfgPPO.runner):
        policy_class_name = "DAggerActor"
        algorithm_class_name = "DAgger"
        runner_class_name = "DAggerRunner"
        max_iterations = 1_000_002
        warm_iters = 100
        save_interval = 500
        experiment_name = "tienkung_stu_mimic"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        teacher_experiment_name = "tienkung_priv_placeholder"
        teacher_proj_name = "tienkung_priv_mimic"
        teacher_checkpoint = -1
        eval_student = False

    class algorithm:
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1e-4
        max_grad_norm = 1.0
        normalizer_update_iterations = 1000

    class policy:
        actor_hidden_dims = [1024, 1024, 512, 256]
        history_latent_dim = 128
        activation = "silu"


class TienkungMimicStuRLCfgDAgger(TienkungMimicStuRLCfg):
    # 定义 student 强化学习路径的 DaggerPPO 训练配置。
    seed = 1

    class teachercfg(TienkungMimicPrivCfgPPO):
        pass

    class runner(TienkungMimicPrivCfgPPO.runner):
        policy_class_name = "ActorCriticTeleop"
        algorithm_class_name = "DaggerPPO"
        runner_class_name = "OnPolicyDaggerRunner"
        max_iterations = 1_000_002
        warm_iters = 100
        save_interval = 500
        experiment_name = "tienkung_stu_rl"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        teacher_experiment_name = "tienkung_priv_placeholder"
        teacher_proj_name = "tienkung_priv_mimic"
        teacher_checkpoint = -1
        eval_student = False

    class algorithm(HumanoidMimicCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005
        dagger_coef_anneal_steps = 60000
        dagger_coef = 0.2
        dagger_coef_min = 0.1

    class policy(HumanoidMimicCfgPPO.policy):
        action_std = [0.7] * 12 + [0.4] * 4 + [0.5] * 14
        init_noise_std = 1.0
        obs_context_len = 11
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = "silu"
        layer_norm = True
        motion_latent_dim = 128
