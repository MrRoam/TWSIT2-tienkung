from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfgPPO
from .tienkung_mimic_distill_config import (
    TienkungMimicPrivCfg,
    TienkungMimicPrivCfgPPO,
    TienkungMimicStuRLCfg,
)

'''
① 这份文件把 distill/student 配置扩展成 future-student 任务
核心是给 student 额外追加 future obs，同时尽量复用 distill 已经验证过的环境、奖励和随机化设置。
② TienkungMimicStuFutureCfg 只改 student_future 真正需要变化的部分
它主要调整 obs_type、future 帧索引、总观测维度、动作幅度和少量奖励/随机化参数。
③ TienkungMimicStuFutureCfgDAgger 定义 future student 的训练器
它把 teacher、runner、algorithm 和 policy 连接起来，支持带 future encoder 的 DaggerPPO 训练。
④ future 配置仍然依赖 distill teacher
也就是说 privileged teacher 的观测组织、motion 数据和 active dof 定义都继承自 distill_config，而不是重新定义一套。
⑤ 所以这份文件的职责不是重新建环境，而是为 student_future 指定“看什么、怎么训”
环境动力学主体仍由 distill / mimic 主干负责，这里主要改观测接口和训练超参。
'''


TAR_MOTION_STEPS_FUTURE = [0]


class TienkungMimicStuFutureCfg(TienkungMimicStuRLCfg):
    # 定义带 future obs 的 student 环境配置。
    class env(TienkungMimicStuRLCfg.env):
        obs_type = "student_future"
        tar_motion_steps = [0]
        tar_motion_steps_future = TAR_MOTION_STEPS_FUTURE
        pose_termination_dist = 0.85
        root_tracking_termination_dist = 2.5

        n_mimic_obs_single = 6 + TienkungMimicPrivCfg.env.num_actions
        n_mimic_obs = len(tar_motion_steps) * n_mimic_obs_single
        n_proprio = TienkungMimicPrivCfg.env.n_proprio

        n_future_obs_single = 6 + TienkungMimicPrivCfg.env.num_actions
        n_future_obs = len(tar_motion_steps_future) * n_future_obs_single

        n_obs_single = n_mimic_obs + n_proprio
        num_observations = n_obs_single * (TienkungMimicPrivCfg.env.history_len + 1) + n_future_obs

    class control(TienkungMimicPrivCfg.control):
        # Keep the action scale modest while using a softer, more G1-like PD range.
        action_scale = 0.4

    class rewards(TienkungMimicPrivCfg.rewards):
        root_height_diff_threshold = 0.3

        class scales(TienkungMimicPrivCfg.rewards.scales):
            # Align with g1 future first, then add TianGong-specific velocity smoothing.
            action_rate = -0.05
            dof_vel = -2e-4
            dof_acc = -1e-7
            ankle_dof_vel = -4e-4
            ankle_dof_acc = -2e-7

    class domain_rand(TienkungMimicPrivCfg.domain_rand):
        # Teacherless future is especially sensitive to delay at the first control step.
        action_delay = False


class TienkungMimicStuFutureCfgDAgger(TienkungMimicStuFutureCfg):
    # 定义带 future encoder 的 student_future DaggerPPO 训练配置。
    seed = 1

    class teachercfg(TienkungMimicPrivCfgPPO):
        pass

    class runner(TienkungMimicPrivCfgPPO.runner):
        policy_class_name = "ActorCriticFuture"
        algorithm_class_name = "DaggerPPO"
        runner_class_name = "OnPolicyDaggerRunner"
        max_iterations = 30_001
        warm_iters = 100
        save_interval = 500
        experiment_name = "tienkung_stu_future"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        # Align with the repo's default g1_stu_future path: future can start without
        # an explicit teacher checkpoint and OnPolicyDaggerRunner will skip KL/teacher load.
        teacher_experiment_name = "None"
        teacher_proj_name = "tienkung_priv_mimic"
        teacher_checkpoint = -1
        eval_student = False
        save_to_wandb = True

    class algorithm(HumanoidMimicCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [0.6, 0.25, 1500, 1000]
        entropy_coef = 0.005
        dagger_coef_anneal_steps = 60000
        dagger_coef = 0.2
        dagger_coef_min = 0.1
        future_weight_decay = 0.95
        future_consistency_loss = 0.1

    class policy(HumanoidMimicCfgPPO.policy):
        action_std = [
            0.35, 0.45, 0.35, 0.40, 0.25, 0.20,
            0.35, 0.45, 0.35, 0.40, 0.25, 0.20,
            0.15, 0.10, 0.10, 0.10,
            0.30, 0.25, 0.25, 0.25, 0.20, 0.15, 0.15,
            0.30, 0.25, 0.25, 0.25, 0.20, 0.15, 0.15,
        ]
        init_noise_std = 0.6
        obs_context_len = 11
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = "silu"
        layer_norm = True
        motion_latent_dim = 128
        future_encoder_dims = [256, 256, 128]
        future_attention_heads = 4
        future_dropout = 0.1
        temporal_embedding_dim = 64
        future_latent_dim = 128
        num_future_steps = len(TAR_MOTION_STEPS_FUTURE)
        num_future_observations = TienkungMimicStuFutureCfg.env.n_future_obs
        num_experts = 4
        expert_hidden_dims = [256, 128]
        gating_hidden_dim = 128
        moe_temperature = 1.0
        moe_topk = None
        load_balancing_loss_weight = 0.01
