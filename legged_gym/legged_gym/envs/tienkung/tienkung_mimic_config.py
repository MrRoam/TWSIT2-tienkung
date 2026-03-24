from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfg, HumanoidMimicCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR


class TienkungMimicCfg(HumanoidMimicCfg):
    class env(HumanoidMimicCfg.env):
        tar_motion_steps_priv = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                         50, 55, 60, 65, 70, 75, 80, 85, 90, 95,]
        
        num_envs = 2048
        num_actions = 30
        n_priv = 0
        n_mimic_obs = 9 + num_actions
        n_priv_mimic_obs = len(tar_motion_steps_priv) * n_mimic_obs
        n_proprio = len(tar_motion_steps_priv) * n_mimic_obs + 3 + 2 + 3*num_actions
        n_priv_latent = 4 + 1 + 2*num_actions
        extra_critic_obs = 3
        history_len = 10
        
        num_observations = n_proprio + n_priv_latent + history_len*n_proprio + n_priv + extra_critic_obs 
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
        pose_termination_dist = 2.0 # Relaxed from 0.7 to 2.0 to survive initial pose mismatch
        root_tracking_termination_dist = 2.0 # Relaxed from 0.8 to 2.0
        rand_reset = True
        track_root = False
        dof_err_w = [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 0.7, 0.7, 0.5,
            1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.6,
            1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.6,
        ]
        
        global_obs = False
    
    class terrain(HumanoidMimicCfg.terrain):
        mesh_type = 'trimesh'
        height = [0, 0.00]
        horizontal_scale = 0.1
    
    class init_state(HumanoidMimicCfg.init_state):
        pos = [0, 0, 0.80]
        default_joint_angles = {
            'hip_roll_l_joint': 0.0,
            'hip_yaw_l_joint': 0.0,
            'hip_pitch_l_joint': -0.2,
            'knee_pitch_l_joint': 0.4,
            'ankle_pitch_l_joint': -0.2,
            'ankle_roll_l_joint': 0.0,
            'hip_roll_r_joint': 0.0,
            'hip_yaw_r_joint': 0.0,
            'hip_pitch_r_joint': -0.2,
            'knee_pitch_r_joint': 0.4,
            'ankle_pitch_r_joint': -0.2,
            'ankle_roll_r_joint': 0.0,
            'waist_yaw_joint': 0.0,
            'head_yaw_joint': 0.0,
            'head_pitch_joint': 0.0,
            'head_roll_joint': 0.0,
            'shoulder_pitch_l_joint': 0.0,
            'shoulder_roll_l_joint': 0.0,
            'shoulder_yaw_l_joint': 0.0,
            'elbow_pitch_l_joint': 0.0,
            'elbow_yaw_l_joint': 0.0,
            'wrist_pitch_l_joint': 0.0,
            'wrist_roll_l_joint': 0.0,
            'shoulder_pitch_r_joint': 0.0,
            'shoulder_roll_r_joint': 0.0,
            'shoulder_yaw_r_joint': 0.0,
            'elbow_pitch_r_joint': 0.0,
            'elbow_yaw_r_joint': 0.0,
            'wrist_pitch_r_joint': 0.0,
            'wrist_roll_r_joint': 0.0,
        }
    
    class control(HumanoidMimicCfg.control):
        stiffness = {'hip': 100,
                     'knee': 150,
                     'ankle': 40,
                     'waist': 150,
                     'shoulder': 40,
                     'elbow': 40,
                     'wrist': 40,
                     'hand': 10,
                     'thumb': 10,
                     'index': 10,
                     'middle': 10,
                     'ring': 10,
                     'pinky': 10,
                     'left': 40,
                     'right': 40,
                     }
        damping = {'hip': 5,
                   'knee': 10,
                   'ankle': 4,
                   'waist': 10,
                   'shoulder': 10,
                   'elbow': 10,
                   'wrist': 10,
                   'hand': 2,
                   'thumb': 1,
                   'index': 1,
                   'middle': 1,
                   'ring': 1,
                   'pinky': 1,
                   'left': 10,
                   'right': 10,
                   }
        action_scale = 0.5
        decimation = 10
    
    class sim(HumanoidMimicCfg.sim):
        dt = 0.002
        
    class normalization(HumanoidMimicCfg.normalization):
        clip_actions = 5.0
    
    class asset(HumanoidMimicCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/../assets/Tienkung/urdf/walker_tienkung_ei.urdf'
        
        torso_name: str = 'pelvis'
        chest_name: str = 'waist_yaw_link'
        
        thigh_name: str = 'hip_pitch'
        shank_name: str = 'knee_pitch'
        foot_name: str = 'ankle_roll'
        waist_name: list = ['waist_yaw_link']
        upper_arm_name: str = 'shoulder_roll'
        lower_arm_name: str = 'elbow_pitch'
        hand_name: list = ['wrist_roll_l_link', 'wrist_roll_r_link']
        
        feet_bodies = ['ankle_roll_l_link', 'ankle_roll_r_link']
        n_lower_body_dofs: int = 12

        penalize_contacts_on = ["shoulder", "elbow", "hip", "knee"]
        terminate_after_contacts_on = ['pelvis']
        
        dof_armature = [0.0103, 0.0103, 0.0251, 0.0251, 0.003597, 0.003597] * 2 + [0.0103] * 4 + [0.003597] * 14

        collapse_fixed_joints = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
    
    class rewards(HumanoidMimicCfg.rewards):
        regularization_names = [
                        "feet_stumble",
                        "feet_contact_forces",
                        "lin_vel_z",
                        "ang_vel_xy",
                        "orientation",
                        "dof_pos_limits",
                        "dof_torque_limits",
                        "collision",
                        "torque_penalty",
                        "dof_acc",
                        "dof_vel",
                        "action_rate",
                        ]
        regularization_scale = 1.0
        regularization_scale_range = [0.8, 2.0]
        regularization_scale_curriculum = False
        regularization_scale_gamma = 0.0001
        class scales:
            tracking_joint_dof = 0.6
            tracking_joint_vel = 0.2
            tracking_root_translation = 0.6
            tracking_root_rotation = 0.6
            tracking_root_vel = 1.0
            tracking_keybody_pos = 2.0
            
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
            
            # ankle_dof_acc = -5e-8 * 2
            # ankle_dof_vel = -1e-4 * 2

        min_dist = 0.1
        max_dist = 0.4
        max_knee_dist = 0.4
        target_feet_height = 0.07
        only_positive_rewards = False
        tracking_sigma = 0.2
        tracking_sigma_ang = 0.125
        max_contact_force = 350
        soft_torque_limit = 0.95
        torque_safety_limit = 0.9
        
        termination_roll = 1.5
        termination_pitch = 1.5
        root_height_diff_threshold = 0.3

    class domain_rand:
        domain_rand_general = True
        
        randomize_gravity = (True and domain_rand_general)
        gravity_rand_interval_s = 4
        gravity_range = (-0.1, 0.1)
        
        randomize_friction = (True and domain_rand_general)
        friction_range = [0.1, 2.]
        
        randomize_base_mass = (True and domain_rand_general)
        added_mass_range = [-3., 3]
        
        randomize_base_com = (True and domain_rand_general)
        added_com_range = [-0.05, 0.05]
        
        push_robots = (True and domain_rand_general)
        push_interval_s = 4
        max_push_vel_xy = 1.0
        
        push_end_effector = (True and domain_rand_general)
        push_end_effector_interval_s = 2
        max_push_force_end_effector = 20.0

        randomize_motor = (True and domain_rand_general)
        motor_strength_range = [0.8, 1.2]

        action_delay = (True and domain_rand_general)
        action_buf_len = 8
    
    class noise(HumanoidMimicCfg.noise):
        add_noise = True
        noise_increasing_steps = 3000
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.1
            lin_vel = 0.1
            ang_vel = 0.1
            gravity = 0.05
            imu = 0.1
    
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

    class motion(HumanoidMimicCfg.motion):
        motion_curriculum = True
        motion_curriculum_gamma = 0.01
        key_bodies = ["pelvis", "waist_yaw_link",
                      "hip_roll_l_link", "hip_pitch_l_link", "hip_yaw_l_link",
                      "knee_pitch_l_link", "ankle_pitch_l_link", "ankle_roll_l_link",
                      "hip_roll_r_link", "hip_pitch_r_link", "hip_yaw_r_link",
                      "knee_pitch_r_link", "ankle_pitch_r_link", "ankle_roll_r_link",
                      "shoulder_pitch_l_link", "shoulder_roll_l_link", "shoulder_yaw_l_link",
                      "elbow_pitch_l_link", "elbow_yaw_l_link", "wrist_roll_l_link",
                      "shoulder_pitch_r_link", "shoulder_roll_r_link", "shoulder_yaw_r_link",
                      "elbow_pitch_r_link", "elbow_yaw_r_link", "wrist_roll_r_link"]
        upper_key_bodies = []
        
        motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/lafan1_tienkung_train.yaml"
        reset_consec_frames = 30
 

class TienkungMimicCfgPPO(HumanoidMimicCfgPPO):
    seed = 1
    class runner(HumanoidMimicCfgPPO.runner):
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'PPO'
        runner_class_name = 'OnPolicyRunnerMimic'
        max_iterations = 3000

        save_interval = 200
        experiment_name = 'tienkung_mimic'
        run_name = ''
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
        init_noise_std = 0.8
        # obs_context_len = 11  # Not used in ActorCriticMimic
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = 'silu'
        # priv_encoder_dims = [] # Not used
        # tanh_encoder_output = False # Not used
