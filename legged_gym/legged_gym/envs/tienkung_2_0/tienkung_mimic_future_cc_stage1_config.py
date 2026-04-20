from legged_gym import LEGGED_GYM_ROOT_DIR

from .tienkung_mimic_future_cc_config import (
    TienkungMimicStuFutureCCCfg,
    TienkungMimicStuFutureCCCfgDAgger,
)


class TienkungMimicStuFutureCCStage1Cfg(TienkungMimicStuFutureCCCfg):
    """A cleaner locomotion-first future route on top of the cc asset.

    Stage 1 intentionally keeps a small reward set:
    - strong pose/root tracking terms so the policy learns the reference first
    - light smoothness terms so the policy does not immediately explode
    - no contact/slip/air-time shaping yet, so failures stay easier to diagnose
    """

    class motion(TienkungMimicStuFutureCCCfg.motion):
        # Stage 1 uses the curated locomotion-only baseline dataset.
        motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/tienkung_ei_train30.yaml"

    class init_state(TienkungMimicStuFutureCCCfg.init_state):
        default_joint_angles = dict(TienkungMimicStuFutureCCCfg.init_state.default_joint_angles)
        # Locomotion-centered neutral pose estimated from tienkung_ei_train30 start windows.
        default_joint_angles.update(
            {
                "hip_roll_l_joint": -0.0882,
                "hip_pitch_l_joint": -0.1274,
                "hip_yaw_l_joint": 0.0790,
                "knee_pitch_l_joint": 0.3272,
                "ankle_pitch_l_joint": -0.2256,
                "ankle_roll_l_joint": 0.0496,
                "hip_roll_r_joint": -0.0110,
                "hip_pitch_r_joint": -0.1781,
                "hip_yaw_r_joint": -0.0907,
                "knee_pitch_r_joint": 0.3165,
                "ankle_pitch_r_joint": -0.1545,
                "ankle_roll_r_joint": 0.0530,
                "waist_yaw_joint": -0.0061,
            }
        )

    class rewards(TienkungMimicStuFutureCCCfg.rewards):
        class scales(TienkungMimicStuFutureCCCfg.rewards.scales):
            # Stage 1 v3: keep early long episodes net-positive, while closing
            # the "stand + spin/slide" loophole with light global/slip terms.
            # We keep these terms intentionally moderate so they guide behavior
            # without reintroducing penalty-dominant collapse.
            #
            # NOTE: tracking_joint_vel here is the outer reward weight. The
            # inner tracking-joint-velocity error shaping is overridden in
            # TienkungMimicDistill to avoid early saturation.
            #
            # Stage 1 v2 baseline:
            # - tracking_keybody_pos_global = 0.0
            # - feet_slip = 0.0
            # - feet_contact_forces = 0.0
            # - ang_vel_xy = 0.0
            #
            # Stage 1 v3:
            # - tracking_keybody_pos_global = 1.0
            # - feet_slip = -0.05
            # - feet_contact_forces = -1e-4
            # - ang_vel_xy = -0.01
            #
            # This should reduce in-place spinning/sliding while keeping
            # locomotion-tracking incentives dominant.
            # make early long episodes net-positive enough to keep
            # the policy pursuing the reference instead of dying early to dodge penalties.
            tracking_joint_dof = 3.0
            tracking_joint_vel = 0.2
            tracking_root_translation_z = 1.5
            tracking_root_rotation = 1.0
            tracking_root_linear_vel = 1.5
            tracking_root_angular_vel = 1.0
            tracking_keybody_pos = 3.0
            alive = 0.5

            # Restore light global/contact-style shaping to discourage
            # "stand + spin/slide" local optima seen in play-time behavior.
            tracking_keybody_pos_global = 1.0
            feet_slip = -0.05
            feet_contact_forces = -1e-4
            feet_stumble = 0.0
            feet_air_time = 0.0
            dof_pos_limits = 0.0
            dof_torque_limits = 0.0
            ang_vel_xy = -0.01

            # Further relax execution penalties so they stop dominating
            # before the tracking terms have a chance to grow.
            dof_vel = -5e-5
            dof_acc = -5e-8
            action_rate = -0.02
            ankle_dof_vel = -1e-4
            ankle_dof_acc = -5e-8


class TienkungMimicStuFutureCCStage1CfgDAgger(TienkungMimicStuFutureCCCfgDAgger):
    class runner(TienkungMimicStuFutureCCCfgDAgger.runner):
        experiment_name = "tienkung_stu_future_cc_stage1"
        teacher_proj_name = "tienkung_priv_mimic_cc"
