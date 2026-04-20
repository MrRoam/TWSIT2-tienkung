from .tienkung_mimic_distill_config import (
    TienkungMimicPrivCfg,
    TienkungMimicPrivCfgPPO,
    TienkungMimicStuCfg,
    TienkungMimicStuCfgDAgger,
    TienkungMimicStuRLCfg,
    TienkungMimicStuRLCfgDAgger,
)


TIENKUNG_CUSTOM_COLLISION_XML = "/data/shared_folder/GMR/assets/tienkung_ei/mjcf/tienkung_ei_custom_collision.xml"


class TienkungMimicPrivCCCfg(TienkungMimicPrivCfg):
    class asset(TienkungMimicPrivCfg.asset):
        file = TIENKUNG_CUSTOM_COLLISION_XML


class TienkungMimicStuCCCfg(TienkungMimicStuCfg):
    class asset(TienkungMimicStuCfg.asset):
        file = TIENKUNG_CUSTOM_COLLISION_XML


class TienkungMimicStuRLCCCfg(TienkungMimicStuRLCfg):
    class asset(TienkungMimicStuRLCfg.asset):
        file = TIENKUNG_CUSTOM_COLLISION_XML


class TienkungMimicPrivCCCfgPPO(TienkungMimicPrivCfgPPO):
    class runner(TienkungMimicPrivCfgPPO.runner):
        experiment_name = "tienkung_priv_mimic_cc"


class TienkungMimicStuCCCfgDAgger(TienkungMimicStuCfgDAgger):
    class runner(TienkungMimicStuCfgDAgger.runner):
        experiment_name = "tienkung_stu_mimic_cc"
        teacher_proj_name = "tienkung_priv_mimic_cc"


class TienkungMimicStuRLCCCfgDAgger(TienkungMimicStuRLCfgDAgger):
    class runner(TienkungMimicStuRLCfgDAgger.runner):
        experiment_name = "tienkung_stu_rl_cc"
        teacher_proj_name = "tienkung_priv_mimic_cc"
