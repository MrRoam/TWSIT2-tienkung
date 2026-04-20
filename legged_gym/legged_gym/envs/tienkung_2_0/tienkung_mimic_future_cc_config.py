from .tienkung_mimic_future_config import (
    TienkungMimicStuFutureCfg,
    TienkungMimicStuFutureCfgDAgger,
)

from .tienkung_mimic_distill_cc_config import TIENKUNG_CUSTOM_COLLISION_XML


class TienkungMimicStuFutureCCCfg(TienkungMimicStuFutureCfg):
    class asset(TienkungMimicStuFutureCfg.asset):
        file = TIENKUNG_CUSTOM_COLLISION_XML


class TienkungMimicStuFutureCCCfgDAgger(TienkungMimicStuFutureCfgDAgger):
    class runner(TienkungMimicStuFutureCfgDAgger.runner):
        experiment_name = "tienkung_stu_future_cc"
        teacher_proj_name = "tienkung_priv_mimic_cc"
