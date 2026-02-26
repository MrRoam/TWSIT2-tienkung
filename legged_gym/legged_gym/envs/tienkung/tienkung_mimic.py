from legged_gym.envs.base.humanoid_mimic import HumanoidMimic
from .tienkung_mimic_config import TienkungMimicCfg

class TienkungMimic(HumanoidMimic):
    def __init__(self, cfg: TienkungMimicCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)