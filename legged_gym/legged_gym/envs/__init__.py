# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot

from .base.humanoid import Humanoid
from .base.humanoid_mimic import HumanoidMimic

from .g1.g1_mimic_config import G1MimicCfg, G1MimicCfgPPO
from .g1.g1_mimic import G1Mimic

# DeepMimic (for teleoperation)
from .g1.g1_mimic_distill import G1MimicDistill
from .g1.g1_mimic_distill_config import G1MimicPrivCfg, G1MimicPrivCfgPPO, G1MimicStuCfg, G1MimicStuCfgDAgger
from .g1.g1_mimic_distill_config import G1MimicStuRLCfg, G1MimicStuRLCfgDAgger

from .g1.g1_mimic_future import G1MimicFuture
from .g1.g1_mimic_future_config import G1MimicStuFutureCfg, G1MimicStuFutureCfgDAgger

from .tienkung_2_0.tienkung_mimic import TienkungMimic
from .tienkung_2_0.tienkung_mimic_config import TienkungMimicCfg, TienkungMimicCfgPPO
from .tienkung_2_0.tienkung_mimic_distill import TienkungMimicDistill
from .tienkung_2_0.tienkung_mimic_distill_config import (
    TienkungMimicPrivCfg,
    TienkungMimicPrivCfgPPO,
    TienkungMimicStuCfg,
    TienkungMimicStuCfgDAgger,
    TienkungMimicStuRLCfg,
    TienkungMimicStuRLCfgDAgger,
)
from .tienkung_2_0.tienkung_mimic_future import TienkungMimicFuture
from .tienkung_2_0.tienkung_mimic_future_config import (
    TienkungMimicStuFutureCfg,
    TienkungMimicStuFutureCfgDAgger,
)
from .tienkung_2_0.tienkung_mimic_distill_cc_config import (
    TienkungMimicPrivCCCfg,
    TienkungMimicPrivCCCfgPPO,
    TienkungMimicStuCCCfg,
    TienkungMimicStuCCCfgDAgger,
    TienkungMimicStuRLCCCfg,
    TienkungMimicStuRLCCCfgDAgger,
)
from .tienkung_2_0.tienkung_mimic_future_cc_config import (
    TienkungMimicStuFutureCCCfg,
    TienkungMimicStuFutureCCCfgDAgger,
)
from .tienkung_2_0.tienkung_mimic_future_cc_stage1_config import (
    TienkungMimicStuFutureCCStage1Cfg,
    TienkungMimicStuFutureCCStage1CfgDAgger,
)

from legged_gym.gym_utils.task_registry import task_registry


# DeepMimic (for teleoperation)
task_registry.register("g1_mimic", G1Mimic, G1MimicCfg(), G1MimicCfgPPO())
task_registry.register("g1_stu_mimic", G1MimicDistill, G1MimicStuCfg(), G1MimicStuCfgDAgger())
task_registry.register("g1_priv_mimic", G1MimicDistill, G1MimicPrivCfg(), G1MimicPrivCfgPPO())
task_registry.register("g1_stu_rl", G1MimicDistill, G1MimicStuRLCfg(), G1MimicStuRLCfgDAgger())
task_registry.register("g1_stu_future", G1MimicFuture, G1MimicStuFutureCfg(), G1MimicStuFutureCfgDAgger())
task_registry.register("tienkung_2_0_mimic", TienkungMimic, TienkungMimicCfg(), TienkungMimicCfgPPO())
task_registry.register("tienkung_priv_mimic", TienkungMimicDistill, TienkungMimicPrivCfg(), TienkungMimicPrivCfgPPO())
task_registry.register("tienkung_stu_mimic", TienkungMimicDistill, TienkungMimicStuCfg(), TienkungMimicStuCfgDAgger())
task_registry.register("tienkung_stu_rl", TienkungMimicDistill, TienkungMimicStuRLCfg(), TienkungMimicStuRLCfgDAgger())
task_registry.register("tienkung_stu_future", TienkungMimicFuture, TienkungMimicStuFutureCfg(), TienkungMimicStuFutureCfgDAgger())
task_registry.register("tienkung_priv_mimic_cc", TienkungMimicDistill, TienkungMimicPrivCCCfg(), TienkungMimicPrivCCCfgPPO())
task_registry.register("tienkung_stu_mimic_cc", TienkungMimicDistill, TienkungMimicStuCCCfg(), TienkungMimicStuCCCfgDAgger())
task_registry.register("tienkung_stu_rl_cc", TienkungMimicDistill, TienkungMimicStuRLCCCfg(), TienkungMimicStuRLCCCfgDAgger())
task_registry.register("tienkung_stu_future_cc", TienkungMimicFuture, TienkungMimicStuFutureCCCfg(), TienkungMimicStuFutureCCCfgDAgger())
task_registry.register("tienkung_stu_future_cc_stage1", TienkungMimicFuture, TienkungMimicStuFutureCCStage1Cfg(), TienkungMimicStuFutureCCStage1CfgDAgger())
