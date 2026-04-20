# TWIST2 项目说明

## 1. 项目定位
TWIST2 是一个“人形机器人全身遥操作 + 低层控制策略训练与部署 + 数据采集”的完整工程。当前仓库聚焦以下能力：
- 通过 Isaac Gym + `legged_gym` + `rsl_rl` 训练低层控制策略（G1 机器人）。
- 将策略导出为 ONNX，在 MuJoCo（sim2sim）和真机（sim2real）运行。
- 通过 Redis 将“高层动作流”（离线动作文件或 PICO 在线遥操作）与“低层策略控制”解耦。
- 支持数据录制（图像 + 状态 + 动作）用于后续学习。

代码与论文对应：
- 论文：`README.md` 中给出了 arXiv 与项目主页链接。
- 当前仓库已包含可直接部署的 ONNX 权重：`assets/ckpts/twist2_1017_20k.onnx`。

---

## 2. 顶层目录结构（按职责）
- `README.md`：安装与标准使用流程（官方入口）。
- `train.sh` / `eval.sh` / `to_onnx.sh`：训练、评估、导出入口脚本。
- `sim2sim.sh` / `sim2real.sh` / `run_motion_server.sh` / `teleop.sh` / `data_record.sh`：部署与数据采集快捷脚本。
- `gui.py` / `gui.sh`：可视化控制中心（按钮化启动多个服务）。
- `legged_gym/`：环境定义、任务注册、训练脚本（主训练框架）。
- `rsl_rl/`：强化学习算法与 actor-critic 网络（PPO/DAgger 等）。
- `deploy_real/`：低层控制服务（仿真/真机）、高层动作服务、数据录制、机器人接口。
- `pose/`：动作库/姿态工具（MotionLib 等）。
- `assets/`：机器人模型（URDF/XML/mesh）、示例动作、预训练 ONNX。
- `doc/`：实机部署与 teleop 补充文档。

---

## 3. 核心系统架构

### 3.1 双层控制解耦
TWIST2 把控制拆成两层，通过 Redis 通信：
- 高层（动作生成/遥操作）
  - 离线动作流：`deploy_real/server_motion_lib.py`
  - 在线 VR 遥操作：`deploy_real/xrobot_teleop_to_robot_w_hand.py`
- 低层（RL 策略执行）
  - 仿真：`deploy_real/server_low_level_g1_sim.py`
  - 真机：`deploy_real/server_low_level_g1_real.py`

低层服务周期性读取高层动作，并结合本体 proprio 历史输入 ONNX 策略，输出关节目标到仿真或真机。

### 3.2 Redis 数据通道（关键键）
在 `deploy_real/*.py` 中可以看到主要键约定：
- 状态发布：
  - `state_body_unitree_g1_with_hands`
  - `state_hand_left_unitree_g1_with_hands`
  - `state_hand_right_unitree_g1_with_hands`
- 动作发布：
  - `action_body_unitree_g1_with_hands`
  - `action_hand_left_unitree_g1_with_hands`
  - `action_hand_right_unitree_g1_with_hands`
  - `action_neck_unitree_g1_with_hands`

这种设计允许“高层服务”和“低层策略”独立替换。

### 3.3 观测维度（低层策略）
从 `server_low_level_g1_sim.py` / `server_low_level_g1_real.py` 可见：
- `n_mimic_obs = 35`
- `n_proprio = 92`
- `n_obs_single = 127`
- `history_len = 10`
- 总输入 `total_obs_size = 1402`

即：当前帧 + 历史堆叠 + 未来/目标相关观测，匹配 `g1_stu_future` 任务定义。

---

## 4. 训练与导出链路

### 4.1 训练入口
`train.sh` 实际调用：
- `legged_gym/legged_gym/scripts/train.py`
- 任务名固定为 `g1_stu_future`

任务注册位于：`legged_gym/legged_gym/envs/__init__.py`。
`g1_stu_future` 对应环境：
- 环境类：`G1MimicFuture`
- 配置类：`G1MimicStuFutureCfg` / `G1MimicStuFutureCfgDAgger`

### 4.2 训练配置重点
在 `g1_mimic_future_config.py` 中：
- `obs_type = 'student_future'`
- 动作数据配置文件：`legged_gym/motion_data_configs/twist2_dataset.yaml`
- 包含 reward、curriculum、未来帧观测等配置。

### 4.3 ONNX 导出
`to_onnx.sh` 调用 `save_onnx.py`，将训练得到的 `.pt` 导出为 ONNX，供 `deploy_real` 侧统一推理。

---

## 5. 部署与运行流程

### 5.1 sim2sim（推荐先打通）
1. 启动高层动作流（离线动作）：`bash run_motion_server.sh`
2. 启动低层仿真控制：`bash sim2sim.sh`

`sim2sim.sh` 默认加载：
- XML：`assets/g1/g1_sim2sim_29dof.xml`
- ONNX：`assets/ckpts/twist2_1017_20k.onnx`

### 5.2 sim2real（G1）
1. 机器人与电脑网线连接，电脑网卡配置到同网段（README 给出示例）。
2. 机器人进入开发/阻尼状态（遥控器组合键）。
3. 修改 `sim2real.sh` 的网卡名（`net=eno1` 仅为示例）。
4. 运行 `bash sim2real.sh`。

### 5.3 高层输入两种来源
- 离线动作：`bash run_motion_server.sh`（读取 `.pkl` 动作库）。
- 在线遥操作：`bash teleop.sh`（PICO + GMR + XRoboToolkit 链路）。

### 5.4 数据采集
`bash data_record.sh` 启动 `server_data_record.py`，可采集：
- 图像（vision client）
- 机器人状态（body/hand/neck）
- 高层动作（body/hand/neck）

---

## 6. 依赖与环境
项目采用双 conda 环境（见 `README.md`）：
- `twist2`（Python 3.8）：Isaac Gym 训练、部署、GUI。
- `gmr`（Python 3.10）：在线 retargeting/teleop。

核心依赖包括：`isaacgym`、`mujoco`、`onnxruntime-gpu`、`redis`、`customtkinter` 等。

---

## 7. GUI 控制中心说明
`gui.py` 可一键启动多个服务：
- 低层：`sim2sim.sh`、`sim2real.sh`
- 高层：`run_motion_server.sh`、`teleop.sh`
- 录制：`data_record.sh`
- 以及若干 G1 远端（SSH）相关命令

注意：GUI 中存在硬编码路径/主机别名（例如 SSH 主机 `g1`、某些绝对路径），在你的机器上可能需要先改配置再用。

---

## 8. 当前仓库需要重点关注的“本地化修改项”
以下内容通常需要按你的环境改掉：
- `sim2real.sh`：网卡名 `net=eno1`。
- `run_motion_server.sh`：`motion_file` 可改为你自己的 `.pkl`。
- `teleop.sh`：`redis_ip`、`actual_human_height`。
- `eval.sh`：默认 `motion_file` 指向作者本地绝对路径。
- `deploy_real/server_data_record.py`：默认 `data_folder` 是作者机器路径。
- `gui.py`：远端 SSH 别名和若干绝对路径命令。

---

## 9. 建议的最小可用验证顺序
1. 先在 `twist2` 环境启动 Redis，并跑通 `sim2sim.sh` + `run_motion_server.sh`。
2. 再切到 `gmr` 跑 `teleop.sh`，确认在线动作可驱动 sim2sim。
3. 最后切真机 `sim2real.sh`，并在安全吊装下逐步测试。

---

## 10. 关键文件索引（便于二次开发）
- 任务注册：`legged_gym/legged_gym/envs/__init__.py`
- 学生未来帧环境：`legged_gym/legged_gym/envs/g1/g1_mimic_future.py`
- 学生未来帧配置：`legged_gym/legged_gym/envs/g1/g1_mimic_future_config.py`
- 训练脚本：`legged_gym/legged_gym/scripts/train.py`
- ONNX 导出：`legged_gym/legged_gym/scripts/save_onnx.py`
- 低层仿真控制：`deploy_real/server_low_level_g1_sim.py`
- 低层真机控制：`deploy_real/server_low_level_g1_real.py`
- 高层动作服务：`deploy_real/server_motion_lib.py`
- 在线遥操作入口：`deploy_real/xrobot_teleop_to_robot_w_hand.py`
- 数据录制：`deploy_real/server_data_record.py`
- GUI：`gui.py`

