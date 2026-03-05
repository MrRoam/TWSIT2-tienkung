# TWIST2 路径树（简洁版）

## 云端（AutoDL）关键路径

```text
/root
├─ TWIST2
│  ├─ legged_gym
│  │  ├─ legged_gym
│  │  │  ├─ scripts
│  │  │  │  ├─ train.py
│  │  │  │  ├─ play.py
│  │  │  │  └─ benchmark_policy.py
│  │  │  └─ envs
│  │  │     ├─ g1
│  │  │     └─ tienkung
│  │  ├─ logs
│  │  │  └─ tienkung
│  │  │     └─ v4_file_logging
│  │  │        ├─ model_600.pt
│  │  │        └─ evaluations.txt
│  │  ├─ motion_data_configs
│  │  └─ setup.py
│  ├─ rsl_rl
│  │  └─ rsl_rl
│  │     └─ runners
│  │        └─ on_policy_runner_mimic.py
│  ├─ assets
│  └─ train.sh
└─ isaacgym
```

## 本地（Windows）关键路径

```text
D:\Desktop\TWIST2
├─ legged_gym
│  ├─ legged_gym
│  │  ├─ scripts
│  │  │  └─ benchmark_policy.py
│  │  └─ envs
│  │     └─ tienkung
│  └─ logs
│     └─ tienkung
│        └─ v4_file_logging
├─ rsl_rl
│  └─ rsl_rl
│     └─ runners
│        └─ on_policy_runner_mimic.py
└─ structure.md
```

## 云端常用命令（路径不易错版）

```bash
# 1) 从 /root 运行
python TWIST2/legged_gym/legged_gym/scripts/benchmark_policy.py --task tienkung_mimic --proj_name tienkung --exptid v4_file_logging --checkpoint 600

# 2) 从 /root/TWIST2 运行
python legged_gym/legged_gym/scripts/benchmark_policy.py --task tienkung_mimic --proj_name tienkung --exptid v4_file_logging --checkpoint 600
```
