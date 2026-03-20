# 旧 twist2 环境备份说明

## 1. 已备份环境

- named env: `C:/Users/MrRoam/.conda/envs/twist2`
- path env: `D:/DevTools/envs/twist2`

## 2. 备份文件

- `doc/env_backups/twist2_named_env_export.yml`
- `doc/env_backups/twist2_named_pip_list.txt`

## 3. 删除前判断

- named env 已确认是旧环境，且 editable 安装指向旧工程 `d:/desktop/motion-remap/twist2`
- path env 当前只确认目录存在，计划直接删除整个 Conda 环境目录
- 当前本地主环境已经切换建议为 `base`

## 4. 说明

- 如需恢复 named env，可基于 yml 重新创建
- 如需核对旧 pip 包，可查看 pip list 备份
