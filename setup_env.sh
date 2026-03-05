#!/bin/bash
# TWIST2 环境变量自动配置脚本
# 用法: bash setup_env.sh

set -e  # 遇到错误立即退出

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检测项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

if [ ! -d "$PROJECT_ROOT/legged_gym" ]; then
    echo_error "未找到 legged_gym 目录，请确认脚本在项目根目录"
    exit 1
fi

echo_info "检测到项目根目录: $PROJECT_ROOT"

# 检测 Shell 类型
SHELL_NAME="$(basename "$SHELL")"
if [ "$SHELL_NAME" = "bash" ]; then
    SHELL_RC="$HOME/.bashrc"
elif [ "$SHELL_NAME" = "zsh" ]; then
    SHELL_RC="$HOME/.zshrc"
else
    echo_warn "未知的 Shell 类型: $SHELL_NAME"
    SHELL_RC="$HOME/.bashrc"
fi

echo_info "检测到 Shell: $SHELL_NAME"
echo_info "配置文件: $SHELL_RC"

# 备份原配置文件
BACKUP_FILE="${SHELL_RC}.backup.$(date +%Y%m%d_%H%M%S)"
cp "$SHELL_RC" "$BACKUP_FILE"
echo_info "已备份原配置到: $BACKUP_FILE"

# 环境变量配置
ENV_CONFIG="
# ===== TWIST2 Project Environment =====
# Added by setup_env.sh on $(date)
export TWIST2_ROOT=$PROJECT_ROOT
export PYTHONPATH=\$TWIST2_ROOT:\$PYTHONPATH

# IsaacGym (如果存在)
if [ -d \"\$HOME/isaacgym\" ]; then
    export ISAACGYM_PATH=\$HOME/isaacgym
    export PYTHONPATH=\$ISAACGYM_PATH/python/isaacgym:\$PYTHONPATH
fi

# CUDA (如果存在)
if [ -d \"/usr/local/cuda\" ]; then
    export CUDA_HOME=/usr/local/cuda
    export LD_LIBRARY_PATH=\${CUDA_HOME}/lib64:\$LD_LIBRARY_PATH
    export PATH=\${CUDA_HOME}/bin:\$PATH
fi
# ===== END TWIST2 Environment =====
"

# 检查是否已经配置过
if grep -q "TWIST2 Project Environment" "$SHELL_RC"; then
    echo_warn "检测到已有 TWIST2 环境变量配置"
    read -p "是否要覆盖？(y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo_info "跳过配置"
        exit 0
    fi

    # 删除旧的配置
    sed -i '/# ===== TWIST2 Project Environment =====/,/# ===== END TWIST2 Environment =====/d' "$SHELL_RC"
    echo_info "已删除旧配置"
fi

# 添加新配置
echo "$ENV_CONFIG" >> "$SHELL_RC"
echo_info "已添加环境变量配置到 $SHELL_RC"

# 立即生效
export TWIST2_ROOT="$PROJECT_ROOT"
export PYTHONPATH="$TWIST2_ROOT:$PYTHONPATH"

if [ -d "$HOME/isaacgym" ]; then
    export ISAACGYM_PATH="$HOME/isaacgym"
    export PYTHONPATH="$ISAACGYM_PATH/python/isaacgym:$PYTHONPATH"
fi

if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"
    export PATH="${CUDA_HOME}/bin:$PATH"
fi

# 验证配置
echo ""
echo_info "===== 环境变量验证 ====="
echo "TWIST2_ROOT = $TWIST2_ROOT"
echo "PYTHONPATH = $PYTHONPATH"
[ -n "$ISAACGYM_PATH" ] && echo "ISAACGYM_PATH = $ISAACGYM_PATH"
[ -n "$CUDA_HOME" ] && echo "CUDA_HOME = $CUDA_HOME"
echo_info "========================"
echo ""

# 测试 Python 导入
echo_info "测试 Python 模块导入..."
python3 -c "
import sys
import os

# 检查项目路径
if '$PROJECT_ROOT' in sys.path:
    print('  ✅ 项目路径已添加到 Python 搜索路径')
else:
    print('  ❌ 项目路径未添加到 Python 搜索路径')

# 检查 pose 模块
try:
    from pose.utils import torch_utils
    print('  ✅ pose.utils.torch_utils 导入成功')
except ImportError as e:
    print(f'  ❌ pose 模块导入失败: {e}')
    print('  提示: 请确保 pose 目录存在于项目根目录')

# 检查 IsaacGym
try:
    from isaacgym import gymapi
    print('  ✅ isaacgym 导入成功')
except ImportError:
    print('  ❌ isaacgym 导入失败')
"

echo ""
echo_info "✅ 环境配置完成！"
echo ""
echo_warn "请执行以下命令使配置在当前终端生效："
echo "  source $SHELL_RC"
echo ""
echo_warn "或者重新打开新终端，配置会自动生效"
