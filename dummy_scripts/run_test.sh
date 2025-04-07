#!/bin/bash

# 设置 CUDA 相关环境变量
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_MULTIPROCESSING=0
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 检查 conda 是否安装
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到 conda 命令"
    exit 1
fi

# 获取 conda 基础路径
CONDA_BASE=$(conda info --base)
if [ -z "$CONDA_BASE" ]; then
    echo "错误: 无法获取 conda 基础路径"
    exit 1
fi

# 加载 conda 配置
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 检查 VeilAttack 环境是否存在
if conda env list | grep -q "VeilAttack"; then
    echo "找到 VeilAttack 环境，正在激活..."
    conda activate VeilAttack
    echo "已激活 VeilAttack 环境"
else
    echo "错误: 未找到 VeilAttack 环境"
    echo "请先运行 setup.sh 创建环境"
    exit 1
fi

# 运行测试脚本
python test_memory.py 