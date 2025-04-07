#!/bin/bash

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0  # Use first GPU, modify as needed
# 从环境变量读取 token，如果没有设置则提示用户
if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "警告: 未设置 HUGGING_FACE_HUB_TOKEN 环境变量"
    echo "请设置环境变量: export HUGGING_FACE_HUB_TOKEN='你的token'"
fi

# Hugging Face authentication
# Method 1: Set Hugging Face token via environment variable
# Replace with your actual token from https://huggingface.co/settings/tokens
# Method 2: Use huggingface-cli login (requires interactive input)
# Uncomment the line below if you prefer this method
# huggingface-cli login

# Method 3: Use pre-saved token file
# If you've already run `huggingface-cli login`, the token is saved in ~/.huggingface/token
# This method doesn't require any additional action as the transformers library will find this file automatically

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

# Run test script
echo "Starting AttackModel test..."
python pre_attack.py

echo "Test completed!" 