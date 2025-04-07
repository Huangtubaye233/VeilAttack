#!/bin/bash

# VeilAttack 环境设置脚本
# 此脚本负责创建 conda 环境并安装所有依赖

# 环境名称
ENV_NAME="VeilAttack"

# 检查 CUDA 可用性
check_cuda() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "警告: 未检测到 NVIDIA GPU。vllm 需要 CUDA 支持。"
        read -p "是否继续？(y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# 检查并安装 Miniconda
check_conda() {
    if ! command -v conda &> /dev/null; then
        echo "未找到 conda，正在安装 Miniconda..."
        
        # 根据操作系统选择安装包
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            if [[ $(uname -m) == "arm64" ]]; then
                MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
            else
                MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
            fi
        else
            echo "不支持的操作系统。请从 https://docs.conda.io/en/latest/miniconda.html 手动安装 Miniconda"
            exit 1
        fi
        
        # 下载并安装 Miniconda
        MINICONDA_INSTALLER="miniconda_installer.sh"
        echo "正在从 $MINICONDA_URL 下载 Miniconda..."
        curl -o $MINICONDA_INSTALLER $MINICONDA_URL
        bash $MINICONDA_INSTALLER -b -p $HOME/miniconda
        rm $MINICONDA_INSTALLER
        
        # 添加到 PATH
        export PATH="$HOME/miniconda/bin:$PATH"
        echo "Miniconda 安装完成。请重新运行此脚本。"
        exit 0
    fi
}

# 创建或更新 conda 环境
setup_conda_env() {
    if conda env list | grep -q $ENV_NAME; then
        echo "环境 $ENV_NAME 已存在。"
        read -p "是否删除并重新创建？(y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "正在删除现有环境..."
            conda env remove -n $ENV_NAME
        else
            echo "使用现有环境。"
            source $(conda info --base)/etc/profile.d/conda.sh
            conda activate $ENV_NAME
            return
        fi
    fi
    
    echo "正在创建 conda 环境: $ENV_NAME"
    conda create -n $ENV_NAME python=3.10 -y
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate $ENV_NAME
}

# 安装系统依赖
install_system_deps() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "正在安装系统依赖..."
        conda install -c conda-forge c-compiler cxx-compiler make -y
        conda install -c nvidia cuda-toolkit -y
    fi
}

# 安装 PyTorch 和 CUDA
install_pytorch() {
    echo "正在安装 PyTorch..."
    # 使用 pip 安装指定版本的 PyTorch
    pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
}

# 安装其他依赖
install_deps() {
    echo "正在安装其他依赖..."
    
    # 安装基础依赖
    echo "正在安装基础依赖..."
    pip install numpy pandas IPython matplotlib
    
    # 安装机器学习相关依赖
    echo "正在安装机器学习相关依赖..."
    pip install transformers==4.47.0 accelerate peft datasets
    
    # 安装 flash-attn
    echo "正在安装 flash-attn..."
    pip install flash-attn==2.3.6 --no-build-isolation
    
    # 安装其他依赖
    echo "正在安装其他依赖..."
    pip install gym gym_sokoban codetiming dill hydra-core pybind11 ray wandb gymnasium gymnasium[toy-text]
    
    # 安装 tensordict（指定版本）
    echo "正在安装 tensordict..."
    pip install "tensordict<0.6"
    
    # 安装 vllm
    echo "正在安装 vllm..."
    pip install vllm==0.6.3
    
    # 安装当前包
    echo "正在安装 VeilAttack..."
    pip install -e .
}

# 主流程
echo "开始设置 VeilAttack 环境..."

# 检查 CUDA
check_cuda

# 检查 conda
check_conda

# 设置 conda 环境
setup_conda_env

# 安装系统依赖
install_system_deps

# 安装 PyTorch
install_pytorch

# 安装其他依赖
install_deps

echo "环境设置完成！"
echo "环境已激活并准备就绪。"
echo "下次使用时，运行: conda activate $ENV_NAME"
echo ""
echo "注意: vllm 需要 CUDA 支持。请确保您的系统有兼容的 GPU。" 