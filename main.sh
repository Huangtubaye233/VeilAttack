#!/bin/bash

# 加载 .env 文件
if [ -f ".env" ]; then
    echo "正在加载环境变量..."
    source .env
else
    echo "警告: 未找到 .env 文件"
    echo "请创建 .env 文件并设置 HUGGING_FACE_HUB_TOKEN"
    exit 1
fi

# 检查 token 是否设置
if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "错误: 未设置 HUGGING_FACE_HUB_TOKEN"
    echo "请在 .env 文件中设置你的 token"
    exit 1
fi

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0  # 使用第一个 GPU，根据需要修改

# 设置默认参数
ATTACK_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
VICTIM_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
LIMIT=1
OUTPUT_DIR="./output"
USE_VLLM=false

# 解析命令行参数
while getopts "a:v:l:o:V" opt; do
  case $opt in
    a) ATTACK_MODEL="$OPTARG" ;;
    v) VICTIM_MODEL="$OPTARG" ;;
    l) LIMIT="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    V) USE_VLLM=true ;;
    \?) echo "无效选项: -$OPTARG" >&2; exit 1 ;;
  esac
done

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

# 打印 Python 和环境信息用于调试
echo "========================================"
echo "Python 版本:"
python --version
echo "当前工作目录:"
pwd
echo "检查 main.py 是否存在:"
if [ -f "main.py" ]; then
  echo "找到 main.py"
else
  echo "错误: 未找到 main.py!"
  exit 1
fi
echo "========================================"

# 构建命令行参数
ARGS=""
if [ ! -z "$ATTACK_MODEL" ]; then
  ARGS="$ARGS --attack_model $ATTACK_MODEL"
fi

if [ ! -z "$VICTIM_MODEL" ]; then
  ARGS="$ARGS --victim_model $VICTIM_MODEL"
fi

if [ ! -z "$LIMIT" ]; then
  ARGS="$ARGS --limit $LIMIT"
fi

if [ ! -z "$OUTPUT_DIR" ]; then
  ARGS="$ARGS --output_dir $OUTPUT_DIR"
fi

if [ "$USE_VLLM" = true ]; then
  ARGS="$ARGS --use_vllm"
fi

# 创建输出目录结构
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/pre_attack"
mkdir -p "$OUTPUT_DIR/attack"
mkdir -p "$OUTPUT_DIR/final"

echo "已创建输出目录结构:"
echo "  - $OUTPUT_DIR"
echo "  - $OUTPUT_DIR/pre_attack"
echo "  - $OUTPUT_DIR/attack"
echo "  - $OUTPUT_DIR/final"

# 运行 Python 脚本
echo "正在运行完整攻击流程..."
echo "攻击模型: $ATTACK_MODEL"
echo "受害模型: $VICTIM_MODEL"
echo "查询限制: $LIMIT"
echo "输出目录: $OUTPUT_DIR"
if [ "$USE_VLLM" = true ]; then
  echo "使用 vllm 进行推理"
fi
echo "----------------------------------------"

python main.py $ARGS

# 检查运行是否成功
if [ $? -eq 0 ]; then
  echo "Shell: 攻击流程成功完成！"
  echo "结果保存在: $OUTPUT_DIR"
else
  echo "Shell: 攻击流程失败，错误代码: $?"
fi 