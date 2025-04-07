#!/bin/bash

# 设置默认参数
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
LIMIT=1
OUTPUT_FILE="comparison_results.json"

# 解析命令行参数
while getopts "m:l:o:" opt; do
  case $opt in
    m) MODEL="$OPTARG" ;;
    l) LIMIT="$OPTARG" ;;
    o) OUTPUT_FILE="$OPTARG" ;;
    \?) echo "无效选项: -$OPTARG" >&2; exit 1 ;;
  esac
done

# 构建命令行参数
ARGS=""
if [ ! -z "$MODEL" ]; then
  ARGS="$ARGS --model $MODEL"
fi

if [ ! -z "$LIMIT" ]; then
  ARGS="$ARGS --limit $LIMIT"
fi

if [ ! -z "$OUTPUT_FILE" ]; then
  ARGS="$ARGS --output $OUTPUT_FILE"
fi

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

# 运行Python脚本
echo "正在运行直接攻击比较..."
echo "模型: $MODEL"
echo "查询限制: $LIMIT"
echo "输出文件: ${OUTPUT_FILE:-自动生成}"
echo "----------------------------------------"

python comparison.py $ARGS 