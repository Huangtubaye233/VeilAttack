#!/bin/bash

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0  # Use the first GPU
# 从环境变量读取 token，如果没有设置则提示用户
if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "警告: 未设置 HUGGING_FACE_HUB_TOKEN 环境变量"
    echo "请设置环境变量: export HUGGING_FACE_HUB_TOKEN='你的token'"
fi

# Default values
INPUT_FILE="decomposed_queries_20250316_134842.json"
OUTPUT_FILE="test_attack.json"
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
LIMIT=1 # Default limit to process only 3 query sets for testing

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -i|--input)
      INPUT_FILE="$2"
      shift 2
      ;;
    -o|--output)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    -m|--model)
      MODEL_NAME="$2"
      shift 2
      ;;
    -l|--limit)
      LIMIT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if input file is provided
if [ -z "$INPUT_FILE" ]; then
  # Try to find the most recent decomposed_queries file
  INPUT_FILE=$(ls -t decomposed_queries_*.json 2>/dev/null | head -1)
  
  if [ -z "$INPUT_FILE" ]; then
    echo "Error: No input file specified and no decomposed_queries_*.json files found."
    echo "Usage: ./attack.sh -i input_file.json [-o output_file.json] [-m model_name] [-l limit]"
    exit 1
  else
    echo "Using most recent decomposed queries file: $INPUT_FILE"
  fi
fi

# Set output file if not provided
if [ -z "$OUTPUT_FILE" ]; then
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  OUTPUT_FILE="attack_results_${TIMESTAMP}.json"
fi

# Print configuration
echo "========================================"
echo "Running attack with the following configuration:"
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Model: $MODEL_NAME"
echo "Limit: $LIMIT"
echo "========================================"

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

# Print Python and environment info for debugging
echo "========================================"
echo "Python version:"
python --version
echo "Current working directory:"
pwd
echo "Checking if attack.py exists:"
if [ -f "attack.py" ]; then
  echo "attack.py found"
else
  echo "ERROR: attack.py not found!"
  exit 1
fi
echo "========================================"

# Run the attack script
echo "Starting attack process..."
python attack.py --input "$INPUT_FILE" --output "$OUTPUT_FILE" --model "$MODEL_NAME" --limit "$LIMIT"

# Check if the attack was successful
if [ $? -eq 0 ]; then
  echo "Attack completed successfully!"
  echo "Results saved to: $OUTPUT_FILE"
  
  # Print a sample of the results
  echo "========================================"
  echo "Sample of results (first few lines):"
  head -n 20 "$OUTPUT_FILE"
  echo "========================================"
else
  echo "Attack failed with error code: $?"
fi
