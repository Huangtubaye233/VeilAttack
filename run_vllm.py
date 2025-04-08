#!/usr/bin/env python3

# 在导入任何其他模块之前设置 multiprocessing 启动方法
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

# 解析命令行参数
import argparse
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--use_vllm", action="store_true")
parser.add_argument("--output_dir", "-o", type=str)
parser.add_argument("--limit", "-l", type=int)
parser.add_argument("--attack_model", "-am", type=str)
parser.add_argument("--victim_model", "-vm", type=str)
args, _ = parser.parse_known_args()

# 现在可以安全地导入其他模块
from main_vllm import main

if __name__ == "__main__":
    main() 