#!/usr/bin/env python3

import os
import argparse
import json
import logging
from datetime import datetime
from typing import List, Optional

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import torch
import torch.distributed as dist
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    print(f"Device count: {torch.cuda.device_count()}")

from pre_attack import AttackModel, load_toxic_queries
from attack import VictimModel, attack_with_sub_queries
from config import Config

def save_query_response_pairs(attack_results_file: str, output_file: Optional[str] = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_file is None:
        output_file = os.path.join(Config.FINAL_DIR, f"query_response_pairs_{timestamp}.json")
    with open(attack_results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    pairs = []
    for r in results:
        for item in r.get("responses", []):
            pairs.append({
                "query_idx": r.get("query_idx", -1),
                "query_key": item.get("query_key", ""),
                "query": item.get("query", ""),
                "response": item.get("response", "")
            })
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    return output_file

def custom_process_queries(queries: List[str], model: AttackModel,
                           output_file: Optional[str] = None,
                           num_samples: Optional[int] = None) -> str:
    print("\n=== Debug: Starting custom_process_queries ===")
    if torch.cuda.is_available():
        print(f"CUDA device in use: {torch.cuda.current_device()}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_file is None:
        output_file = os.path.join(Config.PRE_ATTACK_DIR, f"decomposed_queries_{timestamp}.json")

    if num_samples and num_samples < len(queries):
        queries = queries[:num_samples]

    results = []
    for idx, q in enumerate(queries):
        raw, js = model.decompose_query(q)
        if "error" not in js:
            results.append(js)
            print(json.dumps(js, indent=2, ensure_ascii=False))
        if (idx + 1) % 5 == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return output_file

def main():
    print("\n=== Debug: Starting main function ===")
    print(f"Process ID: {os.getpid()}")
    if torch.cuda.is_available():
        print(f"CUDA device in use: {torch.cuda.current_device()}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default=Config.OUTPUT_DIR)
    parser.add_argument("--limit", "-l", type=int, default=1)
    parser.add_argument("--attack_model", "-am", type=str, default=Config.DEFAULT_ATTACK_MODEL)
    parser.add_argument("--victim_model", "-vm", type=str, default=Config.DEFAULT_VICTIM_MODEL)
    parser.add_argument("--use_vllm", action="store_true")
    args = parser.parse_args()

    Config.set_vllm(args.use_vllm)
    Config.set_output_dir(args.output_dir)

    print("使用 vllm 进行推理" if Config.USE_VLLM else "使用 transformers 进行推理")

    os.makedirs(args.output_dir, exist_ok=True)
    for d in [Config.PRE_ATTACK_DIR, Config.ATTACK_DIR, Config.FINAL_DIR]:
        os.makedirs(d, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n=== Step 1: Loading toxic queries ===")
    queries = load_toxic_queries()
    print(f"Loaded {len(queries)} unique toxic queries")
    if args.limit < len(queries):
        queries = queries[:args.limit]
        print(f"Limited to processing {args.limit} queries")

    print("\n=== Step 2: Initializing attack model for query decomposition ===")
    print(f"Process ID before attack model init: {os.getpid()}")
    if torch.cuda.is_available():
        print(f"CUDA device before attack model init: {torch.cuda.current_device()}")
    attack_model = AttackModel(model_name=args.attack_model)
    print(f"Process ID after attack model init: {os.getpid()}")
    if torch.cuda.is_available():
        print(f"CUDA device after attack model init: {torch.cuda.current_device()}")
    print(f"Initialized attack model: {args.attack_model}")
    if Config.USE_VLLM:
        print("Using vllm for attack model")

    print("\n=== Step 3: Decomposing harmful queries into harmless sub-queries ===")
    decomp_file = os.path.join(Config.PRE_ATTACK_DIR, f"decomposed_queries_{timestamp}.json")
    clean_file = custom_process_queries(queries, attack_model, decomp_file, args.limit)
    print(f"Decomposed queries saved to: {clean_file}")

    del attack_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Cleared CUDA cache after attack model")

    print("\n=== Step 4: Initializing victim model for attack ===")
    print(f"Process ID before victim model init: {os.getpid()}")
    if torch.cuda.is_available():
        print(f"CUDA device before victim model init: {torch.cuda.current_device()}")
    victim_model = VictimModel(model_name=args.victim_model)
    print(f"Process ID after victim model init: {os.getpid()}")
    if torch.cuda.is_available():
        print(f"CUDA device after victim model init: {torch.cuda.current_device()}")
    print(f"Initialized victim model: {args.victim_model}")
    if Config.USE_VLLM:
        print("Using vllm for victim model")

    print("\n=== Step 5: Loading sub-queries for attack ===")
    with open(clean_file, 'r', encoding='utf-8') as f:
        sub_queries_list = json.load(f)
    print(f"Loaded {len(sub_queries_list)} sets of sub-queries")

    print("\n=== Step 6: Attacking victim model with sub-queries ===")
    print(f"Process ID before attack: {os.getpid()}")
    if torch.cuda.is_available():
        print(f"CUDA device before attack: {torch.cuda.current_device()}")
    attack_results_file = os.path.join(Config.ATTACK_DIR, f"attack_results_{timestamp}.json")
    attack_with_sub_queries(
        victim_model=victim_model,
        sub_queries_list=sub_queries_list,
        output_file=attack_results_file
    )
    print(f"Process ID after attack: {os.getpid()}")
    if torch.cuda.is_available():
        print(f"CUDA device after attack: {torch.cuda.current_device()}")
    print(f"Attack results saved to: {attack_results_file}")

    print("\n=== Step 7: Extracting query-response pairs ===")
    pairs_file = os.path.join(Config.FINAL_DIR, f"query_response_pairs_{timestamp}.json")
    save_query_response_pairs(attack_results_file, pairs_file)
    print(f"Query-response pairs saved to: {pairs_file}")

    print("\n=== Attack pipeline completed successfully ===")
    print(f"All results saved to directory structure under: {args.output_dir}")

    # 清理分布式训练资源
    if dist.is_initialized():
        print("\n=== Debug: Cleaning up distributed training resources ===")
        dist.destroy_process_group()
        print("Distributed process group destroyed")


if __name__ == "__main__":
    print("\n=== Debug: Starting main_vllm.py as main ===")
    print(f"Process ID: {os.getpid()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.current_device()}")
    main()
