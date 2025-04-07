#!/usr/bin/env python3

# Standard library imports
import argparse
import json
import os
import logging
import multiprocessing
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import torch

# Configure logging - set to ERROR level to only log errors
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Import functions from pre_attack and attack modules
from pre_attack import AttackModel, load_toxic_queries, process_queries
from attack import VictimModel, attack_with_sub_queries
from config import Config

def save_query_response_pairs(attack_results_file: str, output_file: Optional[str] = None) -> str:
    """
    Extract query-response pairs from attack results and save to a new file
    
    Args:
        attack_results_file: Path to the attack results JSON file
        output_file: Path to save the extracted pairs, if None a timestamp-based name is used
        
    Returns:
        str: Path to the output file
    """
    # Create output file name if not provided
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_file is None:
        output_file = os.path.join(Config.FINAL_DIR, f"query_response_pairs_{timestamp}.json")
    
    try:
        # Load attack results
        with open(attack_results_file, 'r', encoding='utf-8') as f:
            attack_results = json.load(f)
        
        # Extract query-response pairs
        pairs = []
        
        for result in attack_results:
            query_idx = result.get("query_idx", -1)
            sub_queries = result.get("sub_queries", {})
            responses = result.get("responses", [])
            
            # Create a record for each query-response pair
            for response_data in responses:
                query_key = response_data.get("query_key", "")
                query = response_data.get("query", "")
                response = response_data.get("response", "")
                
                pair = {
                    "query_idx": query_idx,
                    "query_key": query_key,
                    "query": query,
                    "response": response
                }
                
                pairs.append(pair)
        
        # Save to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
        
        return output_file
        
    except Exception as e:
        logger.error(f"Error extracting query-response pairs: {str(e)}")
        raise

def custom_process_queries(queries: List[str], 
                          model: AttackModel,
                          output_file: Optional[str] = None,
                          num_samples: Optional[int] = None) -> str:
    """
    Process harmful queries, decompose them into harmless sub-queries, and save results
    
    Args:
        queries: List of queries to process
        model: Model for decomposing queries
        output_file: Output filename, creates one with timestamp if None
        num_samples: Number of samples to process, processes all queries if None
        
    Returns:
        str: Path to the output file
    """
    # Create output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_file is None:
        output_file = os.path.join(Config.PRE_ATTACK_DIR, f"decomposed_queries_{timestamp}.json")
    
    # If sample count specified, only use a subset of queries
    if num_samples and num_samples < len(queries):
        queries = queries[:num_samples]
    
    # Process records
    clean_results = []
    
    for query_idx, query in enumerate(queries):
        # Get model response
        raw_response, json_response = model.decompose_query(query)
        
        # Check for errors
        if "error" in json_response:
            logger.error(f"Error processing query {query_idx+1}: {json_response['error']}")
        else:
            # For clean output, directly use the JSON response
            clean_results.append(json_response)
            
            # Print successful JSON responses
            print(json.dumps(json_response, indent=2, ensure_ascii=False))
        
        # Periodically save to file
        if (query_idx + 1) % 5 == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(clean_results, f, ensure_ascii=False, indent=2)
    
    # Final save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(clean_results, f, ensure_ascii=False, indent=2)
    
    return output_file

def main():
    """
    Main function to run the complete attack pipeline:
    1. Decompose harmful queries into harmless sub-queries
    2. Attack victim model with the sub-queries
    3. Extract and save query-response pairs
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the complete attack pipeline")
    
    # General arguments
    parser.add_argument("--output_dir", "-o", type=str, default=Config.OUTPUT_DIR,
                        help="Base directory to save all output files")
    parser.add_argument("--limit", "-l", type=int, default=1,
                        help="Limit the number of queries to process")
    
    # Pre-attack arguments
    parser.add_argument("--attack_model", "-am", type=str, default=Config.DEFAULT_ATTACK_MODEL,
                        help="Model to use for query decomposition")
    
    # Attack arguments
    parser.add_argument("--victim_model", "-vm", type=str, default=Config.DEFAULT_VICTIM_MODEL,
                        help="Victim model to attack with sub-queries")
    
    # VLLM arguments
    parser.add_argument("--use_vllm", action="store_true",
                        help="Use vllm for inference")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set VLLM configuration
    Config.set_vllm(args.use_vllm)
    
    # 如果使用 VLLM，设置多进程启动方法为 'spawn'
    if Config.USE_VLLM:
        multiprocessing.set_start_method('spawn', force=True)
    
    # Create output directories if they don't exist
    base_output_dir = args.output_dir
    pre_attack_dir = os.path.join(base_output_dir, "pre_attack")
    attack_dir = os.path.join(base_output_dir, "attack")
    final_dir = os.path.join(base_output_dir, "final")
    
    os.makedirs(base_output_dir, exist_ok=True)
    os.makedirs(pre_attack_dir, exist_ok=True)
    os.makedirs(attack_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    
    # Generate timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Step 1: Load toxic queries
    print("\n=== Step 1: Loading toxic queries ===")
    toxic_queries = load_toxic_queries()
    print(f"Loaded {len(toxic_queries)} unique toxic queries")
    
    # Limit the number of queries if specified
    if args.limit and args.limit < len(toxic_queries):
        toxic_queries = toxic_queries[:args.limit]
        print(f"Limited to processing {args.limit} queries")
    
    # Step 2: Initialize attack model for query decomposition
    print("\n=== Step 2: Initializing attack model for query decomposition ===")
    attack_model = AttackModel(model_name=args.attack_model)
    print(f"Initialized attack model: {args.attack_model}")
    if Config.USE_VLLM:
        print("Using vllm for attack model")
    
    # Step 3: Process queries to decompose them into harmless sub-queries
    print("\n=== Step 3: Decomposing harmful queries into harmless sub-queries ===")
    decomposed_queries_file = os.path.join(pre_attack_dir, f"decomposed_queries_{timestamp}.json")
    
    clean_file = custom_process_queries(
        queries=toxic_queries,
        model=attack_model,
        output_file=decomposed_queries_file,
        num_samples=args.limit
    )
    print(f"Decomposed queries saved to: {clean_file}")
    
    # 释放 attack model 的显存
    print("\n=== Releasing attack model memory ===")
    del attack_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Step 4: Initialize victim model for attack
    print("\n=== Step 4: Initializing victim model for attack ===")
    victim_model = VictimModel(model_name=args.victim_model)
    print(f"Initialized victim model: {args.victim_model}")
    if Config.USE_VLLM:
        print("Using vllm for victim model")
    
    # Step 5: Load sub-queries from the decomposed queries file
    print("\n=== Step 5: Loading sub-queries for attack ===")
    try:
        with open(clean_file, 'r', encoding='utf-8') as f:
            sub_queries_list = json.load(f)
        print(f"Loaded {len(sub_queries_list)} sets of sub-queries")
    except Exception as e:
        logger.error(f"Error loading sub-queries: {str(e)}")
        raise
    
    # Step 6: Attack victim model with sub-queries
    print("\n=== Step 6: Attacking victim model with sub-queries ===")
    attack_results_file = os.path.join(attack_dir, f"attack_results_{timestamp}.json")
    attack_results_file = attack_with_sub_queries(
        victim_model=victim_model,
        sub_queries_list=sub_queries_list,
        output_file=attack_results_file
    )
    print(f"Attack results saved to: {attack_results_file}")
    
    # Step 7: Extract and save query-response pairs
    print("\n=== Step 7: Extracting query-response pairs ===")
    pairs_file = os.path.join(final_dir, f"query_response_pairs_{timestamp}.json")
    pairs_file = save_query_response_pairs(
        attack_results_file=attack_results_file,
        output_file=pairs_file
    )
    print(f"Query-response pairs saved to: {pairs_file}")
    
    print("\n=== Attack pipeline completed successfully ===")
    print(f"All results saved to directory structure under: {base_output_dir}")
    print(f"  - Pre-attack outputs: {pre_attack_dir}")
    print(f"  - Attack outputs: {attack_dir}")
    print(f"  - Final outputs: {final_dir}")
    
    return pairs_file

if __name__ == "__main__":
    main()
