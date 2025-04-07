import os
from typing import Dict, Any

class Config:
    """配置类，用于集中管理所有设置"""
    
    # 基础配置
    MODEL_DIR = "./models"  # 模型目录路径
    OUTPUT_DIR = "./output"  # 基础输出目录
    PRE_ATTACK_DIR = os.path.join(OUTPUT_DIR, "pre_attack")  # 预攻击输出目录
    ATTACK_DIR = os.path.join(OUTPUT_DIR, "attack")  # 攻击输出目录
    FINAL_DIR = os.path.join(OUTPUT_DIR, "final")  # 最终输出目录
    
    # 默认模型
    DEFAULT_ATTACK_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # 用于查询分解的模型
    DEFAULT_VICTIM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # 受害模型
    
    # VLLM 配置
    USE_VLLM = False  # 是否使用 VLLM
    VLLM_CONFIG = {
        "trust_remote_code": True,
        "gpu_memory_utilization": 0.8,
        "max_model_len": 2048,
        "enforce_eager": True,
        "quantization": None,
        "seed": 42,
        "revision": "main",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "disable_custom_all_reduce": True,
    }
    
    # 生成参数配置
    GENERATION_CONFIG = {
        "temperature": 0.3,
        "top_p": 0.9,
        "max_new_tokens": 200,  # 改用 max_new_tokens 替代 max_tokens
        "repetition_penalty": 1.2,
        "top_k": 50,
        "pad_token_id": None,  # 将在运行时设置
        "eos_token_id": None,  # 将在运行时设置
    }
    
    # 预攻击生成参数配置
    PRE_ATTACK_GENERATION_CONFIG = {
        "temperature": 0.3,
        "top_p": 0.9,
        "max_new_tokens": 200,  # 改用 max_new_tokens 替代 max_tokens
        "repetition_penalty": 1.2,
        "top_k": 50,
        "pad_token_id": None,  # 将在运行时设置
        "eos_token_id": None,  # 将在运行时设置
    }
    
    @classmethod
    def set_vllm(cls, use_vllm: bool):
        """设置是否使用 VLLM"""
        cls.USE_VLLM = use_vllm
    
    @classmethod
    def update_vllm_config(cls, **kwargs):
        """更新 VLLM 配置"""
        cls.VLLM_CONFIG.update(kwargs)
    
    @classmethod
    def update_generation_config(cls, **kwargs):
        """更新生成参数配置"""
        cls.GENERATION_CONFIG.update(kwargs)
    
    @classmethod
    def update_pre_attack_generation_config(cls, **kwargs):
        """更新预攻击生成参数配置"""
        cls.PRE_ATTACK_GENERATION_CONFIG.update(kwargs) 