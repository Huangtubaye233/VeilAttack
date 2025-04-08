from typing import Dict, Any

class Config:
    """配置类，用于集中管理所有设置"""
    
    # 基础配置
    MODEL_DIR = "./models"  # 模型目录路径
    OUTPUT_DIR = "./output"  # 基础输出目录
    PRE_ATTACK_DIR = None  # 预攻击输出目录
    ATTACK_DIR = None  # 攻击输出目录
    FINAL_DIR = None  # 最终输出目录
    
    # 默认模型
    DEFAULT_ATTACK_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # 用于查询分解的模型
    DEFAULT_VICTIM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # 受害模型
    
    # VLLM 配置
    USE_VLLM = False  # 是否使用 VLLM
    VLLM_CONFIG = {
        "trust_remote_code": True,
        "gpu_memory_utilization": 0.8,  # 适中的内存使用率
        "max_model_len": 1024,  # 适中的最大长度
        "enforce_eager": True,
        "quantization": None,
        "seed": 42,
        "revision": "main",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "disable_custom_all_reduce": True,
        "max_num_seqs": 16,  # 适中的序列数
    }
    
    # 标准 Transformers 配置
    TRANSFORMERS_CONFIG = {
        "device_map": "auto",  # 让 accelerate 处理设备映射
        "trust_remote_code": True,
        "use_flash_attention_2": False,  # 对于 Qwen 模型禁用滑动窗口注意力
    }
    
    # 基础生成参数配置 (VLLM)
    GENERATION_CONFIG = {
        "temperature": 0.5,  # 适中的温度
        "top_p": 0.9,
        "max_tokens": 256,  # 适中的生成长度
        "repetition_penalty": 1.1,
        "top_k": 50,
    }
    
    # 基础预攻击生成参数配置 (VLLM)
    PRE_ATTACK_GENERATION_CONFIG = {
        "temperature": 0.5,  # 适中的温度
        "top_p": 0.9,
        "max_tokens": 256,  # 适中的生成长度
        "repetition_penalty": 1.1,
        "top_k": 50,
    }
    
    # 标准 Transformers 生成参数配置
    TRANSFORMERS_GENERATION_CONFIG = {
        "temperature": 0.5,  # 适中的温度
        "top_p": 0.9,
        "max_new_tokens": 256,  # 适中的生成长度
        "repetition_penalty": 1.1,
        "top_k": 50,
        "pad_token_id": None,  # 将在运行时设置
        "eos_token_id": None,  # 将在运行时设置
    }
    
    # 标准 Transformers 预攻击生成参数配置
    TRANSFORMERS_PRE_ATTACK_GENERATION_CONFIG = {
        "temperature": 0.5,  # 适中的温度
        "top_p": 0.9,
        "max_new_tokens": 256,  # 适中的生成长度
        "repetition_penalty": 1.1,
        "top_k": 50,
        "pad_token_id": None,  # 将在运行时设置
        "eos_token_id": None,  # 将在运行时设置
    }
    
    @classmethod
    def set_vllm(cls, use_vllm: bool):
        """设置是否使用 VLLM"""
        cls.USE_VLLM = use_vllm
        
    @classmethod
    def set_output_dir(cls, output_dir: str):
        """设置输出目录"""
        import os
        cls.OUTPUT_DIR = output_dir
        cls.PRE_ATTACK_DIR = os.path.join(output_dir, "pre_attack")
        cls.ATTACK_DIR = os.path.join(output_dir, "attack")
        cls.FINAL_DIR = os.path.join(output_dir, "final")
    
