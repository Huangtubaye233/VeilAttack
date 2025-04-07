import os
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import multiprocessing

# 设置多进程启动方法为 spawn
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

# 禁用 VLLM 的多进程功能
os.environ["VLLM_USE_MULTIPROCESSING"] = "0"

def print_gpu_memory():
    """打印 GPU 内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        cached = torch.cuda.memory_reserved() / (1024**3)  # GB
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        print(f"\nGPU内存使用情况:")
        print(f"已分配: {allocated:.2f}GB")
        print(f"缓存: {cached:.2f}GB")
        print(f"最大分配: {max_allocated:.2f}GB")
    else:
        print("未检测到 GPU")

def test_vllm():
    """测试 VLLM 模型"""
    print("\n=== 测试 VLLM 模型 ===")
    print_gpu_memory()
    
    # 设置环境变量
    os.environ["HF_HOME"] = "./cache"
    os.environ["TRANSFORMERS_CACHE"] = "./cache"
    
    print("\n开始加载模型...")
    start_time = time.time()
    
    try:
        # 优化 VLLM 初始化参数
        model = LLM(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            trust_remote_code=True,
            gpu_memory_utilization=0.8,  # 限制 GPU 内存使用
            enforce_eager=True,  # 禁用编译优化
            max_model_len=2048,  # 限制最大序列长度
            quantization=None,  # 禁用量化
            seed=42,  # 固定随机种子
            revision="main",  # 使用主分支版本
            tensor_parallel_size=1,  # 禁用张量并行
            pipeline_parallel_size=1,  # 禁用流水线并行
            disable_custom_all_reduce=True,  # 禁用自定义 all-reduce
            download_dir="./models",  # 指定模型下载目录
        )
        
        load_time = time.time() - start_time
        print(f"模型加载完成，耗时: {load_time:.2f}秒")
        print_gpu_memory()
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.9,
            max_tokens=200,
            repetition_penalty=1.2,
            top_k=50,
            presence_penalty=0.0,
            frequency_penalty=0.0,
        )
        
        print("\n开始推理...")
        start_time = time.time()
        
        # 测试推理
        prompt = "你好，请介绍一下你自己。"
        outputs = model.generate(prompt, sampling_params)
        response = outputs[0].outputs[0].text
        
        inference_time = time.time() - start_time
        print(f"推理完成，耗时: {inference_time:.2f}秒")
        print(f"响应: {response}")
        print_gpu_memory()
        
        # 清理内存
        del model
        torch.cuda.empty_cache()
        print("\n清理内存后:")
        print_gpu_memory()
        
    except Exception as e:
        print(f"VLLM 测试出错: {str(e)}")
        raise

def test_transformers():
    """测试 Transformers 模型"""
    print("\n=== 测试 Transformers 模型 ===")
    print_gpu_memory()
    
    print("\n开始加载模型...")
    start_time = time.time()
    
    try:
        # 优化 Transformers 加载参数
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            use_cache=True,  # 启用 KV 缓存
            max_memory={0: "16GB"},  # 限制 GPU 内存使用
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            trust_remote_code=True
        )
        
        load_time = time.time() - start_time
        print(f"模型加载完成，耗时: {load_time:.2f}秒")
        print_gpu_memory()
        
        print("\n开始推理...")
        start_time = time.time()
        
        # 测试推理
        prompt = "你好，请介绍一下你自己。"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=200,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        inference_time = time.time() - start_time
        print(f"推理完成，耗时: {inference_time:.2f}秒")
        print(f"响应: {response}")
        print_gpu_memory()
        
        # 清理内存
        del model
        del tokenizer
        torch.cuda.empty_cache()
        print("\n清理内存后:")
        print_gpu_memory()
        
    except Exception as e:
        print(f"Transformers 测试出错: {str(e)}")
        raise

if __name__ == "__main__":
    print("开始内存测试...")
    test_vllm()
    test_transformers()
    print("\n测试完成!") 