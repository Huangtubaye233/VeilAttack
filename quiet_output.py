#!/usr/bin/env python3
"""
导入此模块以抑制各种库的冗长输出
"""

import os
import sys
import io
import logging
import warnings

# 抑制 Python 警告
warnings.filterwarnings('ignore')

# 设置环境变量来控制各个库的输出
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['VLLM_LOG_LEVEL'] = 'error'

# 配置日志级别
logging.basicConfig(level=logging.ERROR)

# 抑制特定库的日志
for logger_name in [
    'transformers', 'vllm', 'torch', 'tensorflow', 
    'tensorboard', 'ray', 'accelerate', 'huggingface_hub',
    'numpy', 'numexpr', 'matplotlib'
]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# 创建一个重定向类来过滤输出
class OutputFilter:
    def __init__(self, original_stream, filter_strings=None):
        self.original_stream = original_stream
        self.filter_strings = filter_strings or []
        
    def write(self, text):
        # 检查文本是否包含任何需要过滤的字符串
        if not any(s in text for s in self.filter_strings):
            self.original_stream.write(text)
            
    def flush(self):
        self.original_stream.flush()
        
    def isatty(self):
        return self.original_stream.isatty()

# 定义需要过滤的字符串列表
filter_strings = [
    "Creating a tensor",
    "Warning: using slow implementation",
    "This is a warning",
    "Model was loaded with trust_remote_code",
    "Explicitly passing a `revision`",
    "cuda:",
    "Using pad_token, but it is not set yet",
    "This model is a base model",
    "[0m",
    "tensor(",
    "The model weights are not tied",
    "You are using",
    "Special tokens have been added",
    "Loading checkpoint",
    "This tokenizer was incorrectly instantiated",
    "You seem to have cloned a model",
    "Overriding CNN embedding"
]

# 应用过滤器
sys.stdout = OutputFilter(sys.stdout, filter_strings)
sys.stderr = OutputFilter(sys.stderr, filter_strings)

def enable():
    """启用输出过滤"""
    sys.stdout = OutputFilter(sys.stdout, filter_strings)
    sys.stderr = OutputFilter(sys.stderr, filter_strings)

def disable():
    """禁用输出过滤，恢复正常输出"""
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

# 定义一个上下文管理器来临时抑制输出
class SuppressOutput:
    def __init__(self, suppress_stdout=True, suppress_stderr=True):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self._stdout = None
        self._stderr = None

    def __enter__(self):
        if self.suppress_stdout:
            self._stdout = sys.stdout
            sys.stdout = io.StringIO()
        
        if self.suppress_stderr:
            self._stderr = sys.stderr
            sys.stderr = io.StringIO()
            
        return self

    def __exit__(self, *args):
        if self.suppress_stdout:
            sys.stdout = self._stdout
        
        if self.suppress_stderr:
            sys.stderr = self._stderr 