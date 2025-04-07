from setuptools import setup, find_packages
import sys
import os

def read_requirements(filename='requirements.txt'):
    """读取 requirements.txt 文件中的依赖"""
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

def check_python_version():
    """检查 Python 版本"""
    if sys.version_info < (3, 10):
        print("错误: 需要 Python 3.10 或更高版本（因为 vllm 依赖）")
        print(f"当前 Python 版本: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        sys.exit(1)
    print(f"Python 版本 {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} 符合要求")

# 检查 Python 版本
check_python_version()

# 读取依赖
pip_requirements = read_requirements()

# 定义包信息
setup(
    name="veil_attack",
    version="0.2.0",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    python_requires=">=3.10",
    install_requires=pip_requirements,
    author="Kefan",
    author_email="kefan.xyz@gmail.com",
    description="A framework for studying LLM safety through query decomposition attacks",
    long_description=open("README.md", encoding='utf-8').read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/KF-Audio/VeilAttack",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: POSIX :: Linux",
    ],
)

print("\nsetup.py 配置完成")
print("请使用 ./setup.sh 创建 conda 环境并安装所有依赖") 