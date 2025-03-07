from setuptools import setup, find_packages
import subprocess
import sys
import os

def read_requirements(filename):
    """Read requirements from file"""
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

def create_conda_env():
    """Create a new conda environment and install dependencies"""
    env_name = "veil_attack"
    
    # Create conda environment
    subprocess.run(["conda", "create", "-n", env_name, "python=3.9", "-y"])
    
    # Activate environment and install requirements
    if sys.platform == "win32":
        activate_cmd = f"conda activate {env_name} && "
    else:
        activate_cmd = f"source activate {env_name} && "
    
    # Install requirements
    subprocess.run(activate_cmd + "pip install -r requirements.txt", shell=True)
    
    print(f"\nConda environment '{env_name}' created and dependencies installed.")
    print(f"To activate the environment, run: conda activate {env_name}")

def install_dependencies():
    """Install dependencies in current environment"""
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Dependencies installed successfully.")

if __name__ == "__main__":
    if "--conda" in sys.argv:
        create_conda_env()
    else:
        install_dependencies()

# Read requirements from requirements.txt
requirements = read_requirements('requirements.txt')

setup(
    name="veil_attack",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.9",
    author="Your Name",
    author_email="your.email@example.com",
    description="A framework for studying LLM safety through query decomposition attacks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/VeilAttack",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 