#!/bin/bash

# Setup script for VeilAttack
# This script creates a conda environment and installs all dependencies

# Environment name
ENV_NAME="VeilAttack"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Check if the environment already exists
if conda info --envs | grep -q $ENV_NAME; then
    echo "Environment $ENV_NAME already exists."
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n $ENV_NAME
    else
        echo "Using existing environment. Will update packages."
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate $ENV_NAME
        pip install -r requirements.txt
        echo "Dependencies updated successfully."
        echo "Environment is now active and ready to use."
        exit 0
    fi
fi

# Create a new conda environment
echo "Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.9 -y

# Activate the environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Install the package in development mode
echo "Installing VeilAttack in development mode..."
pip install -e .

echo "Setup completed successfully!"
echo "Environment is now active and ready to use."
echo "Next time you need to use this environment, run: conda activate $ENV_NAME" 