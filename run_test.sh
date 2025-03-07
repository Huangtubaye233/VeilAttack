#!/bin/bash

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0  # Use first GPU, modify as needed

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate VeilAttack

# Run main program
echo "Starting VeilAttack test..."
python main.py

# Optional: Add command line arguments support
# python main.py --model_name "your_model_name" --dataset "your_dataset" --epochs 10

echo "Test completed!" 