#!/bin/bash

source /beegfs/scratch/project/humans/mamba/miniforge3/etc/profile.d/conda.sh
source /beegfs/scratch/project/humans/mamba/miniforge3/etc/profile.d/mamba.sh
conda activate qwenvl

export CUDA_HOME=/nfs/core/cuda/12.6


BASE_MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
LORA_WEIGHTS_PATH="/beegfs/scratch/user/lmaria/social_3d_pose/results/sft_shape_vlm/resampled-lora-qwen25vl-3b-utkface-all_ages_faces-ageonly"
OUTPUT_PATH=$LORA_WEIGHTS_PATH"/merge_models"

mkdir -p $OUTPUT_PATH

python tools/merge_lora_weights.py \
    --base_model_name $BASE_MODEL_NAME \
    --lora_model_path $LORA_WEIGHTS_PATH \
    --output_path $OUTPUT_PATH