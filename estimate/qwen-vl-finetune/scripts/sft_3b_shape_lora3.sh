#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --cpus-per-task=3
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --output=/beegfs/scratch/user/%u/logs/qwenvl/%j.log
#SBATCH --job-name=all_lora_3b
#SBATCH --constraint="gpu_h100"

source /beegfs/scratch/project/humans/mamba/miniforge3/etc/profile.d/conda.sh
source /beegfs/scratch/project/humans/mamba/miniforge3/etc/profile.d/mamba.sh
conda activate qwenvl

export CUDA_HOME=/nfs/core/cuda/12.6

export WANDB_API_KEY="e975cab3491c8ecee4315cb1713bd4e759d77627"
export WANDB_PROJECT="sft_shape_vlm"

echo $(nvidia-smi --list-gpus)

# # Distributed training configuration
# # NPROC_PER_NODE=1
# NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
# NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=Qwen/Qwen2.5-VL-3B-Instruct  # Using HuggingFace model ID

# Training hyperparameters
lr=1e-4
batch_size=16
grad_accum_steps=1

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets=utkface_resampled_train_all%100,all_ages_faces_resampled_train_all%100
eval_datasets=all_ages_faces_resampled_val_all%50

# Output configuration
run_name="all-resampled-lora-qwen25vl-3b-utkface-all_ages_faces"
output_dir="/home/lmaria/scratch_lmaria/social_3d_pose/results/sft_shape_vlm/"${run_name}
mkdir -p ${output_dir}

# for LoRA usage base component has to be True e.g. --tune_mm_llm True
# for MLP --lora_target_modules ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --eval_dataset_use ${eval_datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm True \
    --lora_enable True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 10 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "steps" \
    --eval_steps 500 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 10 \
    --learning_rate ${lr} \
    --mm_projector_lr 1e-5 \
    --vision_tower_lr 1e-6 \
    --optim adamw_torch \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 3 \
    --run_name ${run_name} \
    --report_to wandb"

# # Launch training
# torchrun --nproc_per_node=${NPROC_PER_NODE} \
#          --master_addr=${MASTER_ADDR} \
#          --master_port=${MASTER_PORT} \
#          ${entry_file} ${args}

python ${entry_file} ${args}