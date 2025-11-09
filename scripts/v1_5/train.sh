#!/bin/bash
cd /root/LLaVA-LayerRouter-ca
export PYTHONWARNINGS="ignore"
export PYTHONPATH=/root/LLaVA-LayerRouter-ca

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /hy-tmp/vicuna-7b-v1.5 \
    --version plain \
    --data_path /hy-tmp/Train/pretrain.json \
    --image_folder /hy-tmp/Train/images \
    --vision_tower /hy-tmp/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /hy-tmp/checkpoints/llava-v1.5-7b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name pretrain-ca \

deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /hy-tmp/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /hy-tmp/Train/finetune.json \
    --image_folder /hy-tmp/Train/images \
    --vision_tower /hy-tmp/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /hy-tmp/checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /hy-tmp/checkpoints/llava-v1.5-7b-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name finetune

python ./scripts/merge_lora_weights.py \
    --model-path /hy-tmp/checkpoints/llava-v1.5-7b-pretrain \
    --model-base /hy-tmp/vicuna-7b-v1.5 \
    --save-model-path /hy-tmp/checkpoints/llava-v1.5-7b-pretrain-full

python ./scripts/merge_lora_weights.py \
    --model-path /hy-tmp/checkpoints/llava-v1.5-7b-lora \
    --model-base /hy-tmp/checkpoints/llava-v1.5-7b-pretrain-full \
    --save-model-path /hy-tmp/checkpoints/llava-v1.5-7b