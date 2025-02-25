#!/bin/bash

# python dataset_extract.py \
#         --dataset mvtec_ad_setA \
#         --output-dir ./output_A \
#         --seed 99 \


# bash fine_tune_A.sh 0 4 99
# bash fine_tune_A.sh 0 8 99
# bash fine_tune_A.sh 0 16 99


python fine_tune_mvtec_normal.py \
--dataset=mvtec_ad_normal --output_dir=./output_normal_2.1 \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--seed=99 \
--resolution=512 --train_batch_size=8 --lr_warmup_steps=0 \
--gradient_accumulation_steps=1 \
--learning_rate=5.0e-04 --scale_lr --lr_scheduler="constant" \
--mixed_precision=fp16 --revision=fp16 --gradient_checkpointing \
--only_save_embeds --num-trials 1 --examples-per-class 8 16 \
--save_steps 1000 \
--num_train_epochs 100 

# --max_train_steps=10000 \
# --checkpointing_steps 1000 \
# "stabilityai/stable-diffusion-2-1"
# "CompVis/stable-diffusion-v1-4"


# CUDA_VISIBLE_DEVICES=3 python fine_tune_mvtec_normal.py \
# --dataset=mvtec_ad_normal --output_dir=./output_normal_2.1 \
# --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
# --seed=99 \
# --resolution=512 --train_batch_size=8 --lr_warmup_steps=0 \
# --gradient_accumulation_steps=1 \
# --learning_rate=5.0e-04 --scale_lr --lr_scheduler="constant" \
# --mixed_precision=fp16 --revision=fp16 --gradient_checkpointing \
# --only_save_embeds --num-trials 1 --examples-per-class 4 \
# --save_steps 1000 \
# --max_train_steps=10 

CUDA_VISIBLE_DEVICES=3 python fine_tune_mvtec_normal.py \
--dataset=mvtec_ad_normal --output_dir=./output_normal \
--pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
--seed=99 \
--resolution=512 --train_batch_size=8 --lr_warmup_steps=0 \
--gradient_accumulation_steps=1 \
--learning_rate=5.0e-04 --scale_lr --lr_scheduler="constant" \
--mixed_precision=fp16 --revision=fp16 --gradient_checkpointing \
--only_save_embeds --num-trials 1 --examples-per-class 4 8 16 \
--save_steps 1000 \
--num_train_epochs 100 