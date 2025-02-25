#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python dataset_extract.py \
        --dataset mvtec_ad_normal \
        --output-dir ./output_normal \
        --seed 99 \
        --examples-per-class 4 8 16

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
--num_train_epochs 150 


