#!/bin/bash

GPU=$1
EX=$2
SEED=$3

CUDA_VISIBLE_DEVICES=${GPU} \
python fine_tune_mvtec_A.py --dataset=mvtec_ad_setA --output_dir=./output_A \
--pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
--resolution=512 --train_batch_size=1 --lr_warmup_steps=0 \
--gradient_accumulation_steps=1 \
--learning_rate=5.0e-04 --scale_lr --lr_scheduler="constant" \
--mixed_precision=fp16 --revision=fp16 --gradient_checkpointing \
--only_save_embeds --num-trials 1 \
--examples-per-class ${EX} \
--num_train_epochs 5 \
--save_steps 2000 \
--seed ${SEED}

# --checkpointing_steps 1000 \
# --max_train_steps=2
# --max_train_steps=10000 \
# --examples-per-class 4 8 16 \
