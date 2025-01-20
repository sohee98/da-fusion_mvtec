#!/bin/bash

NUM_TRIALS=1    # 고정 
# EXAMPLES_PER_CLASS=8
EXAMPLES_PER_CLASS=$1
DATASET=mvtec_ad_subset_3
# mvtec_ad / mvtec_ad_subset
GPU=$2
# SEED=100
# SEED=0
SEED=$3

## 이미지 추출
# CUDA_VISIBLE_DEVICES=${GPU} \
# python fine_tune_extract.py --dataset=${DATASET} \
#         --output_dir=./output_${DATASET} \
#         --num-trials ${NUM_TRIALS} --examples-per-class ${EXAMPLES_PER_CLASS} \
#         --seed ${SEED} \

## word embedding 학습
# CUDA_VISIBLE_DEVICES=${GPU} \
# python fine_tune_mvtec_new.py --dataset ${DATASET} --output_dir "./output_${DATASET}" \
#         --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
#         --resolution 512 --train_batch_size 8 --lr_warmup_steps 0 \
#         --gradient_accumulation_steps 1 \
#         --learning_rate 5.0e-04 --scale_lr --lr_scheduler "constant" \
#         --mixed_precision fp16 --revision fp16 --gradient_checkpointing \
#         --num-trials ${NUM_TRIALS} --examples-per-class ${EXAMPLES_PER_CLASS}  \
#         --checkpointing_steps 1000 --save_steps 1000 \
#         --max_train_steps 1000 \
#         --seed ${SEED} \
        # --only_save_embeds \
# --num-trials 1 --examples-per-class 4  \


## embedding .bin 파일 -> .pt 파일로 결합
# CUDA_VISIBLE_DEVICES=${GPU} \
# python aggregate_embeddings_mvtec.py \
#         --num-trials ${NUM_TRIALS} \
#         --examples-per-class ${EXAMPLES_PER_CLASS} \
#         --root-path output_${DATASET} \
#         --dataset ${DATASET} \
#         --embed-path "{dataset}-tokens/{dataset}-{seed}-{examples_per_class}.pt" \
#         --seed ${SEED}

# ## augmentation 이미지 생성
# TODO: seed 별로 반복문 구현
# CUDA_VISIBLE_DEVICES=${GPU} \
# python generate_augmentations_mvtec.py \
#         --embed-path ${DATASET}-tokens/${DATASET}-${SEED}-${EXAMPLES_PER_CLASS}.pt \
#         --dataset ${DATASET} \
#         --out output_${DATASET}/aug \
#         --examples-per-class ${EXAMPLES_PER_CLASS} \
#         --num-synthetic 10 \
#         --guidance-scale 10 \
#         --strength 0.5 \
#         --seed ${SEED} 

# CUDA_VISIBLE_DEVICES=${GPU} \
# python generate_augmentations_mvtec.py \
#         --embed-path ${DATASET}-tokens/${DATASET}-${SEED}-${EXAMPLES_PER_CLASS}.pt \
#         --dataset ${DATASET} \
#         --out output_${DATASET}/aug \
#         --examples-per-class ${EXAMPLES_PER_CLASS} \
#         --num-synthetic 10 \
#         --guidance-scale 10 \
#         --strength 0.3 \
#         --seed ${SEED} 


