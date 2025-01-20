#!/bin/bash

NUM_TRIALS=1    # 고정 
# EXAMPLES_PER_CLASS=8
EXAMPLES_PER_CLASS=$1
DATASET=mvtec_ad
# mvtec_ad / mvtec_ad_subset
GPU=$2

## 이미지 추출
CUDA_VISIBLE_DEVICES=${GPU} \
python fine_tune_extract.py --dataset=${DATASET} \
        --output_dir=./output_new_${DATASET} \
        --num-trials ${NUM_TRIALS} --examples-per-class ${EXAMPLES_PER_CLASS}

## word embedding 학습
CUDA_VISIBLE_DEVICES=${GPU} \
python fine_tune_mvtec_new.py --dataset ${DATASET} --output_dir "./output_new_${DATASET}" \
        --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
        --resolution 512 --train_batch_size 8 --lr_warmup_steps 0 \
        --gradient_accumulation_steps 1 \
        --learning_rate 5.0e-04 --scale_lr --lr_scheduler "constant" \
        --mixed_precision fp16 --revision fp16 --gradient_checkpointing \
        --num-trials ${NUM_TRIALS} --examples-per-class ${EXAMPLES_PER_CLASS}  \
        --checkpointing_steps 1000 \
        --max_train_steps 1000
        # --only_save_embeds \
# --num-trials 1 --examples-per-class 4  \


## embedding .bin 파일 -> .pt 파일로 결합
CUDA_VISIBLE_DEVICES=${GPU} \
python aggregate_embeddings_mvtec.py \
        --num-trials ${NUM_TRIALS} \
        --examples-per-class ${EXAMPLES_PER_CLASS} \
        --root-path output_new_${DATASET} \
        --dataset ${DATASET} \
        --embed-path "{dataset}-tokens-new/{dataset}-{seed}-{examples_per_class}.pt"

# ## augmentation 이미지 생성
# TODO: seed 별로 반복문 구현
CUDA_VISIBLE_DEVICES=${GPU} \
python generate_augmentations_mvtec.py \
        --embed-path ${DATASET}-tokens-new/${DATASET}-0-${EXAMPLES_PER_CLASS}.pt \
        --dataset ${DATASET} \
        --out output_new_${DATASET}/aug \
        --examples-per-class ${EXAMPLES_PER_CLASS} \
        --seed 0 \
        --num-synthetic 10 \
        --guidance-scale 7.5 \
        --strength 0.5
