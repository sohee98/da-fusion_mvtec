#!/bin/bash

NUM_TRIALS=1    # 고정 
EXAMPLES_PER_CLASS=4
DATASET=real

## 이미지 추출
# CUDA_VISIBLE_DEVICES=2 \
# python fine_tune_extract.py --dataset=${DATASET} \
#         --output_dir=./output_${DATASET} \
#         --num-trials ${NUM_TRIALS} --examples-per-class ${EXAMPLES_PER_CLASS}

## word embedding 학습
# CUDA_VISIBLE_DEVICES=2 \
# python finetune_real.py --output_dir "./output_real" \
#         --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
#         --resolution 512 --train_batch_size 8 --lr_warmup_steps 0 \
#         --gradient_accumulation_steps 1 \
#         --learning_rate 5.0e-04 --scale_lr --lr_scheduler "constant" \
#         --mixed_precision fp16 --revision fp16 --gradient_checkpointing \
#         --only_save_embeds \
#         --num-trials ${NUM_TRIALS} --examples-per-class ${EXAMPLES_PER_CLASS}  \
#         --checkpointing_steps 1000 \
#         --max_train_steps 1000


## embedding .bin 파일 -> .pt 파일로 결합
# CUDA_VISIBLE_DEVICES=2 \
# python aggregate_embeddings_mvtec.py \
#         --num-trials ${NUM_TRIALS} \
#         --examples-per-class ${EXAMPLES_PER_CLASS} \
#         --root-path output_${DATASET} \
#         --dataset ${DATASET} \
#         --embed-path "{dataset}-tokens/{dataset}-{seed}-{examples_per_class}.pt"

# ## augmentation 이미지 생성
# TODO: seed 별로 반복문 구현
CUDA_VISIBLE_DEVICES=1 \
python generate_augmentations_real.py \
        --embed-path ${DATASET}-tokens/${DATASET}-0-${EXAMPLES_PER_CLASS}.pt \
        --dataset ${DATASET} \
        --out output_${DATASET}/aug \
        --examples-per-class ${EXAMPLES_PER_CLASS} \
        --seed 0 \
        --num-synthetic 10 \
        --guidance-scale 7.5 \
        --strength 0.3 
