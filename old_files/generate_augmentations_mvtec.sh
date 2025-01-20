#!/bin/bash

# CUDA_VISIBLE_DEVICES=1 \
# python generate_augmentations_mvtec.py \
#         --embed-path mvtec_ad-tokens/mvtec_ad-0-1.pt \
#         --dataset mvtec_ad \
        
NUM_TRIALS=1    # 고정 
EXAMPLES_PER_CLASS=4
DATASET=mvtec_ad_subset

# CUDA_VISIBLE_DEVICES=1 \
# python generate_augmentations_mvtec.py \
#         --embed-path ${DATASET}-tokens/${DATASET}-0-${EXAMPLES_PER_CLASS}.pt \
#         --dataset ${DATASET} \
#         --out output_${DATASET}/aug \
#         --examples-per-class ${EXAMPLES_PER_CLASS} \
#         --seed 0 \
#         --num-synthetic 10


# CUDA_VISIBLE_DEVICES=1 python generate_augmentations_mvtec.py \
# --embed-path mvtec_ad_subset-tokens/mvtec_ad_subset-0-4.pt \
# --dataset mvtec_ad_subset \
# --out output_mvtec_ad_subset/aug \
# --examples-per-class 4 --seed 0 --num-synthetic 2

CUDA_VISIBLE_DEVICES=2 python generate_augmentations_mvtec.py \
--embed-path mvtec_ad_subset_2-tokens/mvtec_ad_subset-0-4.pt \
--dataset mvtec_ad_subset_2 \
--out output_mvtec_ad_subset_2/aug \
--examples-per-class 4 --seed 0 --num-synthetic 20 \
--guidance-scale 10 \
--strength 0.5

# CUDA_VISIBLE_DEVICES=1 python generate_augmentations_mvtec.py \
# --embed-path mvtec_ad-tokens/mvtec_ad-0-8.pt \
# --dataset mvtec_ad \
# --out output_mvtec_ad/aug \
# --examples-per-class 8 --seed 0 --num-synthetic 10

CUDA_VISIBLE_DEVICES=1 python generate_augmentations_mvtec.py \
--embed-path mvtec_ad-tokens/mvtec_ad-0-4.pt \
--dataset mvtec_ad \
--out output_mvtec_ad/aug \
--examples-per-class 4 --seed 0 --num-synthetic 4

## default
# --guidance_scale 7.5 \
# --strength 0.5

CUDA_VISIBLE_DEVICES=1 python generate_augmentations_mvtec.py \
--embed-path mvtec_ad-tokens/mvtec_ad-0-4.pt \
--dataset mvtec_ad \
--out output_mvtec_ad/aug \
--examples-per-class 4 --seed 0 --num-synthetic 4 \
--guidance_scale 7.5 \
--strength 0.1