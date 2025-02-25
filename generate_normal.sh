#!/bin/bash

## prompt로만 생성성
# for gs in 7.5 5.0 10.0; do
for gs in 7.5; do
    CUDA_VISIBLE_DEVICES=2 python generate_images_normal.py \
        --out-dir output_normal_2.1/fine-tuned_ep10/mvtec_ad_normal-99-4_images \
        --embed-path output_normal_2.1/fine-tuned_ep10/mvtec_ad_normal-99-4/step_1/mvtec_ad_normal-99-4.pt \
        --num-generate 10 \
        --seed 99 \
        --model-path "stabilityai/stable-diffusion-2-1" \
        --guidance-scale ${gs} 

    CUDA_VISIBLE_DEVICES=1 \
    python generate_images_normal.py \
        --out-dir output_normal_2.1/fine-tuned_ep10/mvtec_ad_normal-99-8_images \
        --embed-path output_normal_2.1/fine-tuned_ep10/mvtec_ad_normal-99-8/step_1/mvtec_ad_normal-99-8.pt \
        --num-generate 10 \
        --seed 99 \
        --model-path "stabilityai/stable-diffusion-2-1" \
        --guidance-scale ${gs}

    CUDA_VISIBLE_DEVICES=1 \
    python generate_images_normal.py \
        --out-dir output_normal_2.1/fine-tuned_ep10/mvtec_ad_normal-99-16_images \
        --embed-path output_normal_2.1/fine-tuned_ep10/mvtec_ad_normal-99-16/step_1/mvtec_ad_normal-99-16.pt \
        --num-generate 10 \
        --seed 99 \
        --model-path "stabilityai/stable-diffusion-2-1" \
        --guidance-scale ${gs}
done