#!/bin/bash
ROOT_FOLDER=output_mvtec/fine-tuned/mvtec_ad-0-2_epoch100

CUDA_VISIBLE_DEVICES=2 \
python generate_images_mvtec.py \
        --embed-base-path ${ROOT_FOLDER} \
        --out-base-path ${ROOT_FOLDER}_images \
