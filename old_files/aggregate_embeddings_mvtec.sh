#!/bin/bash
# ROOT_FOLDER=output_mvtec/fine-tuned/mvtec_ad-0-1

python aggregate_embeddings_mvtec.py \
        --num-trials 1 \
        --examples-per-class 4 \
        --root-path output_mvtec \
        --dataset mvtec_ad