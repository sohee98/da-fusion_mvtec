#!/bin/bash

## prompt로만 생성성
for gs in 7.5 5.0 10.0; do
    python generate_images_mvtec_A.py \
        --out-dir output_A/fine-tuned_gen/mvtec_ad_setA-0-4 \
        --embed-path output_A/fine-tuned/mvtec_ad_setA-0-4/learned_embeds-steps-8000.pt \
        --num-generate 10 \
        --guidance-scale ${gs}

    python generate_images_mvtec_A.py \
        --out-dir output_A/fine-tuned_gen/mvtec_ad_setA-0-8 \
        --embed-path output_A/fine-tuned/mvtec_ad_setA-0-8/learned_embeds-steps-16000.pt \
        --num-generate 10 \
        --guidance-scale ${gs}

    python generate_images_mvtec_A.py \
        --out-dir output_A/fine-tuned_gen/mvtec_ad_setA-0-16 \
        --embed-path output_A/fine-tuned/mvtec_ad_setA-0-16/learned_embeds-steps-32000.pt \
        --num-generate 10 \
        --guidance-scale ${gs}
done



## 정상 이미지 + prompt 로 생성
for gs in 7.5 5.0 10.0; do
    for strength in 0.5 0.3 0.7; do
        python generate_augmentations_mvtec_A.py \
            --dataset mvtec_ad_setA \
            --out output_A/aug \
            --embed-path output_A/fine-tuned/mvtec_ad_setA-0-4/learned_embeds-steps-8000.pt \
            --examples-per-class 4 \
            --guidance-scale ${gs} \
            --strength ${strength}
        
        python generate_augmentations_mvtec_A.py \
            --dataset mvtec_ad_setA \
            --out output_A/aug \
            --embed-path output_A/fine-tuned/mvtec_ad_setA-0-8/learned_embeds-steps-16000.pt \
            --examples-per-class 8 \
            --guidance-scale ${gs} \
            --strength ${strength}

        python generate_augmentations_mvtec_A.py \
            --dataset mvtec_ad_setA \
            --out output_A/aug \
            --embed-path output_A/fine-tuned/mvtec_ad_setA-0-16/learned_embeds-steps-32000.pt \
            --examples-per-class 16 \
            --guidance-scale ${gs} \
            --strength ${strength}
    done
done
