#!/bin/bash

NUM_TRIALS=1    # 고정 
# EXAMPLES_PER_CLASS=8
# EXAMPLES_PER_CLASS=$1
DATASET=mvtec_ad_subset_3
# mvtec_ad / mvtec_ad_subset
GPU=0
# SEED=100
# SEED=0
# SEED=$3

# for ex in 8 16
for ex in 16
do  
        for params in "7.5 0.5" "10.0 0.3" "10.0 0.5"
        do
                gscale=$(echo $params | cut -d' ' -f1)
                strength=$(echo $params | cut -d' ' -f2)

                CUDA_VISIBLE_DEVICES=${GPU} \
                python fid.py \
                        --dataset ${DATASET} \
                        --out output_${DATASET}/aug \
                        --examples-per-class ${ex} \
                        --seed 0 \
                        --guidance-scale ${gscale} \
                        --strength ${strength}
        done
done

# for params in "7.5 0.5" "10.0 0.3" "10.0 0.5"
# do
#         gscale=$(echo $params | cut -d' ' -f1)
#         strength=$(echo $params | cut -d' ' -f2)

#         CUDA_VISIBLE_DEVICES=${GPU} \
#         python fid.py \
#                 --dataset ${DATASET} \
#                 --out output_${DATASET}/aug \
#                 --examples-per-class 5 \
#                 --guidance-scale ${gscale} \
#                 --strength ${strength} \
#                 --seed 100
# done