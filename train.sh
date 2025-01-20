#!/bin/bash

python dataset_extract.py \
        --dataset mvtec_ad_setA \
        --output-dir ./output_A \
        --seed 99 \


bash fine_tune_A.sh 0 4 99
bash fine_tune_A.sh 0 8 99
bash fine_tune_A.sh 0 16 99
