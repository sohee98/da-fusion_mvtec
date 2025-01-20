#!/bin/bash

# JSON 파일 경로 설정
JSON_FILE="prompt_mapping.json"

# jq가 설치되어 있는지 확인
if ! command -v jq &> /dev/null; then
    echo "jq 명령어가 필요합니다. 설치 후 다시 시도하세요."
    exit 1
fi

# JSON 파일이 존재하는지 확인
if [ ! -f "$JSON_FILE" ]; then
    echo "JSON 파일을 찾을 수 없습니다: $JSON_FILE"
    exit 1
fi

# JSON 파일에서 키 목록 가져오기 (폴더 이름들)
keys=$(jq -r 'keys[]' "$JSON_FILE")
ROOT_FOLDER=output_mvtec/fine-tuned/mvtec_ad-0-1

for key in $keys; do
    # CLASS_NAME 설정 (폴더 이름)
    CLASS_NAME="$key"
    # PROMPT 가져오기
    PROMPT=$(jq -r --arg k "$key" '.[$k]' "$JSON_FILE")

    if [ -d "${ROOT_FOLDER}_images/${CLASS_NAME}" ]; then
        echo "이미 '${CLASS_NAME}'에 대한 폴더가 존재합니다."
        continue
    fi

    # 명령 실행 전에 CLASS_NAME과 PROMPT 확인
    echo "CLASS_NAME: $CLASS_NAME"
    echo "PROMPT: $PROMPT"

    CUDA_VISIBLE_DEVICES=1 \
    python generate_images.py \
            --embed-path ${ROOT_FOLDER}/${CLASS_NAME}/learned_embeds.bin \
            --prompt ${PROMPT} \
            --out ${ROOT_FOLDER}_images/${CLASS_NAME}/ \

    echo "----------------------------------------"
done