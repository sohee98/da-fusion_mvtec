#!/usr/bin/env python3

import os
import json

def generate_anomaly_prompt(class_name, anomaly_name):
    # mvtec ad dataset
    original_class_name = class_name  # 나중에 사용할 수 있도록 원본 클래스 이름 저장
    if class_name == 'cable':
        class_name = 'three cables'      # 복수형으로 전환

    # a/an 생략 조건 처리
    no_article_classes = ['carpet', 'grid', 'leather', 'tile', 'wood', 'cable']
    a_none = '' if class_name in no_article_classes else 'a '

    if anomaly_name != 'good':
        prompt = f"{a_none}{class_name} with {anomaly_name} anomaly"
    else:
        prompt = f"{a_none}{class_name}"
    
    return prompt.replace(" ", "_")

def main():
    # 대상 디렉토리 경로 설정
    target_dir = "output_mvtec/fine-tuned/mvtec_ad-0-1"
    
    # 디렉토리 내의 폴더 이름들을 저장할 리스트
    folder_names = []

    # 대상 디렉토리의 폴더들을 가져오기
    for item in os.listdir(target_dir):
        item_path = os.path.join(target_dir, item)
        if os.path.isdir(item_path):
            folder_names.append(item)

    # 클래스-이상치와 프롬프트 매핑을 저장할 딕셔너리
    prompt_mapping = {}

    for folder_name in folder_names:
        # 폴더 이름을 클래스명과 이상치명으로 분리
        try:
            class_name, anomaly_name = folder_name.split('-', 1)
        except ValueError:
            print(f"폴더 이름 형식이 올바르지 않습니다: {folder_name}")
            continue

        # 프롬프트 생성
        prompt = generate_anomaly_prompt(class_name, anomaly_name)

        # 매핑 저장
        prompt_mapping[folder_name] = prompt

    # 결과를 JSON 파일로 저장
    output_file = "prompt_mapping.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(prompt_mapping, f, ensure_ascii=False, indent=4)

    print(f"프롬프트 매핑이 '{output_file}' 파일로 저장되었습니다.")

    # 매핑 내용을 출력 (선택 사항)
    for folder, prompt in prompt_mapping.items():
        print(f"{folder}: {prompt}")

if __name__ == "__main__":
    main()
