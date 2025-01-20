import os
import torch
import argparse
import numpy as np
import random
import json
import warnings
import logging
from tqdm import trange
from torch import autocast
from PIL import Image

from semantic_aug.augmentations.textual_inversion import TextualInversion
from diffusers import StableDiffusionPipeline

# 모든 경고를 무시
warnings.filterwarnings("ignore")
# 로깅 레벨을 ERROR 이상으로 설정
logging.getLogger("diffusers").setLevel(logging.ERROR)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Stable Diffusion batch image generation script")

    parser.add_argument("--model-path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--embed-base-path", type=str, default="output_mvtec/fine-tuned/mvtec_ad-0-1")
    parser.add_argument("--mapping-file", type=str, default="prompt_mapping.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-generate", type=int, default=10)
    parser.add_argument("--out-base-path", type=str, default="output_mvtec/fine-tuned/mvtec_ad-0-1_images")
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--erasure-ckpt-name", type=str, default=None)

    args = parser.parse_args()

    # 랜덤 시드 설정
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 모델 로드 (한 번만 실행)
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path,
        use_auth_token=True,
        torch_dtype=torch.float16,
        revision="fp16",
        # token=True  # 필요한 경우 토큰 사용
    ).to('cuda')

    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None

    if args.erasure_ckpt_name is not None:
        pipe.unet.load_state_dict(torch.load(
            args.erasure_ckpt_name, map_location='cuda'))
    
    available_folders = [
        name for name in os.listdir(args.embed_base_path)
        if os.path.isdir(os.path.join(args.embed_base_path, name))
    ]
    
    # prompt_mapping.json 파일 로드
    with open(args.mapping_file, 'r', encoding='utf-8') as f:
        prompt_mapping = json.load(f)

    # 각 클래스-프롬프트 쌍에 대해 반복
    for class_name in available_folders:
        if class_name not in prompt_mapping:
            print(f"'{class_name}'에 대한 프롬프트가 없습니다. 스킵합니다.")
            continue
        
        prompt = prompt_mapping[class_name]
        print(f"-----------------Processing {class_name} with prompt: {prompt}-----------------")

        # 출력 디렉토리 설정
        output_dir = os.path.join(args.out_base_path, class_name)
        os.makedirs(output_dir, exist_ok=True)

        # 이미 출력 폴더가 존재하고 이미지가 있다면 스킵 
        if len(os.listdir(output_dir)) >= args.num_generate:
            print(f"이미 '{class_name}'에 대한 이미지가 존재합니다. 스킵합니다.")
            continue

        # 해당 클래스의 임베딩 경로 설정
        embed_path = os.path.join(args.embed_base_path, class_name, "learned_embeds.bin")

        if not os.path.exists(embed_path):
            print(f"임베딩 파일을 찾을 수 없습니다: {embed_path}")
            continue

        # 임베딩 로드 및 파이프라인 업데이트
        aug = TextualInversion(embed_path, model_path=args.model_path)
        pipe.tokenizer = aug.pipe.tokenizer
        pipe.text_encoder = aug.pipe.text_encoder

        # 이미지 생성 및 저장
        for idx in trange(args.num_generate, desc=f"Generating Images for {class_name}"):
            with autocast('cuda'):
                image = pipe(
                    prompt,
                    guidance_scale=args.guidance_scale
                ).images[0]

            image.save(os.path.join(output_dir, f"{idx}.png"))

    print("모든 이미지 생성이 완료되었습니다.")
