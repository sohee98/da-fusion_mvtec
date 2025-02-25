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

# from semantic_aug.augmentations.textual_inversion import TextualInversion
from diffusers import StableDiffusionPipeline

# 모든 경고를 무시
warnings.filterwarnings("ignore")
# 로깅 레벨을 ERROR 이상으로 설정
logging.getLogger("diffusers").setLevel(logging.ERROR)

def load_learned_embeddings(pipe, embed_path):
    """
    학습된 임베딩을 .pt 파일에서 로드하고, 토크나이저 및 텍스트 인코더에 반영
    learned_embeds == 추가된 embeds : [768] shape
       {'<metal_nut>': tensor([-4.5464e-03, -5.9878e-03,  1.1627e-02, -4.9869e-03,  7.3673e-03, 1.9...1.0250e-02, -1.3794e-03], device='cuda:0'), 
        '<pill>': tensor([-4.6234e-03, -6.4240e-03,  1.1360e-02, -4.5929e-03,  6.9275e-03, 3.4...1.0002e-02, -1.1415e-03], device='cuda:0'), 
        '<scratch>': tensor([-4.6832e-03, -6.2059e-03,  1.1276e-02, -4.7374e-03,  6.8365e-03, 3.9...1.0045e-02, -1.1737e-03], device='cuda:0'), 
        '<color>': tensor([-4.6234e-03, -6.4240e-03,  1.1360e-02, -4.5929e-03,  6.9275e-03, 3.4...1.0002e-02, -1.1415e-03], device='cuda:0')}
    """
    learned_embeds = torch.load(embed_path, map_location="cuda")

    # 학습된 토큰 추가 및 임베딩 반영
    added_tokens = []
    for token, embedding in learned_embeds.items():
        pipe.tokenizer.add_tokens(token)
        added_tokens.append(token)

    # 텍스트 인코더의 임베딩 크기 조정
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))

    # 추가된 토큰에 대한 임베딩 반영
    for token, embedding in learned_embeds.items():
        token_id = pipe.tokenizer.convert_tokens_to_ids(token)
        pipe.text_encoder.get_input_embeddings().weight.data[token_id] = embedding.cuda()
    # pipe.text_encoder.get_input_embeddings() : Embedding(49412, 768) -> 원래 49408에서 4개 더해짐 

    print(f"Loaded embeddings for tokens: {', '.join(added_tokens)}")   #  <metal_nut>, <pill>, <scratch>, <color>
    return added_tokens

def generate_anomaly_prompt(added_tokens):
    """
    학습된 토큰 중 하나를 선택하여 프롬프트 생성.
    """
    # if len(added_tokens) < 2:
    #     raise ValueError("At least two tokens (class and anomaly) are required for prompt generation.")

    # class_token = added_tokens[0]
    # anomaly_token = added_tokens[1]
    # return f"photo of {class_token} with {anomaly_token} anomaly"

    """
    added_tokens를 절반으로 나눠 앞부분은 class_tokens, 뒷부분은 anomaly_tokens로 분리하고,
    조합하여 프롬프트 생성.
    """
    # 토큰 리스트를 반으로 나누기
    midpoint = len(added_tokens) // 2
    class_tokens = added_tokens[:midpoint]
    anomaly_tokens = added_tokens[midpoint:]

    # 프롬프트 생성
    prompts = []
    for class_token in class_tokens:
        for anomaly_token in anomaly_tokens:
            prompt = f"a photo of {class_token} with {anomaly_token} anomaly"
            prompts.append(prompt)

    return prompts, class_tokens, anomaly_tokens

def generate_normal_prompt(added_tokens):
    prompts = []
    for class_token in added_tokens:
        prompt = f"a photo of {class_token}"
        prompts.append(prompt)
    return prompts, added_tokens

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Stable Diffusion batch image generation script")

    # parser.add_argument("--model-path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--model-path", type=str, default="stabilityai/stable-diffusion-2-1")
    # parser.add_argument("--embed-path", type=str, default="output_A/fine-tuned/mvtec_ad_setA-0-4/mvtec_ad_setA-0-4.pt")
    parser.add_argument("--embed-path", type=str)

    # parser.add_argument("--mapping-file", type=str, default="prompt_mapping.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-generate", type=int, default=10)
    parser.add_argument("--out-dir", type=str, default="output_normal/fine-tuned/mvtec_ad_normal-0-4_images")
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    # parser.add_argument("--erasure-ckpt-name", type=str, default=None)

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

    if args.embed_path is None:
        # args.embed_path = "output_normal/fine-tuned/mvtec_ad_normal-0-4/step_1/mvtec_ad_normal-0-4.pt"
        # args.out_dir = "output_normal/fine-tuned/mvtec_ad_normal-0-4_images"
        args.out_dir = "output_normal_2.1/fine-tuned_ep10/mvtec_ad_normal-99-4_images"
        args.embed_path = "output_normal_2.1/fine-tuned_ep10/mvtec_ad_normal-99-4/step_1/mvtec_ad_normal-99-4.pt"
  
    out_dir = os.path.join(args.out_dir, 'step_1_normal', 'ep_10')
    os.makedirs(out_dir, exist_ok=True)

    # 학습된 임베딩 로드
    added_tokens = load_learned_embeddings(pipe, args.embed_path)

    # 학습된 토큰만 포함된 프롬프트 생성

    prompts, class_tokens = generate_normal_prompt(added_tokens)
    # prompts, class_tokens, anomaly_tokens = generate_anomaly_prompt(added_tokens)
    # 로깅: 토큰 분리 결과 확인
    print(f"Class Tokens: {class_tokens}")
    # print(f"Anomaly Tokens: {anomaly_tokens}")
    print(f"Generated Prompts: {prompts}")

    # 각 프롬프트로 이미지 생성
    for prompt in prompts:
        prompt_dir = os.path.join(out_dir, prompt.replace(" ", "_"), str(args.guidance_scale))

        os.makedirs(prompt_dir, exist_ok=True)

        for idx in trange(args.num_generate, desc=f"Generating Images for prompt: {prompt}"):
            with torch.no_grad(), autocast('cuda'):
                image = pipe(
                    prompt,
                    guidance_scale=args.guidance_scale
                ).images[0]

            image.save(os.path.join(prompt_dir, f"{idx}.png"))
            # print(f"Saved image to {prompt_dir}/{idx}.png")


    # available_folders = [
    #     name for name in os.listdir(args.embed_base_path)
    #     if os.path.isdir(os.path.join(args.embed_base_path, name))
    # ]

    # # prompt_mapping.json 파일 로드
    # with open(args.mapping_file, 'r', encoding='utf-8') as f:
    #     prompt_mapping = json.load(f)

    # # 각 클래스-프롬프트 쌍에 대해 반복
    # for class_name in available_folders:
    #     if class_name not in prompt_mapping:
    #         print(f"'{class_name}'에 대한 프롬프트가 없습니다. 스킵합니다.")
    #         continue
        
    #     prompt = prompt_mapping[class_name]
    #     print(f"-----------------Processing {class_name} with prompt: {prompt}-----------------")

    #     # 출력 디렉토리 설정
    #     output_dir = os.path.join(args.out_base_path, class_name)
    #     os.makedirs(output_dir, exist_ok=True)

    #     # 이미 출력 폴더가 존재하고 이미지가 있다면 스킵 
    #     if len(os.listdir(output_dir)) >= args.num_generate:
    #         print(f"이미 '{class_name}'에 대한 이미지가 존재합니다. 스킵합니다.")
    #         continue

    #     # 해당 클래스의 임베딩 경로 설정
    #     embed_path = os.path.join(args.embed_base_path, class_name, "learned_embeds.bin")

    #     if not os.path.exists(embed_path):
    #         print(f"임베딩 파일을 찾을 수 없습니다: {embed_path}")
    #         continue

    #     # 임베딩 로드 및 파이프라인 업데이트
    #     aug = TextualInversion(embed_path, model_path=args.model_path)
    #     pipe.tokenizer = aug.pipe.tokenizer
    #     pipe.text_encoder = aug.pipe.text_encoder

    #     # 이미지 생성 및 저장
    #     for idx in trange(args.num_generate, desc=f"Generating Images for {class_name}"):
    #         with autocast('cuda'):
    #             image = pipe(
    #                 prompt,
    #                 guidance_scale=args.guidance_scale
    #             ).images[0]

    #         image.save(os.path.join(output_dir, f"{idx}.png"))

    print("모든 이미지 생성이 완료되었습니다.")
