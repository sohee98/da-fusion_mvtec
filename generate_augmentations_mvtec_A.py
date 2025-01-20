from semantic_aug.datasets.coco import COCODataset
from semantic_aug.datasets.spurge import SpurgeDataset
from semantic_aug.datasets.imagenet import ImageNetDataset
from semantic_aug.datasets.pascal import PASCALDataset
from semantic_aug.datasets.mvtec import MvtecDataset
# from semantic_aug.datasets.mvtec_subset import MvtecDataset_subset

from semantic_aug.augmentations.compose import ComposeParallel
from semantic_aug.augmentations.compose import ComposeSequential
from semantic_aug.augmentations.real_guidance import RealGuidance
from semantic_aug.augmentations.textual_inversion import TextualInversion
from diffusers import StableDiffusionPipeline
from itertools import product
from torch import autocast
from PIL import Image

from tqdm import tqdm
import os
import torch
import argparse
import numpy as np
import random
import json 

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pdb 

DATASETS = {
    "spurge": SpurgeDataset, 
    "coco": COCODataset, 
    "pascal": PASCALDataset,
    "imagenet": ImageNetDataset,
    "mvtec_ad": MvtecDataset,
    "mvtec_ad_subset": MvtecDataset,
    "mvtec_ad_subset_2": MvtecDataset,
    "mvtec_ad_subset_3": MvtecDataset,
    "mvtec_ad_setA": MvtecDataset,
}

COMPOSE = {
    "parallel": ComposeParallel,
    "sequential": ComposeSequential
}

AUGMENT = {
    "real-guidance": RealGuidance,
    "textual-inversion": TextualInversion
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Inference script")
    
    parser.add_argument("--out", type=str, default="output_A/aug")
    parser.add_argument("--model-path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--embed-path", type=str, default="output_A/fine-tuned/mvtec_ad_setA-0-4/learned_embeds-steps-8000.pt")
    parser.add_argument("--dataset", type=str, default="mvtec_ad_setA")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples-per-class", type=int, default=4)
    parser.add_argument("--num-synthetic", type=int, default=10)
    # parser.add_argument("--prompt", type=str, default="a photo of {name}")
    parser.add_argument("--aug", nargs="+", type=str, default=["textual-inversion"], 
                        choices=["real-guidance", "textual-inversion"])
    parser.add_argument("--guidance-scale", nargs="+", type=float, default=[7.5])
    parser.add_argument("--strength", nargs="+", type=float, default=[0.5])
    parser.add_argument("--mask", nargs="+", type=int, default=[0], choices=[0, 1])
    parser.add_argument("--inverted", nargs="+", type=int, default=[0], choices=[0, 1])
    parser.add_argument("--probs", nargs="+", type=float, default=None)
    parser.add_argument("--compose", type=str, default="parallel", 
                        choices=["parallel", "sequential"])

    # parser.add_argument("--class-name", type=str, default=None)
    parser.add_argument("--erasure-ckpt-path", type=str, default=None)

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load the Stable Diffusion model
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path,
        use_auth_token=True,
        torch_dtype=torch.float16,
    ).to('cuda')

    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None

    # Load learned embeddings
    learned_embeds = torch.load(args.embed_path, map_location="cuda")
    class_tokens = []
    anomaly_tokens = []

    for i, (token, embedding) in enumerate(learned_embeds.items()):
        pipe.tokenizer.add_tokens(token)
        token_id = pipe.tokenizer.convert_tokens_to_ids(token)
        pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
        pipe.text_encoder.get_input_embeddings().weight.data[token_id] = embedding.cuda()

        if i < len(learned_embeds) // 2:
            class_tokens.append(token)
        else:
            anomaly_tokens.append(token)


    # prompt_json = json.load(open('prompt_mapping.json', 'r'))
    # args.prompt = "a photo of {prompt_json[name]}"
    # prompt_json[name]

    # aug = TextualInversion(
    #     embed_path=args.embed_path, 
    #     model_path=args.model_path,
    #     prompt="",  # Prompt will be set dynamically
    #     strength=args.strength, 
    #     guidance_scale=args.guidance_scale
    # )

    aug = COMPOSE[args.compose]([
        AUGMENT[aug](
            embed_path=args.embed_path, 
            model_path=args.model_path, 
            prompt="", 
            strength=strength, 
            guidance_scale=guidance_scale,
            mask=mask, 
            inverted=inverted,
            erasure_ckpt_path=args.erasure_ckpt_path
        )
        for (aug, guidance_scale, 
             strength, mask, inverted) in zip(
            args.aug, args.guidance_scale, 
            args.strength, args.mask, args.inverted
        )
    ], probs=args.probs)

    # train_dataset = DATASETS[
    #     args.dataset](split="train", seed=args.seed, 
    #                   examples_per_class=args.examples_per_class)
    # train_dataset = DATASETS[args.dataset](args=args, examples_per_class=args.examples_per_class, seed=args.seed)
    train_dataset = MvtecDataset(args=args, examples_per_class=args.examples_per_class, seed=args.seed)

    # normal_image_num = len(train_dataset.normal_image)


    # options = product(range(len(train_dataset)), range(args.num_synthetic))
    options = product(range(0,len(train_dataset),args.examples_per_class),          # class당 한장씩만 하도록 
                      range(args.num_synthetic))
    # (range(0, 88), range(0, 10)) - 1장씩 일 때 
    # args.examples_per_class

    # pdb.set_trace()
    # save_path = os.path.join(args.out, f"{args.dataset}-{args.seed}-{args.examples_per_class}")
    # os.makedirs(save_path, exist_ok=True)

    dirname = f"{args.dataset}-{args.seed}-{args.examples_per_class}"
    setting_name = f'gscale_{args.guidance_scale[0]}-strength_{args.strength[0]}'
    save_path = os.path.join(args.out, dirname, setting_name)
    os.makedirs(save_path, exist_ok=True)

    for idx, num in tqdm(list(options), desc="Generating Augmentations"):
        normal_image = train_dataset.get_normal_image_by_idx(idx)
        label = train_dataset.get_label_by_idx(idx)         # (class_name, anomaly_type)
        metadata = train_dataset.get_metadata_by_idx(idx)

        class_name, anomaly_name = label
        class_token = '<' + class_name + '>'
        # anomaly_name = metadata['name'].split('-')[1]
        anomaly_token = '<' + anomaly_name + '>'
        prompt = f"a photo of {class_token} with {anomaly_token} anomaly"

        # Create class-specific output directory
        class_name = metadata['name'].split('-')[0]
        class_dir = os.path.join(save_path, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Save the original normal image
        normal_image.save(os.path.join(class_dir, "normal.png"))

        # Generate synthetic image using the prompt
        with torch.autocast('cuda'):
            aug_image, label = aug(normal_image, label, metadata)

        # Save the synthetic image
        synthetic_dir = os.path.join(class_dir, f"{class_token}_{anomaly_token}")
        os.makedirs(synthetic_dir, exist_ok=True)
        aug_image.save(os.path.join(synthetic_dir, f"{idx}-{num}.png"))


        # image = train_dataset.get_image_by_idx(idx)
        # label = train_dataset.get_label_by_idx(idx)         # (class_name, anomaly_type)
        # metadata = train_dataset.get_metadata_by_idx(idx)   # (name, path, prompt)
        # # {"name": f"{class_name}-{anomaly_type}", "path": image_path}
        # # {'name': 'carpet-thread', 'path': '/SSD1/datasets/mvtec_ad/carpet/test/thread/009.png', 'prompt': 'carpet with thread anomaly'}
        # normal_image = train_dataset.get_normal_image_by_idx(idx)   # class 이미지들 중 정상 이미지 첫번째걸로 고정 추출 
        # # normal_image = train_dataset.get_random_normal_image_by_idx(idx)   # class 이미지들 중 무작위로 정상 이미지 추출 

        # save_path_class = os.path.join(save_path, f"{label[0]}")
        # os.makedirs(save_path_class, exist_ok=True)
        # normal_image.save(os.path.join(save_path_class, f"normal.png"))    # class 당 한장만 

        # # if args.class_name is not None: 
        # #     if metadata["name"] != args.class_name: continue
        
        # # prompt가 placeholder token 이기 때문에 name prompt로 변경
        # # metadata['name'] = metadata['prompt'].replace(" ", "_")    
        # name = metadata['name']
        # os.makedirs(os.path.join(save_path_class, f"{name}"), exist_ok=True)
        # ## 원본이미지 저장
        # # normal_image.save(os.path.join(save_path, f"{name}/{idx}-{num}-normal.png"))    # random 으로 불러올 때

        # ## 이미지 생성 
        # aug_image, label = aug(normal_image, label, metadata)

        # pil_image, image_path = aug_image, os.path.join(save_path_class, f"{name}/{idx}-{num}.png")
        # pil_image.save(image_path)