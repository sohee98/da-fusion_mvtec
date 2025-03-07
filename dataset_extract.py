import argparse
import logging
import math
import os
import gc
import shutil
import random
from pathlib import Path
from typing import Optional
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

from semantic_aug.datasets.coco import COCODataset
from semantic_aug.datasets.spurge import SpurgeDataset
from semantic_aug.datasets.imagenet import ImageNetDataset
from semantic_aug.datasets.pascal import PASCALDataset
from semantic_aug.datasets.mvtec import MvtecDataset
from semantic_aug.datasets.mvtec_normal import MvtecDataset_Normal

# import datasets
# import diffusers
# import PIL
# import transformers
# from accelerate import Accelerator
from accelerate.logging import get_logger
# from accelerate.utils import set_seed
# from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
# from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
# from diffusers.utils.import_utils import is_xformers_available
# from huggingface_hub import HfFolder, Repository, whoami


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")


logger = get_logger(__name__)

DATASETS = {
    "spurge": SpurgeDataset, 
    "coco": COCODataset, 
    "pascal": PASCALDataset,
    "imagenet": ImageNetDataset,
    "mvtec_ad": MvtecDataset,
    "mvtec_ad_setA": MvtecDataset,
    "mvtec_ad_normal": MvtecDataset_Normal,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Extract dataset for training")

    parser.add_argument("--examples-per-class", nargs='+', type=int, default=[4, 8, 16])
    parser.add_argument("--dataset", type=str, default="mvtec_ad_normal", choices=["mvtec_ad", "mvtec_ad_setA", "mvtec_ad_normal"])
    parser.add_argument("--output-dir", type=str, default="output_normal_2.1")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-trials", type=int, default=1)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    output_dir = args.output_dir

    options = product(range(args.num_trials), args.examples_per_class)
    options = np.array(list(options))
    # options = np.array_split(options, world_size)[rank]
    # [[0, 1],[0, 2],[0, 4],[0, 8],[0, 16], [1, 1],....., [7, 16]] (num_trials=8, examples_per_class=1 2 4 8 16)
    # seed, class별 이미지 개수 -> 모든 경우의수 별 모델 저장
    for seed, examples_per_class in options.tolist():

        if args.seed:
            seed = args.seed
            
        os.makedirs(os.path.join(output_dir, "extracted"), exist_ok=True)

        train_dataset = DATASETS[
            args.dataset](args=args, examples_per_class=examples_per_class, seed=seed)

        for idx in range(len(train_dataset)):

            image = train_dataset.get_image_by_idx(idx)
            metadata = train_dataset.get_metadata_by_idx(idx)
            name = metadata["name"].replace(" ", "_")
            setting = f"{args.dataset}-{seed}-{examples_per_class}"

            if metadata['is_anomaly']:  # 비정상 이미지
                path = os.path.join(output_dir, "extracted", setting, "anomaly", name, f"{idx}.png")
            else:                       # 정상 이미지 
                path = os.path.join(output_dir, "extracted", setting, "good", name, f"{idx}.png")
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            image.save(path)
        
        # [extracted/dataset-seed-img개수] 폴더에 각각 class별 이미지 폴더 저장
