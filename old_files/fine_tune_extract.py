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
# from semantic_aug.datasets.mvtec_subset import MvtecDataset_subset

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
    "mvtec_ad_subset": MvtecDataset,
    "mvtec_ad_subset_2": MvtecDataset,
    "mvtec_ad_subset_3": MvtecDataset,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--only_save_embeds",
        action="store_true",
        default=False,
        help="Save only the embeddings for the new concept.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument("--learnable_property", type=str, default="object", help="Choose between 'object' and 'style'")
    parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    parser.add_argument("--num-trials", type=int, default=8)
    parser.add_argument("--examples-per-class", nargs='+', type=int, default=[1, 2, 4, 8, 16])
    
    parser.add_argument("--dataset", type=str, default="coco", 
                        choices=["spurge", "imagenet", "coco", "pascal", "mvtec_ad", "mvtec_ad_subset", "mvtec_ad_subset_2", "mvtec_ad_subset_3"])

    parser.add_argument("--unet-ckpt", type=str, default=None)

    parser.add_argument("--erase-concepts", action="store_true", 
                        help="erase text inversion concepts first")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


if __name__ == "__main__":

    args = parse_args()
    output_dir = args.output_dir

    rank = int(os.environ.pop("RANK", 0))
    world_size = int(os.environ.pop("WORLD_SIZE", 1))

    device_id = rank % torch.cuda.device_count()        
    torch.cuda.set_device(rank % torch.cuda.device_count())

    print(f'Initialized process {rank} / {world_size}')

    options = product(range(args.num_trials), args.examples_per_class)
    options = np.array(list(options))
    options = np.array_split(options, world_size)[rank]
    # [[0, 1],[0, 2],[0, 4],[0, 8],[0, 16], [1, 1],....., [7, 16]] (num_trials=8, examples_per_class=1 2 4 8 16)
    # seed, class별 이미지 개수 -> 모든 경우의수 별 모델 저장
    for seed, examples_per_class in options.tolist():

        if args.seed:
            seed = args.seed
            
        os.makedirs(os.path.join(output_dir, "extracted"), exist_ok=True)

        # train_dataset = DATASETS[
        #     args.dataset](split="train", seed=seed, 
        #                   examples_per_class=examples_per_class)
        train_dataset = DATASETS[
            args.dataset](args=args, examples_per_class=examples_per_class, seed=seed)

        for idx in range(len(train_dataset)):

            image = train_dataset.get_image_by_idx(idx)
            metadata = train_dataset.get_metadata_by_idx(idx)

            name = metadata["name"].replace(" ", "_")
            path = f"{args.dataset}-{seed}-{examples_per_class}"

            path = os.path.join(output_dir, "extracted", path, name, f"{idx}.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            image.save(path)
        
        # [extracted/dataset-seed-img개수] 폴더에 각각 class별 이미지 폴더 저장
