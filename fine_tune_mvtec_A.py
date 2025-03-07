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

import datasets
import diffusers
import PIL
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, whoami

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import datetime

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


if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")


logger = get_logger(__name__)


# def save_progress(text_encoder, placeholder_token_id, accelerator, args, save_path):
#     logger.info("Saving embeddings")
#     learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
#     learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}
#     torch.save(learned_embeds_dict, save_path)
def save_progress(text_encoder, placeholder_token_ids, accelerator, args, save_path):
    """
    Save embeddings for multiple placeholder tokens.
    Args:
        text_encoder: The text encoder model.
        placeholder_token_ids: List of token IDs for the placeholders.
        accelerator: The Hugging Face Accelerator instance.
        args: Arguments containing placeholder token names.
        save_path: Path to save the embeddings.
    """
    logger.info("Saving embeddings")
    # Ensure placeholder_token_ids is a list
    if isinstance(placeholder_token_ids, int):  # 단일 값인 경우
        placeholder_token_ids = [placeholder_token_ids]  # 리스트로 변환

    # Unwrap the text encoder and get input embeddings
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight

    # Create a dictionary to store embeddings for each placeholder token
    learned_embeds_dict = {
        token: learned_embeds[token_id].detach().cpu()
        for token, token_id in zip(args.placeholder_token_list, placeholder_token_ids)
    }

    # Save the embeddings dictionary
    torch.save(learned_embeds_dict, save_path)
    logger.info(f"Saved embeddings to {save_path}")


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
        default="CompVis/stable-diffusion-v1-4",
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
        default="./output_A/",
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
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
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
    parser.add_argument("--examples-per-class", nargs='+', type=int, default=[4, 8, 16])
    
    parser.add_argument("--dataset", type=str, default="mvtec_ad_setA", 
                        choices=["spurge", "imagenet", "coco", "pascal", "mvtec_ad", "mvtec_ad_setA"])

    parser.add_argument("--unet-ckpt", type=str, default=None)

    parser.add_argument("--erase-concepts", action="store_true", 
                        help="erase text inversion concepts first")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

    
class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        mode="train",
        placeholder_token_list=["*"],       # list로 수정
        center_crop=False,
        class_token_list=None,              # add
        anomaly_token_list=None,            # add
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token_list = placeholder_token_list
        self.center_crop = center_crop
        self.flip_p = flip_p

        '''
        self.image_paths
        ['./output_A/extracted/mvtec_ad_setA-0-4/metal_nut-scratch', 
        './output_A/extracted/mvtec_ad_setA-0-4/pill-scratch', 
        './output_A/extracted/mvtec_ad_setA-0-4/pill-color', 
        './output_A/extracted/mvtec_ad_setA-0-4/metal_nut-color']
        '''

        # 이미지 경로, class_token, anomaly_token 매핑
        self.image_paths = []
        self.class_tokens = []
        self.anomaly_tokens = []

        # data_root 내부의 하위 디렉토리를 순회하며 매핑 생성
        for folder in os.listdir(data_root):
            folder_path = os.path.join(data_root, folder)
            if not os.path.isdir(folder_path):
                continue  # 폴더가 아닌 경우 건너뜀

            # 폴더 이름에서 class와 anomaly 추출
            class_name, anomaly_name = folder.split("-")
            class_token = f"<{class_name}>"
            anomaly_token = f"<{anomaly_name}>"

            # 폴더 내 이미지 경로 추가
            for image_file in os.listdir(folder_path):
                if image_file.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(folder_path, image_file))
                    self.class_tokens.append(class_token)
                    self.anomaly_tokens.append(anomaly_token) 

        self.num_images = len(self.image_paths)     # 전체 이미지 갯수

        # 유효성 검사
        if class_token_list:    # ['<metal_nut>', '<pill>']
            assert set(self.class_tokens) <= set(class_token_list), \
                "Some class tokens in the dataset are not in the class_token_list."
        if anomaly_token_list:  # ['<scratch>', '<color>']
            assert set(self.anomaly_tokens) <= set(anomaly_token_list), \
                "Some anomaly tokens in the dataset are not in the anomaly_token_list."

        self._length = self.num_images * repeats if mode == "train" else self.num_images
        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

        # self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small

    def __len__(self):
        return self._length     # 전체 이미지 갯수 * repeats


    def __getitem__(self, i):
        # 인덱스 순환
        image_index = i % self.num_images

        # 이미지 경로와 대응되는 class, anomaly token
        image_path = self.image_paths[image_index]
        class_token = self.class_tokens[image_index]
        anomaly_token = self.anomaly_tokens[image_index]

        # 이미지 로드 및 전처리
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # 각 토큰에 대한 텍스트 생성
        text = f"a photo of {class_token} with {anomaly_token} anomaly"

        input_ids = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        img = np.array(image).astype(np.uint8)
        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img).resize((self.size, self.size), resample=self.interpolation)
        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        pixel_values = (image / 127.5 - 1.0).astype(np.float32)

        return {
            "pixel_values": torch.from_numpy(pixel_values).permute(2, 0, 1),
            "input_ids": input_ids,
            "class_token": class_token,         # class 토큰 추가
            "anomaly_token": anomaly_token,     # anomaly 토큰 추가
        }


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main(args):

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    os.makedirs(logging_dir, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        # logging_dir=logging_dir,                # 에러남
        log_with="wandb",                           # tensorboard 에러나서 wandb로 기록 
        # log_with=args.report_to,                # tensorboard, wandb, ... 알아서 log 기록 => accelerator.trackers[0].log(logs, step=step)
    )


    # Make one log on every process with the configuration for debugging. logging 설정.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(logging_dir, "training.log")),  # 로그 파일 저장
            logging.StreamHandler()  # 터미널 출력
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(accelerator.state)
    # logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # ## 추가 - Tracker 초기화
    # if args.report_to == "tensorboard":
    #     from torch.utils.tensorboard import SummaryWriter
    #     writer = SummaryWriter(log_dir=logging_dir)
    #     logger.info(f"TensorBoard initialized at {logging_dir}")
    # elif args.report_to == "wandb":
    #     import wandb
    #     wandb.init(project=f"{args.dataset}-{seed}-{examples_per_class}", dir=logging_dir)
    #     logger.info("Weights & Biases initialized")

    logger.info(f"Logging directory: {logging_dir}")

    # if accelerator.is_main_process:
    #     accelerator.init_trackers("textual_inversion", config=vars(args))
    if accelerator.is_main_process:
        accelerator.init_trackers(
            "textual_inversion",
            config=vars(args),
            init_kwargs={
                "wandb": {
                    # "project": "textual_inversion",  # 새 프로젝트 이름 설정
                    "name": f"{args.dataset[9:]}-{args.seed}-{args.examples_per_class}-{datetime.datetime.now().strftime('%m%d_%H%M')}",  # 실험 이름
                },
                "tensorboard": {
                    "logging_dir": logging_dir    # TensorBoard 로그 디렉토리 설정
                }
            }
        )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:            # hugging face hub에 푸시할지 여부 
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore: # gitignore에 output 파일 추가
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:    # "CompVis/stable-diffusion-v1-4"
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    if args.unet_ckpt is not None:
        unet.load_state_dict(torch.load(args.unet_ckpt))
        print(f"Loaded UNET from {args.unet_ckpt}")

    # Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens(args.placeholder_token_list)     # class_placeholder_tokens + anomaly_placeholder_tokens
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token_list}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    # Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)  # initializer_token = "the"
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]                                             # "the"

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))    # 새로 추가한 token 때문에 token embedding 크기 조절
    
    # Initialise each new placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    for placeholder_token in args.placeholder_token_list:
        placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)   # 각 placeholder_token의 ID를 가져옴
        token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]     # initializer_token("the")의 임베딩으로 초기화

    # placeholder_token_id = tokenizer.convert_tokens_to_ids(args.placeholder_token)  # class_name
    # token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]         # "the"의 embedding 으로 초기화 



    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    # Freeze all parameters except for the token embeddings in text encoder. 토큰 임베딩만 학습가능하도록 고정.
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    if args.gradient_checkpointing:
        # Keep unet in train mode if we are using gradient checkpointing to save memory.
        # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        unet.train()
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings 임베딩만 학습.
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = TextualInversionDataset(
        data_root=args.train_data_dir,          # extracted 폴더. 데이터셋 새로 저장한 폴더.
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_token_list=args.placeholder_token_list,
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=args.center_crop,
        mode="train",
        class_token_list=args.class_token_list,         # add
        anomaly_token_list=args.anomaly_token_list      # add
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    steps_per_epoch = math.ceil(len(train_dataloader.dataset) / train_dataloader.batch_size)
    print(f"Steps per epoch: {steps_per_epoch}")    
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("textual_inversion", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Placeholder token list = {args.placeholder_token_list}")
    
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path))
        global_step = int(path.split("-")[1])

        resume_global_step = global_step * args.gradient_accumulation_steps
        first_epoch = resume_global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # keep original embeddings as reference
    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()

    for epoch in range(first_epoch, args.num_train_epochs):

        # token별 업데이트 횟수 저장
        update_count = {"class_token": {}, "anomaly_token": {}}
        for token in args.class_token_list:
            update_count["class_token"][token] = 0
        for token in args.anomaly_token_list:
            update_count["anomaly_token"][token] = 0

        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(text_encoder):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()    # image [B, 3, 512, 512] -> latents [B, 4, 64, 64]
                latents = latents * 0.18215      # scaling   # [B, 4, 64, 64]   

                # Add noise to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)     # random timestep. shape=[4]
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)    # noise 추가된 latents  [B, 4, 64, 64]

                # Class and Anomaly embeddings
                encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)   # [B, 77, 768]
                ### "a photo of {class_token} with {anomaly_token} anomaly" -> tokenizer -> input_ids -> text_encoder -> hidden_states
                '''
                batch["input_ids"].shape [4, 77]   : 각 text에 대한 77개의 token id
                    tensor([[49406,   320,  1125,   539, 49409,   593, 49410, 46811, 49407, 49407,...49407 ], ...4개])
                -> text encoder -> [4, 77, 768]
                '''

                # Predict the noise residual using class and anomaly embeddings
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample     # [B, 4, 64, 64]
                ### unet에 noisy_latents, timesteps, hidden_states 입력 -> noise 예측

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":     # default
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")     # 노이즈 예측 값과 실제 노이즈 값의 차이로 loss 계산

                accelerator.backward(loss)

                # Update embeddings
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                ### 현재 해당하는 class, anomaly token만 update
                with accelerator.accumulate(text_encoder):
                    class_tokens = batch["class_token"]  
                    anomaly_tokens = batch["anomaly_token"] 

                    # 업데이트된 토큰들의 횟수 증가
                    for token in class_tokens:
                        update_count["class_token"][token] += 1
                    for token in anomaly_tokens:
                        update_count["anomaly_token"][token] += 1
                    # logger.info(f"Updated class token: {class_tokens} / anomaly token: {anomaly_tokens}")

                    update_token_ids = [
                        tokenizer.convert_tokens_to_ids(token) for token in class_tokens + anomaly_tokens
                    ]

                    # 기존 임베딩 고정 및 특정 토큰 업데이트
                    index_no_updates = torch.ones(len(tokenizer), dtype=torch.bool, device=accelerator.device)
                    for token_id in update_token_ids:           # 학습할 토큰만 False로 설정
                        index_no_updates[token_id] = False

                    with torch.no_grad():
                        accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                            index_no_updates
                        ] = orig_embeds_params[index_no_updates]

                    # 학습 중 업데이트된 임베딩 확인
                    # current_embedding = text_encoder.get_input_embeddings().weight.data.clone()
                    # with torch.no_grad():
                    #     for token_id in update_token_ids:
                    #         print(f"Updated embedding for token {tokenizer.convert_ids_to_tokens(token_id)}")
                            # print(f"Updated embedding values: {current_embedding[token_id]}")
                    

                ## 새롭게 추가된 모든 placeholder token의 인덱스 가져오기
                # placeholder_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in args.placeholder_token_list]   # [49408. 49409. 49410. 49411]

                ### 새롭게 추가된 토큰만 update
                # index_no_updates = torch.ones(len(tokenizer), dtype=torch.bool, device=accelerator.device)  # [True, ...] 49408 + 4(추가토큰개수) = 49412개
                # for token_id in placeholder_token_ids:
                #     index_no_updates[token_id] = False  # 새롭게 추가된 토큰만 False로 설정. 마지막 4개만 False
                ## 임베딩 업데이트 - 새로 추가된 토큰 제외 원래 임베딩으로 고정.
                # with torch.no_grad():
                #     accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                #         index_no_updates
                #     ] = orig_embeds_params[index_no_updates]

                ## 학습 중 업데이트된 임베딩 확인                
                # original_embedding = orig_embeds_params.clone()
                # current_embedding = text_encoder.get_input_embeddings().weight.data.clone()
                # with torch.no_grad():
                #     for idx in range(len(tokenizer)):
                #         if idx not in placeholder_token_ids:
                #             assert torch.equal(original_embedding[idx], current_embedding[idx]), \
                #                 f"Token ID {idx} (non-placeholder) has been updated!"
                                
                #     # 학습하는 토큰 확인
                #     for token_id in placeholder_token_ids:
                #         print(f"Updated embedding for token {tokenizer.convert_ids_to_tokens(token_id)}")
                #         # print(f"Updated embedding for token {tokenizer.convert_ids_to_tokens(token_id)}: {current_embedding[token_id]}")




            ## 새롭게 추가된 모든 placeholder token의 인덱스 가져오기
            placeholder_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in args.placeholder_token_list]   # [49408. 49409. 49410. 49411]

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    # save_path = os.path.join(args.output_dir, f"learned_embeds-steps-{global_step}.bin")
                    save_path = os.path.join(args.output_dir, f"learned_embeds-steps-{global_step}.pt")     # bin -> pt로 저장
                    save_progress(text_encoder, placeholder_token_ids, accelerator, args, save_path)        # 여러 토큰 전달
                    # save_progress(text_encoder, placeholder_token_id, accelerator, args, save_path)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            accelerator.trackers[0].log(logs, step=global_step)
            progress_bar.set_postfix(**logs)
            # accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        
        # 학습 종료 후 로깅
        logger.info("Class token update counts:")
        for token, count in update_count["class_token"].items():
            logger.info(f"{token}: {count} updates")
        logger.info("Anomaly token update counts:")
        for token, count in update_count["anomaly_token"].items():
            logger.info(f"{token}: {count} updates")

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        # Save the newly trained embeddings
        # save_path = os.path.join(args.output_dir, "learned_embeds.bin")
        save_path = os.path.join(args.output_dir, f"{args.dataset}-{seed}-{examples_per_class}.pt")         # bin -> pt로 저장
        save_progress(text_encoder, placeholder_token_ids, 
                      accelerator, args, save_path)

    accelerator.end_training()
    accelerator.free_memory()

    del accelerator, vae, unet, text_encoder

    gc.collect()
    torch.cuda.empty_cache()

def generate_anomaly_prompt(class_name, anomaly_name):
    # mvtec ad dataset
    if class_name == 'cable':
        class_name = 'three cables'      # 복수형으로 전환

    # a/an 생략 조건 처리
    no_article_classes = ['carpet', 'grid', 'leather', 'tile', 'wood', 'cable']
    a_none = '' if class_name in no_article_classes else 'a '

    if anomaly_name != 'good':
        prompt = f"{a_none}{class_name} with {anomaly_name} anomaly"
    else:
        prompt = f"{a_none}{class_name}"
    
    return prompt


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
        train_dataset = DATASETS[args.dataset](args=args, examples_per_class=examples_per_class, seed=seed)

        ### 미리 이미지 추출함. extracted/{dataset}-{seed}-{examples_per_class}

        # Placeholder 토큰 리스트 초기화
        class_placeholder_tokens = []
        anomaly_placeholder_tokens = []

        for class_name in train_dataset.class_list:
            # Class 이름으로 토큰 생성
            class_name = class_name.replace(" ", "_")
            class_placeholder_token = f"<{class_name}>"  # e.g., "<bottle>"
            class_placeholder_tokens.append(class_placeholder_token)

        for class_name in train_dataset.class_list:
            anomaly_names_list = train_dataset.anomaly_dict[class_name]
            for anomaly_name in anomaly_names_list:
                # Anomaly 이름으로 토큰 생성
                anomaly_name = anomaly_name.replace(" ", "_")
                anomaly_placeholder_token = f"<{anomaly_name}>"  # e.g., "<crack>"
                if anomaly_placeholder_token not in anomaly_placeholder_tokens:
                    anomaly_placeholder_tokens.append(anomaly_placeholder_token)

        print(f"Class tokens: {class_placeholder_tokens}")
        print(f"Anomaly tokens: {anomaly_placeholder_tokens}")

        args.class_token_list = class_placeholder_tokens
        args.anomaly_token_list = anomaly_placeholder_tokens
        args.placeholder_token_list = class_placeholder_tokens + anomaly_placeholder_tokens
        
        args.initializer_token = "the"
        dirname = f"{args.dataset}-{seed}-{examples_per_class}"

        args.train_data_dir = os.path.join(output_dir, "extracted", dirname)
        args.output_dir = os.path.join(output_dir, "fine-tuned", dirname)
        args.seed = seed

        main(args)
