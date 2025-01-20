from semantic_aug.few_shot_dataset import FewShotDataset
from collections import defaultdict
from typing import Tuple, Dict, Any
import os
import torch
import json
import torchvision.transforms as transforms
from PIL import Image
# from torch.utils.data import Dataset
import numpy as np


MVTEC_DIR = "/SSD1/datasets/mvtec_ad"
# dataset_name = "mvtec_ad"

def generate_class_info(dataset_name):
    class_name_map_class_id = {}
    if dataset_name == 'mvtec_ad':
        class_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
        
        anomaly_dict = {
            'bottle': ['broken small', 'broken large', 'contamination'],
            'cable': ['cut inner insulation', 'missing cable', 'combined', 'cable swap', 'missing wire', 'poke insulation', 'bent wire', 'cut outer insulation'],
            # 'three cables': ['cut inner insulation', 'missing cable', 'combined', 'cable swap', 'missing wire', 'poke insulation', 'bent wire', 'cut outer insulation'],
            'capsule': ['faulty imprint', 'scratch', 'squeeze', 'poke', 'crack'],
            'carpet': ['thread', 'hole', 'metal contamination', 'cut', 'color'],
            'grid': ['thread', 'bent', 'glue', 'metal contamination', 'broken'],
            'hazelnut': ['hole', 'cut', 'print', 'crack'],
            'leather': ['poke', 'glue', 'fold', 'cut', 'color'],
            'metal_nut': ['scratch', 'flip', 'bent', 'color'],
            'pill': ['faulty imprint', 'scratch', 'pill type', 'combined', 'color', 'contamination', 'crack'],
            'screw': ['scratch head', 'thread side', 'scratch neck', 'manipulated front', 'thread top'],
            'tile': ['oil', 'rough', 'gray stroke', 'glue strip', 'crack'],
            'toothbrush': ['defective'],
            'transistor': ['bent lead', 'damaged case', 'misplaced', 'cut lead'],
            'wood': ['scratch', 'combined', 'hole', 'liquid', 'color'],
            'zipper': ['combined', 'rough', 'squeezed teeth', 'fabric interior', 'split teeth', 'broken teeth', 'fabric border']
        }
    elif dataset_name == 'mvtec_loco':
        class_list = ['breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag', 'splicing_connectors']
    elif dataset_name == 'mvtec_ad_subset':
        class_list = ['screw']
        anomaly_dict = {
            # 'grid': ['thread', 'bent', 'glue', 'metal contamination', 'broken'],
            'screw': ['scratch head', 'thread side', 'scratch neck', 'manipulated front', 'thread top'],
        }
    elif dataset_name == 'mvtec_ad_subset_2':
        class_list = ['grid', 'hazelnut']
        anomaly_dict = {
            'grid': ['thread', 'bent', 'glue', 'metal contamination', 'broken'],
            'hazelnut': ['hole', 'cut', 'print', 'crack'],
        }
    elif dataset_name == 'mvtec_ad_subset_3':
        class_list = ['hazelnut', 'screw', 'grid']
        anomaly_dict = {
            'screw': ['scratch head', 'thread side', 'scratch neck', 'manipulated front', 'thread top'],
            'hazelnut': ['hole', 'cut', 'print', 'crack'],
            'grid': ['thread', 'bent', 'glue', 'metal contamination', 'broken'],
        }
    elif dataset_name == 'mvtec_ad_setA':
        class_list = ['metal_nut', 'pill']
        anomaly_dict = {
            'metal_nut': ['scratch', 'color'],
            'pill': ['scratch','color']
        }
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
            
    for k, index in zip(class_list, range(len(class_list))):
        class_name_map_class_id[k] = index

    return class_list, class_name_map_class_id, anomaly_dict

def generate_anomaly_prompt(dataset_name, class_name, anomaly_name):
    if dataset_name == 'mvtec_ad_setA':
        text = f"a photo of <{class_name}> with <{anomaly_name}> anomaly"
        return text
    else:
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

class MvtecDataset(FewShotDataset):
    # class_names = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
    #                'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    # num_classes: int = len(class_names)

    def __init__(self, args, root_dir=MVTEC_DIR, examples_per_class=None, seed=0, image_size=(256, 256)):
        """
        :param root_dir: MVTec 데이터셋의 루트 경로.
        :param examples_per_class: anomaly 종류별로 가져올 이미지 수.
        :param seed: 랜덤 시드.
        :param image_size: 이미지 크기 (H, W).
        """

        #### 전체 데이터셋 불러오기
        meta_info = json.load(open(f'{root_dir}/meta.json', 'r'))
        meta_info = meta_info['test']

        self.data_all = []
        self.cls_names = list(meta_info.keys())
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)
        ####

        #### textual inversion. 
        dataset_name = args.dataset
        self.class_list, self.class_name_map_class_id, self.anomaly_dict = generate_class_info(dataset_name)
        
        self.root_dir = root_dir
        self.examples_per_class = examples_per_class
        self.image_size = image_size
        self.rng = np.random.default_rng(seed)

        # 데이터셋 클래스 및 anomaly 폴더 구조 로드
        self.class_to_images = defaultdict(list)
        self.prompt = defaultdict(str)
        self.normal_image = defaultdict(list)
        # for class_name in os.listdir(root_dir):
        for class_name in self.class_list:
            class_path = os.path.join(root_dir, class_name, "test")
            if not os.path.isdir(class_path):
                continue
            # for anomaly_type in os.listdir(class_path):     # anomaly 폴더 
            for anomaly_type in self.anomaly_dict[class_name]:
                anomaly_type = anomaly_type.replace(" ","_")
                anomaly_path = os.path.join(class_path, anomaly_type)
                if not os.path.isdir(anomaly_path):
                    continue
                image_files = [
                    os.path.join(anomaly_path, f)
                    for f in os.listdir(anomaly_path)
                    if f.endswith(('.png', '.jpg', '.jpeg'))
                ]
                self.class_to_images[(class_name, anomaly_type)].extend(image_files)
                prompt = generate_anomaly_prompt(dataset_name, class_name, anomaly_type)  # 'carpet with thread anomaly'
                self.prompt[(class_name, anomaly_type)] = prompt
            # normal image 추가 
            normal_path = os.path.join(class_path, 'good')
            if not os.path.isdir(normal_path):
                continue
            image_files = [
                os.path.join(normal_path, f)
                for f in os.listdir(normal_path)
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ]
            self.normal_image[class_name].extend(image_files)

        # examples_per_class 적용
        if examples_per_class is not None:
            self.class_to_images = {
                key: self.rng.choice(images, examples_per_class, replace=False).tolist()
                for key, images in self.class_to_images.items()
                if len(images) >= examples_per_class
            }

        self.all_images = [
            (class_name, anomaly_type, image_path)
            for (class_name, anomaly_type), image_paths in self.class_to_images.items()
            for image_path in image_paths
        ]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])



    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                              data['specie_name'], data['anomaly']
        img_origin = Image.open(os.path.join(self.root, img_path)).convert("RGB")

        img = self.transform(img_origin)

        img_origin_transformed = self.default_transform(img_origin)

        return {'img': img, 'cls_name': cls_name, 'anomaly': anomaly,
                'img_path': os.path.join(self.root, img_path), "cls_id": self.class_name_map_class_id[cls_name], "img_origin" : img_origin_transformed}    

    def get_image_by_idx(self, idx: int) -> Image.Image:
        """지정된 인덱스의 원본 이미지를 반환합니다."""
        _, _, image_path = self.all_images[idx]
        return Image.open(image_path).convert('RGB')
    
    def get_label_by_idx(self, idx: int) -> torch.Tensor:
        class_name, anomaly_type, _ = self.all_images[idx]
        return (class_name, anomaly_type)
    
    def get_metadata_by_idx(self, idx: int) -> Dict[str, Any]:
        """지정된 인덱스의 메타데이터를 반환합니다."""
        class_name, anomaly_type, image_path = self.all_images[idx]
        prompt = self.prompt[(class_name, anomaly_type)]
        # return {"name": f"{class_name}-{anomaly_type}", "path": image_path}
        return {"name": f"{class_name}-{anomaly_type}", "path": image_path, "prompt": prompt}
    
    def get_normal_image_by_idx(self, idx: int) -> Image.Image:
        """지정된 인덱스의 원본 이미지를 반환합니다."""
        class_name, _, _ = self.all_images[idx]
        image_path = self.normal_image[class_name][0]
        return Image.open(image_path).convert('RGB')
    
    def get_random_normal_image_by_idx(self, idx: int) -> Image.Image:
        """지정된 인덱스의 원본 이미지를 반환합니다."""
        class_name, _, _ = self.all_images[idx]
        random_idx = self.rng.integers(len(self.normal_image[class_name]))
        image_path = self.normal_image[class_name][random_idx]
        return Image.open(image_path).convert('RGB')