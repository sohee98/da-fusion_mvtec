import os
import sys
import torch.utils.data as data
import json
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch

from ..mvtec_ad import class_state_abnormal       #mvtec ad dataset abnormal state


NONE_SINGULAR = ['carpet', 'grid', 'leather', 'tile', 'wood', 'cable']


class MvTec_Dataset(data.Dataset):
    def __init__(self, root, tokenizer = None, image_size=224, max_seq_length=300, img_transform=None, pair_criterion = None):
        self.root = root
        self.tokenizer = tokenizer
        self.data_all = []
        self.img_transform = img_transform or transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
        self.max_seq_length = max_seq_length

        self.pair_criterion = pair_criterion


        self.image_extensions = ('.png', '.jpg', '.jpeg')

        for root, dirs, files in os.walk(self.root):
            for file in files:
                if (file.endswith(self.image_extensions)) and (not file.endswith('_mask.png')):
                    file_path = os.path.join(root, file)

                    # normal cat and dog image for validation
                    if file_path.endswith('cat_wallpapers_dog_cat_kissing.png'):
                        self.data_all.append((file_path, 'A photo of a dog and cat kissing', "dog and cat", "good"))
                        continue

                    # mvtec ad dataset
                    object_name = file_path.split('/')[-4]
                    if object_name == 'cable':
                        object_name = 'three cables'      # 복수형으로 전환

                    state_name = file_path.split('/')[-2].replace('_', ' ')

                    a_none = ''

                    if object_name in NONE_SINGULAR:
                        a_none = ''
                    else:
                        a_none = 'a '

                    if state_name != 'good':
                        prompt = f"A photo of {a_none}{object_name} with {state_name} anomaly"
                    else:
                        prompt = f"A photo of {a_none}{object_name}"
                    
                    self.data_all.append((file_path, prompt, object_name, state_name))

        self.length = len(self.data_all)



    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        data = self.data_all[idx]
        img_path, prompt, object_name, state_name = data[0], data[1], data[2], data[3]

        # Pair selection. init via random selection
        random_pair_idx = np.random.randint(self.length)
        paired_data = self.data_all[random_pair_idx]
        paired_img_path, paired_prompt, paired_object_name, paired_state_name = paired_data[0], paired_data[1], paired_data[2], paired_data[3]


        # Image processing
        img_origin = Image.open(img_path).convert("RGB")
        current_image = self.img_transform(img_origin, return_tensors='pt')['pixel_values'].squeeze(0)

        # Paired image processing
        paired_img_origin = Image.open(paired_img_path).convert("RGB")
        paired_image = self.img_transform(paired_img_origin, return_tensors='pt')['pixel_values'].squeeze(0)


        # Tokenize captions
        current_label_tokens = self.tokenizer(
            prompt, 
            max_length=self.max_seq_length,
            padding="max_length", 
            truncation=True,
            return_tensors="pt"
        ).input_ids # (max_seq_length,)

        # Paired tokenize captions
        paired_label_tokens = self.tokenizer(
            paired_prompt, 
            max_length=self.max_seq_length,
            padding="max_length", 
            truncation=True,
            return_tensors="pt"
        ).input_ids # (max_seq_length,)

        normality = ''
        if state_name == 'good':
            normality = 'normal'
        else:
            normality = 'abnoraml'



        return (
                torch.tensor([idx, random_pair_idx]),
                torch.stack([current_image, paired_image]),
                torch.stack([current_label_tokens, paired_label_tokens]),
                self.get_contrastiveness(object_name, state_name, paired_object_name, paired_state_name),
                normality,
                (object_name, paired_object_name, state_name, paired_state_name)
        )
    
    def get_contrastiveness(self, object_name, state_name, paired_object_name, paired_state_name):
        # it returns 1 if the paired is positve, on the other hand, it returns 0 if the paired is negative.
        # the setting 1

        if self.pair_criterion == 'A':
            if state_name == 'good':
                if paired_state_name == 'good':
                    return 1    # same good state
                else:
                    return -1    # different state
            else:
                if paired_state_name == 'good':
                    return -1   # different state
                else:
                    return 1    # same anomaly state
                
        elif self.pair_criterion == 'OA':
            if object_name == paired_object_name:
                if state_name == 'good':
                    if paired_state_name == 'good':
                        return 1    # same object, same good state
                    else:
                        return -1    # same object, different state
                else:
                    if paired_state_name == 'good':
                        return -1   # same object, different state
                    else:
                        return 1    # same object, same anomaly state
            else:
                return -1        # different object

        elif self.pair_criterion == 'OAC' or 'QOAC':
            if object_name == paired_object_name:
                if state_name == paired_state_name:
                    return 1        # same object, same anoamly class
                else:
                    return -1        # same object, different anomaly class
            else:
                return -1        # different object
            
        else:
            raise ValueError('Invalid pair criterion. It should be A, OA, OAC or QOAC')

