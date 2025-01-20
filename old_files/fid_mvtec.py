import torch
from torchvision import transforms
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import argparse

# 이미지에서 Inception 모델의 feature vector를 추출
def get_features(image_dir, model, device):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    model.eval()
    features = []
    with torch.no_grad():
        for img_name in tqdm(os.listdir(image_dir), desc="Extracting Features"):
            img_path = os.path.join(image_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0).to(device)
            feat = model(img)[0].cpu().numpy()
            features.append(feat)
    return np.array(features)

# FID 계산 함수
def calculate_fid(features1, features2):
    mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)
    
    # Calculate mean and covariance difference
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Handle numerical issues
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

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
    
    parser = argparse.ArgumentParser("FID script")
    
    parser.add_argument("--out", type=str, default="output_mvtec/aug")
    parser.add_argument("--dataset", type=str, default="mvtec_ad")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples-per-class", type=int, default=1)
    parser.add_argument("--guidance-scale", nargs="+", type=float, default=[7.5])
    parser.add_argument("--strength", nargs="+", type=float, default=[0.5])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.fc = torch.nn.Identity()  # 마지막 layer를 제거하여 feature extraction 전용으로 설정

    dataset_name = args.dataset
    if dataset_name == 'mvtec_ad':
        class_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
        anomaly_dict = {
            'bottle': ['broken small', 'broken large', 'contamination'],
            'cable': ['cut inner insulation', 'missing cable', 'combined', 'cable swap', 'missing wire', 'poke insulation', 'bent wire', 'cut outer insulation'],
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
    elif dataset_name == 'mvtec_ad_subset':
        class_list = ['screw']
        anomaly_dict = {
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

    # dataset = 'mvtec_ad_subset_3'
    aug_dir = f"{args.dataset}-{args.seed}-{args.examples_per_class}/gscale_{args.guidance_scale}-strength_{args.strength}"
    out_dir = os.path.join(args.out, aug_dir)

    MVTEC_DIR = "/SSD1/datasets/mvtec_ad"

    output_file = os.path.join(args.out, f"{args.dataset}-{args.seed}-{args.examples_per_class}_fid_scores.txt")

    with open(output_file, "a") as f:
        f.write("\n" + aug_dir + "\n")
        for class_name in class_list:
            for anomaly_name in anomaly_dict[class_name]:
                anomaly_name = anomaly_name.replace(" ", "_")
                real_images_dir = os.path.join(MVTEC_DIR, class_name, 'test', anomaly_name)
                prompt = generate_anomaly_prompt(class_name, anomaly_name).replace(" ", "_")
                generated_images_dir = os.path.join(out_dir, class_name, prompt)
                
                real_features = get_features(real_images_dir, model, device)
                generated_features = get_features(generated_images_dir, model, device)
                
                fid_score = calculate_fid(real_features, generated_features)

                result_line = f"FID Score for {class_name}-{anomaly_name}: {fid_score}\n"
                print(result_line.strip())
                f.write(result_line)
