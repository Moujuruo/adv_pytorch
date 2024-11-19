import torch
from architecture.iresnet import get_model
from torchvision import transforms
from PIL import Image
import os
import pandas as pd

# 初始化模型
model_path = 'assets/model.pt'
device = 'cuda:0'
model = get_model('r50').eval()
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.to(device)

# 图像预处理
transform = transforms.Compose([
    transforms.Resize([112,112]),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

def get_feature(img_path):
    """获取图像特征"""
    image = Image.open(img_path)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(image)
    return feat

def compare_pairs():
    orig_dir = 'images_ch1/no_target/images'
    adv_dir = './advimages'
    results = []

    for person_folder in os.listdir(orig_dir):
        person_orig_path = os.path.join(orig_dir, person_folder)
        person_adv_path = os.path.join(adv_dir, person_folder)
        
        if not os.path.isdir(person_orig_path) or not os.path.isdir(person_adv_path):
            continue

        # 获取两张原始图片和生成图片的路径
        orig_imgs = sorted([f for f in os.listdir(person_orig_path) if f.endswith('.png')])
        adv_imgs = sorted([f for f in os.listdir(person_adv_path) if f.endswith('.png')])

        if len(orig_imgs) != 2 or len(adv_imgs) != 2:
            print(f"Skipping {person_folder}: incorrect number of images")
            continue

        # 第一张生成图与第二张原图比对
        feat1 = get_feature(os.path.join(person_adv_path, adv_imgs[0]))
        feat2 = get_feature(os.path.join(person_orig_path, orig_imgs[1]))
        similarity1 = torch.cosine_similarity(feat1, feat2).item()

        # 第二张生成图与第一张原图比对
        feat3 = get_feature(os.path.join(person_adv_path, adv_imgs[1]))
        feat4 = get_feature(os.path.join(person_orig_path, orig_imgs[0]))
        similarity2 = torch.cosine_similarity(feat3, feat4).item()

        # 保存结果
        results.append({
            'person': person_folder,
            'adv1_orig2_similarity': similarity1,
            'adv2_orig1_similarity': similarity2
        })
        
        print(f"Processed {person_folder}")

    # 保存结果到CSV
    df = pd.DataFrame(results)
    df.to_csv('comparison_results.csv', index=False)
    print("Results saved to comparison_results.csv")

if __name__ == '__main__':
    compare_pairs()
