import importlib
import torch
from advfaces import AdvFaces
from utils.dataset import MyDataset
import imageio 
import numpy as np  
import matplotlib.pyplot as plt
import utils.utils as utils
from torchvision import transforms
import os

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

output_dir = './advimages'
# mkdir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

config = importlib.import_module("configs.default")
network = AdvFaces(config)

state_dict = torch.load('logs/obfuscation/model_epoch_220.pth')
network.load_state_dict(state_dict, strict=False)

network.eval()
network.to('cuda')


transform = transforms.Compose([
    transforms.Resize([112,112]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def denormalize(tensor):
    """将归一化的tensor转换回正常图片范围"""
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor.clamp_(0, 1)

def save_image(tensor, path):
    """保存tensor为图片
    Args:
        tensor: 形状为[3,H,W]的tensor，值范围[0,1]
        path: 保存路径
    """
    # 转换为PIL图像
    img = transforms.ToPILImage()(tensor)
    # 保存图片
    img.save(path, format='PNG')

# 遍历测试集目录
test_dir = 'images_ch1/no_target/images'
for person_folder in os.listdir(test_dir):
    person_path = os.path.join(test_dir, person_folder)
    if not os.path.isdir(person_path):
        continue
        
    # 创建对应的输出目录
    out_person_dir = os.path.join(output_dir, person_folder)
    if not os.path.exists(out_person_dir):
        os.makedirs(out_person_dir)
    
    # 处理每张图片
    for img_name in os.listdir(person_path):
        if not img_name.endswith('.png'):
            continue
            
        img_path = os.path.join(person_path, img_name)
        
        # 读取并预处理图片
        img = imageio.imread(img_path)
        if len(img.shape) == 2:  # 处理灰度图
            img = np.stack([img] * 3, axis=2)
        img_tensor = transform(transforms.ToPILImage()(img))
        
        # 生成对抗样本
        with torch.no_grad():
            result, _ = network.generate(img_tensor.unsqueeze(0).cuda())
        
        # 逆变换并保存
        adv_img = denormalize(result.cpu()[0])
        out_path = os.path.join(out_person_dir, img_name)
        save_image(adv_img, out_path)  # 使用新定义的save_image函数
        
        print(f'Generated: {out_path}')

print('All images generated successfully!')


