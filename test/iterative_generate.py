import sys
import os
# 将项目根目录添加到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
import torch
from model.advfaces import AdvFaces
from model.architecture.iresnet import get_model
import imageio 
import numpy as np  
from torchvision import transforms
import os
from PIL import Image

# 设置随机种子
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 配置路径和参数
output_dir = './result_data/advimages'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 加载生成模型
config = importlib.import_module("model.configs.default")
network = AdvFaces(config)
state_dict = torch.load('model/assets/target/model_epoch_112.pth')
network.load_state_dict(state_dict, strict=False)
network.eval()
network.to('cuda')

# 加载相似度计算模型
model_path = 'model/assets/model.pt'
similarity_model = get_model('r50').eval()
similarity_model.load_state_dict(torch.load(model_path, map_location='cuda'), strict=False)
similarity_model.to('cuda')

# 图像预处理
transform = transforms.Compose([
    transforms.Resize([112,112]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

similarity_transform = transforms.Compose([
    transforms.Resize([112,112]),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

def denormalize(tensor):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor.clamp_(0, 1)

def get_similarity(img_tensor, target_tensor):
    """计算两个图片tensor的相似度"""
    with torch.no_grad():
        feat1 = similarity_model(img_tensor)
        feat2 = similarity_model(target_tensor)
        similarity = torch.cosine_similarity(feat1, feat2)
    return similarity.item()

def iterative_generate(img_path, target_path, max_iterations=5):
    """迭代生成对抗样本直到达到目标相似度"""
    # 读取目标图片
    target_img = imageio.imread(target_path)
    target_tensor = transform(transforms.ToPILImage()(target_img))
    target_sim_tensor = similarity_transform(Image.fromarray(target_img)).unsqueeze(0).cuda()
    
    # 读取原始图片
    current_img = imageio.imread(img_path)
    if len(current_img.shape) == 2:
        current_img = np.stack([current_img] * 3, axis=2)
    
    for iteration in range(max_iterations):
        # 生成对抗样本
        current_tensor = transform(transforms.ToPILImage()(current_img))
        with torch.no_grad():
            result, _ = network.generate(current_tensor.unsqueeze(0).cuda(), 
                                      target_tensor.unsqueeze(0).cuda())
        
        # 计算相似度
        adv_img = denormalize(result.cpu()[0])
        adv_sim_tensor = similarity_transform(transforms.ToPILImage()(adv_img)).unsqueeze(0).cuda()
        similarity = get_similarity(adv_sim_tensor, target_sim_tensor)
        
        print(f"Iteration {iteration + 1}, Similarity: {similarity:.4f}")
        
        if similarity >= 0.3:
            print("Target similarity reached!")
            return adv_img, similarity
        
        # 使用当前生成结果作为下一次输入
        current_img = (adv_img.permute(1,2,0).numpy() * 255).astype(np.uint8)
    
    print("Max iterations reached.")
    return adv_img, similarity

def main():
    test_dir = 'test/images_ch2/target/images'
    target_path = 'test/images_ch2/target/TargetPerson/Aaron_Eckhart_0001.png'
    
    for person_folder in os.listdir(test_dir):
        person_path = os.path.join(test_dir, person_folder)
        if not os.path.isdir(person_path):
            continue
            
        out_person_dir = os.path.join(output_dir, person_folder)
        if not os.path.exists(out_person_dir):
            os.makedirs(out_person_dir)
        
        for img_name in os.listdir(person_path):
            if not img_name.endswith('.png'):
                continue
                
            img_path = os.path.join(person_path, img_name)
            out_path = os.path.join(out_person_dir, img_name)
            
            print(f"\nProcessing: {img_path}")
            adv_img, final_similarity = iterative_generate(img_path, target_path)
            
            # 保存生成的图片
            transforms.ToPILImage()(adv_img).save(out_path, format='PNG')
            print(f"Saved to: {out_path} with final similarity: {final_similarity:.4f}")

if __name__ == '__main__':
    main()