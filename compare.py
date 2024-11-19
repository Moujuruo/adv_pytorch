import importlib
from tkinter import Image
import torch
from advfaces import AdvFaces
from utils.dataset import MyDataset
import imageio  # 替换 scipy.misc
import numpy as np  # 添加 numpy 导入
import matplotlib.pyplot as plt
import utils.utils as utils
from torchvision import transforms
from architecture.iresnet import get_model


model_path = 'assets/model.pt'
device = 'cuda:0'

# load model
model = get_model('r50').eval()
model.load_state_dict(torch.load(model_path,map_location=device), strict=False)
model.to(device)

image1_path = './001.jpg'

# image load and preprocess
transform = transforms.Compose([
    transforms.Resize([112,112]),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

image1 = Image.open(image1_path)
image1 = transform(image1).unsqueeze(0) #拓展维度
image1 = image1.to(device)

feat1 = model(image1)

image2 = '013.jpg'
image2 = Image.open(image2)
image2 = transform(image2).unsqueeze(0) #拓展维度
image2 = image2.to(device)
feat2 = model(image2)

image3 = '014.jpg'
image3 = Image.open(image3)
image3 = transform(image3).unsqueeze(0) #拓展维度
image3 = image3.to(device)
feat3 = model(image3)


print(torch.cosine_similarity(feat1, feat2).item())