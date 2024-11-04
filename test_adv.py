import importlib
import torch
from advfaces import AdvFaces
from utils.dataset import MyDataset
import imageio  # 替换 scipy.misc
import numpy as np  # 添加 numpy 导入
import matplotlib.pyplot as plt
import utils.utils as utils



path = './archive/lfw-subset'

config = importlib.import_module("configs.default")
dataset = MyDataset(path, config.mode, is_train=False)

network = AdvFaces(config)
state_dict = torch.load('logs/model_epoch_3_1.pth') 
network.load_state_dict(state_dict)



network.eval()
# network.to('cuda')


# 取 config.batch_size 个样本
batch_size = config.batch_size
num_samples = batch_size // 2
data = []
for i in range(batch_size):
    data.append(dataset[i][0])
data = torch.stack(data)

# 生成对抗样本
results, perturbations = network.generate(data)


train_example_one = data[1]
result_example_one = results[1]

one_feats = network.aux_matcher_extract_feature(train_example_one.unsqueeze(0))
two_feats = network.aux_matcher_extract_feature(result_example_one.unsqueeze(0))

scores_a_t = utils.cosine_pair_torch(one_feats, two_feats)

print(scores_a_t)

import matplotlib.pyplot as plt

plt.figure(figsize=(40, 40))
for i in range(num_samples):
    plt.subplot(2, num_samples, i + 1)
    img = data[i].permute(1, 2, 0).cpu() * 128 + 127.5  # 移动到 CPU
    plt.imshow(img.numpy().astype(np.uint8))
    plt.axis('off')
    plt.subplot(2, num_samples, num_samples + i + 1)
    img_result = results[i].permute(1, 2, 0).cpu() * 128 + 127.5  # 移动到 CPU
    plt.imshow(img_result.numpy().astype(np.uint8)) 
    plt.axis('off')

plt.show()






