import importlib
import torch
from advfaces import AdvFaces
from utils.dataset import MyDataset
import imageio  # 替换 scipy.misc
import numpy as np  # 添加 numpy 导入
import matplotlib.pyplot as plt
import utils.utils as utils
from torchvision import transforms

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

path = './archive/aligned_imagesv3'

config = importlib.import_module("configs.default")
dataset = MyDataset(path, config.mode, is_train=False)

network = AdvFaces(config)

aux_matcher_state_dict = {}
for key in network.aux_matcher_model.state_dict():
    aux_matcher_state_dict[key] = network.aux_matcher_model.state_dict()[key]
    
state_dict = torch.load('logs/target/model_epoch_88.pth') 
# state_dict = torch.load('logs/model_epoch_100_improved.pth')
network.load_state_dict(state_dict, strict=False)

# 比较加载完的模型和原始模型的参数
# for key in network.aux_matcher_model.state_dict():
#     if key in state_dict:
#         if not torch.equal(aux_matcher_state_dict[key], state_dict[key]):
#             print(f"Key: {key} is not equal")

# state_dict = torch.load('logs/target/model_epoch_499.pth') 
# network.load_state_dict(state_dict, strict=False)
# for key in network.aux_matcher_model.state_dict():
#     if key in state_dict:
#         if not torch.equal(aux_matcher_state_dict[key], state_dict[key]):
#             print(f"Key: {key} is not equal")


network.eval()
network.to('cuda')



### attack

original = dataset[0][0]
original = original.unsqueeze(0).to('cuda')
target = dataset[18][0]
target = target.unsqueeze(0).to('cuda')

result, perturbation = network.generate(original, target)

# 输出3张图片
plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.imshow(original.squeeze(0).permute(1, 2, 0).cpu() * 0.5 + 0.5)  # 移动到 CPU
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(result.squeeze(0).permute(1, 2, 0).cpu() * 0.5 + 0.5)
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(target.squeeze(0).permute(1, 2, 0).cpu() * 0.5 + 0.5)
plt.axis('off')
plt.show()

feat_original = network.aux_matcher_extract_feature(original, batch_size=1)
feat_result = network.aux_matcher_extract_feature(result, batch_size=1)
feat_target = network.aux_matcher_extract_feature(target, batch_size=1)

print(f"Similarity score between original and result: {utils.cosine_pair_torch(feat_original, feat_result)}")
print(f"Similarity score between original and target: {utils.cosine_pair_torch(feat_original, feat_target)}")
print(f"Similarity score between result and target: {utils.cosine_pair_torch(feat_result, feat_target)}")


# 取 config.batch_size 个样本
# batch_size = config.batch_size
# num_samples = batch_size // 2
# data = []
# for i in range(batch_size):
#     data.append(dataset[i][0])
# data = torch.stack(data)

# 生成对抗样本
# results, perturbations = network.generate(data)

### obfuscation

# train_example_one = dataset[200][0]
# result, perturbation = network.generate(train_example_one.unsqueeze(0))

# one_feats = network.aux_matcher_extract_feature(train_example_one.unsqueeze(0))
# two_feats = network.aux_matcher_extract_feature(result)

# # 输出两张图片
# plt.figure(figsize=(10, 10))
# plt.subplot(1, 2, 1)
# plt.imshow(train_example_one.permute(1, 2, 0).cpu() * 0.5 + 0.5)  # 移动到 CPU
# plt.axis('off')
# plt.subplot(1, 2, 2)
# plt.imshow(result.squeeze(0).permute(1, 2, 0).cpu() * 0.5 + 0.5) 
# plt.axis('off')
# plt.show()

# scores_a_t = utils.cosine_pair_torch(one_feats, two_feats)

# print(scores_a_t)

# result_example_one = results[1]

# transform = transforms.Compose([
#     transforms.Resize([112,112]),
#     # transforms.ToTensor(),
#     # transforms.Normalize(mean=0.5, std=0.5)
#     # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
# ])

# train_example_one = transform(train_example_one)

# 输出 train_example_one 的图片
# plt.imshow(train_example_one.permute(1, 2, 0).cpu() * 0.5 + 0.5)  # 移动到 CPU
# plt.axis('off')
# plt.show()

# result_example_one = transform(result_example_one)

# one_feats = network.aux_matcher_extract_feature(train_example_one.unsqueeze(0))
# two_feats = network.aux_matcher_extract_feature(result_example_one.unsqueeze(0))

# scores_a_t = utils.cosine_pair_torch(one_feats, two_feats)

# print(scores_a_t)

# import matplotlib.pyplot as plt

# plt.figure(figsize=(40, 40))
# for i in range(num_samples):
#     plt.subplot(2, num_samples, i + 1)
#     d1 = transform(data[i])
#     one_feats = network.aux_matcher_extract_feature(d1.unsqueeze(0))
#     img = data[i].permute(1, 2, 0).cpu() * 128 + 127.5  # 移动到 CPU
#     plt.imshow(img.numpy().astype(np.uint8))
#     plt.axis('off')
#     plt.subplot(2, num_samples, num_samples + i + 1)
#     d2 = transform(results[i])
#     two_feats = network.aux_matcher_extract_feature(d2.unsqueeze(0))
#     img_result = results[i].permute(1, 2, 0).cpu() * 128 + 127.5  # 移动到 CPU
#     plt.imshow(img_result.numpy().astype(np.uint8)) 
#     plt.axis('off')
#     print(f"Similarity score: {utils.cosine_pair_torch(one_feats, two_feats)}")

# plt.show()






