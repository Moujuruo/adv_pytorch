import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Tuple
from typing import Optional, Tuple
from PIL import Image
import torchvision.transforms as transforms

class DataClass:
    def __init__(self, class_name: str, indices: List[int], label: int):
        self.class_name = class_name
        self.indices = indices
        self.label = label

    def random_pair(self) -> np.ndarray:
        return np.random.choice(self.indices, 2, replace=False)
    
    def random_samples(self, num_samples_per_class: int, exception: Optional[int] = None) -> np.ndarray:
        indices_tmp = list(set(self.indices) - set([exception]))
        return np.random.choice(indices_tmp, num_samples_per_class, replace=False)
    
class MyDataset(Dataset):
    def __init__(self, path: Optional[str] = None, mode: Optional[str] = None, is_train: bool = True):
        self.DataClass = DataClass
        self.mode = mode
        self.classes = None
        self.images = None
        self.labels = None
        self.targets = None
        self.features = None
        self.idx2cls = None
       
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])



        if path is not None:
            self.init_from_path(path)

    def __len__(self) -> int:
        return len(self.images) if self.images is not None else 0
    
    def __getitem__(self, idx: int) -> Tuple[str, str, int]:
        # return self.images[idx], self.targets[idx], self.labels[idx]
        image_path = self.images[idx]
        target_path = self.targets[idx]

        image = Image.open(image_path).convert('RGB')
        target = Image.open(target_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
            target = self.transform(target)

        return image, target, self.labels[idx]
    
    def init_from_path(self, path: str):
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            self.init_from_folder(path)
        print(f'{len(self.images)} images of {self.num_classes} classes loaded')

    def init_from_folder(self, path: str):
        # folder = os.path.expanduser(folder)
        folder = os.path.abspath(path)
        class_names = sorted(os.listdir(folder))
        images = []
        labels = []
        label = 0

        if os.path.isdir(os.path.join(folder, class_names[0])):
            # 多类别目录结构
            for class_name in class_names:
                classdir = os.path.join(folder, class_name)
                if os.path.isdir(classdir):
                    images_class = sorted(os.listdir(classdir))
                    images_class = [os.path.join(classdir, img) for img in images_class]
                    if len(images_class) < 2:
                        continue
                    images.extend(images_class)
                    labels.extend([label] * len(images_class))
                    label += 1
        else:
            # 单类别目录结构
            images = [os.path.join(folder, c) for c in class_names]
            labels = [1] * len(images)

        self.images = np.array(images, dtype=np.object_)
        self.labels = np.array(labels, dtype=np.int32)
        
        self.init_classes()
        
        # 为每个样本选择目标样本
        self.targets = []
        valid_indices = []
        for i, c in enumerate(self.labels):
            if self.mode == 'target':
                idx_to_choose = np.where(self.labels != c)[0]
            else:
                idx_to_choose = np.where(self.labels == c)[0]
                idx_to_choose = np.delete(idx_to_choose, np.argwhere(idx_to_choose == i))
            if len(idx_to_choose) == 0:
                continue  # 如果没有可选的目标，跳过该样本
            self.targets.append(self.images[random.choice(idx_to_choose)])
            valid_indices.append(i)
        self.targets = np.array(self.targets, dtype=np.object_)
        self.images = self.images[valid_indices]
        self.labels = self.labels[valid_indices]

    def init_classes(self):
        """初始化类别信息"""
        dict_classes = {}
        classes = []
        self.idx2cls = np.ndarray((len(self.labels),), dtype=np.object_)
        
        for i, label in enumerate(self.labels):
            if label not in dict_classes:
                dict_classes[label] = [i]
            else:
                dict_classes[label].append(i)
                
        for label, indices in dict_classes.items():
            classes.append(self.DataClass(str(label), indices, label))
            self.idx2cls[indices] = classes[-1]
            
        self.classes = np.array(classes, dtype=np.object_)
        self.num_classes = len(classes)


    
if __name__ == '__main__':
    path = './archive/casia-subset'
    dataset = Dataset(path)
    print(dataset[0])
