import sys
import time
from architecture.advfaces import Generator, NormalDiscriminator

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn import functional as F
import importlib.util
import utils.utils as utils


class AdvFaces(nn.Module):
    def __init__(self, config):
        super(AdvFaces, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.mode = config.mode

        spec = importlib.util.spec_from_file_location("network_model", config.aux_matcher_definition)
        self.aux_matcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.aux_matcher)
        self.aux_matcher_model = self.aux_matcher.InceptionResnetV1(pretrained='vggface2', device=self.device)
        
        for param in self.aux_matcher_model.parameters():
            param.requires_grad = False

        self.setup_network_model()
        
        # 设置优化器
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config.lr,
            betas=(0.5, 0.9)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config.lr,
            betas=(0.5, 0.9)
        )
        
            
        self.global_step = 0
        
    def setup_network_model(self):
        self.generator = Generator().to(self.device)
        self.discriminator = NormalDiscriminator().to(self.device)
        
    def forward(self, images, targets=None):
        if self.mode == "target" and targets is not None:
            self.perturb, self.g_output = self.generator(images, targets)
        else:
            self.perturb, self.g_output = self.generator(images)
            
        return self.perturb, self.g_output
    

    
    def save_checkpoint(self, path):
        """保存模型检查点"""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'global_step': self.global_step,
        }, path)
    
    def load_checkpoint(self, path):
        """加载模型检查点"""
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.global_step = checkpoint['global_step']

    @torch.no_grad()
    def generate(self, images, targets=None, batch_size=128, return_target=False):
        self.eval()
        num_images = images.shape[0]
        c, h, w = images.shape[1:]

        result = torch.empty((num_images, c, h, w), dtype=images.dtype, device=self.device)
        pertubations = torch.empty((num_images, c, h, w), dtype=images.dtype, device=self.device)

        for start_idx in range(0, num_images, batch_size):
            end_idx = min(start_idx + batch_size, num_images)

            im = images[start_idx:end_idx].to(self.device)

            if self.mode == "target":
                t = targets[start_idx:end_idx].to(self.device)
                p, g = self.forward(im, t)
            else:
                p, g = self.forward(im)

            result[start_idx:end_idx] = g
            pertubations[start_idx:end_idx] = p

        # self.train()

        return result, pertubations

    @torch.no_grad()
    def aux_matcher_extract_feature(self, images, batch_size=512, bottelneck_layer=512, verbose=True):
        self.eval()
        num_images = images.shape[0]

        fake = torch.empty((num_images, bottelneck_layer), dtype=torch.float32, device=self.device)
        start_time = time.time()

        for start_idx in range(0, num_images, batch_size):
            if verbose:
                elapsed_time = time.strftime('%H:%M:%S', 
                    time.gmtime(time.time() - start_time))
                sys.stdout.write('# of images: %d Current image: %d Elapsed time: %s \t\r' 
                    % (num_images, start_idx, elapsed_time))
                sys.stdout.flush()
            end_idx = min(start_idx + batch_size, num_images)

            im = images[start_idx:end_idx].to(self.device)

            f = self.aux_matcher_model(im)

            fake[start_idx:end_idx] = f


        if verbose:
            print('\n')

        return fake


if __name__ == '__main__':

    config = importlib.import_module('configs.default')
    advfaces = AdvFaces(config)

    # 生成一个 [1*3*112*112] 的随机张量
    images = torch.randn(1, 3, 112, 112)
    images = images.to(advfaces.device)
    perturb, g_output = advfaces(images)
    print(perturb, g_output)

    print(advfaces.compute_losses(images))




