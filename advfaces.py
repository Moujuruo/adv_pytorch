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

    def load_aux_matcher(self):
        self.aux_matcher.load_weights(self.aux_matcher_model, 'vggface2')
        self.aux_matcher_model.eval()
        
    def forward(self, images, targets=None):
        
        if self.mode == "target" and targets is not None:
            self.perturb, self.g_output = self.generator(images, targets)
        else:
            self.perturb, self.g_output = self.generator(images)
            
        return self.perturb, self.g_output
    
    def compute_losses(self, images, targets=None):

        d_real = self.discriminator(images)
        d_fake = self.discriminator(self.g_output)

        ## GAN LOSS ##
        d_loss_real = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
        d_loss_fake = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
        self.d_loss = d_loss_real + d_loss_fake

        self.adv_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))

        ## IDENTITY LOSS ##
        with torch.no_grad():
            self.fake_feat = self.aux_matcher_model(self.g_output)
            if self.mode == "target":
                self.real_feat = self.aux_matcher_model(targets)
            else:
                self.real_feat = self.aux_matcher_model(images)

        if self.mode == "target":
            identity_loss = torch.mean(
                1.0 - (utils.cosine_pair_torch(self.fake_feat, self.real_feat) + 1.0) / 2.0
            )
        else:
            identity_loss = torch.mean(
                utils.cosine_pair_torch(self.fake_feat, self.real_feat) + 1.0
            )
        self.identity_loss = self.config.identity_loss_weight * identity_loss

        ## perturbation LOSS ##
        perturb_norm = torch.norm(self.perturb, p=2, dim=(1, 2, 3))
        perturbation_loss = torch.mean(
            torch.maximum(
                perturb_norm - self.config.perturbation_threshold,
                torch.zeros_like(perturb_norm)
            )
        )
        self.perturbation_loss = self.config.perturbation_loss_weight * perturbation_loss

        ## pixel loss ##
        self.pixel_loss = self.config.pixel_loss_weight * F.l1_loss(self.g_output, images)

        self.g_loss = self.adv_loss + self.identity_loss + self.perturbation_loss

        return {
            "d_loss": self.d_loss,
            "g_loss": self.g_loss,
            "adv_loss": self.adv_loss,
            "identity_loss": self.identity_loss,
            "perturbation_loss": self.perturbation_loss,
            "pixel_loss": self.pixel_loss,
        }

    
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

        result = np.ndarray((num_images, c, h, w), dtype=np.float32)
        pertubations = np.ndarray((num_images, c, h, w), dtype=np.float32)

        for start_idx in range(0, num_images, batch_size):
            end_idx = min(start_idx + batch_size, num_images)

            im = torch.tensor(images[start_idx:end_idx]).to(self.device)

            if self.mode == "target":
                t = torch.tensor(targets[start_idx:end_idx]).to(self.device)
                p, g = self.forward(im, t)

            else:
                p, g = self.forward(im)

            result[start_idx:end_idx] = g.cpu().numpy()
            pertubations[start_idx:end_idx] = p.cpu().numpy()

        self.train()

        return result, pertubations

    @torch.no_grad()
    def aux_matcher_extract_feature(self, images, batch_size=512, bottelneck_layer=512, verbose=True):
        self.eval()
        num_images = images.shape[0]
        c, h, w = images.shape[1:]

        fake = np.ndarray((num_images, bottelneck_layer), dtype=np.float32)
        start_time = time.time()

        for start_idx in range(0, num_images, batch_size):
            if verbose:
                elapsed_time = time.strftime('%H:%M:%S', 
                    time.gmtime(time.time() - start_time))
                sys.stdout.write('# of images: %d Current image: %d Elapsed time: %s \t\r' 
                    % (num_images, start_idx, elapsed_time))
                sys.stdout.flush()
            end_idx = min(start_idx + batch_size, num_images)

            im = torch.tensor(images[start_idx:end_idx]).to(self.device)

            f = self.aux_matcher_model(im)

            fake[start_idx:end_idx] = f.cpu().numpy()

        self.train()

        if verbose:
            print('\n')

        return fake


if __name__ == '__main__':

    config = importlib.import_module('configs.default')
    advfaces = AdvFaces(config, 10)

    # 生成一个 [1*3*112*112] 的随机张量
    images = torch.randn(1, 3, 112, 112)
    images = images.to(advfaces.device)
    perturb, g_output = advfaces(images)
    print(perturb, g_output)

    print(advfaces.compute_losses(images))




