import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as init

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1), 
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )            

    def forward(self, x):
        return x + self.conv_block(x)
    
class Generator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, use_dropout=False):
        super(Generator, self).__init__()
        self.apply(weights_init_normal)

        self.k = base_channels 

        # encoder
        self.conv0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, self.k, 7),
            nn.InstanceNorm2d(self.k),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.k, 2*self.k, 4, 2, 1),
            nn.InstanceNorm2d(2*self.k),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(2*self.k, 4*self.k, 4, 2, 1),
            nn.InstanceNorm2d(4*self.k),
            nn.ReLU(inplace=True),
        )

        # residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(4*self.k) for _ in range(3)]
        )

        # decoder
        self.deconv0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(2),
            nn.Conv2d(4*self.k, 2*self.k, 5),
            # nn.LayerNorm(2*self.k),
            nn.GroupNorm(32, 2*self.k),
            nn.ReLU(inplace=True),
        )

        self.deconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(2),
            nn.Conv2d(2*self.k, self.k, 5),
            # nn.LayerNorm(self.k),
            nn.GroupNorm(32, self.k),
            nn.ReLU(inplace=True),
        )

        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(self.k, in_channels, 7),
        )

    def forward(self, x, target=None):
        input_image = x

        if target is not None:
            x = torch.cat([x, target], dim=1) 

        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.residual_blocks(x)

        encoded = x

        x = self.deconv0(x)
        x = self.deconv1(x)
        x = self.output(x)
        x = torch.tanh(x)

        # perturb the output
        perturb = torch.clamp(x, -1.0, 1.0)
        output = 2 * torch.clamp(perturb + (input_image + 1.0) / 2.0, 0, 1) - 1.0

        return x, output
    
class NormalDiscriminator(nn.Module):
    def __init__(self, in_channels=3, dropout_rate=0.0):
        super(NormalDiscriminator, self).__init__()
        self.apply(weights_init_normal)

        def discriminator_block(in_channels, out_channels, kernel_size=4, stride=2, normalize=True):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1, bias=not normalize)
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(in_channels, 32, normalize=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x, train=True):
        if train:
            self.train()
        else:
            self.eval()

        output = self.model(x)
        output = output.view(x.size(0), -1) # [B, 1]
        return output
            
