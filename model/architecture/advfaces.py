import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as init

# def weights_init_normal(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1:
#         init.normal_(m.weight.data, 1.0, 0.02)
#         init.constant_(m.bias.data, 0.0)

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
    
class LAuReL_RW_LR_ResidualBlock(nn.Module):
    def __init__(self, in_features, rank=16):
        super(LAuReL_RW_LR_ResidualBlock, self).__init__()

        # Original convolutional block (f(x_i))
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1), 
            nn.Conv2d(in_features, in_features, kernel_size=3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, kernel_size=3),
            nn.InstanceNorm2d(in_features),
        )

        # Learnable weights α and β with normalization
        self.weights = nn.Parameter(torch.ones(2))  # Initialize α and β to 1.0

        # Low-rank matrices A and B for W = AB^T + I
        self.A = nn.Conv2d(in_channels=in_features, out_channels=rank, kernel_size=1, bias=False)
        self.B = nn.Conv2d(in_channels=rank, out_channels=in_features, kernel_size=1, bias=False)

    def forward(self, x):
        # Normalize α and β using softmax to keep them bounded and summing to 1
        weights = F.softmax(self.weights, dim=0)
        alpha = weights[0]
        beta = weights[1]

        # Compute the residual connection (f(x_i))
        residual = self.conv_block(x)

        # Compute the low-rank skip connection W x_i = (AB^T + I) x_i
        W_x = x + self.B(self.A(x))

        # Combine with learnable weights
        out = alpha * residual + beta * W_x
        return out
    
class Generator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, use_dropout=False, use_target=False):
        super(Generator, self).__init__()
        # self.apply(weights_init_normal)

        self.k = base_channels 

        if use_target:
            in_channels *= 2

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
            *[LAuReL_RW_LR_ResidualBlock(4*self.k) for _ in range(4)]
        )

        # decoder
        self.deconv0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(2),
            nn.Conv2d(4*self.k, 2*self.k, 5),
            # nn.LayerNorm(2*self.k),
            nn.GroupNorm(1, 2*self.k),
            nn.ReLU(inplace=True),
        )

        self.deconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(2),
            nn.Conv2d(2*self.k, self.k, 5),
            # nn.LayerNorm(self.k),
            nn.GroupNorm(1, self.k),
            nn.ReLU(inplace=True),
        )

        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(self.k, 3, 7),
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
    
class ImprovedResidualBlock(nn.Module):
    def __init__(self, in_features, rank=16, squeeze_factor=4):
        super(ImprovedResidualBlock, self).__init__()
        
        # SE (Squeeze-and-Excitation) block
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features, in_features // squeeze_factor, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features // squeeze_factor, in_features, 1),
            nn.Sigmoid()
        )
        
        # Original convolutional block with split branches
        self.conv_block1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features//2, kernel_size=3),
            nn.InstanceNorm2d(in_features//2),
            nn.ReLU(inplace=True)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features//2, in_features, kernel_size=3),
            nn.InstanceNorm2d(in_features),
        )

        # Learnable weights with normalization
        self.weights = nn.Parameter(torch.ones(3))  # α, β, and γ for SE attention

        # Low-rank matrices A and B with improved initialization
        self.A = nn.Conv2d(in_channels=in_features, out_channels=rank, kernel_size=1, bias=False)
        self.B = nn.Conv2d(in_channels=rank, out_channels=in_features, kernel_size=1, bias=False)
        
        # Initialize low-rank matrices
        nn.init.kaiming_normal_(self.A.weight)
        nn.init.kaiming_normal_(self.B.weight)
        
        # Gradient scaling factor
        self.grad_scale = 0.1
        
    def forward(self, x):
        # Normalize weights using softmax
        weights = F.softmax(self.weights, dim=0)
        alpha, beta, gamma = weights[0], weights[1], weights[2]

        # SE attention
        se_weight = self.se(x)
        
        # Split path residual computation
        residual = self.conv_block1(x)
        residual = self.conv_block2(residual)

        # Low-rank skip connection with gradient scaling
        W_x = x + self.grad_scale * self.B(self.A(x))

        # Combine all paths with learnable weights
        out = alpha * residual + beta * W_x + gamma * (x * se_weight)
        return out

class ImprovedGenerator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_residual_blocks=4, use_target=False, dropout_rate=0.1):
        super(ImprovedGenerator, self).__init__()
        
        self.k = base_channels

        if use_target:
            in_channels *= 2
        
        # Encoder with dropout
        self.conv0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, self.k, 7),
            nn.InstanceNorm2d(self.k),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate)
        )

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.k, 2*self.k, 4, stride=2),
            nn.InstanceNorm2d(2*self.k),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate)
        )

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(2*self.k, 4*self.k, 4, stride=2),
            nn.InstanceNorm2d(4*self.k),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate)
        )

        # Residual blocks with dropout
        self.residual_blocks = nn.Sequential(
            *[ImprovedResidualBlock(4*self.k) for _ in range(num_residual_blocks)]
        )

        # Decoder with dropout
        self.pre_deconv0 = nn.Sequential(
            nn.Conv2d(4*self.k, 4*self.k, 1),
            nn.GroupNorm(4, 4*self.k),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate)
        )
        
        self.upsample0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReflectionPad2d(2),
            nn.Conv2d(4*self.k, 2*self.k, 5),
            nn.GroupNorm(4, 2*self.k),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate)
        )
        
        self.skip_fusion0 = nn.Sequential(
            nn.Conv2d(4*self.k, 2*self.k, 1),
            nn.GroupNorm(4, 2*self.k),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate)
        )
        
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReflectionPad2d(2),
            nn.Conv2d(2*self.k, self.k, 5),
            nn.GroupNorm(4, self.k),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate)
        )
        
        self.skip_fusion1 = nn.Sequential(
            nn.Conv2d(2*self.k, self.k, 1),
            nn.GroupNorm(4, self.k),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate)
        )

        # Output layers (no dropout in final layers to ensure stable output)
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(self.k, 3, 7),
        )
        
        self.noise_modulation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.k, self.k//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.k//4, 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x, target=None):
        input_image = x
        
        if target is not None:
            x = torch.cat([x, target], dim=1)

        # Encoder path with skip connections
        skip0 = self.conv0(x)
        skip1 = self.conv1(skip0)
        x = self.conv2(skip1)
        
        x = self.residual_blocks(x)
        x = self.pre_deconv0(x)
        
        x = self.upsample0(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.skip_fusion0(x)
        
        x = self.upsample1(x)
        x = torch.cat([x, skip0], dim=1)
        x = self.skip_fusion1(x)
        
        noise = self.output(x)
        noise = torch.tanh(noise)
        
        noise_amplitude = self.noise_modulation(x)
        noise = noise * noise_amplitude
        
        perturb = torch.clamp(noise, -1.0, 1.0)
        output = 2 * torch.clamp((perturb + (input_image + 1.0) / 2.0), 0, 1) - 1.0

        return noise, output
    
class NormalDiscriminator(nn.Module):
    def __init__(self, in_channels=3, dropout_rate=0.0, phase_train=True):
        super(NormalDiscriminator, self).__init__()
        self.phase_train = phase_train
        
        def discriminator_block(in_channels, out_channels, kernel_size=4, stride=2, normalize=True, activation=True):
            layers = [
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, bias=not normalize)
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels, track_running_stats=phase_train))
            if activation:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(in_channels, 32, normalize=True, activation=False),
            *discriminator_block(32, 64, normalize=True, activation=False),
            *discriminator_block(64, 128, normalize=True, activation=True),
            *discriminator_block(128, 256, normalize=True, activation=True),
            *discriminator_block(256, 512, normalize=True, activation=True),
            nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        batch_size = x.size(0)
        out = self.model(x)
        return out.view(batch_size, -1)

import torch
import torch.nn as nn

class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class MultiScaleUNetGenerator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, use_target=False):
        super(MultiScaleUNetGenerator, self).__init__()
        
        if use_target:
            in_channels *= 2

        # Encoder
        self.enc1 = MultiScaleBlock(in_channels, base_channels)     # 112x112
        self.enc2 = MultiScaleBlock(base_channels, base_channels*2) # 56x56
        self.enc3 = MultiScaleBlock(base_channels*2, base_channels*4) # 28x28
        self.enc4 = MultiScaleBlock(base_channels*4, base_channels*8) # 14x14
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bridge
        self.bridge = MultiScaleBlock(base_channels*8, base_channels*16)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(base_channels*16, base_channels*8, 2, stride=2)
        self.dec4 = MultiScaleBlock(base_channels*16, base_channels*8)
        
        self.upconv3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, stride=2)
        self.dec3 = MultiScaleBlock(base_channels*8, base_channels*4)
        
        self.upconv2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.dec2 = MultiScaleBlock(base_channels*4, base_channels*2)
        
        self.upconv1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        self.dec1 = MultiScaleBlock(base_channels*2, base_channels)
        
        # Output noise generator
        self.noise_out = nn.Sequential(
            nn.Conv2d(base_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1),
            nn.Tanh()
        )
        
        # Noise amplitude modulation
        self.noise_modulation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_channels, base_channels//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels//4, 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x, target=None):
        input_image = x
        
        if target is not None:
            x = torch.cat([x, target], dim=1)
            
        # Encoding
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bridge
        b = self.bridge(self.pool(e4))
        
        # Decoding
        d4 = self.upconv4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        
        d3 = self.upconv3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.upconv2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.upconv1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        # Generate noise
        noise = self.noise_out(d1)
        
        # Modulate noise amplitude
        noise_amplitude = self.noise_modulation(d1)
        noise = noise * noise_amplitude
        
        # Apply perturbation with clamping
        perturb = torch.clamp(noise, -1.0, 1.0)
        output = 2 * torch.clamp((perturb + (input_image + 1.0) / 2.0), 0, 1) - 1.0
        
        return noise, output