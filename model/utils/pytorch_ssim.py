import torch
import torch.nn.functional as F


def gaussian_kernel(size=11, sigma=1.5):
    """
    生成高斯核用于SSIM计算
    
    Args:
        size (int): 核大小，应该是奇数
        sigma (float): 标准差
        
    Returns:
        torch.Tensor: 2D高斯核
    """
    coords = torch.arange(size).float() - size // 2
    coords = coords.repeat(size, 1)
    
    # 计算2D高斯核
    g = torch.exp(-(coords ** 2 + coords.t() ** 2) / (2 * sigma ** 2))
    
    # 归一化
    g = g / g.sum()
    
    return g


def _ssim_kernel(data1, data2, kernel, k1=0.01, k2=0.03, L=1):
    """
    计算SSIM的核心函数
    
    Args:
        data1 (torch.Tensor): 第一张图像
        data2 (torch.Tensor): 第二张图像
        kernel (torch.Tensor): 高斯核
        k1 (float): SSIM公式中的常数1
        k2 (float): SSIM公式中的常数2
        L (int): 像素值的动态范围
        
    Returns:
        torch.Tensor: SSIM值
    """
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    
    # 将核扩展为4D的权重
    kernel = kernel.expand(data1.size(1), 1, kernel.size(0), kernel.size(1))
    kernel = kernel.to(data1.device)
    
    # 计算均值
    mu1 = F.conv2d(data1, kernel, groups=data1.size(1), padding=kernel.size(-1)//2)
    mu2 = F.conv2d(data2, kernel, groups=data2.size(1), padding=kernel.size(-1)//2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # 计算方差和协方差
    sigma1_sq = F.conv2d(data1 * data1, kernel, groups=data1.size(1), padding=kernel.size(-1)//2) - mu1_sq
    sigma2_sq = F.conv2d(data2 * data2, kernel, groups=data2.size(1), padding=kernel.size(-1)//2) - mu2_sq
    sigma12 = F.conv2d(data1 * data2, kernel, groups=data1.size(1), padding=kernel.size(-1)//2) - mu1_mu2
    
    # SSIM公式
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    
    ssim = numerator / denominator
    
    return ssim


def ssim(img1, img2, window_size=11, sigma=1.5):
    """
    计算两张图片之间的SSIM
    
    Args:
        img1 (torch.Tensor): 第一张图像 (B, C, H, W)
        img2 (torch.Tensor): 第二张图像 (B, C, H, W)
        window_size (int): 窗口大小
        sigma (float): 高斯核的标准差
        
    Returns:
        torch.Tensor: SSIM值，范围在[-1, 1]之间
    """
    # 检查输入
    if not torch.is_tensor(img1) or not torch.is_tensor(img2):
        raise TypeError('输入应该是torch.Tensor')
        
    if img1.shape != img2.shape:
        raise ValueError('两张图片的尺寸应该相同')
    
    # 确保输入是4D tensor
    if len(img1.shape) != 4:
        raise ValueError('输入应该是4D tensor (B, C, H, W)')
    
    # 生成高斯核
    kernel = gaussian_kernel(window_size, sigma)
    
    # 计算SSIM
    ssim_map = _ssim_kernel(img1, img2, kernel)
    
    # 返回均值
    return ssim_map.mean()


class SSIM(torch.nn.Module):
    """
    SSIM损失模块
    """
    def __init__(self, window_size=11, sigma=1.5):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        
    def forward(self, img1, img2):
        return ssim(img1, img2, self.window_size, self.sigma)


# 使用示例
if __name__ == "__main__":
    # 创建两个随机图像
    img1 = torch.randn(1, 3, 64, 64)
    img2 = torch.randn(1, 3, 64, 64)
    
    # 方法1：直接使用函数
    ssim_value = ssim(img1, img2)
    print(f"SSIM value: {ssim_value.item()}")
    
    # 方法2：使用nn.Module
    ssim_module = SSIM()
    ssim_value = ssim_module(img1, img2)
    print(f"SSIM value (using module): {ssim_value.item()}")
    
    # 计算SSIM损失
    ssim_loss = 1 - ssim_value
    print(f"SSIM loss: {ssim_loss.item()}")