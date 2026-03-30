import sys
import os
# 将项目根目录添加到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import datetime
import importlib
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from model.advfaces import AdvFaces
import model.utils.utils as utils
from model.utils.dataset import MyDataset
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



def train_step(images, targets, network, epoch):


    # 训练判别器
    network.d_optimizer.zero_grad()
    
    # 前向传播
    perturb, g_output = network(images, targets)
    
    # 计算判别器损失
    d_real = network.discriminator(images)
    d_fake = network.discriminator(g_output.detach())  
    
    d_loss_real = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
    d_loss_fake = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
    d_loss = d_loss_real + d_loss_fake
    
    d_loss.backward()
    network.d_optimizer.step()

    # 训练生成器
    network.g_optimizer.zero_grad() 
    
    d_fake = network.discriminator(g_output)  
    adv_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
    
    # 计算其他损失
    fake_feat = network.aux_matcher_model(g_output)
    fake_feat_fr = network.fr_model(g_output)
    fake_feat_incep = network.incep_model(g_output)
    # fake_feat_ar = network.ar_model(g_output)

    if network.mode == "target":
        real_feat = network.aux_matcher_model(targets)
        real_feat_fr = network.fr_model(targets)
        real_feat_incep = network.incep_model(targets)
        identity_loss1 = torch.mean(
            1.0 - (utils.cosine_pair_torch(fake_feat, real_feat) + 1.0) / 2.0
        )
        identity_loss2 = torch.mean(
            1.0 - (utils.cosine_pair_torch(fake_feat_fr, real_feat_fr) + 1.0) / 2.0
        )
        identity_loss3 = torch.mean(
            1.0 - (utils.cosine_pair_torch(fake_feat_incep, real_feat_incep) + 1.0) / 2.0
        )
    else:
        real_feat = network.aux_matcher_model(images)
        real_feat_fr = network.fr_model(images)
        real_feat_incep = network.incep_model(images)
        # real_feat_ar = network.ar_model(images)
        identity_loss1 = torch.mean(
            utils.cosine_pair_torch(fake_feat, real_feat) + 1.0
        )
        identity_loss2 = torch.mean(
            utils.cosine_pair_torch(fake_feat_fr, real_feat_fr) + 1.0
        )
        identity_loss3 = torch.mean(
            utils.cosine_pair_torch(fake_feat_incep, real_feat_incep) + 1.0
        )
        # identity_loss4 = torch.mean(
        #     utils.cosine_pair_torch(fake_feat_ar, real_feat_ar) + 1.0
        # )

    identity_loss1 = network.config.identity_loss_weight * identity_loss1
    identity_loss2 = 6 * identity_loss2
    identity_loss3 = 6 * identity_loss3
    # identity_loss4 = 6 * identity_loss4
    identity_loss = identity_loss1 + identity_loss2 + identity_loss3

    perturb_norm = torch.norm(perturb, p=2, dim=(1, 2, 3) )
    perturbation_loss = torch.mean(
        torch.maximum(
            perturb_norm,
            torch.full_like(perturb_norm, network.config.perturbation_threshold)
        )
    )
    warm_up = 10
    if epoch < warm_up:
        perturbation_loss_weight = network.config.perturbation_loss_weight * (epoch / warm_up)
        # perturbation_loss_weight = network.config.perturbation_loss_weight
    else:
        perturbation_loss_weight = network.config.perturbation_loss_weight

    # perturbation_loss = network.config.perturbation_loss_weight * perturbation_loss
    perturbation_loss = perturbation_loss_weight * perturbation_loss
    
    pixel_loss = network.config.pixel_loss_weight * F.l1_loss(g_output, images)
    
    g_loss = adv_loss + identity_loss + perturbation_loss + pixel_loss 
    
    g_loss.backward()

    network.g_optimizer.step()
    
    return {
        "d_loss": d_loss.item(),
        "g_loss": g_loss.item(),
        "adv_loss": adv_loss.item(),
        "identity_loss": identity_loss.item(),
        "perturbation_loss": perturbation_loss.item(),
        "pixel_loss": pixel_loss.item()
    }

# def main(rank, world_size):
if __name__ == "__main__":


    config = importlib.import_module("model.configs.default")

    print("loading trainset...")
    trainset = MyDataset(config.train_dataset_path, config.mode)
    print("loading testset...")
    # testset = MyDataset(config.test_dataset_path, config.mode, is_train=False)


    train_loader = DataLoader(trainset, batch_size=config.batch_size, drop_last=True, num_workers=8, pin_memory=True, shuffle=True) 
    # test_loader = DataLoader(testset, batch_size=config.batch_size, drop_last=True, num_workers=8, pin_memory=True)    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    network = AdvFaces(config)
    network = network.to(device)
    network.train()
    
    log_dir = './logs/' + config.mode
    os.makedirs(log_dir, exist_ok=True) 
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f'./logs/{config.mode}/{current_time}'
    writer = SummaryWriter(log_dir)


    print("\nStart Training\n# epochs: {}\nepoch_size: {}\nbatch_size: {}\n".format(
        config.num_epochs, config.epoch_size, config.batch_size))

    global_step = 0

    for epoch in range(config.num_epochs):
        for step, batch in enumerate(train_loader):
            images, targets, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            losses = train_step(images, targets, network, epoch + 1)

            global_step = epoch * len(train_loader) + step
            for loss_name, loss_value in losses.items():
                writer.add_scalar(f'Loss/{loss_name}', loss_value, global_step)
                # print(f"{loss_name}: {loss_value}")
            
            writer.add_scalar('Learning Rate/discriminator', network.d_optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Learning Rate/generator', network.g_optimizer.param_groups[0]['lr'], epoch)
        
        if epoch % 3 == 0:
            model_save_path = os.path.join(log_dir, f'model_epoch_{epoch+1}.pth')
            # torch.save(network.state_dict(), model_save_path)
            torch.save(network.state_dict(), model_save_path)
            print(f"模型已保存到 {model_save_path}")



