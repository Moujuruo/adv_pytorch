import datetime
import importlib
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from advfaces import AdvFaces
import utils.utils as utils
from utils.dataset import MyDataset
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def success_rate(
    network,
    config,
    original_images,
    targets,
    target_feats,
    log_dir,
    step,
):
    network.eval()  
    with torch.no_grad():
        if config.mode == "target":
            fakes, _ = network.generate(original_images, targets)
        else:
            fakes, _ = network.generate(original_images)

        gen_feats = network.aux_matcher_extract_feature(fakes)

        scores_a_t = utils.cosine_pair_torch(gen_feats, target_feats)
        if config.mode == 'target':
            sr = (sum(scores_a_t > config.aux_matcher_threshold) / len(scores_a_t)) * 100
        else:
            sr = (sum(scores_a_t <= config.aux_matcher_threshold) / len(scores_a_t)) * 100
        print(f"Success Rate: {sr}%")
        print("Mean Sim. Score (adv v. target): {}", format(torch.mean(scores_a_t)))
        with open(log_dir + "/accuracy.txt", "a") as f:
            f.write("{}: {}\n".format(sr, step))
    network.train()  
    return sr, torch.mean(scores_a_t)

def train_step(images, targets, ddp_network, epoch):

    network = ddp_network.module

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
    if network.mode == "target":
        real_feat = network.aux_matcher_model(targets)
        identity_loss = torch.mean(
            1.0 - (utils.cosine_pair_torch(fake_feat, real_feat) + 1.0) / 2.0
        )
    else:
        real_feat = network.aux_matcher_model(images)
        identity_loss = torch.mean(
            utils.cosine_pair_torch(fake_feat, real_feat) + 1.0
        )
    identity_loss = network.config.identity_loss_weight * identity_loss

    perturb_norm = torch.norm(perturb, p=2, dim=(1, 2, 3))
    perturbation_loss = torch.mean(
        torch.maximum(
            perturb_norm,
            torch.full_like(perturb_norm, network.config.perturbation_threshold)
        )
    )
    warm_up = 20
    if epoch < warm_up:
        perturbation_loss_weight = network.config.perturbation_loss_weight * (epoch / warm_up)
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

def main(rank, world_size):
    setup(rank, world_size)

    config = importlib.import_module("configs.default")

    torch.cuda.set_device(rank)
    print("loading trainset...")
    trainset = MyDataset(config.train_dataset_path, config.mode)
    print("loading testset...")
    testset = MyDataset(config.test_dataset_path, config.mode, is_train=False)

    train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(testset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(trainset, batch_size=config.batch_size, sampler=train_sampler, drop_last=True, num_workers=8, pin_memory=True) 
    test_loader = DataLoader(testset, batch_size=config.batch_size, sampler=test_sampler, drop_last=True, num_workers=8, pin_memory=True)    

    network = AdvFaces(config)
    network = network.to(rank)
    network = DDP(network, device_ids=[rank])
    network.train()
    
    if rank == 0:
        log_dir = './logs/' + config.mode
        os.makedirs(log_dir, exist_ok=True) 

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f'./logs/{config.mode}/{current_time}'
        writer = SummaryWriter(log_dir)


    print("\nStart Training\n# epochs: {}\nepoch_size: {}\nbatch_size: {}\n".format(
        config.num_epochs, config.epoch_size, config.batch_size))

    global_step = 0

    for epoch in range(config.num_epochs):
        train_sampler.set_epoch(epoch)
        for step, batch in enumerate(train_loader):
            images, targets, labels = batch[0].to(rank), batch[1].to(rank), batch[2].to(rank)
            losses = train_step(images, targets, network, epoch + 1)

            global_step = epoch * len(train_loader) + step
            if rank == 0:
                for loss_name, loss_value in losses.items():
                    writer.add_scalar(f'Loss/{loss_name}', loss_value, global_step)
                    # print(f"{loss_name}: {loss_value}")
            
        if rank == 0:
            writer.add_scalar('Learning Rate/discriminator', network.module.d_optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Learning Rate/generator', network.module.g_optimizer.param_groups[0]['lr'], epoch)
        
        if epoch % 3 == 0:
            if rank == 0:
                model_save_path = os.path.join(log_dir, f'model_epoch_{epoch+1}.pth')
                # torch.save(network.state_dict(), model_save_path)
                torch.save(network.module.state_dict(), model_save_path)
                print(f"模型已保存到 {model_save_path}")

    cleanup()

if __name__ == "__main__":
    world_size = 4 # 使用2个GPU
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)


