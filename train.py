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

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

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

def train_step(images, targets, network):
    # 训练判别器
    network.d_optimizer.zero_grad()
    
    # 前向传播
    perturb, g_output = network(images, targets)
    
    # 计算判别器损失
    d_real = network.discriminator(images)
    d_fake = network.discriminator(g_output.detach())  # 注意这里使用detach()
    
    d_loss_real = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
    d_loss_fake = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
    d_loss = d_loss_real + d_loss_fake
    
    d_loss.backward()
    network.d_optimizer.step()

    # 训练生成器
    network.g_optimizer.zero_grad()
    
    # 重新计算生成器的损失
    d_fake = network.discriminator(g_output)  # 注意这里不使用detach()
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
    perturbation_loss = network.config.perturbation_loss_weight * perturbation_loss
    
    pixel_loss = network.config.pixel_loss_weight * F.l1_loss(g_output, images)
    
    g_loss = adv_loss + identity_loss + perturbation_loss + pixel_loss
    
    # 生成器反向传播
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

if __name__ == "__main__":
    config = importlib.import_module("configs.default")
    trainset = MyDataset(config.train_dataset_path, config.mode)
    testset = MyDataset(config.test_dataset_path, config.mode, is_train=False)

    train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, drop_last=True) 
    test_loader = DataLoader(testset, batch_size=config.batch_size, shuffle=False, drop_last=True)   

    network = AdvFaces(config)
    network = network.to(network.device)
    network.train()
    
    log_dir = './logs/' + config.mode
    os.makedirs(log_dir, exist_ok=True) 


    print("\nStart Training\n# epochs: {}\nepoch_size: {}\nbatch_size: {}\n".format(
        config.num_epochs, config.epoch_size, config.batch_size))

    global_step = 0
    start_time = time.time()

    for epoch in range(config.num_epochs):
        if epoch == 0:
            print("Loading Test Set")
            originals, targets, labels = next(iter(test_loader))  
            originals = originals.to(network.device)
            targets = targets.to(network.device)
            labels = labels.to(network.device)  
            print('Done loading test set')
            target_feats = network.aux_matcher_extract_feature(targets)
            output_dir = os.path.join(log_dir, "samples")
            os.makedirs(output_dir, exist_ok=True)
            test_images = originals[labels < 5]  
            test_images = test_images.to(network.device)  

            print("Computing initial success rates..")
            success_rate(network, config, originals, targets, target_feats, log_dir, global_step)

        for step, batch in enumerate(train_loader):
            images, targets, labels = batch[0].to(network.device), \
                                      batch[1].to(network.device), \
                                      batch[2].to(network.device)
            wl = train_step(images, targets, network)

            

            # Display and logging
            if step % config.summary_interval == 0:
                duration = time.time() - start_time
                start_time = time.time()
                utils.display_info(epoch, step, duration, wl)
                # 输出各个loss
                print(f"Epoch [{epoch+1}/{config.num_epochs}], Step [{step}/{len(train_loader)}], "
                      f"D Loss: {wl['d_loss']:.4f}, G Loss: {wl['g_loss']:.4f}, "
                      f"Adv Loss: {wl['adv_loss']:.4f}, Identity Loss: {wl['identity_loss']:.4f}, "
                      f"Perturbation Loss: {wl['perturbation_loss']:.4f}, Pixel Loss: {wl['pixel_loss']:.4f}")

        success_rate(network, config, originals, targets, target_feats, log_dir, global_step)
        
        if epoch % 10 == 0:
            model_save_path = os.path.join(log_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(network.state_dict(), model_save_path)
            print(f"模型已保存到 {model_save_path}")



