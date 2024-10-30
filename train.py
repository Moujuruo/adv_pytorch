import importlib
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from advfaces import AdvFaces
import utils.utils as utils
from utils.dataset import Dataset

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

        scores_a_t = utils.cosine_pair(gen_feats, target_feats)
        if config.mode == 'target':
            sr = (sum(scores_a_t > config.aux_matcher_threshold) / len(scores_a_t)) * 100
        else:
            sr = (sum(scores_a_t <= config.aux_matcher_threshold) / len(scores_a_t)) * 100
        print(f"Success Rate: {sr}%")
        print("Mean Sim. Score (adv v. target): {}", format(np.mean(scores_a_t)))
        with open(log_dir + "/accuracy.txt", "a") as f:
            f.write("{}: {}\n".format(sr, step))
    network.train()  
    return sr, np.mean(scores_a_t)

if __name__ == "__main__":
    config = importlib.import_module("configs.default")
    trainset = Dataset(config.train_dataset_path, config.mode)
    testset = Dataset(config.test_dataset_path, config.mode, is_train=False)

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
            originals, targets, labels = next(iter(test_loader))  # 获取标签
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
            
            learning_rate = utils.get_updated_learning_rate(global_step, config)
            
            network.g_optimizer.zero_grad()
            network.d_optimizer.zero_grad()

            # Forward pass and compute losses
            wl = network.compute_losses(images, targets)
            
            # Backward pass
            wl["g_loss"].backward(retain_graph=True)
            network.g_optimizer.step()
            wl["d_loss"].backward()
            network.d_optimizer.step()

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



