import math
import sys
import numpy as np

# compare every row of x1 with every row of x2
def cosine_pair(x1, x2):
    assert x1.shape == x2.shape
    epison = 1e-8
    x1 = x1 / (np.linalg.norm(x1, axis=1, keepdims=True) + epison)
    x2 = x2 / (np.linalg.norm(x2, axis=1, keepdims=True) + epison)
    return np.dot(x1, x2.T)

import torch

def cosine_pair_torch(x1, x2):
    eps = 1e-10
    x1_norm = torch.sqrt(torch.sum(x1 ** 2, dim=1, keepdim=True))
    x2_norm = torch.sqrt(torch.sum(x2 ** 2, dim=1, keepdim=True))
    x1 = x1 / (x1_norm + eps)
    x2 = x2 / (x2_norm + eps)
    dist = torch.sum(x1 * x2, dim=1)
    return dist

def get_updated_learning_rate(global_step, config):
    if config.learning_rate_strategy == 'step':
        max_step = -1
        learning_rate = 0.0
        for step, lr in config.learning_rate_schedule.items():
            if global_step >= step and step > max_step:
                learning_rate = lr
                max_step = step
        if max_step == -1:
            raise ValueError('cannot find learning rate for step %d' % global_step)
    elif config.learning_rate_strategy == 'cosine':
        initial = config.learning_rate_schedule['initial']
        interval = config.learning_rate_schedule['interval']
        end_step = config.learning_rate_schedule['end_step']
        step = math.floor(float(global_step) / interval) * interval
        assert step <= end_step
        learning_rate = initial * 0.5 * (math.cos(math.pi * step / end_step) + 1)
    return learning_rate

def display_info(epoch, step, duration, watch_list):
    sys.stdout.write('[%d][%d] time: %2.2f' % (epoch+1, step+1, duration))
    for item in watch_list.items():
        if type(item[1]) in [float, np.float32, np.float64]:
            sys.stdout.write('   %s: %2.3f' % (item[0], item[1]))
        elif type(item[1]) in [int, bool, np.int32, np.int64, np.bool]:
            sys.stdout.write('   %s: %d' % (item[0], item[1]))
    sys.stdout.write('\n')