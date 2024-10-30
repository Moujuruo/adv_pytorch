import os
import sys


mode = 'obfuscation'


aux_matcher_definition = 'architecture/inception_resnet_v1.py'
# aux_matcher_path = 'pretrained/facenet/model-20180402-114759.ckpt-275'
aux_matcher_path = 'assets/20180402-114759-vggface2.pt'
aux_matcher_scope = 'InceptionResnetV1'
# Matching Threshold. !!!!  CAREFUL -- By default, we assume scores are un-normalized between [-1, 1]
aux_matcher_threshold = 0.45	


####### LOSS FUNCTION #######
pixel_loss_factor = 1.0
perturb_loss_factor = 1.0
idt_loss_factor = 10.0
MAX_PERTURBATION = 3.0

# learning rate strategy
learning_rate_strategy = 'step'

# learning rate schedule
lr = 0.0001
learning_rate_schedule = {
    0: 1 * lr,
}
learning_rate_multipliers = {}
# Number of samples per batch
batch_size = 32

# Number of batches per epoch
epoch_size = 200

# Number of epochs
num_epochs = 500

identity_loss_weight = 10.0

perturbation_threshold = 3.0

perturbation_loss_weight = 1.0

pixel_loss_weight = 1.0

summary_interval = 100


learning_rate_strategy = 'step'


train_dataset_path = './archive/casia-subset'
test_dataset_path = './archive/lfw-subset'