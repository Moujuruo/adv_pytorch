# adv_pytorch: Hashen Face Adversarial Attack Solution

This repository contains our team's **technical solution** for the **Hashen AI security track**, implemented in **PyTorch**. The code focuses on two face-recognition adversarial generation tasks:

- **Task 1 / obfuscation**: generate non-targeted adversarial faces to reduce identity recognizability.
- **Task 2 / target**: generate targeted adversarial faces whose feature representation moves closer to a target identity.

---

## 1. Method Overview

The main model wrapper is implemented in `model/advfaces.py`. The overall solution follows a generator-discriminator adversarial training pipeline.

### 1.1 Generator

The main generator is `ImprovedGenerator` in `model/architecture/advfaces.py`. Its design includes:

- an encoder-residual-decoder architecture;
- `ImprovedResidualBlock`, which combines:
  - **SE (Squeeze-and-Excitation)** channel attention,
  - **low-rank skip connections**,
  - **learnable softmax path weights** for residual / skip / attention fusion,
  - gradient scaling for more stable training;
- a targeted mode where both the source image and target image are used as input.

### 1.2 Discriminator

The discriminator is implemented as `NormalDiscriminator`, using stacked convolutional blocks to distinguish generated images from real ones.

### 1.3 Loss Design

The training pipeline combines several objectives:

- **adversarial loss**;
- **identity matching loss**, using an auxiliary matcher together with `IR_152` and `InceptionResnetV1`;
- **perturbation constraint loss**;
- **pixel-level reconstruction loss**;
- additional losses used by the multi-GPU training script:
  - **SSIM loss**
  - **L2 loss**

---

## 2. Repository Structure

```text
.
├── README.md
├── requirements.txt
├── model
│   ├── advfaces.py                      # Main model wrapper
│   ├── architecture
│   │   ├── advfaces.py                  # Generator / discriminator implementations
│   │   ├── inception_resnet_v1.py       # Feature extractor
│   │   ├── iresnet.py                   # Auxiliary matcher backbone
│   │   ├── models.py                    # Additional face model components
│   │   └── utils
│   │       └── download.py
│   ├── assets                           # Model weight directory
│   │   ├── obfuscation
│   │   └── target
│   ├── configs
│   │   ├── default.py                   # Main training / target config
│   │   └── default1.py                  # Obfuscation generation config
│   ├── train
│   │   ├── train.py                     # Single-GPU training entry
│   │   ├── train_multigpu.py            # Multi-GPU training entry
│   │   └── extra_data                   # Small training subset
│   └── utils
│       ├── dataset.py
│       ├── pytorch_ssim.py
│       └── utils.py
└── test
    ├── run.py                           # Unified testing entry
    ├── generate.py                      # Task 1 generation script
    ├── iterative_generate.py            # Task 2 iterative generation script
    ├── images_ch1                       # Sample evaluation images for Task 1
    └── images_ch2                       # Sample evaluation images for Task 2
```

---

## 3. Environment and Installation

Reference environment:

- Python 3.9+
- CUDA 12.4
- PyTorch 2.4.1

Install dependencies with:

```bash
pip install -r requirements.txt
```

Notes:

- The current inference / generation scripts are written for **CUDA-based execution** and are not adapted for CPU-only deployment.
- `requirements.txt` contains both runtime and experiment-related dependencies; it can be simplified further if needed.

---

## 4. Data Preparation

### 4.1 Training Data

Training is based on **aligned face datasets**. The repository keeps a small subset under `model/train/extra_data/aligned_imagesv3` for code validation and directory-structure reference.

### 4.2 Test Data

The repository includes two sample evaluation sets:

- `test/images_ch1/no_target/images`
- `test/images_ch2/target/images`

These are used for quick reproduction and sanity-checking of both tasks.

---

## 5. Model Weights

The following files are expected by the current codebase:

### 5.1 Third-Party / Base Weights

Place the following files under `model/assets/`:

| File | Description |
| --- | --- |
| `20180402-114759-vggface2.pt` | FaceNet / `facenet-pytorch` related weight file |
| `ir152.pth` | `IR_152` face-recognition model weight |
| `model.pt` | Auxiliary matcher weight |

### 5.2 Task-Specific Weights Trained by Our Team

Place the following files in:

- `model/assets/obfuscation/task_1_multimodel_drop_ssim_early.pth`
- `model/assets/target/model_epoch_112.pth`

Without these weights, the provided generation scripts cannot run end-to-end.

---

## 6. Quick Start

### 6.1 Task 1: Obfuscation

Run from the repository root:

```bash
python test/run.py --mode obfuscation
```

### 6.2 Task 2: Targeted Attack

Run from the repository root:

```bash
python test/run.py --mode target
```

Generated outputs are written to:

```text
result_data/advimages/
```

---

## 7. Training

### 7.1 Single-GPU Training

Run from the repository root:

```bash
python model/train/train.py
```

Before training, check `model/configs/default.py` for:

- `mode = 'obfuscation'` or `mode = 'target'`
- learning rate, batch size, epoch count, and other hyperparameters
- dataset paths

### 7.2 Multi-GPU Training

The multi-GPU script uses imports that assume `model/` as the working directory, so the recommended command is:

```bash
cd model
CUDA_VISIBLE_DEVICES=0,1,2,3 python train/train_multigpu.py
```

Additional notes:

- adjust `world_size` in `train/train_multigpu.py` to match the actual number of GPUs;
- NCCL is required for distributed training;
- training logs are written to `logs/`, and checkpoints are saved under timestamped subdirectories.

---

## 8. References

```bibtex
@InProceedings{Huang2012a,
  author =    {Gary B. Huang and Marwan Mattar and Honglak Lee and Erik Learned-Miller},
  title =     {Learning to Align from Scratch},
  booktitle = {NIPS},
  year =      {2012}
}

@article{yi2014learning,
  title={Learning face representation from scratch},
  author={Yi, Dong and Lei, Zhen and Liao, Shengcai and Li, Stan Z},
  journal={arXiv preprint arXiv:1411.7923},
  year={2014}
}

@inproceedings{deb2020advfaces,
  title={Advfaces: Adversarial face synthesis},
  author={Deb, Debayan and Zhang, Jianbang and Jain, Anil K},
  booktitle={2020 IEEE International Joint Conference on Biometrics (IJCB)},
  pages={1--10},
  year={2020},
  organization={IEEE}
}

@article{menghani2024laurel,
  title={LAUREL: Learned Augmented Residual Layer},
  author={Menghani, Gaurav and Kumar, Ravi and Kumar, Sanjiv},
  journal={arXiv preprint arXiv:2411.07501},
  year={2024}
}
```
