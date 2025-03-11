# Exposure-slot
<p align="center">
  <h1 align="center">Exposure-slot: Exposure-centric representations learning with Slot-in-Slot Attention for Region-aware Exposure Correction</h1>
  
  <p align="center">Donggoo Jung*, Daehyun Kim*, Guanghui Wang, Tae Hyun Kim†.
  </p>
  <h2 align="center">CVPR 2025</h2>

  <h3 align="center">
    <a href="https://github.com/kdhRick2222/Exposure-slot/" target='_blank'><img src="https://img.shields.io/badge/🐳-Project%20Page-blue"></a>
<!--     <a href="https://www.arxiv.org/pdf/2407.16125" target='_blank'><img src="https://img.shields.io/badge/arXiv-2407.16125-b31b1b.svg"></a> -->
  </h3>

</p>

This repository contains the official PyTorch implementation of **_Exposure-slot_**: **Exposure**-centric representations learning with **Slot-in-Slot Attention** for Region-aware Exposure Correction accepted at **CVPR 2025.**

Exposure-slot is the first approach to leverage Slot Attention mechanism for optimized exposure-specific feature partitioning. 111
• We introduce the slot-in-slot attention that enables sophisticated feature partitioning and learning and exposure-aware prompts that enhance the exposure-centric characteristics of each image feature. 

Our proposing method is **the first approach to leverage Slot Attention mechanism** for optimized exposure-specific feature partitioning. We introduce the **slot-in-slot attention** that enables sophisticated feature partitioning and learning and exposure-aware prompts that enhance the exposure-centric characteristics of each image feature. We provide validation code, training code, and pre-trained weights on three benchmark datasets (**MSEC, SICE, LCDP**).

<div align="center">
  <img src="asset/main.png" width="700px" />
</div>

## Setting

Please follow these steps to set up the repository.

### 1. Clone the Repository

```
git clone https://github.com/mlvlab/Exposure-slot.git
cd Exposure-slot
```

### 2. Download Pre-trained models and Official Checkpoints

We utilize pre-trained models from [FFHQ (ffhq_10m.pt)](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh).

- Download the checkpoints for FFHQ and ImageNet from this [Google Drive Link](https://drive.google.com/drive/folders/1h8vKYwTYSshljBuW9NdBRJQSp_HBPZdA).
- Place the pre-trained models into the `ckpt/` directory.

### 3. Prepare Data

For amortized optimization, we use the FFHQ 49K dataset and the ImageNet 130K dataset, which are subsets of the training datasets used for the pre-trained models. These subsets are distinct from the validation datasets (ffhq_1K and imagenet_val_1K) used for evaluation.

- ### FFHQ 256x256

  - `data/ffhq_1K` and `data/ffhq_49K`.

  We downloaded the FFHQ dataset and resized it to 256x256, following the instructions on the [ffhq-dataset
  public site](https://github.com/NVlabs/ffhq-dataset).

  We use 00000-00999 as the validation set (1K) and 01000-49999 (49K) as the training set.

- ### ImageNet 256x256

  - `data/imagenet_val_1K` and `data/imagenet_130K`.

  We downloaded the [ImageNet 100 dataset](https://www.kaggle.com/datasets/ambityga/imagenet100) and use its training set.

- ### Measurements as numpy format

  - `data/y_npy`

  During amortized training, we load a subset of the training set to monitor the convergence of the training process.

  You can specify degradation types using the `--deg` option.

  - Gaussian Deblur `gaussian`
  - 4x Super-resolution `sr_averagepooling`
  - Box inpainting `inpainting`
  - Denoising `deno`
  - Colorization `colorization`

  ```
  python utils/get_measurements.py --deg gaussian --data_dir data/ffhq_49K
  ```

## Overall directory

```
├── results
│
├── models
│ ├── ffhq_10m.pt # FFHQ for training
│ ├── 256x256_diffusion_uncond.pt # ImageNet for training
│ └── official_ckpt # For Evaluation
│     ├── ffhq
│     │   ├── gaussian_ema.pt
│     │   ├── sr_averagepooling_ema.pt
│     │   ├── ...
│     │   ├── ...
│     ├── imagenet
│     │   ├── gaussian_ema.pt
│     │   ├── sr_averagepooling_ema.pt
│     │   ├── ...
│     └── └── ...
│
├── data # including training set and evaluation set
│ ├── ffhq_1K # FFHQ evluation
│ ├── imagenet_val_1K # ImageNet evluation
│ ├── ffhq_49K # FFHQ training
│ ├── imagenet_130K # ImageNet training
│ └── y_npy
│         ├── ffhq_1k_npy
│         │   ├── gaussian
│         │   ├── sr_averagepooling
│         │   ├── ...
│         │   └── ...
│         ├── imagenet_val_1k_npy
│         │   ├── gaussian
│         │   ├── sr_averagepooling
│         │   ├── ...
└─────────└── └── ...
```

## Evaluation

### 1. Restore degraded images

- You can specify the directory of measurements with `--y_dir data/y_npy`

```
accelerate launch --num_processes=1 eval.py --eval_dir data/ffhq_1K --deg gaussian --perturb_h 0.1 --ckpt model/official_ckpt/ffhq/gaussian_ema.pt
```

### 2. Evaluate PSNR,LPIPS and FID

- PSNR and LPIPS
  ```
  python utils/eval_psnr_lpips.py
  ```
- FID: [pytorch-fid](https://github.com/mseitzer/pytorch-fid)
  ```
  python -m pytorch_fid source_dir recon_dir
  ```

## Train with MultiGPU

- To check training logs, use the `--use_wandb` flag.

```
accelerate launch --multi_gpu --num_processes=4 train.py --data_dir data/ffhq_49K/ --model_path model/ffhq_10m.pt --deg gaussian --t_ikl 400 --weight_con 0.5 --reg_coeff 0.25 --perturb_h 0.1
```
