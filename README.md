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

We utilize pre-trained models from [Exposure-slot_ckpt.zip)](https://1drv.ms/u/c/1acaeb9b8ad3b4e8/ESoJibo6AeBNpjmZjVYWBqcB7Chlw8_Wdtw0bmz9jkZxsg?e=GTbKrU).

- Place the pre-trained models into the `ckpt/` directory.

### 3. Prepare Data

For training and validating our model, we used SICE, MSEC, and LCDP dataset

- ### SICE dataset

  We downloaded the SICE dataset from [here](https://github.com/csjcai/SICE).
  ```
  python prepare_SICE.py
  ```

- ### MSEC dataset

  We downloaded the MSEC dataset from [here](https://github.com/mahmoudnafifi/Exposure_Correction).
  ```
  python prepare_MSEC.py
  ```
  
- ### LCDP dataset

  We downloaded the LCDP dataset from [here](https://github.com/onpix/LCDPNet).
  ```
  python prepare_LCDP.py
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
