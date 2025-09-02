<p align="center">
  <h1 align="center">Exposure-slot: Exposure-centric representations learning with Slot-in-Slot Attention for Region-aware Exposure Correction (Official)</h1>
  
  <p align="center">
    <a href="https://github.com/dgjung0220">Donggoo Jung</a>*, 
    <a href="https://github.com/kdhRick2222">Daehyun Kim</a>*, 
    <a href="https://scholar.google.com/citations?hl=ko&user=I_5aoAwAAAAJ">Guanghui Wang</a>,
    <a href="https://sites.google.com/view/lliger9/">Tae Hyun Kim</a>†.
      (*Equal Contribution, †Corresponding author)
  </p>
  <h2 align="center">CVPR 2025</h2>

  <h3 align="center">
    <!-- GitHub Project -->
    <a href="https://github.com/kdhRick2222/Exposure-slot" target="_blank"><img src="https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white"></a>
    <!-- CVPR Paper -->
    <a href="https://openaccess.thecvf.com/content/CVPR2025/papers/Jung_Exposure-slot_Exposure-centric_Representations_Learning_with_Slot-in-Slot_Attention_for_Region-aware_Exposure_CVPR_2025_paper.pdf" target="_blank"><img src="https://img.shields.io/badge/CVPR%20Paper-003B6F?logo=readthedocs&logoColor=white"></a>
    <!-- Hugging Face Demo -->
    <a href="https://huggingface.co/kdh2b/Exposure-slot/tree/main" target="_blank"><img src="https://img.shields.io/badge/🤗%20HuggingFace-FFAC45?logo=huggingface&logoColor=white"></a>
<!--     <a href="https://paperswithcode.com/sota/image-enhancement-on-exposure-errors?p=exposure-slot-exposure-centric" target="_blank">
  <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/exposure-slot-exposure-centric/image-enhancement-on-exposure-errors"> -->
</a>
  </h3>

</p>

This repository contains the official PyTorch implementation of "**_Exposure-slot_**: *Exposure-centric representations learning with Slot-in-Slot Attention for Region-aware Exposure Correction*" accepted at **CVPR 2025.**


<div align="center">
  <img src="images/concept_figure.png" width="500px" />
</div>


**Exposure-slot** is the first approach to leverage *Slot Attention mechanism* for optimized exposure-specific feature partitioning. We introduce the slot-in-slot attention that enables sophisticated feature partitioning and learning and exposure-aware prompts that enhance the exposure-centric characteristics of each image feature. 


Our proposing method is **the first approach to leverage Slot Attention mechanism** for optimized exposure-specific feature partitioning. We introduce the **slot-in-slot attention** that enables sophisticated feature partitioning and learning and exposure-aware prompts that enhance the exposure-centric characteristics of each image feature. We provide validation code, training code, and pre-trained weights on three benchmark datasets (**MSEC, SICE, LCDP**).

## Setting

Please follow these steps to set up the repository.

### 1. Clone the Repository

```
git clone https://github.com/kdhRick2222/Exposure-slot.git
cd Exposure-slot
```

### 2. Download Pre-trained models and Official Checkpoints

We utilize pre-trained models from [Exposure-slot_ckpt.zip](https://1drv.ms/u/c/1acaeb9b8ad3b4e8/ESoJibo6AeBNpjmZjVYWBqcBo1RC2pXZO3S13wEwiMqZQg?e=LQkgJo).

- Place the pre-trained models into the `ckpt/` directory.

### 3. Prepare Data

For training and validating our model, we used SICE, MSEC, and LCDP dataset.

- ### SICE dataset

  We downloaded the SICE dataset from [here](https://github.com/csjcai/SICE). 
  ```
  python prepare_SICE.py
  ```
  Make .Dataset_txt/SICE_Train.txt and .Dataset_txt/SICE_Test.txt for validation and training.

- ### MSEC dataset

  We downloaded the MSEC dataset from [here](https://github.com/mahmoudnafifi/Exposure_Correction).
  ```
  python prepare_MSEC.py
  ```
  Make .Dataset_txt/MSEC_Train.txt and .Dataset_txt/MSEC_Test.txt for validation and training.
  
- ### LCDP dataset

  We downloaded the LCDP dataset from [here](https://github.com/onpix/LCDPNet).
  ```
  python prepare_LCDP.py
  ```
  Make .Dataset_txt/LCDP_Train.txt and .Dataset_txt/LCDP_Test.txt for validation and training.


## Inference and Evaluation
We provide *2-level* and *3-level* Exposure-slot model for each dataset (SICE, MSEC, LCDP).
  ```
  python test.py --level=2 --dataset="MSEC"
  ```

## Training
  ```
  python train.py --gpu_num=0 --level=2 --dataset="MSEC"
  ```

## Overall directory

```
├── ckpts
│ ├── LCDP_level2.pth
│ ├── LCDP_level3.pth
│ ├── MSEC_level2.pth
│ ├── MSEC_level3.pth
│ ├── SICE_level2.pth
│ └── SICE_level3.pth
│
├── config
│ ├── basic.py
│
├── data
│ ├── dataloaders.py
│ └── datasets.py
|
├── Dataset_txt
│ ├── LCDP_Train.txt
│ ├── LCDP_Test.txt
│ ├── MSEC_Train.txt
│ ├── MSEC_Test.txt
│ ├── SICE_Train.txt
│ └── SICE_Test.txt
|
├── utils
│ ├── scheduler_util.py
│ └── util.py
|
├── network_level2.py
├── network_level3.py
├── prepare_LCDP.py
├── prepare_MSEC.py
├── prepare_SICE.py
├── test.py
└── train.py
```

## Citation
If you find our work useful in your research, please cite:

```
@inproceedings{jung2025Exposureslot,
  title={Exposure-slot: Exposure-centric representations learning with Slot-in-Slot Attention for Region-aware Exposure Correction},
  author={Donggoo Jung, Daehyun Kim, Guanghui Wang, Tae Hyun Kim},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```
