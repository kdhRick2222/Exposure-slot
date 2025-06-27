import os
import sys
import numpy as np
import random
import wandb
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import argparse
# import pyiqa

from network_level2_bayesian import Slot_model as Slot_model_level2
# from network_level2_bayesian import Amortized
from network_level2_bayesian import Amortized_alpha
from config.basic import ConfigBasic
from utils.util import make_dir, set_wandb
from utils.util import AverageMeter, compute_psnr_ssim
from data.dataloaders import get_datasets


def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_local_config(cfg, level, dataset, gpu_num):
    # Level
    cfg.level = level
    # Dataset
    cfg.dataset = dataset

    # # SICE alpha_bayesian - pretrained
    # cfg.ckpt_path = './ckpt/SICE_level2.pth'
    # cfg.bayesian_ckpt_path = './ckpt/SICE_level2_bayesian.pth'
    # # LCDP alpha_bayesian - pretrained
    # cfg.ckpt_path = './ckpt/LCDP_level2.pth'
    # cfg.bayesian_ckpt_path = './ckpt/LCDP_level2_bayesian.pth'
    # # MSEC alpha_bayesian - pretrained
    # cfg.ckpt_path = './ckpt/MSEC_level2.pth'
    # cfg.bayesian_ckpt_path = './ckpt/MSEC_level2_bayesian.pth'


    cfg.set_dataset()
    cfg.learning_rate = 2e-4
    cfg.batch_size = 8

    # Log
    cfg.wandb = False
    cfg.save_folder = f'./{cfg.dataset}/Level{cfg.level}_alpha_Bayesian_Results'
    make_dir(cfg.save_folder)

    cfg.n_gpu = torch.cuda.device_count()
    cfg.device = "cuda:"+gpu_num
    cfg.num_workers = 1
    return cfg


def main(level=3, dataset='SICE', gpu_num='0'):
    cfg = ConfigBasic()
    cfg = set_local_config(cfg, level, dataset, gpu_num)

    # dataloader
    loader_dict = get_datasets(cfg)

    # load model
    if level==2:
        model = Slot_model_level2(cfg)
        bayesian_net = Amortized_alpha(level=2)
    else:
        model = Slot_model_level3(cfg)
        bayesian_net = Amortized_alpha(level=3)

    ckpt = torch.load(cfg.ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    bayesian_ckpt = torch.load(cfg.bayesian_ckpt_path, map_location='cpu', weights_only=False)
    bayesian_net.load_state_dict(bayesian_ckpt['model'])

    if cfg.wandb:
        set_wandb(cfg)
        wandb.watch(model)

    model = model.to(cfg.device)
    bayesian_net = bayesian_net.to(cfg.device)

    for i in range(1):
        val_psnr, val_ssim = validate(cfg, loader_dict, model, bayesian_net)


def validate(cfg, loader_dict, model, bayesian_net):

    model.eval()

    psnr_list = []
    
    PSNRs_under = AverageMeter()
    SSIMs_under = AverageMeter()    
    PSNRs_over = AverageMeter()
    SSIMs_over = AverageMeter()

    w_over = 0
    w_under = 0

    test_loader = loader_dict['val']

    with torch.no_grad():
        for idx, (x_base, gt_base, label, item) in enumerate(test_loader):
            # print(idx, '/', len(test_loader))
            x_base = x_base.to(cfg.device)
            gt_base = gt_base.to(cfg.device)
            
            factor = 4
            h, w = x_base.shape[2], x_base.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            x_base = F.pad(x_base, (0, padw, 0, padh), 'reflect')
            gt_base = F.pad(gt_base, (0, padw, 0, padh), 'reflect')

            preds, slot_recon, feature_loss, mu_list, log_var_list = model(x_base, x_base, bayesian_net, inference=True)

            preds = preds[:, :, :h, :w]
            gt_base = gt_base[:,:,:h,:w]
            x_base = x_base[:,:,:h,:w]
            
            temp_psnr, temp_ssim, N = compute_psnr_ssim(torch.clip(preds, 0, 1), gt_base)
            
            # under
            if cfg.dataset == 'SICE':
                picture_name = item[0].split('/')[-1][:-4]

                if label == 0: # Under-exposed images
                    PSNRs_under.update(temp_psnr, N)
                    SSIMs_under.update(temp_ssim, N)
                    w_under += 1
                    name_path = "/" + picture_name + f"_under_{temp_psnr:.2f}_{temp_ssim:.4f}.png"
                    save_image(preds, cfg.save_folder + name_path)
                elif label == 2: # Over-exposed images
                    PSNRs_over.update(temp_psnr, N)
                    SSIMs_over.update(temp_ssim, N)
                    w_over += 1
                    name_path = "/" + picture_name + f"_over_{temp_psnr:.2f}_{temp_ssim:.4f}.png"
                    save_image(preds, cfg.save_folder + name_path)

            elif cfg.dataset == 'LCDP':
                PSNRs_under.update(temp_psnr, N)
                SSIMs_under.update(temp_ssim, N)
                w_under += 1
                picture_name = item[0].split('/')[-1][:-4]
                name_path = "/" + picture_name + f"_{temp_psnr:.2f}_{temp_ssim:.4f}.png"
                save_image(preds, cfg.save_folder + name_path)

            # MSEC dataset
            else:
                picture_name = item[0].split('/')[-1][:-4]

                if label < 0:
                    PSNRs_under.update(temp_psnr, N)
                    SSIMs_under.update(temp_ssim, N)
                    w_under += 1
                    name_path = "/" + picture_name + f"_under_{temp_psnr:.2f}_{temp_ssim:.4f}.png"
                    save_image(preds, cfg.save_folder + name_path)
                else:
                    PSNRs_over.update(temp_psnr, N)
                    SSIMs_over.update(temp_ssim, N)
                    w_over += 1
                    name_path = "/" + picture_name + f"_over_{temp_psnr:.2f}_{temp_ssim:.4f}.png"
                    save_image(preds, cfg.save_folder + name_path)


    PSNRs = (w_over*PSNRs_over.avg + w_under*PSNRs_under.avg) / (w_over + w_under)
    SSIMs = (w_over*SSIMs_over.avg + w_under*SSIMs_under.avg) / (w_over + w_under)
    
    if cfg.dataset == 'SICE':
        print(f'SICEV2 - PSNR : {PSNRs:.2f},  SSIM : {SSIMs:.4f}  /  SICEV2 over - PSNR : {PSNRs_over.avg:.2f},  SSIM : {SSIMs_over.avg:.4f}  /  SICEV2 under - PSNR : {PSNRs_under.avg:.2f},  SSIM : {SSIMs_under.avg:.4f}')
    elif cfg.dataset == 'LCDP':
        print(f'LCDP - PSNR : {PSNRs:.2f},  SSIM : {SSIMs:.4f}')
    else:
        print(f'MSEC - PSNR : {PSNRs:.2f},  SSIM : {SSIMs:.4f},  /  MSEC over - PSNR : {PSNRs_over.avg:.2f},  SSIM : {SSIMs_over.avg:.4f}  /  MSEC under - PSNR : {PSNRs_under.avg:.2f},  SSIM : {SSIMs_under.avg:.4f}')
    sys.stdout.flush()
    return PSNRs, SSIMs


# python test_extention.py

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exposure-slot')
    parser.add_argument('--gpu_num', default='3', type=str, metavar='D', help='choose gpu')
    parser.add_argument('--level', default='2', type=int, metavar='D', help='2 or 3')
    parser.add_argument('--dataset', default='LCDP', type=str, metavar='D', help='SICE, MSEC, LCDP')
    args = parser.parse_args()

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    main(level=args.level, dataset=args.dataset, gpu_num=args.gpu_num)
