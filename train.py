import os
import time
import sys
from copy import deepcopy
import argparse

import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

# sys.path.append('../')
from network_level2 import Slot_model as Slot_model_level2
from network_level3 import Slot_model as Slot_model_level3
from config.basic import ConfigBasic
from utils.util import write_log, make_dir, log_configs, save_ckpt
from utils.util import AverageMeter, compute_psnr_ssim
from data.dataloaders import get_datasets
from torchvision.utils import save_image

train_data = 'train'

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
    cfg.set_dataset()

    # Training
    cfg.epochs = 1000
    if cfg.dataset == 'MSEC':
        cfg.val_freq = 1000
    else:
        cfg.val_freq = 5

    cfg.learning_rate = 2e-4
    cfg.batch_size = 8

    # Log
    cfg.wandb = False
    cfg.save_folder = f'./{cfg.dataset}/Level{cfg.level}'
    make_dir(cfg.save_folder)

    cfg.n_gpu = 1 
    cfg.device = "cuda:"+gpu_num
    cfg.num_workers = 1
    return cfg


def main(level=3, dataset='SICE', gpu_num='0'):
    np.random.seed(999)
    seed_torch(42)

    cfg = ConfigBasic()
    cfg = set_local_config(cfg, level, dataset, gpu_num)
    cfg.logfile = log_configs(cfg, log_file='train_log.txt')

    # dataloader
    loader_dict = get_datasets(cfg)

    # model
    if level == 2:
        model = Slot_model_level2(cfg)
    elif level == 3:
        model = Slot_model_level3(cfg)
    else:
        print("Please check level again.")

    if cfg.adam:
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=cfg.learning_rate,
                              momentum=cfg.momentum,
                              weight_decay=cfg.weight_decay)

    if cfg.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs, eta_min=cfg.learning_rate*0.001)
    elif cfg.scheduler == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.lr_decay_epochs, gamma=cfg.lr_decay_rate)

    if torch.cuda.is_available():
        if cfg.n_gpu > 1:
            model = nn.DataParallel(model)
        model = model.to(cfg.device)
        cudnn.benchmark = True


    # init loss matrix
    loss_record = dict()

    for epoch in range(cfg.epochs):
        print("==> training...")

        time1 = time.time()
        train_loss, loss_record = train(cfg, epoch, loader_dict[train_data], model, optimizer, prev_loss_record=loss_record)

        if cfg.scheduler:
            scheduler.step()
        time2 = time.time()
        print('epoch {}, loss {:.4f}, total time {:.2f}'.format(epoch, train_loss, time2 - time1))

        # torch.cuda.empty_cache()

        if (epoch+1) % cfg.val_freq == 0:
            print('==> validation...')
            val_psnr, val_ssim = validate(loader_dict, model, cfg)
            save_ckpt(cfg, model, f'ep_{epoch}_val_psnr_{val_psnr:.2f}_val_ssim_{val_ssim:.4f}.pth')

        if cfg.dataset == 'MSEC':
            save_ckpt(cfg, model, f'ep_{epoch}.pth')

    print('[*] Training ends')


def train(cfg, epoch, train_loader, model, optimizer, prev_loss_record):
    """One epoch training"""

    model = model.to(cfg.device)
    model.train()

    l1_loss = nn.L1Loss().to(cfg.device)
    Color_loss = nn.CosineSimilarity(dim=1, eps=1e-6).to(cfg.device)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    recon_losses = AverageMeter()

    loss_record = deepcopy(prev_loss_record)
    end = time.time()

    # base_img, ref_img, base_gt, order_label, [base_rank, ref_rank], item
    for idx, (x_base, gt_base, _, _, _) in enumerate(train_loader):

        # if torch.cuda.is_available():  
        x_base = x_base.to(cfg.device)
        gt_base = gt_base.to(cfg.device)

        data_time.update(time.time() - end)

        # ===================forward=====================
        # preds, attn_list, slot_loss, color_map_loss, feature_loss = model(x_base, gt_base, rank_base, inference=False)
        preds, slot_recon, feature_loss = model(x_base, gt_base, inference=False)

        # =====================loss======================
        recon_loss = l1_loss(preds, gt_base) 
        total_loss = recon_loss + feature_loss 

        losses.update(total_loss.item(), x_base.size(0))
        recon_losses.update(recon_loss.item(), x_base.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % cfg.print_freq == 0:
            write_log(cfg.logfile,
                      f'Epoch [{epoch}][{idx}/{len(train_loader)}]\t' 
                      f'Loss {losses.val:.4f}\t' 
                      f'Recon-Loss {recon_loss:.4f}\t' 
                      f'Feature-Loss {feature_loss:.4f}\t'
                      )
            sys.stdout.flush()
    write_log(cfg.logfile, f' * Total_loss: {losses.avg:.3f}')

    return losses.avg, loss_record


def validate(loader_dict, model, cfg):
    model.eval()
    data_time = AverageMeter()
    
    PSNRs_under = AverageMeter()
    SSIMs_under = AverageMeter()    
    PSNRs_over = AverageMeter()
    SSIMs_over = AverageMeter()
    w_over = 0
    w_under = 0

    test_loader = loader_dict['val']

    with torch.no_grad():

        for idx, (x_base, gt_base, label, _) in enumerate(test_loader):
            x_base = x_base.to(cfg.device)
            gt_base = gt_base.to(cfg.device)
                        
            factor = 4
            h, w = x_base.shape[2], x_base.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            x_base = F.pad(x_base, (0, padw, 0, padh), 'reflect')
            gt_base = F.pad(gt_base, (0, padw, 0, padh), 'reflect')

            model = model.to(cfg.device)
            preds, slot_recon, feature_loss = model(x_base, x_base, inference=True)
            B, _, H_, W_ = preds.shape
            
            temp_psnr, temp_ssim, N = compute_psnr_ssim(torch.clip(preds, 0, 1), gt_base)

            # under
            if cfg.dataset == 'SICE':
                make_dir(cfg.save_folder+'/SICEV2_results')
                if label == 0:
                    PSNRs_under.update(temp_psnr, N)
                    SSIMs_under.update(temp_ssim, N)
                    w_under += 1
                    save_image(torch.cat([x_base, gt_base, preds], dim=0), cfg.save_folder + "/SICEV2_results/" + str(idx) + "_under.png")
                elif label == 2:
                    PSNRs_over.update(temp_psnr, N)
                    SSIMs_over.update(temp_ssim, N)
                    w_over += 1
                    save_image(torch.cat([x_base, gt_base, preds], dim=0), cfg.save_folder + "/SICEV2_results/" + str(idx) + "_over.png")

            elif cfg.dataset == 'LCDP':
                make_dir(cfg.save_folder+'/LCDP_results')
                if label < 0:
                    PSNRs_under.update(temp_psnr, N)
                    SSIMs_under.update(temp_ssim, N)
                    w_under += 1
                    save_image(torch.cat([x_base, gt_base, preds], dim=0), cfg.save_folder + "/LCDP_results/" + str(idx) + ".png")
                else:
                    PSNRs_over.update(temp_psnr, N)
                    SSIMs_over.update(temp_ssim, N)
                    w_over += 1
                    save_image(torch.cat([x_base, gt_base, preds], dim=0), cfg.save_folder + "/LCDP_results/" + str(idx) + ".png")

            else: # MSEC
                make_dir(cfg.save_folder+'/MSEC_results')
                if label < 0:
                    PSNRs_under.update(temp_psnr, N)
                    SSIMs_under.update(temp_ssim, N)
                    w_under += 1
                    save_image(torch.cat([x_base, gt_base, preds], dim=0), cfg.save_folder + "/MSEC_results/" + str(idx) + "_under.png")
                else:
                    PSNRs_over.update(temp_psnr, N)
                    SSIMs_over.update(temp_ssim, N)
                    w_over += 1
                    save_image(torch.cat([x_base, gt_base, preds], dim=0), cfg.save_folder + "/MSEC_results/" + str(idx) + "_over.png")
        
    # PSNRs = 0.5*PSNRs_over.avg + 0.5*PSNRs_under.avg
    # SSIMs = 0.5*SSIMs_over.avg + 0.5*SSIMs_under.avg
    PSNRs = (w_over*PSNRs_over.avg + w_under*PSNRs_under.avg) / (w_over + w_under)
    SSIMs = (w_over*SSIMs_over.avg + w_under*SSIMs_under.avg) / (w_over + w_under)
    write_log(cfg.logfile, f'SICEV2 - PSNR : {PSNRs:.2f},  SSIM : {SSIMs:.4f}  /  SICEV2 over - PSNR : {PSNRs_over.avg:.2f},  SSIM : {SSIMs_over.avg:.4f}  /  SICEV2 under - PSNR : {PSNRs_under.avg:.2f},  SSIM : {SSIMs_under.avg:.4f}')
    sys.stdout.flush()
    return PSNRs, SSIMs



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exposure-slot')
    parser.add_argument('--gpu_num', default='0', type=str, metavar='D', help='choose gpu')
    parser.add_argument('--level', default='2', type=int, metavar='D', help='2 or 3')
    parser.add_argument('--dataset', default='SICE', type=str, metavar='D', help='SICE, MSEC, LCDP')
    args = parser.parse_args()

    main(level=args.level, dataset=args.dataset, gpu_num=args.gpu_num)
