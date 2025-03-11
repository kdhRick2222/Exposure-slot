import pickle
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from data.datasets import Basic, Trainingset


def get_datasets(cfg):
    te_std = None
       
    if cfg.dataset =='SICE':
        tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter)
        tr_list = np.array(tr_list)
        tr_imgs = [i_path for i_path in tr_list[:, cfg.img_idx]]
        tr_gt = [i_path for i_path in tr_list[:, cfg.gt_idx]]
        tr_exp = tr_list[:, cfg.lb_idx]

        te_list = pd.read_csv(cfg.test_file, sep=cfg.delimeter)
        te_list = np.array(te_list)
        te_imgs = [i_path for i_path in te_list[:, cfg.img_idx]]
        te_gt = [i_path for i_path in te_list[:, cfg.gt_idx]]
        te_exp = te_list[:, cfg.lb_idx]
       
    elif cfg.dataset =='MSEC':
        tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter)
        tr_list = np.array(tr_list)
        tr_imgs = [i_path for i_path in tr_list[:, cfg.img_idx]]
        tr_gt = [i_path for i_path in tr_list[:, cfg.gt_idx]]
        tr_exp = tr_list[:, cfg.lb_idx]

        te_list = pd.read_csv(cfg.test_file, sep=cfg.delimeter)
        te_list = np.array(te_list)
        te_imgs = [i_path for i_path in te_list[:, cfg.img_idx]]
        te_gt = [i_path for i_path in te_list[:, cfg.gt_idx]]
        te_exp = te_list[:, cfg.lb_idx]
           
    elif cfg.dataset =='LCDP':
        tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter)
        tr_list = np.array(tr_list)
        tr_imgs = [i_path for i_path in tr_list[:, cfg.img_idx]]
        tr_gt = [i_path for i_path in tr_list[:, cfg.gt_idx]]
        tr_exp = tr_list[:, cfg.lb_idx]

        te_list = pd.read_csv(cfg.test_file, sep=cfg.delimeter)
        te_list = np.array(te_list)
        te_imgs = [i_path for i_path in te_list[:, cfg.img_idx]]
        te_gt = [i_path for i_path in te_list[:, cfg.gt_idx]]
        te_exp = te_list[:, cfg.lb_idx]

    else:
        print("Please check dataset again.")

    loader_dict = dict()
    
    loader_dict['train'] = DataLoader(Trainingset(tr_imgs, tr_gt, tr_exp, cfg.transform, is_filelist=cfg.is_filelist),
                                      batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)

    loader_dict['val'] = DataLoader(Basic(te_imgs, te_gt, te_exp, cfg.transform, is_filelist=cfg.is_filelist),
                                     batch_size=1, shuffle=False, drop_last=False, num_workers=cfg.num_workers)


    return loader_dict





