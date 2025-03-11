import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import glob
import random
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor, Resize
from PIL import Image


def load_one_image(img_path, width=256, height=256, rot_deg=0):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((width, height), Image.LANCZOS)
    return img


def load_one_image_test(img_path):
    img = Image.open(img_path).convert('RGB')
    return img


class Basic(Dataset):
    def __init__(self, imgs, gt_imgs, labels, transform, norm_age=True, is_filelist=False, return_ranks=False, std=None):
        super(Dataset, self).__init__()
        self.transform = transform
        self.imgs = imgs
        self.gt_imgs = gt_imgs
        self.labels = labels

        # self.n_imgs = len(sorted(self.imgs))
        self.n_imgs = len(self.imgs)
        self.is_filelist = is_filelist
        # if norm_age:
        #     self.labels = self.labels - min(self.labels)
        # self.return_ranks = return_ranks
        # self.std = std

        # rank = 0
        # self.mapping = dict()
        # for cls in np.unique(self.labels):
        #     self.mapping[cls] = rank
        #     rank += 1
        # self.ranks = np.array([self.mapping[l] for l in self.labels])

    def __getitem__(self, item):
        if self.is_filelist:
            img = np.asarray(load_one_image_test(self.imgs[item]))
            gt_img = np.asarray(load_one_image_test(self.gt_imgs[item]))
        else:
            img = np.asarray(self.imgs[item]).astype('uint8')
            gt_img = np.asarray(self.gt_imgs[item]).astype('uint8')
        img = self.transform(img)
        gt_img = self.transform(gt_img)
        return img, gt_img, self.labels[item], self.imgs[item]

    def __len__(self):
        return len(self.imgs)



class Trainingset(Dataset):
    def __init__(self, imgs, gt_imgs, labels, transform, is_filelist=False):
        super(Dataset, self).__init__()
        self.imgs = imgs
        self.gt_imgs = gt_imgs
        self.labels = labels
        self.transform = transform
        self.n_imgs = len(self.imgs)
        self.min_age_bf_norm = self.labels.min()
        self.labels = self.labels - min(self.labels)
        self.is_filelist = is_filelist

        rank = 0
        self.mapping = dict()
        for cls in np.unique(self.labels):
            self.mapping[cls] = rank
            rank += 1
        self.ranks = np.array([self.mapping[l] for l in self.labels])

    def _crop_patch(self, img_1, img_2):
        patch_size = 256
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - patch_size)
        ind_W = random.randint(0, W - patch_size)

        patch_1 = img_1[ind_H:ind_H + patch_size, ind_W:ind_W + patch_size]
        patch_2 = img_2[ind_H:ind_H + patch_size, ind_W:ind_W + patch_size]

        return patch_1, patch_2

    def __getitem__(self, item):

        if self.is_filelist:
            base_img = np.asarray(load_one_image(self.imgs[item]))
            base_gt = np.asarray(load_one_image(self.gt_imgs[item]))
        else:
            base_img = np.asarray(self.imgs[item]).astype('uint8')
            base_gt = np.asarray(self.gt_imgs[item]).astype('uint8')

        base_img, base_gt = self._crop_patch(base_img, base_gt)
        base_img = self.transform(base_img)
        base_gt = self.transform(base_gt)

        # gt ranks
        base_rank = self.ranks[item]

        # return base_img, ref_img, base_gt, ref_gt, self.labels[item], [base_rank, ref_rank], item
        return base_img, base_gt, self.labels[item], [base_rank], item

    def __len__(self):
        return self.n_imgs
       


