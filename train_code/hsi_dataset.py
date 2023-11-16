from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import os
import scipy.io
#pathlib
from pathlib import Path
import argparse

def get_base_name(file_name):
    return os.path.splitext(file_name)[0]

def remove_unmatched_files(path, reference_list):
    for file_name in os.listdir(path):
        base_name = get_base_name(file_name)
        if base_name not in reference_list:
            file_path = os.path.join(path, file_name)
            os.remove(file_path)
            print(f"Removed: {file_path}")
class TrainDataset(Dataset):
    def __init__(self, train_spec_path, train_rgb_path, crop_size, arg=True, bgr2rgb=True, stride=8):
        self.train_spec_path = train_spec_path
        self.train_rgb_path = train_rgb_path
        self.crop_size = crop_size
        self.hypers = []
        self.bgrs = []
        self.arg = arg
        h,w = 482,512
        self.stride = stride
        self.patch_per_line = (w-crop_size)//stride+1
        self.patch_per_colum = (h-crop_size)//stride+1
        self.patch_per_img = self.patch_per_line*self.patch_per_colum

        hyper_data_path = Path(train_spec_path)
        bgr_data_path = Path(train_rgb_path)
        hyper_list = os.listdir(hyper_data_path)
        bgr_list = os.listdir(bgr_data_path)
        # hyper_list.sort()
        # bgr_list.sort()
        # #remove all files with non matching basename
        # hyper_list = [x for x in hyper_list if x.split('.')[0] in [y.split('.')[0] for y in bgr_list]]
        # bgr_list = [x for x in bgr_list if x.split('.')[0] in [y.split('.')[0] for y in hyper_list]]
        print(f'len(hyper) of ntire2022 dataset:{len(hyper_list)}')
        print(f'len(bgr) of ntire2022 dataset:{len(bgr_list)}')
        for i in range(len(hyper_list)):
            hyper_path = hyper_data_path / hyper_list[i]
            bgr_path = bgr_data_path / bgr_list[i]

            if 'mat' not in str(hyper_path):
                continue

            hyper = np.float32(scipy.io.loadmat(hyper_path)['hsi'])
            #select the first 31 bands
            hyper = hyper[:,:,:31]
            hyper = np.transpose(hyper, [2, 0, 1])
            
            bgr = cv2.imread(str(bgr_path))
            if bgr is None or bgr.size == 0:
                print(f"Failed to load image: {bgr_path}")
                continue

            if bgr2rgb and bgr.ndim == 3:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            elif bgr.ndim != 3:
                print(f"Invalid image shape: {bgr.shape} for file: {bgr_path}")
                continue

            bgr = np.float32(bgr) / 255.0
            if bgr.ndim == 3:
                bgr = np.transpose(bgr, [2, 0, 1])  # [3, height, width]
            if bgr.shape[1] != h or bgr.shape[2] != w or hyper.shape[1] != h or hyper.shape[2] != w:
                print(f"Skipping: {bgr_path}, Invalid shape: {bgr.shape}, {hyper.shape}")
                continue
            self.hypers.append(hyper)
            self.bgrs.append(bgr)
            print(f'Ntire2022 scene {i} is loaded.')
        self.img_num = len(self.hypers)
        self.length = self.patch_per_img * self.img_num

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        stride = self.stride
        crop_size = self.crop_size
        img_idx, patch_idx = idx//self.patch_per_img, idx%self.patch_per_img
        h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line
        bgr = self.bgrs[img_idx]
        hyper = self.hypers[img_idx]
        bgr = bgr[:,h_idx*stride:h_idx*stride+crop_size, w_idx*stride:w_idx*stride+crop_size]
        hyper = hyper[:, h_idx * stride:h_idx * stride + crop_size,w_idx * stride:w_idx * stride + crop_size]
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return self.patch_per_img*self.img_num

class ValidDataset(Dataset):
    def __init__(self,test_spec_path, test_rgb_path, bgr2rgb=True):
        self.hypers = []
        self.bgrs = []
        h,w = 482,512
        # hyper_data_path = r'C:\\Users\\joeli\\Dropbox\\Code\\MST-plus-plus\\dataset\\Val_Spec\\'
        # bgr_data_path = r'C:\\Users\\joeli\\Dropbox\\Code\\MST-plus-plus\\dataset\\Val_RGB\\'
        # hyper_list = os.listdir(hyper_data_path)
        # bgr_list = os.listdir(bgr_data_path)
        # hyper_list.sort()
        # bgr_list.sort()
        self.test_spec_path = test_spec_path
        self.test_rgb_path = test_rgb_path
        hyper_data_path = Path(test_spec_path)
        bgr_data_path = Path(test_rgb_path)
        hyper_list = os.listdir(hyper_data_path)
        bgr_list = os.listdir(bgr_data_path)
        # hyper_list.sort()
        # bgr_list.sort()
        # hyper_list = [x for x in hyper_list if x.split('.')[0] in [y.split('.')[0] for y in bgr_list]]
        # bgr_list = [x for x in bgr_list if x.split('.')[0] in [y.split('.')[0] for y in hyper_list]]
        print(f'len(hyper_valid) of ntire2022 dataset:{len(hyper_list)}')
        print(f'len(bgr_valid) of ntire2022 dataset:{len(bgr_list)}')
        for i in range(len(hyper_list)):
            hyper_path = hyper_data_path / hyper_list[i]
            bgr_path = bgr_data_path / bgr_list[i]

            if 'mat' not in str(hyper_path):
                continue

            hyper = np.float32(scipy.io.loadmat(hyper_path)['hsi'])
            #select the first 31 bands
            hyper = hyper[:,:,:31]
            hyper = np.transpose(hyper, [2, 0, 1])
            
            bgr = cv2.imread(str(bgr_path))
            if bgr is None or bgr.size == 0:
                print(f"Failed to load image: {bgr_path}")
                continue

            if bgr2rgb and bgr.ndim == 3:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            elif bgr.ndim != 3:
                print(f"Invalid image shape: {bgr.shape} for file: {bgr_path}")
                continue

            bgr = np.float32(bgr) / 255.0
            if bgr.ndim == 3:
                bgr = np.transpose(bgr, [2, 0, 1])  # [3, height, width]
            if bgr.shape[1] != h or bgr.shape[2] != w or hyper.shape[1] != h or hyper.shape[2] != w:
                print(f"Skipping: {bgr_path}, Invalid shape: {bgr.shape}, {hyper.shape}")
                continue
            self.hypers.append(hyper)
            self.bgrs.append(bgr)

    def __getitem__(self, idx):
        hyper = self.hypers[idx]
        bgr = self.bgrs[idx]
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hypers)