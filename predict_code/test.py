import torch
import argparse
import torch.backends.cudnn as cudnn
import os
from architecture import *
# from utils import save_matv73
import cv2
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib import gridspec
    
import spectral2rgb
import scipy
import h5py
RGB_DIR = r"D:\Datasets\HSI_DATASET\train_vis\RGB_CIE"
VIS_DIR = r"D:\Datasets\HSI_DATASET\train_vis\VIS"
rgbs = [os.path.join(RGB_DIR, f) for f in os.listdir(RGB_DIR)]
vis = [os.path.join(VIS_DIR, f) for f in os.listdir(VIS_DIR)]
# rgb  is first file name containing p012
rgb = [f for f in rgbs if 'p027' in f][0]
#hsi is first file name containing p012
vis = [f for f in vis if 'p027' in f][0]
# vis = r"C:\Users\joeli\Dropbox\Code\Python Projects\MST_JJ\output_pre\hsi_images\p027_neutral_front.mat"
parser = argparse.ArgumentParser(description="SSR")
parser.add_argument('--method', type=str, default='mst_plus_plus')
parser.add_argument('--pretrained_model_path', type=str, default=r'C:\Users\joeli\Dropbox\Code\Python Projects\MST-plus-plus\exp\mst_plus_plus\2024_03_11_19_33_47\epoch_3_train_loss0.07470536977052689_test_loss_0.06304560353358586.pth')
parser.add_argument('--rgb_path', type=str, default=rgb)
#hsi path
parser.add_argument('--hsi_path', type=str, default=vis)
parser.add_argument('--outf', type=str, default='./exp/mst_plus_plus/')
parser.add_argument('--ensemble_mode', type=str, default='mean')
parser.add_argument("--gpu_id", type=str, default='0')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)
wavelengths = np.linspace(400, 700, 31)
def main():
    cudnn.benchmark = True
    pretrained_model_path = opt.pretrained_model_path
    method = opt.method
    # Initialize your model
    model = MST_Plus_Plus(in_channels=3, out_channels=31, n_feat=31, stage=3).cuda()
    # Load the checkpoint
    checkpoint = torch.load(pretrained_model_path)
    # Since checkpoint does not contain 'state_dict', load it directly
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()}, strict=False)

    test(model, opt.rgb_path, opt.hsi_path, opt.outf)
def load_mat_v73(filename):
    with h5py.File(filename, 'r') as file:
        hsi_original = None
        for key in file.keys():
            if key == 'hsi':
                hsi_original = np.array(file[key])
            if key == 'cube':
                hsi_original = np.array(file[key])
        return hsi_original


def test(model, rgb_path, hsi_path, save_path):
    print(f'Loading {rgb_path}')
    # hsi_original = scipy.io.loadmat(hsi_path)
    hsi_original = load_mat_v73(hsi_path)
    hsi_original = np.transpose(hsi_original, [2,1, 0])
    hsi_original = np.float32(hsi_original)

    var_name = 'cube'
    bgr = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = np.float32(rgb)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    rgb = np.expand_dims(np.transpose(rgb, [2, 0, 1]), axis=0).copy()
    rgb = torch.from_numpy(rgb).float().cuda()
    print(f'Reconstructing {rgb_path}')
    with torch.no_grad():
        result = forward_ensemble(rgb, model, opt.ensemble_mode)
    result = result.cpu().numpy() * 1.0
    result = np.transpose(np.squeeze(result), [1, 2, 0])
    result = np.minimum(result, 1.0)
    result = np.maximum(result, 0)
    mat_name = rgb_path.split('/')[-1][:-4] + '.mat'
    rgb_rec = spectral2rgb.Get_RGB(result, np.linspace(400, 700, 31))
    #normalize
    rgb_rec = (rgb_rec - rgb_rec.min()) / (rgb_rec.max() - rgb_rec.min())
    rgb_rec = np.uint8(rgb_rec * 255)
    rgb_rec = cv2.cvtColor(rgb_rec, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb.cpu().numpy().squeeze().transpose(1, 2, 0))
    plt.title('Original RGB')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(rgb_rec)
    plt.title('Reconstructed RGB')
    plt.axis('off')
    plt.show()


    width = 2 * 25
    height = 4 * 25
    plt.clf()
    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(4, 2, wspace=0, hspace=0)  # 6 rows and 2 columns

    hsi_pred = result  # your result variable
    count = 0

    # Ensure that 'hsi' is extracted from hsi_original before the loop if it's a dict
   
    #plot wavelengths 400, 450, 500, 550, 600, 650, 700
    waves = [400, 500, 600, 700]
    for i, wave in enumerate(waves):
        #ensure that hsi_original is a numpy array shape (w,h,c)
        print(f'hsi_original.shape: {hsi_original.shape}, wave: {wave} , i: {i} , type(hsi_original): {type(hsi_original)}')
        print(f'hsi_pred.shape: {hsi_pred.shape}')
        ax1 = fig.add_subplot(gs[i, 0])  # create subplot in left column
        ax2 = fig.add_subplot(gs[i, 1])  # create subplot in right column
        ax1.imshow(hsi_original[:, :, i], cmap='viridis')
        ax2.imshow(hsi_pred[:, :, i], cmap='viridis')
        if i == 0:
            ax1.set_title("Original Band")
            ax2.set_title("Predicted Band")

        ax1.set_ylabel(wavelengths[i])
        ax1.axis('off')
        ax2.axis('off')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])

    plt.show()
def forward_ensemble(x, forward_func, ensemble_mode = 'mean'):
    def _transform(data, xflip, yflip, transpose, reverse=False):
        if not reverse:  # forward transform
            if xflip:
                data = torch.flip(data, [3])
            if yflip:
                data = torch.flip(data, [2])
            if transpose:
                data = torch.transpose(data, 2, 3)
        else:  # reverse transform
            if transpose:
                data = torch.transpose(data, 2, 3)
            if yflip:
                data = torch.flip(data, [2])
            if xflip:
                data = torch.flip(data, [3])
        return data

    outputs = []
    opts = itertools.product((False, True), (False, True), (False, True))
    for xflip, yflip, transpose in opts:
        data = x.clone()
        data = _transform(data, xflip, yflip, transpose)
        data = forward_func(data)
        outputs.append(
            _transform(data, xflip, yflip, transpose, reverse=True))
    if ensemble_mode == 'mean':
        return torch.stack(outputs, 0).mean(0)
    elif ensemble_mode == 'median':
        return torch.stack(outputs, 0).median(0)[0]


if __name__ == '__main__':
    main()