import numpy as np
import onnxruntime as ort
import cv2
import scipy.io
import matplotlib.pyplot as plt
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
import time
import spectral2rgb
import scipy
import h5py
RGB_DIR = r"D:\Datasets\HSI_DATASET\train_vis\RGB_CIE"
VIS_DIR = r"D:\Datasets\HSI_DATASET\train_vis\VIS"
rgbs = [os.path.join(RGB_DIR, f) for f in os.listdir(RGB_DIR)]
vis = [os.path.join(VIS_DIR, f) for f in os.listdir(VIS_DIR)]
# rgb  is first file name containing p012
rgb = [f for f in rgbs if 'p019' in f][0]
#hsi is first file name containing p012
vis = [f for f in vis if 'p019' in f][0]
def load_and_preprocess_image(image_path):
    bgr = cv2.imread(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (512, 512))
    rgb = np.float32(rgb)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    rgb = np.expand_dims(np.transpose(rgb, [2, 0, 1]), axis=0)
    return rgb

def convert_rgb_to_hyperspectral(onnx_model_path, rgb_path):
    # Load the ONNX model
    ort_session = ort.InferenceSession(onnx_model_path)

    # Load and preprocess the image
    rgb = load_and_preprocess_image(rgb_path)

    # Convert to tensor
    ort_inputs = {ort_session.get_inputs()[0].name: rgb}
    ort_outs = ort_session.run(None, ort_inputs)

    # Postprocess the output
    hyperspectral_cube = np.squeeze(ort_outs[0])
    
    # Saving the hyperspectral cube to a .mat file (optional)
    mat_name = rgb_path.split('/')[-1][:-4] + '_hyperspectral.mat'
    scipy.io.savemat(mat_name, {'cube': hyperspectral_cube})
    
    return hyperspectral_cube

# Use the function
onnx_model_path = 'predict_code/rgb2hsi.onnx'  # Update this path
rgb_path = rgb
start = time.time()
hyperspectral_cube = convert_rgb_to_hyperspectral(onnx_model_path, rgb_path)
end = time.time()
print(f'Inference time: {end - start} seconds')
#reshape to 512x512x31
hyperspectral_cube = np.transpose(hyperspectral_cube, [1, 2, 0])
wavelengths = np.linspace(400, 700, 31)
reconstructed_rgb = spectral2rgb.Get_RGB(hyperspectral_cube, wavelengths)
# reconstructed_rgb = (reconstructed_rgb - reconstructed_rgb.min()) / (reconstructed_rgb.max() - reconstructed_rgb.min())

original_rgb = cv2.imread(rgb)
original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_BGR2RGB)
original_rgb = cv2.resize(original_rgb, (512, 512))
#plot original,  400, 500, 600, 700 nm, reconstructed
fig, ax = plt.subplots(1, 6, figsize=(25, 5))
plt.subplot(1, 6, 1)
plt.imshow(original_rgb)
plt.title('Original RGB')
plt.axis('off')
plt.subplot(1, 6, 2)
plt.imshow(hyperspectral_cube[:,:,0])
plt.title('400 nm')
plt.axis('off')
plt.subplot(1, 6, 3)
plt.imshow(hyperspectral_cube[:,:,10])
plt.title('500 nm')
plt.axis('off')
plt.subplot(1, 6, 4)
plt.imshow(hyperspectral_cube[:,:,20])
plt.title('600 nm')
plt.axis('off')
plt.subplot(1, 6, 5)
plt.imshow(hyperspectral_cube[:,:,30])
plt.title('700 nm')
plt.axis('off')
plt.subplot(1, 6, 6)
plt.imshow(reconstructed_rgb)
plt.title('Reconstructed RGB')
plt.axis('off')
plt.show()
