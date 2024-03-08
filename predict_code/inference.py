import sys
import cv2
import os
sys.path.append(r'C:\Users\joeli\Dropbox\Code\MST-plus-plus\predict_code\architecture')

import torch
# import MST_Plus_Plus
import sys
sys.path.append('MST_Plus_Plus.py')
from MST_Plus_Plus import MST_Plus_Plus  # Corrected import statement
import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from pathlib import Path
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

import spectral2rgb

# Define the preprocessing steps
def preprocess_true_rgb_path(true_rgb_path_path, expected_size=(482, 512)):
    # Read the image
    bgr_image = cv2.imread(true_rgb_path_path)
    bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Resize the image if needed
    bgr_image = cv2.resize(bgr_image, expected_size[::-1])  # OpenCV uses width, height

    # Normalize the image
    bgr_image = np.float32(bgr_image) / 255.0
    bgr_image = np.transpose(bgr_image, (2, 0, 1))  # Change data layout to CxHxW
    
    # Create a batch dimension and convert to tensor
    input_tensor = torch.from_numpy(bgr_image).unsqueeze(0)
    
    return input_tensor

# Predict function
def predict_hsi_from_rgb(model, true_rgb_path_path):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Assume the model is on CUDA, move input tensor to the same device
        input_tensor = preprocess_true_rgb_path(true_rgb_path_path).cuda()
        
        # Inference
        output_tensor = model(input_tensor)
        
        # The output is on GPU, move it back to CPU and convert to numpy array
        hsi_output = output_tensor.cpu().numpy().squeeze(0)  # Remove batch dimension
        return hsi_output

def plot_specific_bands(predicted_hsi, true_hsi, true_rgb, band_indices):
    fig, axs = plt.subplots(4, 2, figsize=(4,8))
    wavelengths = np.arange(400, 1001, 10)[:31]
    true_hsi = np.transpose(true_hsi, (2, 0, 1))
    true_rgb = np.float32(true_rgb) 
    #black background
    plt.style.use('dark_background')
    #no axis
    plt.axis('off')
    print(f'predicted_hsi.shape: {predicted_hsi.shape}')
    # Plot RGB image
    axs[0, 0].imshow(true_rgb)
    axs[0, 0].set_title('RGB Image')
    axs[0, 0].axis('off')

    # Plot true band
    axs[1, 0].imshow(true_hsi[0], cmap='gray')
    axs[1, 0].set_title(f'True Band {wavelengths[0]} nm')
    axs[1, 0].axis('off')

    axs[2, 0].imshow(true_hsi[3], cmap='gray')
    axs[2, 0].set_title(f'True Band {wavelengths[3]}')
    axs[2, 0].axis('off')

    axs[3, 0].imshow(true_hsi[8], cmap='gray')
    axs[3, 0].set_title(f'True Band {wavelengths[8]}')
    axs[3, 0].axis('off')

    #recovered rgb
    #resize to -1,31
    predicted_hsi = np.reshape(predicted_hsi, (31, 482, 512))
    #to w,h,c
    predicted_hsi = np.transpose(predicted_hsi, (1,2,0))
    print(f'predicted_hsi.shape: {predicted_hsi.shape}')
    #crop to square
    h,w,c = predicted_hsi.shape
    if h!=w:
        predicted_hsi = predicted_hsi[:min(h,w),:min(h,w),:]
    print(f'predicted_hsi.shape: {predicted_hsi.shape}')
    #resize to -1,31
    # predicted_hsi = np.reshape(predicted_hsi, (-1, 31))
    # print(f'predicted_hsi.shape: {predicted_hsi.shape}')
    rgb_pred = spectral2rgb.Get_RGB(predicted_hsi, wavelengths)
    #normalize 0-1
    rgb_pred = (rgb_pred - np.min(rgb_pred))/(np.max(rgb_pred) - np.min(rgb_pred))
    #gamma correction using opencv
    # rgb_pred = cv2.cvtColor(rgb_pred, cv2.COLOR_RGB2BGR)
    rgb_pred = np.float32(rgb_pred) 

    axs[0, 1].imshow(cv2.cvtColor(rgb_pred, cv2.COLOR_BGR2RGB))
    axs[0, 1].set_title('Recovered RGB')
    axs[0, 1].axis('off')
    # Plot predicted band   

    axs[1, 1].imshow(predicted_hsi[:,:,0], cmap='gray')
    axs[1, 1].set_title(f'Predicted Band {wavelengths[0]} nm')
    axs[1, 1].axis('off')

    axs[2, 1].imshow(predicted_hsi[:,:,3], cmap='gray')
    axs[2, 1].set_title(f'Predicted Band {wavelengths[3]}')
    axs[2, 1].axis('off')

    axs[3, 1].imshow(predicted_hsi[:,:,8], cmap='gray')
    axs[3, 1].set_title(f'Predicted Band {wavelengths[8]}')
    axs[3, 1].axis('off')





    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    model_path = r"C:\Users\joeli\Dropbox\Code\MST-plus-plus\exp\mst_plus_plus\2023_11_21_13_36_52\net_14epoch.pth"
    model_path = r"C:\Users\joeli\Dropbox\Code\MST-plus-plus\exp\mst_plus_plus\2023_11_16_20_46_31\net_14epoch.pth"
    model_path = r"C:\Users\joeli\Dropbox\Code\Python Projects\MST-plus-plus\exp\mst_plus_plus\2023_11_21_13_36_52\net_14epoch.pth"
    # model_path = r"C:\Users\joeli\Dropbox\Code\Python Projects\MST-plus-plus\exp\mst_plus_plus\net_1epoch.pth"
    model = MST_Plus_Plus(in_channels=3, out_channels=31, n_feat=31, stage=3)
    
    # Load the model onto the GPU
    model = model.cuda()  # Assuming you are using a GPU

    # Load the saved model weights
    checkpoint = torch.load(model_path)
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}, strict=True)

    val_path = r'C:\Users\joeli\Dropbox\Code\Python Projects\MST-plus-plus\dataset\Val_Spec'
    # rgb_path = r"C:\Users\joeli\Dropbox\UE5Exports\FaceColor_CM1.PNG"
    rgb_path = r"C:\Users\joeli\Dropbox\Data\models_4k\m53_4k.png"
    # "C:\Users\joeli\Dropbox\Data\models_4k\m32_4k.png"
    rgb_path = r'C:\Users\joeli\Dropbox\Data\models_4k\m32_4k.png'
    rgb_path = r"C:\Users\joeli\Dropbox\Code\Python Projects\MST-plus-plus\dataset\Val_RGB\p021_neutral_front.jpg"
    # rgb_path = r"C:\Users\joeli\Dropbox\Code\MST-plus-plus\dataset\Train_RGB\p021_neutral_front.jpg"
    val_path = Path(val_path)
    rgb_path = Path(rgb_path)
    val_list = os.listdir(val_path) 
    # rgb_list = os.listdir(rgb_path)
    val_list.sort()
    # rgb_list.sort()
    hsi_true = os.path.join(val_path, val_list[0])
    rgb_true = rgb_path
    hsi_true = np.float32(scipy.io.loadmat(hsi_true)['hsi'])
    rgb_true = Image.open(rgb_true).convert('RGB')
    rgb_true = np.array(rgb_true)/255.0
    # rgb_true = cv2.cvtColor(rgb_true, cv2.COLOR_BGR2RGB)

    print(f'first val_list: {val_list[0]}')
    # print(f'first rgb_list: {rgb_list[0]}')
    # Prediction
    predicted_hsi = predict_hsi_from_rgb(model, os.path.join(rgb_path, rgb_path))
    print(predicted_hsi.shape)
    plot_specific_bands(predicted_hsi, hsi_true, rgb_true, [0,4,8])
    rgb_corrected = spectral2rgb.Get_RGB(predicted_hsi[:,:,:31], np.linspace(400, 700, 31))
    #reshape to width, height, channels
    rgb_corrected = np.transpose(rgb_corrected, (1,2,0)).reshape(482,512,3)
    plt.imshow(rgb_corrected)
    plt.show()
    