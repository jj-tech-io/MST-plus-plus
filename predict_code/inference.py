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
from MST_Plus_Plus import MST_Plus_Plus
from MST import MST
import spectral2rgb

def preprocess_rgb_image(rgb_image, expected_size=(512, 512)):


    # Resize the image if needed
    rgb_image = cv2.resize(rgb_image, expected_size[::-1])  # OpenCV uses width, height

    # Normalize the image
    rgb_image = np.float32(rgb_image) / 255.0
    rgb_image = np.transpose(rgb_image, (2, 0, 1))  # Change data layout to CxHxW

    # Create a batch dimension and convert to tensor
    input_tensor = torch.from_numpy(rgb_image).unsqueeze(0)

    return input_tensor

# Predict function
def predict_hsi_from_rgb(model, true_rgb_image_path):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Preprocess the RGB image
        input_tensor = preprocess_rgb_image(true_rgb_image_path).cuda()

        # Inference
        output_tensor = model(input_tensor)

        # The output is on GPU, move it back to CPU and convert to numpy array
        hsi_output = output_tensor.cpu().numpy().squeeze(0)  # Remove batch dimension

    return hsi_output

# Predict function
# def predict_hsi_from_rgb(model, true_rgb_path_path):
#     model.eval()  # Set the model to evaluation mode
#     with torch.no_grad():
#         # Assume the model is on CUDA, move input tensor to the same device
#         input_tensor = preprocess_true_rgb_path(true_rgb_path_path).cuda()
        
#         # Inference
#         output_tensor = model(input_tensor)
        
#         # The output is on GPU, move it back to CPU and convert to numpy array
#         hsi_output = output_tensor.cpu().numpy().squeeze(0)  # Remove batch dimension
#         return hsi_output
def predict_hsi_from_rgb(model, rgb_tensor):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # The input_tensor is assumed to be preprocessed, normalized and on the same device as the model
        # Inference
        output_tensor = model(rgb_tensor)

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
    model_path = r"predict_code\mst_plus_plus.pth"
    # model = MST_Plus_Plus(in_channels=3, out_channels=31, n_feat=31, stage=3)

    model = MST(dim=31, stage=2, num_blocks=[2, 2, 2])
    # model = model.cuda()
    # Load the model onto the GPU
    # model = model.cuda()  # Assuming you are using a GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_path is not None:
        try:
            print(f'load model from {model_path}')
            #print model keys and shapes
            print(model.state_dict().keys())
            checkpoint = torch.load(model_path, strict=False)
            print(checkpoint.keys())
            model.load_state_dict(checkpoint['state_dict'] )
        except:
            print(f'load model from {model_path}')
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            print(checkpoint.keys())
            model.load_state_dict(checkpoint, strict=False)


    model = model.to(device)  # Move model to GPU if available
    model.eval()
    RGB_DIR = r"C:\Users\joeli\Dropbox\Code\Python Projects\MST_JJ\rgb_images.pkl"
    HSI_DIR = r"C:\Users\joeli\Dropbox\Code\Python Projects\MST_JJ\hsi_images.pkl"
    import pickle
    rgb_images = []
    hsi_images = []
    with open(RGB_DIR, 'rb') as f:
        rgb_images = pickle.load(f)
    with open(HSI_DIR, 'rb') as f:
        hsi_images = pickle.load(f)
    print(f'rgb_images: {len(rgb_images)}')
    print(f'hsi_images: {len(hsi_images)}')
    rgb_true = rgb_images[0]
    hsi_true = hsi_images[0]
    if np.max(rgb_true) > 1:
        rgb_true = np.float32(rgb_true) / 255.0

    rgb_tensor = preprocess_rgb_image(rgb_true)
    predicted_hsi = predict_hsi_from_rgb(model, rgb_tensor)
    print(predicted_hsi.shape)
    plot_specific_bands(predicted_hsi, hsi_true, rgb_true, [0,4,8])
    rgb_corrected = spectral2rgb.Get_RGB(predicted_hsi[:,:,:31], np.linspace(400, 700, 31))
    rgb_corrected = spectral2rgb.convert_to_RGB(predicted_hsi[:,:,:31], np.linspace(400, 700, 31))
    from os.path import splitext , join 
    import cv2 as cv
    # Save image file
    fileName = f'{splitext(rgb_path.name)[0]}_realWorld'
    path = join(rgb_path.parent, f'{fileName}.jpg')
    # Display RGB image
    # img = cv2.cvtColor(rgb_corrected, cv2.COLOR_BGR2RGB)
