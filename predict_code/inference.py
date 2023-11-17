import sys
import cv2
sys.path.append(r'C:\Users\joeli\Dropbox\Code\MST-plus-plus\predict_code\architecture')

import torch
from MST_Plus_Plus import MST_Plus_Plus  # Corrected import statement
import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from pathlib import Path
import scipy.io
import matplotlib.pyplot as plt
import numpy as np

# Define the preprocessing steps
def preprocess_rgb_image(rgb_image_path, expected_size=(482, 512)):
    # Read the image
    bgr_image = cv2.imread(rgb_image_path)
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
def predict_hsi_from_rgb(model, rgb_image_path):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Assume the model is on CUDA, move input tensor to the same device
        input_tensor = preprocess_rgb_image(rgb_image_path).cuda()
        
        # Inference
        output_tensor = model(input_tensor)
        
        # The output is on GPU, move it back to CPU and convert to numpy array
        hsi_output = output_tensor.cpu().numpy().squeeze(0)  # Remove batch dimension
        return hsi_output

def plot_bands(predicted_hsi, true_hsi, num_bands_to_plot=10):
    # Ensure the predicted and true HSI data have the same shape
    assert predicted_hsi.shape == true_hsi.shape

    # Number of rows and columns for subplot
    num_rows = num_bands_to_plot
    num_cols = 2  # Two columns: one for predicted, one for true

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 2 * num_rows))

    # Select a subset of bands to plot
    band_indices = np.linspace(0, predicted_hsi.shape[0] - 1, num_bands_to_plot, dtype=int)

    for i, band_idx in enumerate(band_indices):
        # Plot predicted band
        axs[i, 0].imshow(predicted_hsi[band_idx], cmap='gray')
        axs[i, 0].set_title(f'Predicted Band {band_idx}')
        axs[i, 0].axis('off')

        # Plot true band
        axs[i, 1].imshow(true_hsi[band_idx], cmap='gray')
        axs[i, 1].set_title(f'True Band {band_idx}')
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    model_path = r'C:\Users\joeli\Dropbox\Code\MST-plus-plus\exp\mst_plus_plus\2023_11_16_17_10_43\net_10epoch.pth'
    model = MST_Plus_Plus(in_channels=3, out_channels=31, n_feat=31, stage=3)
    
    # Load the model onto the GPU
    model = model.cuda()  # Assuming you are using a GPU

    # Load the saved model weights
    checkpoint = torch.load(model_path)
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}, strict=True)

    # Prediction
    predicted_hsi = predict_hsi_from_rgb(model, r"C:\Users\joeli\Dropbox\Code\MST-plus-plus\dataset\Val_RGB\p023_smile_front.jpg")
    print(predicted_hsi.shape)
    plot_bands(predicted_hsi, true_hsi, num_bands_to_plot=10)