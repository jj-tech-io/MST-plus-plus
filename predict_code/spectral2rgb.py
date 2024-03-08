import numpy as np
import scipy.io
from os.path import basename, join, splitext
import numpy as np
import scipy.io



def xFit_1931(wavelengths):
    t1 = np.where(wavelengths < 442.0, (wavelengths - 442.0) * 0.0624, (wavelengths - 442.0) * 0.0374)
    t2 = np.where(wavelengths < 599.8, (wavelengths - 599.8) * 0.0264, (wavelengths - 599.8) * 0.0323)
    t3 = np.where(wavelengths < 501.1, (wavelengths - 501.1) * 0.0490, (wavelengths - 501.1) * 0.0382)
    return 0.362 * np.exp(-0.5 * t1**2) + 1.056 * np.exp(-0.5 * t2**2) - 0.065 * np.exp(-0.5 * t3**2)

def yFit_1931(wavelengths):
    t1 = np.where(wavelengths < 568.8, (wavelengths - 568.8) * 0.0213, (wavelengths - 568.8) * 0.0247)
    t2 = np.where(wavelengths < 530.9, (wavelengths - 530.9) * 0.0613, (wavelengths - 530.9) * 0.0322)
    return 0.821 * np.exp(-0.5 * t1**2) + 0.286 * np.exp(-0.5 * t2**2)

def zFit_1931(wavelengths):
    t1 = np.where(wavelengths < 437.0, (wavelengths - 437.0) * 0.0845, (wavelengths - 437.0) * 0.0278)
    t2 = np.where(wavelengths < 459.0, (wavelengths - 459.0) * 0.0385, (wavelengths - 459.0) * 0.0725)
    return 1.217 * np.exp(-0.5 * t1**2) + 0.681 * np.exp(-0.5 * t2**2)

def gamma_correction(C):
    abs_C = np.abs(C)
    return np.where(abs_C > 0.0031308, 1.055 * np.power(abs_C, 1.0 / 2.4) - 0.055, 12.92 * C)

def XYZ_to_sRGB(xyz_array):
    # Define the transformation matrix from XYZ to sRGB color space
    mat3x3 = np.array([
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570]
    ])

    # Reshape xyz_array if necessary to handle both single XYZ value and 2D array of XYZ values
    original_shape = xyz_array.shape
    if xyz_array.ndim == 3:
        xyz_array = xyz_array.reshape(-1, 3)
    xyz_array /= 10  # Normalize XYZ values to be between 0 and 1
    # Apply the matrix transformation
    rgb_array = np.dot(xyz_array, mat3x3.T)  # Transpose the matrix to align dimensions

    # Apply gamma correction
    rgb_array = gamma_correction(rgb_array)

    # Reshape back to the original image shape with 3 channels for RGB
    if len(original_shape) == 3:
        rgb_array = rgb_array.reshape(original_shape)

    return rgb_array

def Get_RGB(hsi_data, wavelengths):
    # Check if hsi_data is a single pixel or an HSI cube
    if hsi_data.ndim == 1:  # Single pixel
        reflectances = hsi_data[:, np.newaxis]  # Make it 2D for broadcasting
    elif hsi_data.ndim == 3:  # HSI cube
        # Reshape HSI cube into a 2D array where each row is a pixel
        height, width, num_bands = hsi_data.shape
        reflectances = hsi_data.reshape((-1, num_bands))
    else:
        raise ValueError("hsi_data must be either a 1D or 3D array")

    # Check wavelengths length matches HSI data bands
    num_bands = reflectances.shape[1]
    if len(wavelengths) != num_bands:
        s = f"len(wavelengths) = {len(wavelengths)} and num_bands = {num_bands})"
        wavelengths = wavelengths[:num_bands]

    # Calculate the XYZ values for each wavelength and sum them
    x = xFit_1931(wavelengths) * reflectances
    y = yFit_1931(wavelengths)* reflectances
    z = zFit_1931(wavelengths) * reflectances

    total_XYZ = np.stack((x.sum(axis=1), y.sum(axis=1), z.sum(axis=1)), axis=-1)/10

    # Convert XYZ to sRGB
    sRGB = XYZ_to_sRGB(total_XYZ)

    # If it was a cube, reshape the result back to the original cube dimensions with 3 channels for RGB
    if hsi_data.ndim == 3:
        sRGB = sRGB.reshape((height, width, 3))

    return sRGB

def convert_to_RGB(hsi_cube, wavelengths):
    print(f"hsi_cube.shape: {hsi_cube.shape}")
    print(f"wavelengths.shape: {wavelengths.shape}")
    # Ensure wavelengths is a numpy array for broadcasting
    wavelengths = np.array(wavelengths)
    if len(wavelengths) > hsi_cube.shape[2]:
        hsi_cube = hsi_cube[:, :, :len(wavelengths)]
    #-1,31  
    print(f"hsi_cube.shape: {hsi_cube.shape}")
    hsi_cube = np.reshape(hsi_cube, (-1, hsi_cube.shape[2]))
    # Assuming xFit_1931, yFit_1931, and zFit_1931 are functions that can operate over an array of wavelengths
    x = xFit_1931(wavelengths)* hsi_cube  # The shape of x will be (524, 524, 61)
    y = yFit_1931(wavelengths)* hsi_cube
    z = zFit_1931(wavelengths)* hsi_cube

    # Sum over the wavelength dimension to get the total XYZ for each pixel
    total_XYZ = np.stack((x.sum(axis=2), y.sum(axis=2), z.sum(axis=2)), axis=-1)
    
    # Convert the XYZ for each pixel to RGB using the vectorized XYZ_to_sRGB function
    sRGB = XYZ_to_sRGB(total_XYZ)  # This function needs to be vectorized to operate on each pixel's XYZ
    if sRGB.max() < 1:
        sRGB = sRGB * 255
    if sRGB.min() > 255:
        sRGB = sRGB / 255
    # Ensure RGB values are within the 0-255 range and of type uint8
    # sRGB = np.clip(sRGB, 0, 255).astype(np.uint8)
    
    return sRGB


