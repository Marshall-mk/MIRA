import numpy as np
from scipy.ndimage import affine_transform
from skimage.transform import pyramid_gaussian
from scipy.optimize import minimize
from PIL import Image
from matplotlib import pyplot as plt
from skimage.transform import rescale
from skimage.filters import sobel



def ncc(A, B):
    """
    Computes the Normalized Cross-Correlation (NCC) between two images.
    
    Parameters:
    A, B : ndarray
        Input images (grayscale) as NumPy arrays of the same shape.
        
    Returns:
    R : float
        The Normalized Cross-Correlation value.
    """
    # Ensure A and B are of the same size
    assert A.shape == B.shape, "Images must be of the same dimensions"

    # Calculate means of both images
    I_A_mean = np.mean(A)
    I_B_mean = np.mean(B)

    # Calculate numerator and denominator of the NCC formula
    numerator = np.sum((A - I_A_mean) * (B - I_B_mean))
    denominator = np.sqrt(np.sum((A - I_A_mean)**2) * np.sum((B - I_B_mean)**2))
    
    if denominator == 0 or np.isnan(denominator):
        return 0.0
    ncc = numerator / denominator
    
    return -ncc


def ngc(imgA, imgB):
    """
    Computes the Normalized Gradient Correlation (NGC) between two images using Sobel gradients.

    Parameters:
      imgA, imgB : ndarray
        Input images (grayscale) as NumPy arrays of the same shape.

    Returns:
    ngc_value : float
        The Normalized Gradient Correlation value.
    """
    # Compute gradients using Sobel operator
    imgA_x_grad = sobel(imgA, axis=1)
    imgA_y_grad = sobel(imgA, axis=0)
    imgB_x_grad = sobel(imgB, axis=1)
    imgB_y_grad = sobel(imgB, axis=0)

    # Compute numerator
    numerator = np.sum(imgA_x_grad * imgB_x_grad + imgA_y_grad * imgB_y_grad)

    # Compute denominator
    den1 = np.sum(imgA_x_grad**2 + imgA_y_grad**2)
    den2 = np.sum(imgB_x_grad**2 + imgB_y_grad**2)
    denominator = np.sqrt(den1 * den2)

    # Avoid division by zero
    if denominator == 0 or np.isnan(denominator):
        return 0.0

    ngc_value = numerator / denominator
    
    return -ngc_value

# Define the image pyramid
def create_image_pyramid(image, max_level):
    return list(pyramid_gaussian(image, max_layer=max_level, downscale=2))

# Similarity metric (e.g., Mean Squared Error)
def mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

# Apply affine transformation using parameters [a, b, c, d, tx, ty]
def transform_image_affine(image, params):
    a, b, c, d, tx, ty = params
    affine_matrix = np.array([[a, b, tx],
                              [c, d, ty]])
    transformed_image = affine_transform(image, affine_matrix[:2, :2], offset=affine_matrix[:2, 2], order=1)
    return transformed_image

# Optimize transformation parameters for affine transformation
def optimize_affine(fixed_image, moving_image, initial_params):
    def objective_function(params):
        transformed_image = transform_image_affine(moving_image, params)
        return ncc(fixed_image, transformed_image)

    result = minimize(objective_function, initial_params, method='Powell')
    return result.x  # Optimized affine transformation parameters

# Multi-resolution registration with affine transformation
def multi_resolution_registration_affine(fixed_image, moving_image, max_level=3):
    # Create image pyramids
    fixed_pyramid = create_image_pyramid(fixed_image, max_level)
    moving_pyramid = create_image_pyramid(moving_image, max_level)

    # Start with the coarsest level
    # Initial affine parameters: [a, b, c, d, tx, ty] -> Identity matrix and zero translation
    current_params = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    for level in range(max_level, -1, -1):
        # Get the current level images
        fixed_resized = fixed_pyramid[level]
        moving_resized = moving_pyramid[level]

        # Optimize affine parameters at the current level
        current_params = optimize_affine(fixed_resized, moving_resized, current_params)

        # Scale translation part of parameters for the next (finer) level
        if level > 0:
            current_params[4:] *= 2  # Scale tx, ty

    # Final transformation using the optimized parameters
    registered_image = transform_image_affine(moving_image, current_params)

    return registered_image, current_params
# Example usage
if __name__ == "__main__":
    # Load or create test images (fixed_image and moving_image)
    fixed_image = np.array(Image.open("brain2.png").convert("L"))
    moving_image = np.array(Image.open("brain4.png").convert("L"))

    # Perform multi-resolution registration
    registered_image, final_translation = multi_resolution_registration_affine(fixed_image, moving_image)

    print("Final translation:", final_translation)
    
    # Display the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(fixed_image, cmap='gray')
    plt.title("Fixed Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(moving_image, cmap='gray')
    plt.title("Moving Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(registered_image, cmap='gray')
    plt.title("Registered Image")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
