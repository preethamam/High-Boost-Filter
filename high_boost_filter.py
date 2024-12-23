import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve


def high_frequency_boost_filter(
    image,
    use_implementation_1=False,
    scale_factor=1.5,
    hbf_central_value=5.15,
    shbf_central_value=9.15,
    kernel_type="SHBF",
):
    """
    Apply one of two filtering methods to an image.

    Parameters:
        image (numpy.ndarray): Input grayscale or color image.
        use_implementation_1 (bool): Flag to select the implementation. Default is True.
        scale_factor (float): Scale factor for Implementation 1. Default is 1.5.
        hbf_central_value (float): Central value for the High Boost Filter with central value=4 and A=1.15. Default is 5.15.
        shbf_central_value (float): Central value for the HBF with Central value=8 and A=1.15. Default is 9.15.
        kernel_type (str): Type of kernel to use. Default is 'SHBF'. 'SHBF' --> Improves intensities or 'HBF' -> Slighlty dull intensities.
    Returns:
        numpy.ndarray: Filtered image.
    """
    # Ensure the image is a numpy array
    image = np.asarray(image, dtype=np.float32)

    if use_implementation_1:
        # Implementation 1
        # Acknowledgement: This implementation is inspired by the following source (Image Analyst's answer):
        # https://www.mathworks.com/matlabcentral/answers/125062-what-is-the-code-for-high-boost-filter
        laplacian_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        delta_function = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        kernel = laplacian_kernel + scale_factor * delta_function
        kernel = kernel / np.sum(kernel)  # Normalize kernel

        # Apply convolution (handles both grayscale and color images)
        if len(image.shape) == 3:  # Color image
            filtered_image = convolve(image, kernel[:, :, None], mode="reflect")
        else:  # Grayscale image
            filtered_image = convolve(image, kernel, mode="reflect")

    else:
        # Implementation 2
        # Acknowledgement: This implementation is inspired by the following source:
        # https://www.geeksforgeeks.org/image-sharpening-using-laplacian-filter-and-high-boost-filtering-in-matlab/
        # Define the High Boost Filter with central value=4 and A=1.
        HBF = np.array([[0, -1, 0], [-1, hbf_central_value, -1], [0, -1, 0]])

        # Define the HBF with Central value=8 and A=1.
        SHBF = np.array([[-1, -1, -1], [-1, shbf_central_value, -1], [-1, -1, -1]])

        # Apply High Boost Filters sequentially (handles both grayscale and color images)
        if len(image.shape) == 3:  # Color image
            if kernel_type == "HBF":
                filtered_image = convolve(image, HBF[:, :, None], mode="reflect")
            else:
                filtered_image = convolve(image, SHBF[:, :, None], mode="reflect")
        else:  # Grayscale image
            if kernel_type == "HBF":
                filtered_image = convolve(image, HBF, mode="reflect")
            else:
                filtered_image = convolve(image, SHBF, mode="reflect")

    # Clip to valid image range and convert back to uint8
    high_boost_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    return high_boost_image


def main():
    # Example usage:
    image_path = r"image/path/"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load a color image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for processing
    s1 = time.time()
    filtered = high_frequency_boost_filter(
        image, use_implementation_1=False, kernel_type="SHBF"
    )
    s2 = time.time()
    print(f"High boost filter time taken: {s2-s1}")

    # Display the original and filtered images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Filtered Image")
    plt.imshow(filtered)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
