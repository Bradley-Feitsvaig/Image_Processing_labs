
import cv2
import numpy as np
import matplotlib.pyplot as plt


def clean_Gaussian_noise_bilateral(im, radius, stdSpatial, stdIntensity):
    # Your code goes here
    # Convert the image to float64 for precise calculations
    im = im.astype(np.float64)

    # Pad the image to avoid border effects
    padded_im = cv2.copyMakeBorder(im, radius, radius, radius, radius, cv2.BORDER_REFLECT)

    # Adjusted image dimensions to account for padding
    padded_height, padded_width = padded_im.shape

    # Pre-compute the spatial Gaussian function (gs)
    y, x = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
    gs = np.exp(-(x ** 2 + y ** 2) / (2 * stdSpatial ** 2))

    # Initialize the output image with padding
    cleanIm_padded = np.zeros_like(padded_im)

    # Iterate over each pixel in the padded image
    for i in range(radius, padded_height - radius):
        for j in range(radius, padded_width - radius):
            # Extract the local region (window) with padding
            window = padded_im[i - radius:i + radius + 1, j - radius:j + radius + 1]

            # Calculate the intensity Gaussian function (gi)
            gi = np.exp(-((window - padded_im[i, j]) ** 2) / (2 * stdIntensity ** 2))

            # Calculate the final mask
            mask = gi * gs
            mask /= mask.sum()

            # Apply the mask
            cleanIm_padded[i, j] = (mask * window).sum()

    # Crop the padded output image to original size
    cleanIm = cleanIm_padded[radius:-radius, radius:-radius]

    # Convert back to uint8
    cleanIm = np.clip(cleanIm, 0, 255).astype(np.uint8)

    return cleanIm


# change this to the name of the image you'll try to clean up
original_image_path = 'NoisyGrayImage.png'
image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

clear_image_b = clean_Gaussian_noise_bilateral(image, 7, 50, 100)

# Save the cleaned image
clean_image_path = 'clean_NoisyGrayImage.png'
cv2.imwrite(clean_image_path, clear_image_b)

plt.subplot(121)
plt.imshow(image, cmap='gray')

plt.subplot(122)
plt.imshow(clear_image_b, cmap='gray')

plt.show()
