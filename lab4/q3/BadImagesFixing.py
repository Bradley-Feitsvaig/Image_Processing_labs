# Student_Name1, Student_ID1
# Student_Name2, Student_ID2

# Please replace the above comments with your names and ID numbers in the same format.

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import matplotlib.pyplot as plt


def clean_image_in_freq_domain(im, im_name, filter):
    # Compute the 2D Fourier transform
    f_transform = fft2(im)

    # Shift the zero freq component to center
    f_shift = fftshift(f_transform)

    # Visualize the image in Fourier Spectrum and save it
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    magnitude_image = np.array(magnitude_spectrum)
    cv2.imwrite(f'{im_name}_Fourier_Spectrum.png', magnitude_image)

    # Apply the local frequency filter to the shifted Fourier transform
    f_filtered = f_shift * filter

    # Inverse Fourier transform the basis function
    f_ishift = ifftshift(f_filtered)
    img_back = ifft2(f_ishift)
    img_back = np.real(img_back)

    # Normalize the image back into the range [0, 255]
    img_filtered = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)

    return np.uint8(img_filtered)


def clean_baby(im):
    # points in order counterclockwise from the top left corner (for PerspectiveTransform)
    points1 = np.array([[181, 5], [121, 51], [177, 120], [249, 70]], dtype='float32')  # upper right image
    points2 = np.array([[78, 163], [132, 244], [245, 160], [145, 117]], dtype='float32')  # bottom right image
    points3 = np.array([[6, 20], [6, 130], [111, 130], [111, 20]], dtype='float32')  # left image

    # Destination points, needed full size image 256X256
    dst_pts = np.array([[0, 0], [0, 255], [255, 255], [255, 0]], dtype='float32')  # full size image

    # List to hold the corrected images
    corrected_images = []

    height, width = im.shape[:2]

    # Apply median blur to remove salt paper noise
    im = cv2.medianBlur(im, 3)

    for points in [points1, points2, points3]:
        # Calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(points, dst_pts)

        # Apply the perspective transform
        corrected_img = cv2.warpPerspective(im, M, (width, height))

        # Add the corrected image to the list
        corrected_images.append(corrected_img)

    # Compute the average image
    avg_img = np.mean(corrected_images, axis=0).astype(np.uint8)
    return avg_img


def clean_windmill(im):
    # Build frequency filter
    rows, cols = im.shape
    frequency_filter = np.ones((rows, cols), np.uint8)
    frequency_filter[132][156] = 0
    frequency_filter[124][100] = 0
    clean_image = clean_image_in_freq_domain(im, 'windmill', frequency_filter)
    return clean_image


def clean_watermelon(im):
    rows, cols = im.shape  # image shape
    crow, ccol = rows // 2, cols // 2  # Center col and row

    # Create a Gaussian High Pass Filter
    D0 = 10  # D0 cutoff frequency
    H = np.zeros((rows, cols), np.float32)
    for u in range(rows):
        for v in range(cols):
            Duv = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)
            H[u, v] = 1 - np.exp(-(Duv ** 2) / (2 * (D0 ** 2)))
    # Apply the High Frequency Emphasis
    alpha = 1.5  # weight for the high frequencies
    beta = 1.0  # weight for the original frequencies
    HFE = alpha + beta * H
    sharpened_image = clean_image_in_freq_domain(im, 'watermelon', HFE)
    return sharpened_image


def clean_umbrella(im):
    H = np.zeros((5, 80))
    H[0, 0], H[-1, -1] = 0.5, 0.5
    H = fft2(H, im.shape)
    H += 1e-8
    H = 1 / H
    fixed_image = clean_image_in_freq_domain(im, 'umbrella', H)
    return fixed_image

# def clean_USAflag(im):
# 	# Your code goes here
#
# def clean_house(im):
# 	# Your code goes here
#
# def clean_bears(im):
# 	# Your code goes here
