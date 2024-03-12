# Student_Name1, Student_ID1
# Student_Name2, Student_ID2

# Please replace the above comments with your names and ID numbers in the same format.

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import matplotlib.pyplot as plt


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
    # Apply Fourier Transform to the image
    f_transform = fft2(im)
    f_shift = fftshift(f_transform)

    # Visualize the Fourier Transform (magnitude spectrum)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    magnitude_image = np.array(magnitude_spectrum, dtype=np.uint8)
    cv2.imwrite('magnitude_spectrum.png', magnitude_image)

    # Masking: Create a notch filter or use a band-reject filter to remove the noise spikes
    rows, cols = im.shape

    # This is where you define the size and position of the notch filter
    # You have to find the coordinates in the magnitude spectrum where the noise is located
    notch_filter = np.ones((rows, cols), np.uint8)
    notch_filter[132][156] = 0
    notch_filter[124][100] = 0

    # Apply the notch filter to the shifted Fourier transform
    f_filtered = f_shift * notch_filter

    # Inverse shift and inverse Fourier transform to get the image back in spatial domain
    f_ishift = ifftshift(f_filtered)
    img_back = ifft2(f_ishift)

    # Take the real part of the inverse transform
    img_back = np.real(img_back)

    # Normalize the image back into the range [0, 255]
    img_filtered = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)

    return np.uint8(img_filtered)

# def clean_watermelon(im):
# 	# Your code goes here
#
# def clean_umbrella(im):
# 	# Your code goes here
#
# def clean_USAflag(im):
# 	# Your code goes here
#
# def clean_house(im):
# 	# Your code goes here
#
# def clean_bears(im):
# 	# Your code goes here
