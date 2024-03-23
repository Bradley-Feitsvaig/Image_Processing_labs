import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter


def clean_image_in_freq_domain(im, im_name, filter):
    # Compute the 2D Fourier transform
    f_transform = fft2(im)

    # Shift the zero freq component to center
    f_shift = fftshift(f_transform)

    # # Visualize the image in Fourier Spectrum and save it
    # magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    # magnitude_image = np.array(magnitude_spectrum)
    # cv2.imwrite(f'{im_name}_Fourier_Spectrum.png', magnitude_image)

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
    # Build Local Frequency Reject filter to remove frequency noise
    rows, cols = im.shape
    frequency_filter = np.ones((rows, cols), np.uint8)

    # in those indices there is the added frequency noise
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
    beta = 1.0  # weight for the low frequencies
    HFE = alpha + beta * H
    sharpened_image = clean_image_in_freq_domain(im, 'watermelon', HFE)
    return sharpened_image


def clean_umbrella(im):
    # Build filter

    # Initialize a 5x80 filter to zeros (matches the artifact's spacing)
    filter = np.zeros((5, 80))
    # Set the first and the last elements of the filter to 0.5 (for the delta of the original image and the shift image)
    filter[0, 0], filter[-1, -1] = 0.5, 0.5

    # Perform a Fourier transform on the filter and extend it to the size of the image
    mask_f = fft2(filter, im.shape)

    # Compute the Fourier transform of the input image to analyze it in the frequency domain
    im_fourier = fft2(im)

    # Establish a threshold to identify significant frequencies
    threshold = 1e-5

    # Create a boolean mask where the absolute value of the filter's FFT is greater than the threshold
    valid_mask = np.abs(mask_f) > threshold

    # Apply the boolean mask to the Fourier transform of the image
    im_fourier_masked = valid_mask * im_fourier

    # Sanitize the filter by replacing insignificant frequencies with a fixed threshold
    # This prevents division by values that are too close to zero
    mask_f_sanitized = np.where(valid_mask, mask_f, threshold)

    # Divide the masked Fourier transform of the image by the sanitized filter
    # This is Removing the effects of the unwanted frequencies
    clean_im_f = im_fourier_masked / mask_f_sanitized

    # Perform an inverse FFT to convert the processed image back to the spatial domain
    clean_im = np.real(ifft2(clean_im_f))

    return clean_im


def clean_USAflag(im):
    # Coordinates of stars part of the flag
    exclusion_coords = (0, 140, 0, 90)

    # Save copy of the image to bring back the stars part without median filter
    original_image_array = np.copy(im)

    # Apply median filter on the entire flag
    filtered_image_array = median_filter(im, size=(1, 10))

    # Return back the stars part to the filtered image
    filtered_image_array[exclusion_coords[2]:exclusion_coords[3], exclusion_coords[0]:exclusion_coords[1]] = \
        original_image_array[exclusion_coords[2]:exclusion_coords[3], exclusion_coords[0]:exclusion_coords[1]]

    return filtered_image_array


def clean_house(im):
    # Build horizontal blurring (averaging) filter in size of 10
    kernel = np.ones((1, 10), dtype=float) * 0.1

    # Apply the Fourier transform to the image
    transformed_image = np.fft.fft2(im)

    # Apply the Fourier transform to the kernel and adjust its size to match the image
    transformed_kernel = np.fft.fft2(kernel, s=im.shape)

    # Perform division in the frequency domain for filtering
    filtered_image_freq = transformed_image / transformed_kernel

    # Inverse Fourier transform to convert back to the spatial domain
    filtered_image = np.abs(np.fft.ifft2(filtered_image_freq))

    # Clip filtered_image
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    # apply low pass filter to remove frequency noise after the fixing
    cutoff_frequency = 70
    rows, cols = filtered_image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.uint8)
    mask[crow - cutoff_frequency:crow + cutoff_frequency, ccol - cutoff_frequency:ccol + cutoff_frequency] = 1
    cleaned_image = clean_image_in_freq_domain(filtered_image, 'house-low', mask)

    return cleaned_image


def clean_bears(im):
    rows, cols = im.shape
    crow, ccol = rows // 2, cols // 2

    # Initialize the mask with the scaling factor for brightening
    scaling_factor = 2
    mask = np.ones((rows, cols), np.float32) * scaling_factor

    # Adjust the mask for contrast enhancement: amplify high frequencies
    for u in range(rows):
        for v in range(cols):
            distance = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)
            if 60 < distance:
                mask[u, v] *= 2  # Amplify these frequencies

    cleaned_image = clean_image_in_freq_domain(im, 'bears', mask)
    return cleaned_image
