import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from numpy.lib.stride_tricks import sliding_window_view
import warnings
warnings.filterwarnings("ignore")


def non_max_suppression(scores, threshold, dist):
    """
    scores: 2D array of scores
    threshold: Score threshold for considering a match
    dist: Minimum distance between maxima

    return: Filtered coordinates of matches
    """
    # Apply threshold
    scores = np.where(scores >= threshold, scores, 0)

    # Find coordinates of scores above threshold
    coords = np.argwhere(scores > 0)
    if not len(coords):
        return np.array([], dtype=int).reshape(0, 2)  # Return empty array if no matches

    # Sort coordinates based on scores
    idx = np.argsort(scores[coords[:, 0], coords[:, 1]])[::-1]
    coords = coords[idx]

    selected = []
    while len(coords):
        # Add the coordinate with the highest score to the selected list
        selected.append(coords[0])
        if len(coords) == 1:
            break

        # Calculate distances to the rest of the points
        distances = np.sqrt(np.sum((coords[1:] - coords[0]) ** 2, axis=1))

        # Keep only coordinates that are further than `dist` apart
        coords = coords[1:][distances > dist]

    return np.array(selected)


def scale_down(image, resize_ratio):
    # Fourier transform
    f_transform = fft2(image)
    f_transform_shifted = fftshift(f_transform)

    # Calculating the cropping boundaries
    rows, cols = f_transform_shifted.shape
    crow, ccol = rows // 2, cols // 2
    row_min, row_max = int(crow - crow * resize_ratio), int(crow + crow * resize_ratio)
    col_min, col_max = int(ccol - ccol * resize_ratio), int(ccol + ccol * resize_ratio)

    # Cropping and creating a new shifted transform
    f_transform_shifted_cropped = f_transform_shifted[row_min:row_max, col_min:col_max]

    # Inverse Fourier Transform
    f_ishift = ifftshift(f_transform_shifted_cropped)
    img_back = ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back


def scale_up(image, resize_ratio):
    # Fourier transform
    f_transform = fft2(image)
    f_transform_shifted = fftshift(f_transform)

    # Calculating the padded boundaries
    rows, cols = f_transform_shifted.shape
    crow, ccol = rows // 2, cols // 2
    row_min, row_max = int(crow - crow * resize_ratio), int(crow + crow * resize_ratio)
    col_min, col_max = int(ccol - ccol * resize_ratio), int(ccol + ccol * resize_ratio)

    # Scale up with zero padding
    f_padded = np.pad(f_transform_shifted, ((row_min, row_max), (col_min, col_max)), mode='constant',
                      constant_values=0)

    # Inverse Fourier Transform
    f_ishift = ifftshift(f_padded)
    img_back = ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back


def ncc_2d(image, pattern):
    # all possible windows of the same size as the pattern
    windows = sliding_window_view(image, pattern.shape)

    # Subtract mean from pattern and windows
    pattern = pattern - np.mean(pattern)
    windows = windows - np.mean(windows, axis=(2, 3), keepdims=True)

    # Compute NCC
    numerator = np.tensordot(windows, pattern, axes=([2, 3], [0, 1]))
    denominator = np.sqrt(np.sum(windows ** 2, axis=(2, 3)) * np.sum(pattern ** 2))

    # Avoid division by zero
    denominator[denominator == 0] = 1

    # Calculate NCC scores
    ncc_scores = numerator / denominator

    # The resulting NCC map will be smaller than the original image because the edges are not considered.
    # Pad the result, so it will be in the same size as the input image.
    pad_height = (image.shape[0] - ncc_scores.shape[0]) // 2
    pad_width = (image.shape[1] - ncc_scores.shape[1]) // 2
    ncc_padded = np.pad(ncc_scores, ((pad_height, pad_height), (pad_width, pad_width)), 'constant', constant_values=0)

    return ncc_padded


def display(image, pattern):
    plt.subplot(2, 3, 1)
    plt.title('Image')
    plt.imshow(image, cmap='gray')

    plt.subplot(2, 3, 3)
    plt.title('Pattern')
    plt.imshow(pattern, cmap='gray', aspect='equal')

    ncc = ncc_2d(image, pattern)

    plt.subplot(2, 3, 5)
    plt.title('Normalized Cross-Correlation Heatmap')
    plt.imshow(ncc ** 2, cmap='coolwarm', vmin=0, vmax=1, aspect='auto')

    cbar = plt.colorbar()
    cbar.set_label('NCC Values')

    plt.show()


def draw_matches(image, matches, pattern_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for point in matches:
        y, x = point
        top_left = (int(x - pattern_size[1]), int(y - pattern_size[0]))
        bottom_right = (int(x + pattern_size[1] / 2), int(y + pattern_size[0] / 2))
        cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 1)

    plt.imshow(image, cmap='gray')
    plt.show()

    cv2.imwrite(f"{CURR_IMAGE}_result.jpg", image)


CURR_IMAGE = "students"

image = cv2.imread(f'{CURR_IMAGE}.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

pattern = cv2.imread('template.jpg')
pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)

############ DEMO #############
display(image, pattern)

############# Students #############
image_scaled = image
patten_scaled = scale_down(pattern, 0.5)

display(image_scaled, patten_scaled)

ncc = ncc_2d(image_scaled, patten_scaled)
real_matches = non_max_suppression(ncc, 0.55, 10)

######### DONT CHANGE THE NEXT TWO LINES #########
real_matches[:, 0] += patten_scaled.shape[0] // 2  # if pattern was not scaled, replace this with "pattern"
real_matches[:, 1] += patten_scaled.shape[1] // 2  # if pattern was not scaled, replace this with "pattern"

# If you chose to scale the original image, make sure to scale back the matches in the inverse resize ratio.

draw_matches(image, real_matches, patten_scaled.shape)  # if pattern was not scaled, replace this with "pattern"

############# Crew #############
CURR_IMAGE = "thecrew"

image = cv2.imread(f'{CURR_IMAGE}.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


image_scaled = image
patten_scaled = scale_down(pattern, 0.23)

display(image_scaled, patten_scaled)

ncc = ncc_2d(image_scaled, patten_scaled)
real_matches = non_max_suppression(ncc, 0.45, 10)

######### DONT CHANGE THE NEXT TWO LINES #########
real_matches[:, 0] += patten_scaled.shape[0] // 2  # if pattern was not scaled, replace this with "pattern"
real_matches[:, 1] += patten_scaled.shape[1] // 2  # if pattern was not scaled, replace this with "pattern"

# If you chose to scale the original image, make sure to scale back the matches in the inverse resize ratio.

draw_matches(image, real_matches, patten_scaled.shape)  # if pattern was not scaled, replace this with "pattern"

