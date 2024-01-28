# Fares Fares, 311136287
# Bradley Feitsvaig, 311183073


import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import warnings

warnings.filterwarnings("ignore")


# Input: numpy array of images and number of gray levels to quantize the images down to
# Output: numpy array of images, each with only n_colors gray levels
def quantization(imgs_arr, n_colors=4):
    img_size = imgs_arr[0].shape
    res = []

    for img in imgs_arr:
        X = img.reshape(img_size[0] * img_size[1], 1)
        km = KMeans(n_clusters=n_colors)
        km.fit(X)

        img_compressed = km.cluster_centers_[km.labels_]
        img_compressed = np.clip(img_compressed.astype('uint8'), 0, 255)

        res.append(img_compressed.reshape(img_size[0], img_size[1]))

    return np.array(res)


# Input: A path to a folder and formats of images to read
# Output: numpy array of grayscale versions of images read from input folder, and also a list of their names
def read_dir(folder, formats=(".jpg", ".png")):
    image_arrays = []
    lst = [file for file in os.listdir(folder) if file.endswith(formats)]
    for filename in lst:
        file_path = os.path.join(folder, filename)
        image = cv2.imread(file_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_arrays.append(gray_image)
    return np.array(image_arrays), lst


# Input: image object (as numpy array) and the index of the wanted bin (between 0 to 9)
# Output: the height of the idx bin in pixels
def get_bar_height(image, idx):
    # Assuming the image is of the same pixel proportions as images supplied in this exercise, the following values
    # will work
    x_pos = 70 + 40 * idx
    y_pos = 274
    while image[y_pos, x_pos] == 0:
        y_pos -= 1
    return 274 - y_pos


# Input:  source image: grade histogram image, target: image of a digit
# implement the histogram-based pattern matching functionality using the EMD between
# histograms to compare a window to the target.
# Output: whether a region was found with EMD < 260
def compare_hist(src_image, target):
    height, width = target.shape
    # calculate the target’s cumulative histogram
    target_histogram = cv2.calcHist([target], [0], None, [256], [0, 256]).flatten()
    cumulative_target_histogram = np.cumsum(target_histogram)
    windows = np.lib.stride_tricks.sliding_window_view(src_image, (height, width))
    # search the region around the topmost number
    for x in range(25, 56):
        for y in range(100, 141):
            # calculate the window’s cumulative histogram
            window_histogram = cv2.calcHist([windows[y, x]], [0], None, [256], [0, 256]).flatten()
            cumulative_window_histogram = np.cumsum(window_histogram)
            # calculate EMD
            emd = np.sum(np.abs(cumulative_window_histogram - cumulative_target_histogram))
            if emd < 260:
                return True
    return False


# Input:  source image: grade histogram image, numbers: np array of number files
# Calling compare_hist between each number's image in reverse with the source image.
# Output: the topmost_number of the histogram in the src image.
def recognize_topmost_number(src_image, numbers_arr):
    for i in reversed(range(len(numbers_arr))):
        if compare_hist(src_image, numbers_arr[i]):
            return i
    return -1

# Read data
images, names = read_dir('data')
numbers, _ = read_dir('numbers')

# display the first image
# cv2.imshow(names[0], images[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Array of topmost number (highest bin) for each image
topmost_numbers = [recognize_topmost_number(image, numbers) for image in images]

# Quantize all images to 4 levels of gray.
quantized_images = quantization(images, n_colors=4)

# Threshold the quantized image to black & white
threshold_value = 230
threshold_images = [np.where(image < threshold_value, 0, 1) for image in quantized_images]

# Transcribe all images
for id in range(len(images)):
    # Array of bin heights in the image.
    bar_heights_list = [get_bar_height(threshold_images[id], bin_i) for bin_i in range(10)]
    # Max bin height
    max_bin_height = max(bar_heights_list)
    # Transcribe histogram in the image.
    transcribed_histogram = [round(topmost_numbers[id] * bin_height / max_bin_height) for bin_height in bar_heights_list]
    heights = ",".join(map(str, transcribed_histogram))
    print(f'Histogram {names[id]} gave {heights}')

exit()
