import cv2
import numpy as np


def mse(im1_path, im2_path):
    image1 = cv2.imread(im1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(im2_path, cv2.IMREAD_GRAYSCALE)
    mse = np.mean((image1 - image2) ** 2)
    image_name = im2_path.split('.')[0]
    print(f"{image_name} MSE is: {mse}")
    return mse


def recreate_image_with_kernel(input_image, kernel, output_image, borderType):
    recreated = cv2.filter2D(input_image, -1, kernel, borderType=borderType)
    cv2.imwrite(f'recreated_images/{output_image}.jpg', recreated)
    mse(f'recreated_images/{output_image}.jpg', f'{output_image}.jpg')


def recreate_image_mean_value_of_each_row(input_image, output_image_path):
    row_means = np.mean(input_image, axis=1)  # mean value of each row

    # Create an output image where each row has the mean value of the original row
    output_image = np.zeros_like(input_image)
    for i, mean in enumerate(row_means):
        output_image[i, :] = mean
    cv2.imwrite(f'recreated_images/{output_image_path}.jpg', output_image)
    mse(f'recreated_images/{output_image_path}.jpg', f'{output_image_path}.jpg')


def recreate_image_with_medianBlur(input_image, output_image_path):
    median_filtered_image = cv2.medianBlur(input_image, 11)
    cv2.imwrite(f'recreated_images/{output_image_path}.jpg', median_filtered_image)
    mse(f'recreated_images/{output_image_path}.jpg', f'{output_image_path}.jpg')


# Load the image in grayscale
image = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)

# image 1
recreate_image_mean_value_of_each_row(image, 'image_1')

# image 2
kernel_size = 11
kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
recreate_image_with_kernel(image, kernel, 'image_2', borderType=cv2.BORDER_REFLECT)

# image 3
recreate_image_with_medianBlur(image, 'image_3')

# image 6
kernel = np.array([[-1],
                    [0],
                    [1]])
recreate_image_with_kernel(image, kernel, 'image_6', borderType=cv2.BORDER_DEFAULT)

# image 7
# Image dimensions
height, width = image.shape[:2]
kernel = np.zeros((height, width))
np.fill_diagonal(kernel, 1)
print(kernel)
recreate_image_with_kernel(image, kernel, 'image_7', borderType=cv2.BORDER_REFLECT)
