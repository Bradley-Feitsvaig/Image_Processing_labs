import cv2
import numpy as np


def mse(im1_path, im2_path):
    image1 = cv2.imread(im1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(im2_path, cv2.IMREAD_GRAYSCALE)
    mse = np.mean((image1 - image2) ** 2)  # calc MSE
    image_name = im2_path.split('.')[0]
    print(f"{image_name} MSE is: {mse}")
    return mse


def recreate_image_with_kernel(input_image, kernel, border_type, border_value=128):
    # Calculate the border width from the kernel size
    border_width = kernel.shape[0] // 2

    # Apply a constant border if needed
    if border_type == cv2.BORDER_CONSTANT:
        input_image_with_border = cv2.copyMakeBorder(input_image, border_width, border_width,
                                                     border_width, border_width, border_type, value=(border_value,))
        # Apply convolution
        recreated = cv2.filter2D(input_image_with_border, -1, kernel)
        # Returns the recreated image into original size
        recreated = recreated[border_width:-border_width, border_width:-border_width]
    else: # any other type of border
        # For other border types, apply the kernel using filter2D which handles borders internally
        recreated = cv2.filter2D(input_image, -1, kernel, borderType=border_type)

    return recreated


def save_image_and_calculate_mse(recreated, output_image):
    # save the recreated image and calculate the MSE
    cv2.imwrite(f'recreated_images/{output_image}.jpg', recreated)
    mse(f'recreated_images/{output_image}.jpg', f'{output_image}.jpg')


def recreate_image_mean_value_of_each_row(input_image):
    row_means = np.mean(input_image, axis=1)  # mean value of each row

    # Create an output image where each row has the mean value of the original row
    output_image = np.zeros_like(input_image)
    for i, mean in enumerate(row_means):
        output_image[i, :] = mean

    return output_image


def recreate_image_with_medianBlur(input_image):
    median_filtered_image = cv2.medianBlur(input_image, 11)
    return median_filtered_image


def recreate_image_with_motionBlur(input_image, ksize):
    pad_size = ksize // 6
    kernel_motion_blur = np.zeros((kernel_size, kernel_size))
    kernel_motion_blur[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
    kernel_motion_blur = kernel_motion_blur / kernel_size
    # # Apply padding with a constant value
    padded_image = cv2.copyMakeBorder(input_image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT,
                                      value=(0,))

    # Apply Gaussian blur to the padded image
    recreated = cv2.filter2D(padded_image, -1, kernel_motion_blur)

    # # Correctly crop the image back to original size
    blurred_image = recreated[pad_size:-pad_size, pad_size:-pad_size]

    return blurred_image


def recreate_image_with_bilateralFilter(input_image):
    recreated = cv2.bilateralFilter(input_image, d=15, sigmaColor=150, sigmaSpace=0)
    return recreated


# Load the image in grayscale
image = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)

# image 1
recreated = recreate_image_mean_value_of_each_row(image)
save_image_and_calculate_mse(recreated, 'image_1')

# image 2
kernel_size = 11
kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
recreated = recreate_image_with_kernel(image, kernel, border_type=cv2.BORDER_REFLECT)
save_image_and_calculate_mse(recreated, 'image_2')

# image 3
recreated = recreate_image_with_medianBlur(image)
save_image_and_calculate_mse(recreated, 'image_3')

# image 4
kernel_size = 15
kernel_motion_blur = np.zeros((kernel_size, kernel_size))
kernel_motion_blur[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
kernel_motion_blur = kernel_motion_blur / kernel_size
recreated = recreate_image_with_kernel(image, kernel_motion_blur, border_type=cv2.BORDER_CONSTANT)
save_image_and_calculate_mse(recreated, 'image_4')

# image 5
b_g = cv2.GaussianBlur(image, (21, 21), 0, borderType=cv2.BORDER_WRAP)
B_sharp = image - b_g
B_sharp += 128
save_image_and_calculate_mse(B_sharp, 'image_5')

# image 6
kernel = np.array([[-1],
                   [0],
                   [1]])
recreated = recreate_image_with_kernel(image, kernel, border_type=cv2.BORDER_DEFAULT)
save_image_and_calculate_mse(recreated, 'image_6')

# # image 7
H, W = image.shape
kernel = np.zeros((H + 1, W), dtype=np.float32)
kernel[H, W // 2] = 1
recreated = recreate_image_with_kernel(image, kernel, border_type=cv2.BORDER_WRAP)
save_image_and_calculate_mse(recreated, 'image_7')

# image 8
save_image_and_calculate_mse(image, 'image_8')

# image 9

sharpening_kernel = np.array([[-1, -1, -1],
                              [-1, 20, -1],
                              [-1, -1, -1]]) / 12
recreated = recreate_image_with_kernel(image, sharpening_kernel, border_type=cv2.BORDER_DEFAULT)
recreated = recreate_image_with_kernel(recreated, sharpening_kernel, border_type=cv2.BORDER_DEFAULT)
save_image_and_calculate_mse(recreated, 'image_9')
