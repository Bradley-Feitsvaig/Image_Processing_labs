import cv2
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter


def mse(im1_path, im2_path):
    # Calculate MSE between 2 images
    image1 = cv2.imread(im1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(im2_path, cv2.IMREAD_GRAYSCALE)
    mse = np.mean((image1 - image2) ** 2)  # calc MSE
    image_name = im2_path.split('.')[0]
    print(f"{image_name} MSE is: {mse}")
    return mse


def save_image_and_calculate_mse(recreated, output_image):
    # Save the recreated image and calculate the MSE
    cv2.imwrite(f'recreated_images/{output_image}.jpg', recreated)
    mse(f'recreated_images/{output_image}.jpg', f'{output_image}.jpg')


if __name__ == '__main__':
    # Load the image in grayscale
    image = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)

    # image 1
    # Kernel will give to each pixel the mean of pixel's row
    height, width = image.shape
    kernel = np.ones((1, width), dtype=np.float32) / width
    recreated = signal.convolve2d(image, kernel, boundary='wrap', mode='same')
    save_image_and_calculate_mse(recreated, 'image_1')

    # image 2
    # Blurring kernel (averaging kernel)
    kernel_size = 11
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
    recreated = signal.convolve2d(image, kernel, boundary='wrap', mode='same')
    save_image_and_calculate_mse(recreated, 'image_2')

    # image 3
    recreated = cv2.medianBlur(image, 11)
    save_image_and_calculate_mse(recreated, 'image_3')

    # image 4
    kernel_size = 15
    kernel_motion_blur = np.zeros((kernel_size, kernel_size))
    kernel_motion_blur[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
    kernel_motion_blur = kernel_motion_blur / kernel_size
    recreated = signal.convolve2d(image, kernel_motion_blur, boundary='wrap', mode='same')
    save_image_and_calculate_mse(recreated, 'image_4')

    # image 5
    blurred_img = gaussian_filter(image, sigma=4, radius=9, mode='wrap')
    B_sharp = image - blurred_img
    B_sharp += 127
    save_image_and_calculate_mse(B_sharp, 'image_5')

    # image 6
    kernel = np.array([[1], [0], [-1]])
    recreated = signal.convolve2d(image, kernel, mode='same', boundary='wrap')
    save_image_and_calculate_mse(recreated, 'image_6')

    # # image 7
    H, W = image.shape
    kernel = np.zeros((H + 1, W), dtype=np.float32)
    kernel[H, W // 2] = 1
    recreated = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_WRAP)
    save_image_and_calculate_mse(recreated, 'image_7')

    # image 8
    save_image_and_calculate_mse(image, 'image_8')

    # image 9
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1, 20, -1],
                                  [-1, -1, -1]]) / 12
    recreated = signal.convolve2d(image, sharpening_kernel, mode='same', boundary='wrap')
    recreated = signal.convolve2d(recreated, sharpening_kernel, mode='same', boundary='wrap')
    save_image_and_calculate_mse(recreated, 'image_9')
