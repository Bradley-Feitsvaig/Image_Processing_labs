import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_laplacian_pyramid(image, levels, resize_ratio=0.5):
    image = image.astype(np.float32)

    # First level is the original image
    gaussian_pyramid = [image]

    # Build Gaussian Pyramid
    for i in range(1, levels):
        # Gaussian blur
        blurred = cv2.GaussianBlur(gaussian_pyramid[-1], (5, 5), 0)
        # scale down
        scaled_down = cv2.resize(blurred, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_LINEAR)
        # Add to pyramid
        gaussian_pyramid.append(scaled_down)

    # Build Laplacian Pyramid
    laplacian_pyramid = []
    for i in range(levels - 1):
        # Calculate scaled up size
        scaled_up_size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])

        # Scale up
        scaled_up = cv2.resize(gaussian_pyramid[i + 1], scaled_up_size, interpolation=cv2.INTER_LINEAR)

        # Substructure expanded image from current level in the pyramid
        laplacian = cv2.subtract(gaussian_pyramid[i], scaled_up)
        laplacian_pyramid.append(laplacian)

    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid


def restore_from_pyramid(pyramid, resize_ratio=2):
    current_image = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        # Calculate the target size for up scaling
        target_height = int(current_image.shape[0] * resize_ratio)
        target_width = int(current_image.shape[1] * resize_ratio)

        # Expand the current image to the target size
        scaled_up_image = cv2.resize(current_image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        # Add the scaled up image to the current level of the pyramid to reconstruct the detail
        current_image = cv2.add(pyramid[i], scaled_up_image)

    return current_image


def validate_operation(img):
    pyr = get_laplacian_pyramid(img, levels)
    img_restored = restore_from_pyramid(pyr)

    plt.title(f"MSE is {np.mean((img_restored - img) ** 2)}")
    plt.imshow(img_restored, cmap='gray')

    plt.show()


def blend_pyramids(pyr_apple, pyr_orange, levels):
    pyr_result = []
    for curr_level in range(levels):
        # Retrieve the current level images from both pyramids
        apple_layer = pyr_apple[curr_level]
        orange_layer = pyr_orange[curr_level]

        # Define a mask in the size of the current pyramid
        rows, cols = apple_layer.shape[:2]
        mask = np.zeros((rows, cols), dtype=np.float32)

        # Initialize the mask’s columns from the first one up to (0.5 * width - curr_level) to 1.0
        transition_start_col = int(0.5 * cols - curr_level)
        mask[:, :transition_start_col] = 1.0

        # Calculate the mask values in gradual blending part
        for i in range(2 * (curr_level + 1)):
            index = cols // 2 - (curr_level + 1) + i
            mask[:, index] = 0.9 - 0.9 * i / (2 * (curr_level + 1))

        # Blend pyramid level for curr_level
        blended_level = orange_layer * mask + apple_layer * (1 - mask)
        pyr_result.append(blended_level)

    return pyr_result


apple = cv2.imread('apple.jpg')
apple = cv2.cvtColor(apple, cv2.COLOR_BGR2GRAY)

orange = cv2.imread('orange.jpg')
orange = cv2.cvtColor(orange, cv2.COLOR_BGR2GRAY)

levels = 5

validate_operation(apple)
validate_operation(orange)

pyr_apple = get_laplacian_pyramid(apple, levels)
pyr_orange = get_laplacian_pyramid(orange, levels)

# Blend the pyramids
pyr_result = blend_pyramids(pyr_apple, pyr_orange, levels)

final = restore_from_pyramid(pyr_result)
plt.imshow(final, cmap='gray')
plt.show()

cv2.imwrite("result.jpg", final)
