import cv2
import numpy as np
from matplotlib import pyplot as plt


# Load the noisy image
img_path = 'broken.jpg'
noisy_image = cv2.imread(img_path, 0)

# part A
# Apply bilateralFilter
bilateralFilter_image = cv2.bilateralFilter(noisy_image, d=10, sigmaColor=125, sigmaSpace=25)
# Apply median filter
fixed_image = cv2.medianBlur(bilateralFilter_image, 3)

# Save the image after applying both filters
combined_output_path = 'fixed.jpg'
cv2.imwrite(combined_output_path, fixed_image)

# Display the original, bilateral filtered, and combined filtered images
plt.figure(figsize=(7, 15))

plt.subplot(3, 1, 1)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image')
plt.axis('off')

plt.subplot(3, 1, 2)
plt.imshow(bilateralFilter_image, cmap='gray')
plt.title('Bilateral Filter Image')
plt.axis('off')

plt.subplot(3, 1, 3)
plt.imshow(fixed_image, cmap='gray')
plt.title('Median filter Applied after Bilateral Filter')
plt.axis('off')

plt.tight_layout()
plt.show()

print(combined_output_path)

# part B
# Path to the noised images numpy array
noised_images_path = 'noised_images.npy'

# Load the noised images
noised_images = np.load(noised_images_path)

# Compute the average image from the stack of noised images
average_image = np.mean(noised_images, axis=0)

# Convert the average image to uint8 format for proper visualization
average_image_uint8 = np.uint8(average_image)

# Path to save the cleaned (average) image
average_image_path = 'average_image.jpg'

# Save the average image
cv2.imwrite(average_image_path, average_image_uint8)

# Display the path to the saved average image
print(average_image_path)
