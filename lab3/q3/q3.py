
import cv2
import numpy as np
from matplotlib import pyplot as plt


# Load the noisy image
img_path = 'broken.jpg'
noisy_image = cv2.imread(img_path, 0)

# part A
# Apply median filter
median_filtered = cv2.medianBlur(noisy_image, 5)

# Save the image after applying both filters
combined_output_path = 'fixed.jpg'
cv2.imwrite(combined_output_path, median_filtered)

# Display the original, bilateral filtered, and combined filtered images
plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
plt.imshow(noisy_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(median_filtered, cmap='gray')
plt.title('Median Filter Applied')
plt.axis('off')

plt.show()

print(combined_output_path)

# part B
# Path to the noised images numpy array
noised_images_path = 'noised_images.npy'

# Load the noised images
noised_images = np.load(noised_images_path)
# add the broken image to all other imaged
# Expand the dimensions of noisy_image to make it (1, 522, 799)
noisy_image_expanded = np.expand_dims(noisy_image, axis=0)

# Append the new image to the array
noised_images_updated = np.concatenate((noised_images, noisy_image_expanded), axis=0)

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
