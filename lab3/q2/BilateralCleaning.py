# Student_Name1, Student_ID1
# Student_Name2, Student_ID2

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import numpy as np
import matplotlib.pyplot as plt

def clean_Gaussian_noise_bilateral(im, radius, stdSpatial, stdIntensity):
	# Your code goes here

# change this to the name of the image you'll try to clean up
original_image_path = 'path.jpg'
image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

clear_image_b = clean_Gaussian_noise_bilateral(image, 0, 0, 0)

plt.subplot(121)
plt.imshow(image, cmap='gray')

plt.subplot(122)
plt.imshow(clear_image_b, cmap='gray')

plt.show()
