import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import matplotlib.pyplot as plt

image_path = 'zebra.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Your code goes here
h, w = image.shape[0], image.shape[1]

# perform the fft & Shift to the center:
fourier_transform = fftshift(fft2(image))

# Display fourier_transform
magnitude_spectrum = np.log(np.abs(fourier_transform))

# Zero padding
f_padded = np.pad(fourier_transform, ((h // 2, h // 2), (w // 2, w // 2)))

# Apply inverse FFT to the padded frequency domain to get the enlarged image
f_inv_padded = ifft2(f_padded)
image_larger = np.abs(f_inv_padded)

# Display padded spectrum
magnitude_spectrum_padded = 20 * np.log(np.abs(f_padded) + 1)

# Create an empty array of zeros of size (2H, 2W)
image_ft = np.zeros((h * 2, w * 2), np.complex128)
image_ft2 = np.zeros((h * 2, w * 2), np.complex128)

# Place the original Fourier transform values in the new array, skipping every other row and column
image_ft[::2, ::2] = 4 * magnitude_spectrum
image_ft2[::2, ::2] = 4 * fourier_transform

# Compute the inverse Fourier transform of this padded array
image_four = np.abs(ifft2(ifftshift(image_ft2)))

# Display the results
plt.figure(figsize=(10, 10))

plt.subplot(321)
plt.title('Original Grayscale Image')
plt.imshow(image, cmap='gray')

plt.subplot(322)
plt.title('Fourier Spectrum')
plt.imshow(magnitude_spectrum, cmap='gray')

plt.subplot(323)
plt.title('Fourier Spectrum Zero Padding')
plt.imshow(magnitude_spectrum_padded, cmap='gray')

plt.subplot(324)
plt.title('Two Times Larger Grayscale Image')
plt.imshow(image_larger, cmap='gray')
plt.subplot(325)
plt.title('Fourier Spectrum Four Copies')
plt.imshow(np.abs(image_ft), cmap='gray')

plt.subplot(326)
plt.title('Four Copies Grayscale Image')
plt.imshow(image_four, cmap='gray')

plt.tight_layout()
plt.savefig('zebra_scaled.png')
plt.show()
