
import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_fix(image, id):
	"""
    Parameters:
    - image: the input image
    - id: the identifier for the image

    Returns:
    - the processed image
    """
	if id == 1:
		# Apply histogram equalization
		fixed_image = cv2.equalizeHist(image)
	elif id == 2:
		# Apply gamma correction
		gamma = 1.5
		inv_gamma = 1.0 / gamma
		table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
		fixed_image = cv2.LUT(image, table)
	elif id == 3:
		# Convert to float to avoid unsigned byte overflow
		img_float = image.astype(np.float64)
		# Get the minimum and maximum pixel values
		min_val = np.percentile(img_float, 2)
		max_val = np.percentile(img_float, 98)
		# Perform the stretch
		fixed_image = (img_float - min_val) * (255 / (max_val - min_val))
		fixed_image = np.clip(fixed_image, 0, 255).astype(np.uint8)
		return fixed_image


	else:
		raise ValueError(f"Unsupported id: {id}")
	return fixed_image


# Apply the fix to each image and save the results
for i in range(1, 4):
	# Correct the file extension based on the image ID
	ext = 'png' if i == 1 else 'jpg'
	path = f'{i}.{ext}'

	# Read the image with the appropriate flag
	flag = cv2.IMREAD_GRAYSCALE if i != 3 else cv2.IMREAD_COLOR
	image = cv2.imread(path, flag)

	# Check if the image is loaded
	if image is None:
		raise FileNotFoundError(f"No image found at path: {path}")

	# Apply the fix
	fixed_image = apply_fix(image, i)

	# Define the color map: if grayscale, use 'gray', else None for color images
	cmap = 'gray' if i != 3 else None

	# Save the fixed image with appropriate color mapping
	save_path = f'{i}_fixed.jpg'
	plt.imsave(save_path, fixed_image, cmap=cmap, vmin=0, vmax=255)

# Return the paths to the saved images
fixed_image_paths = [f'{i}_fixed.jpg' for i in range(1, 4)]
fixed_image_paths
