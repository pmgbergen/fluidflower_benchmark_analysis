import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
from scipy import ndimage as ndi

mask = np.load("mask.npy")

mask = skimage.img_as_bool(mask)

plt.figure()
plt.imshow(mask)

# Remove small objects
mask = skimage.morphology.remove_small_objects(mask, min_size=5**2)

plt.figure()
plt.imshow(mask)

# Remove small holes
mask = skimage.morphology.remove_small_holes(mask, area_threshold=5**2)
plt.figure()
plt.imshow(mask)

mask = skimage.morphology.remove_small_holes(mask, area_threshold=5**2)
plt.figure()
plt.imshow(mask)

mask = ndi.morphology.binary_opening(mask, structure=np.ones((2, 2)))

plt.figure()
plt.imshow(mask)


plt.show()
