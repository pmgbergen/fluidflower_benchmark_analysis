import cv2
import daria
import matplotlib.pyplot as plt
import numpy as np
import skimage
from scipy.stats import norm

# Choose signal and CO2
signal_70 = np.load("test_img/signal_70.npy")
co2_70 = np.load("co2.npy")

signal_70 = skimage.util.img_as_ubyte(signal_70)
signal_70 = cv2.cvtColor(signal_70, cv2.COLOR_RGB2HSV)

signal = np.load("test_img/signal_125.npy")
co2 = np.load("test_img/co2_125.npy")

signal = skimage.util.img_as_ubyte(signal)
signal = cv2.cvtColor(signal, cv2.COLOR_RGB2HSV)

# signal = signal_70
# co2 = co2_70

# Deactivate region of disintrest
esf = np.load("test_img/esf.npy")
active_roi = np.logical_and(co2, np.logical_not(esf))
signal[np.logical_or(np.logical_not(co2), esf)] = 0

# From 70: blue and dark green
low = (1, 60, 100)
high = (15, 120, 160)
mask_70_blue = cv2.inRange(signal, low, high)
mask_70_blue = skimage.util.img_as_bool(mask_70_blue)
low = (1, 160, 80)
high = (20, 180, 100)
mask_70_green = cv2.inRange(signal, low, high)
mask_70_green = skimage.util.img_as_bool(mask_70_green)

# Mask_norm_lower based on 70 and 125 run.
low = (0, 150 - 34, 80 - 15)
high = (20, 150 + 34, 80 + 15)
mask_norm_lower_70 = cv2.inRange(signal, low, high)
mask_norm_lower_70 = skimage.util.img_as_bool(mask_norm_lower_70)

low = (0, 142 - 47, 70 - 20)
high = (20, 142 + 47, 70 + 20)
mask_norm_lower_125 = cv2.inRange(signal, low, high)
mask_norm_lower_125 = skimage.util.img_as_bool(mask_norm_lower_125)

# Define final mask
mask = np.logical_or(mask_70_green, mask_70_blue)
# mask = mask_70_green
# mask = mask_70_blue
# mask = mask_125_analysis
# mask = mask_norm_lower_125

# Deactivate signal outside mask
signal[~mask] = 0

plt.figure()
plt.imshow(signal)
plt.show()
plt.figure()
plt.imshow(signal)

# Convert to some scalar space - it does not anymore matter to much what space afetr thresholding.
signal = cv2.cvtColor(signal, cv2.COLOR_RGB2GRAY)

print("Start presmoothing")

# Apply presmoothing

## Apply TVD
# resize = 0.5
# signal = cv2.resize(signal.astype(np.float32), None, fx = resize, fy = resize)
# signal = skimage.restoration.denoise_tv_chambolle(
#    signal,
#    weight = 10,
#    eps = 1e-8,
#    max_num_iter = 1000,
# )

# Works
resize = 0.25
signal = cv2.resize(signal.astype(np.float32), None, fx=resize, fy=resize)
signal = skimage.restoration.denoise_tv_bregman(
    signal,
    weight=10,
    eps=1e-5,
    max_num_iter=100,
    isotropic=False,
)

# Resize to original size
signal = cv2.resize(signal, tuple(reversed(co2.shape[:2])))

plt.figure()
plt.imshow(signal)
plt.show()

print("Start thresholding")

# Find automatic threshold value using OTSU
thresh = skimage.filters.threshold_otsu(signal)
print("otsu full thresh", thresh)
mask = signal > thresh

plt.figure()
plt.imshow(mask)
plt.show()

# Label the connected regions first
labels, num_labels = skimage.measure.label(mask, return_num=True)
print("labels", num_labels)
plt.figure()
plt.imshow(labels)
plt.show()
