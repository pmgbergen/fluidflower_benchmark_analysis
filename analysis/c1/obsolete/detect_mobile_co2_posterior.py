import cv2
import daria
import matplotlib.pyplot as plt
import numpy as np
import skimage
from scipy.stats import norm

signal_70 = np.load("test_img/signal_70.npy")
co2_70 = np.load("co2.npy")

signal_70 = skimage.util.img_as_ubyte(signal_70)
signal_70 = cv2.cvtColor(signal_70, cv2.COLOR_RGB2HSV)

signal = np.load("test_img/signal_125.npy")
co2 = np.load("test_img/co2_125.npy")

signal = skimage.util.img_as_ubyte(signal)
signal = cv2.cvtColor(signal, cv2.COLOR_RGB2HSV)

signal = signal_70
co2 = co2_70

esf = np.load("test_img/esf.npy")
active_roi = np.logical_and(co2, np.logical_not(esf))
signal[np.logical_or(np.logical_not(co2), esf)] = 0

# Histogram analysis
roi_70_upper = (slice(1280, 1380), slice(5300, 6000))
roi_70_lower = (slice(2950, 3150), slice(4600, 5800))
roi_125_upper = (slice(2500, 2700), slice(6900, 7150))
roi_125_lower = (slice(2900, 2940), slice(5300, 5500))

roi = roi_125_upper
signal0 = signal[:, :, 0]
signal0 = signal0[roi]
mask0_low = signal0 > 0
mask0_high = signal0 < 255
signal1 = signal[:, :, 1]
signal1 = signal1[roi]
mask1_low = signal1 > 0
mask1_high = signal1 < 255
signal2 = signal[:, :, 2]
signal2 = signal2[roi]
mask2_low = signal2 > 0
mask2_high = signal2 < 255

mask0 = np.logical_and(mask0_low, mask0_high)
mask1 = np.logical_and(mask1_low, mask1_high)
mask2 = np.logical_and(mask2_low, mask2_high)

signal_0 = np.histogram(signal0[mask0], bins=256)[0]
signal_1 = np.histogram(signal1[mask1], bins=256)[0]
signal_2 = np.histogram(signal2[mask2], bins=256)[0]

mu0, std0 = norm.fit(np.ravel(signal0[mask0]))
mu1, std1 = norm.fit(np.ravel(signal1[mask1]))
mu2, std2 = norm.fit(np.ravel(signal2[mask2]))


print(mu0, std0)
print(mu1, std1)
print(mu2, std2)

if False:
    low = (
        max(1, int(mu0 - std0)),
        max(1, int(mu1 - std1)),
        max(1, int(mu2 - std2)),
    )
    high = (
        min(255, mu0 + std0),
        min(255, mu1 + std1),
        min(255, mu2 + std2),
    )
    mask_norm_upper = cv2.inRange(signal, low, high)
    mask_norm_upper = skimage.util.img_as_bool(mask_norm_upper)
    plt.figure()
    plt.plot(np.linspace(np.min(signal0[mask0]), np.max(signal0[mask0]), 256), signal_0)
    plt.figure()
    plt.plot(np.linspace(np.min(signal1[mask1]), np.max(signal1[mask1]), 256), signal_1)
    plt.figure()
    plt.plot(np.linspace(np.min(signal2[mask2]), np.max(signal2[mask2]), 256), signal_2)

roi = roi_125_lower
signal0 = signal[:, :, 0]
signal0 = signal0[roi]
mask0_low = signal0 > 0
mask0_high = signal0 < 255
signal1 = signal[:, :, 1]
signal1 = signal1[roi]
mask1_low = signal1 > 0
mask1_high = signal1 < 255
signal2 = signal[:, :, 2]
signal2 = signal2[roi]
mask2_low = signal2 > 0
mask2_high = signal2 < 255

mask0 = np.logical_and(mask0_low, mask0_high)
mask1 = np.logical_and(mask1_low, mask1_high)
mask2 = np.logical_and(mask2_low, mask2_high)

signal_0 = np.histogram(signal0[mask0], bins=256)[0]
signal_1 = np.histogram(signal1[mask1], bins=256)[0]
signal_2 = np.histogram(signal2[mask2], bins=256)[0]

mu0, std0 = norm.fit(np.ravel(signal0[mask0]))
mu1, std1 = norm.fit(np.ravel(signal1[mask1]))
mu2, std2 = norm.fit(np.ravel(signal2[mask2]))

print(mu0, std0)
print(mu1, std1)
print(mu2, std2)

low = (
    max(1, int(mu0 - std0)),
    max(1, int(mu1 - std1)),
    max(1, int(mu2 - std2)),
)
high = (
    min(255, mu0 + std0),
    min(255, mu1 + std1),
    min(255, mu2 + std2),
)
mask_norm_lower = cv2.inRange(signal, low, high)
mask_norm_lower = skimage.util.img_as_bool(mask_norm_lower)

# Mask_norm_lower based on 70 and 125 run.
low = (0, 150 - 34, 80 - 15)
high = (16 + 38, 150 + 34, 80 + 15)
mask_norm_lower_70 = cv2.inRange(signal, low, high)
mask_norm_lower_70 = skimage.util.img_as_bool(mask_norm_lower_70)

low = (0, 142 - 47, 70 - 20)
high = (90, 142 + 47, 70 + 20)
mask_norm_lower_125 = cv2.inRange(signal, low, high)
mask_norm_lower_125 = skimage.util.img_as_bool(mask_norm_lower_125)

plt.figure()
plt.plot(np.linspace(np.min(signal0[mask0]), np.max(signal0[mask0]), 256), signal_0)
plt.figure()
plt.plot(np.linspace(np.min(signal1[mask1]), np.max(signal1[mask1]), 256), signal_1)
plt.figure()
plt.plot(np.linspace(np.min(signal2[mask2]), np.max(signal2[mask2]), 256), signal_2)
plt.show()
#
# plt.figure()
# plt.imshow(signal)
# plt.show()


# From 70: blue and dark green
low = (1, 60, 100)
high = (15, 120, 160)
mask_70_blue = cv2.inRange(signal, low, high)
mask_70_blue = skimage.util.img_as_bool(mask_70_blue)
low = (1, 160, 80)
high = (20, 180, 100)
mask_70_green = cv2.inRange(signal, low, high)
mask_70_green = skimage.util.img_as_bool(mask_70_green)

low = (1, 25, 1)
high = (25, 75, 50)
mask_125_analysis = cv2.inRange(signal, low, high)
mask_125_analysis = skimage.util.img_as_bool(mask_125_analysis)

low = (1, 100, 125)
high = (25, 200, 175)
mask_70_analysis = cv2.inRange(signal, low, high)
mask_70_analysis = skimage.util.img_as_bool(mask_70_analysis)

low = (1, 100, 75)
high = (25, 200, 125)
mask_125_analysis2 = cv2.inRange(signal, low, high)
mask_125_analysis2 = skimage.util.img_as_bool(mask_125_analysis2)

low = (1, 50, 80)
high = (20, 100, 140)
mask_125_analysis3 = cv2.inRange(signal, low, high)
mask_125_analysis3 = skimage.util.img_as_bool(mask_125_analysis3)

# Define final mask
# mask = np.logical_or(mask_70_green, mask_70_blue)
# mask = mask_70_green
# mask = mask_70_blue
# mask = mask_125_analysis
# mask = mask_norm_lower_125
mask = mask_125_analysis3

# Deactivate signal outside mask
signal[~mask] = 0


# Remove small objects
print("remove small objects")
mask = skimage.morphology.remove_small_objects(mask, min_size=4)

plt.figure()
plt.imshow(signal)
plt.show()
print("Start local covering")
# Loop through patches and fill up
covered_mask = np.zeros(mask.shape[:2], dtype=bool)
size = 10  # in some sense the REV size in pixels
Ny, Nx = mask.shape[:2]
for row in range(int(Ny / size)):
    for col in range(int(Nx / size)):
        roi = (
            slice(row * size, (row + 1) * size),
            slice(col * size, (col + 1) * size),
        )
        covered_mask[roi] = skimage.morphology.convex_hull_image(mask[roi])

# Fill holes
print("Fill holes")
mask = skimage.morphology.remove_small_holes(mask, area_threshold=20**2)

# Update the mask value
# mask = covered_mask

# Use mask as signal
# signal = skimage.util.img_as_float(mask)
signal[~mask] = 0
signal = cv2.cvtColor(signal, cv2.COLOR_RGB2GRAY)

print("Start presmoothing")

# Apply presmoothing
if True:
    # Resize image

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
    resize = 0.5
    signal = cv2.resize(signal.astype(np.float32), None, fx=resize, fy=resize)
    signal = skimage.restoration.denoise_tv_bregman(
        signal,
        weight=1,
        eps=1e-5,
        max_num_iter=100,
        isotropic=False,
    )

    # Resize to original size
    signal = cv2.resize(signal, tuple(reversed(co2.shape[:2])))

print("Start thresholding")

plt.figure()
plt.imshow(signal)
plt.show()

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
