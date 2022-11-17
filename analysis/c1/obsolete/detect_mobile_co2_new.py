import cv2
import daria
import matplotlib.pyplot as plt
import numpy as np
import skimage

signal_70 = np.load("test_img/signal_70.npy")
co2_70 = np.load("co2.npy")
esf = np.load("test_img/esf.npy")

signal_70 = skimage.util.img_as_ubyte(signal_70)
signal_70 = cv2.cvtColor(signal_70, cv2.COLOR_RGB2HSV)
active_roi = np.logical_and(co2_70, np.logical_not(esf))
signal_70[np.logical_or(np.logical_not(co2_70), esf)] = 0

plt.figure()
plt.imshow(signal_70)

signal = np.load("test_img/signal_125.npy")
co2 = np.load("test_img/co2_125.npy")

signal = skimage.util.img_as_ubyte(signal)
signal = cv2.cvtColor(signal, cv2.COLOR_RGB2HSV)
active_roi = np.logical_and(co2, np.logical_not(esf))
signal[np.logical_or(np.logical_not(co2), esf)] = 0

plt.figure()
plt.imshow(signal)
plt.show()

# From 70: blue and dark green
low = (1, 60, 100)
high = (15, 120, 160)
mask_70_blue = cv2.inRange(signal, low, high)
mask_70_blue = skimage.util.img_as_bool(mask_70_blue)
low = (1, 160, 80)
high = (20, 180, 100)
mask_70_green = cv2.inRange(signal, low, high)
mask_70_green = skimage.util.img_as_bool(mask_70_green)

mask = np.logical_or(mask_70_green, mask_70_blue)
# mask = mask_70_blue

signal[~mask] = 0
signal = signal[:, :, 0]
plt.figure()
plt.imshow(signal)
plt.show()

# assert False
#
#
## Current attempt.
# low = (1,1,50)
# high = (255, 200, 255)
# mask = cv2.inRange(signal, low, high)
# mask = skimage.util.img_as_bool(mask)
# plt.figure()
# plt.imshow(mask)
#
# plt.show()
#
# assert False

# print("Start local covering")
#
## Loop through patches and fill up
# covered_mask = np.zeros(mask.shape[:2], dtype=bool)
# size = 20 # in some sense the REV size in pixels
# Ny, Nx = mask.shape[:2]
# for row in range(int(Ny / size)):
#    for col in range(int(Nx / size)):
#        roi = (
#            slice(row * size, (row + 1) * size),
#            slice(col * size, (col + 1) * size),
#        )
#        covered_mask[roi] = skimage.morphology.convex_hull_image(mask[roi])
#
## Fill holes
# print("Fill holes")
# mask = skimage.morphology.remove_small_holes(mask, area_threshold=20**2)

# plt.figure()
# plt.imshow(mask)

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
        weight=0.02,
        eps=1e-5,
        max_num_iter=100,
        isotropic=False,
    )

    ## Slightly smoother - also OK
    # resize = 0.5
    # signal = cv2.resize(signal.astype(np.float32), None, fx = resize, fy = resize)
    # signal = skimage.restoration.denoise_tv_bregman(
    #    signal,
    #    weight = 1,
    #    eps = 1e-5,
    #    max_num_iter = 1000,
    #    isotropic=False,
    # )

    # signal = signal.astype(np.float32)
    # signal = skimage.restoration.denoise_tv_bregman(
    #    signal,
    #    weight = 0.1,
    #    eps = 1e-5,
    #    max_num_iter = 1000,
    #    isotropic=False,
    # )

    # Resize to original size
    signal = cv2.resize(signal, tuple(reversed(co2.shape[:2])))

plt.figure()
plt.imshow(signal)

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
print("labels", num_labels, np.unique(labels))
plt.figure()
plt.imshow(labels)
plt.show()

props = skimage.measure.regionprops(labels)

# Investigate each labeled region separately
for label in range(1, num_labels + 1):
    # Find max value in region with fixed label
    labeled_region = labels == label
    print(label)

    contours, hierarchy = cv2.findContours(
        skimage.util.img_as_ubyte(labeled_region),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    contour = contours[0]
    # contour = np.zeros(contour_tmp.shape, dtype=int)
    # contour[:,0, 0] = contour_tmp[:,0, 1]
    # contour[:,0, 1] = contour_tmp[:,0, 0]

    if np.any(mask[labeled_region]):

        # Deactivate labeled region if the max value it not sufficiently high. Allow for some relaxation.
        restricted_signal = np.copy(signal)
        restricted_signal[~labeled_region] = 0
        dx = daria.forward_diff_x(restricted_signal)
        dy = daria.forward_diff_y(restricted_signal)
        grad = np.sqrt(dx**2 + dy**2)
        print(
            np.sum(np.transpose(grad)[contour]),
            np.max(np.transpose(grad)[contour]),
            contour.shape[0],
        )

        plt.figure()
        plt.imshow(grad)
        plt.show()

print("start postsmoothing")
# Apply postsmoothing
if False:
    # Resize image
    resize = 0.5
    resized_mask = cv2.resize(
        covered_mask.astype(np.float32), None, fx=resize, fy=resize
    )

    smoothed_mask = skimage.restoration.denoise_tv_bregman(
        resized_mask,
        weight=0.02,
        eps=1e-5,
        max_num_iter=1000,
        isotropic=True,
    )
    # Resize to original size
    large_mask = cv2.resize(
        smoothed_mask.astype(np.float32), tuple(reversed(co2.shape[:2]))
    )

    # Apply hardcoded threshold value of 0.5 assuming it is sufficient to turn
    # off small particles and rely on larger marked regions
    mask = large_mask > 0.5

plt.figure()
plt.imshow(large_mask)
plt.show()

plt.figure()
plt.imshow(mask)

plt.figure()
plt.imshow(mask2)

plt.show()

# Clean up and include only connected regions which reach a certain value in
# the cached mask. This step is built on the assumption that we have been
# interested in high-concentration areas.

cached_mask = copy.deepcopy(large_mask)
# final_mask = np.zeros(cached_mask.shape[:2], dtype=bool)

# Label the connected regions first
labels, num_labels = skimage.measure.label(mask, return_num=True)
props = skimage.measure.regionprops(labels)

# Investigate each labeled region separately
for label in range(num_labels):
    # Find max value in region with fixed label
    labeled_region = labels == label

    ## Deactivate labeled region if the max value it not sufficiently high. Allow for some relaxation.
    # max_value_in_cached_mask = np.max(cached_mask[labeled_region])
    # print(label, max_value_in_cached_mask)
    # if max_value_in_cached_mask > 0.8:
    #    final_mask[labeled_region] = True
    #    plt.figure()
    #    plt.imshow(final_mask)
    #    plt.show()

    # Deactivate labeled region if the max value it not sufficiently high. Allow for some relaxation.
    grad = self.gradient_modulus(large_mask, labeled_region)
    plt.figure()
    plt.imshow(labeled_region)
    plt.figure()
    plt.imshow(grad)
    print(np.sum(grad), props[label]["perimeter"], np.sum(labeled_region))
    plt.show()

# plt.figure()
# plt.imshow(large_mask)
##plt.figure()
##plt.imshow(labels)

plt.figure()
plt.imshow(final_mask)
plt.show()
