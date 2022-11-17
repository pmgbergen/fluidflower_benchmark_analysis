import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
from scipy import ndimage as ndi

# Images
base = np.load("base.npy")
original = np.load("original_mid.npy")

# Masks
esf = np.load("esf.npy").astype(bool)
co2 = np.load("co2.npy").astype(bool)

# Full diff
diff = skimage.util.compare_images(base, original, method="diff")

# Active set # TODO is it required to remove ESF?
active = np.logical_and(co2, ~esf)

# Restrict to active set
diff[~active] = 0

# Cache the seemingly only relevant single color spectra

# Hue
hue = cv2.cvtColor(diff.astype(np.float32), cv2.COLOR_RGB2HSV)[:, :, 0]

# ! ---- HUE
img = (np.max(hue) - hue) / np.max(hue)
img[~active] = 0.0

plt.figure()
plt.imshow(original)

plt.figure()
plt.imshow(img)

active_img = np.ravel(img)[np.ravel(active)]
hist = np.histogram(active_img, bins=100)[0]
plt.figure()
plt.plot(hist)

plt.show()

img = skimage.filters.rank.median(img, skimage.morphology.disk(5))
# img = skimage.restoration.denoise_tv_chambolle(img, 0.01, eps = 1e-5, max_num_iter = 1000)
img = skimage.restoration.denoise_tv_bregman(
    img, weight=100, max_num_iter=1000, eps=1e-5, isotropic=False
)

plt.figure()
plt.imshow(img)
plt.show()


# BLUE approach.
active_img = np.ravel(img)[np.ravel(active)]
hist = np.histogram(active_img, bins=100)[0]
plt.figure()
plt.plot(hist)
thresh = skimage.filters.threshold_otsu(active_img)
thresh *= 1.1
print(thresh)
mask = skimage.util.img_as_float(img > thresh)
plt.figure()
plt.imshow(mask)

# Extend to outside
outside = np.copy(mask)
outside = skimage.morphology.binary_dilation(mask, footprint=np.ones((5, 5)))
outside[active] = mask[active]

# Remove small holes and small objects
mask1 = np.copy(mask)
mask1 = skimage.filters.rank.median(mask1, skimage.morphology.disk(10))
plt.figure()
plt.imshow(mask1)
mask1 = skimage.morphology.binary_closing(
    skimage.util.img_as_bool(mask1), footprint=np.ones((10, 10))
)
plt.figure()
plt.imshow(mask1)
mask1 = skimage.morphology.remove_small_holes(mask1, 50**2)
plt.figure()
plt.imshow(mask1)
mask1 = ndi.morphology.binary_opening(mask1, structure=np.ones((5, 5)))
plt.figure()
plt.imshow(mask1)
mask1 = skimage.util.img_as_bool(
    cv2.resize(mask1.astype(np.float32), tuple(reversed(co2.shape)))
)
plt.figure()
plt.imshow(mask1)
plt.show()

## Resize to make TVD feasible
# factor = 8
# img = cv2.resize(img, (factor * 280,factor * 150), interpolation = cv2.INTER_AREA)
#
# plt.figure()
# plt.imshow(img)
#
# plt.show()
#
# plt.figure()
# plt.imshow(img)
# img = skimage.restoration.denoise_tv_chambolle(img, 0.01, eps = 1e-5, max_num_iter = 1000)
#
# plt.figure()
# plt.imshow(img)
#
# plt.show()

## Resize to original size
# img = cv2.resize(img, tuple(reversed(original.shape[:2])))

img = skimage.filters.rank.median(img, skimage.morphology.disk(5))
img = skimage.restoration.denoise_tv_chambolle(img, 0.01, eps=1e-5, max_num_iter=1000)

plt.figure()
plt.imshow(img)
plt.show()


# BLUE approach.
active_img = np.ravel(img)[np.ravel(active)]
hist = np.histogram(active_img, bins=100)[0]
plt.figure()
plt.plot(hist)
thresh = skimage.filters.threshold_otsu(active_img)
print(thresh)
mask = skimage.util.img_as_float(img > thresh)
plt.figure()
plt.imshow(mask)

# Extend to outside
outside = np.copy(mask)
outside = skimage.morphology.binary_dilation(mask, footprint=np.ones((5, 5)))
outside[active] = mask[active]

# Remove small holes and small objects
mask1 = np.copy(mask)
mask1 = skimage.filters.rank.median(mask1, skimage.morphology.disk(10))
plt.figure()
plt.imshow(mask1)
mask1 = skimage.morphology.binary_closing(
    skimage.util.img_as_bool(mask1), footprint=np.ones((10, 10))
)
plt.figure()
plt.imshow(mask1)
mask1 = skimage.morphology.remove_small_holes(mask1, 50**2)
plt.figure()
plt.imshow(mask1)
mask1 = ndi.morphology.binary_opening(mask1, structure=np.ones((5, 5)))
plt.figure()
plt.imshow(mask1)
mask1 = skimage.util.img_as_bool(
    cv2.resize(mask1.astype(np.float32), tuple(reversed(co2.shape)))
)
plt.figure()
plt.imshow(mask1)
plt.show()
