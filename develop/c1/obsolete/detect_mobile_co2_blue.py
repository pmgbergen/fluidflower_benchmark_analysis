import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
from scipy import ndimage as ndi
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture

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

# Blue
blue = diff[:, :, 2]

# ! ---- BLUE

img = blue
img_opening = skimage.morphology.opening(img, footprint=np.ones((5, 5)))

## Choose: R, B, H
# plt.figure()
# plt.imshow(original)
#
# plt.figure()
# plt.imshow(img)
#
# plt.figure()
# plt.imshow(img_opening)
#
#
# plt.show()
# plt.figure()
# plt.imshow(img > 0.2 * np.max(img))

# Construct representative point cloud. The values are not relevant anymore.
# Instead the structure seems more relevant. Aim at segmentation of binary point
# cloud data.
active_img = np.ravel(img)[np.ravel(active)]
thresh = skimage.filters.threshold_otsu(active_img)
mask = skimage.util.img_as_float(img > thresh)
mask_bool = img > thresh
plt.figure()
plt.imshow(mask)

density = np.transpose(np.vstack(np.nonzero(mask_bool)))

# clustering = DBSCAN(eps = 0.1, min_samples=2).fit(density)
# print(np.unique(clustering.labels_))
# clustering2 = SpectralClustering(n_clusters=2).fit(density)
# print(np.unique(clustering2.labels_))

mixture = GaussianMixture(n_components=2, random_state=0).fit(density)
clustering = mixture.predict(density)
print(mixture.score(density))
row, col = np.nonzero(mask_bool)
tst = np.zeros(mask.shape[:2], dtype=int)
tst[row, col] = clustering + 1
plt.figure()
plt.imshow(tst)

active2 = tst == 2
density = np.transpose(np.vstack(np.nonzero(active2)))

mixture2 = GaussianMixture(n_components=2, random_state=0).fit(density)
clustering = mixture2.predict(density)
print(mixture2.score(density))
row, col = np.nonzero(active2)
tst = np.zeros(mask.shape[:2], dtype=int)
tst[row, col] = clustering + 1
plt.figure()
plt.imshow(tst)

active2 = tst == 2
density = np.transpose(np.vstack(np.nonzero(active2)))

mixture2 = GaussianMixture(n_components=2, random_state=0).fit(density)
clustering = mixture2.predict(density)
print(mixture2.score(density))
row, col = np.nonzero(active2)
tst = np.zeros(mask.shape[:2], dtype=int)
tst[row, col] = clustering + 1
plt.figure()
plt.imshow(tst)

plt.show()


# Continue with cluster with value
cluster = tst == 1

plt.figure()
plt.imshow(cluster)

img[~cluster] = 0
plt.figure()
plt.imshow(img)


cluster = skimage.morphology.remove_small_objects(cluster, min_size=20**2)
cluster = skimage.morphology.remove_small_holes(cluster, 50**2)

plt.figure()
plt.imshow(cluster)

##cluster = skimage.filters.rank.median(cluster, skimage.morphology.disk(5))
# cluster = skimage.morphology.binary_closing(skimage.util.img_as_bool(cluster), footprint = np.ones((10,10)))
# cluster = skimage.morphology.remove_small_holes(cluster, 50**2)
# cluster = ndi.morphology.binary_opening(cluster, structure=np.ones((5,5)))
# cluster = skimage.util.img_as_bool(cv2.resize(cluster.astype(np.float32), tuple(reversed(co2.shape))))

img[~cluster] = 0

plt.figure()
plt.imshow(img)

img = skimage.filters.rank.mean(img, skimage.morphology.disk(5))
img = skimage.restoration.denoise_tv_chambolle(img, 0.1, eps=1e-5, max_num_iter=10)

active_img = np.ravel(img)[np.ravel(cluster)]
hist = np.histogram(active_img, bins=100)[0]
plt.figure()
plt.plot(hist)

plt.figure()
plt.imshow(cluster)

plt.figure()
plt.imshow(img)

plt.show()


## Use DBScan to segment mask
# >>> X = np.array([[1, 2], [2, 2], [2, 3],
# ...               [8, 7], [8, 8], [25, 80]])
# >>> clustering = DBSCAN(eps=3, min_samples=2).fit(X)
# >>> clustering.labels_
# array([ 0,  0,  0,  1,  1, -1])
# >>> clustering
# DBSCAN(eps=3, min_samples=2)

print(mask.shape)


assert False

plt.show()


mask0 = np.copy(mask)
mask0 = cv2.resize(mask0, None, fx=0.5, fy=0.5)
mask0 = cv2.resize(mask0, tuple(reversed(mask.shape[:2])))
plt.figure()
plt.imshow(mask0)

# Remove small holes and small objects
mask1 = np.copy(mask)
mask1 = skimage.filters.rank.median(mask1, skimage.morphology.disk(5))
mask1 = skimage.morphology.binary_closing(
    skimage.util.img_as_bool(mask1), footprint=np.ones((10, 10))
)
mask1 = skimage.morphology.remove_small_holes(mask1, 50**2)
mask1 = ndi.morphology.binary_opening(mask1, structure=np.ones((5, 5)))
mask1 = skimage.util.img_as_bool(cv2.resize(mask1, tuple(reversed(co2.shape))))
plt.figure()
plt.imshow(mask1)
plt.show()

# Test plot
contours_co2, hierarchy = cv2.findContours(co2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_mobile_co2, hierarchy = cv2.findContours(
    mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)
original_copy = np.copy(original)
cv2.drawContours(original_copy, contours_co2, -1, (0, 255, 0), 3)
cv2.drawContours(original_copy, contours_mobile_co2, -1, (0, 0, 255), 3)
plt.figure()
plt.imshow(original_copy)
plt.show()


# NOTE: mask1 is the current choice for the blue analysis.
# Do some testing with closig etc.


# Consecutive closing

closed_med_out = skimage.util.img_as_bool(np.copy(med))
for i in range(1, 20):
    closed_med_in = np.copy(closed_med_out)
    closed_med_out = skimage.morphology.binary_closing(
        skimage.util.img_as_bool(closed_med_in), footprint=np.ones((2 * i, 2 * i))
    )
    change = np.sum(np.logical_xor(closed_med_out, closed_med_in)) / np.sum(active)
    rel_change = np.sum(np.logical_xor(closed_med_out, closed_med_in)) / np.sum(
        cloded_med_in
    )
    print(change, rel_change, np.sum(closed_med_out))

    plt.figure()
    plt.imshow(closed_med_out)

plt.show()

mask = cv2.resize(mask, (4 * 280, 4 * 150))
plt.figure()
plt.imshow(mask)

mask = skimage.restoration.denoise_tv_chambolle(mask, 0.1, eps=1e-5, max_num_iter=1000)
plt.figure()
plt.imshow(mask)

plt.show()

# Subset
# roi = (slice(3800, 3600), slice(3900, 4800))
roi = (slice(3000, 3500), slice(2000, 3000))

original_sub = original[roi]
plt.figure()
plt.imshow(original_sub)

sub = img[roi]
plt.figure()
plt.imshow(sub)

sub_opening = img_opening[roi]
plt.figure()
plt.imshow(sub_opening)

sub = skimage.morphology.opening(sub, footprint=np.ones((5, 5)))
plt.figure()
plt.imshow(sub)

# sub = skimage.morphology.closing(sub, footprint = np.ones((10,10)))
# plt.figure()
# plt.imshow(sub)

sub = cv2.resize(sub, None, fx=0.5, fy=0.5)
sub = skimage.restoration.denoise_tv_chambolle(sub, 0.05, eps=1e-5, max_num_iter=1000)
plt.figure()
plt.imshow(sub)

plt.show()


# Resize to make TVD feasible
factor = 2
img = cv2.resize(img, (factor * 280, factor * 150), interpolation=cv2.INTER_AREA)

plt.figure()
plt.imshow(img)

img = skimage.restoration.denoise_tv_chambolle(img, 0.01, eps=1e-8, max_num_iter=5000)
plt.figure()
plt.imshow(img)

# Perform histogram analysis only on the active region
active_coarse = skimage.util.img_as_bool(
    skimage.transform.resize(active, (factor * 150, factor * 280))
)
active_img = np.ravel(img)[np.ravel(active_coarse)]
hist = np.histogram(np.ravel(img)[np.ravel(active_coarse)], bins=100)[0]
plt.figure()
plt.plot(hist)
thresh = skimage.filters.threshold_otsu(active_img)
print(thresh)

# Apply thresholding (identify brighter region)
mask = img > thresh
mask_fine = skimage.util.img_as_ubyte(
    skimage.util.img_as_bool(skimage.transform.resize(mask, original.shape[:2]))
)
plt.figure()
plt.imshow(original)
plt.imshow(mask, alpha=0.1)

contours, hierarchy = cv2.findContours(
    mask_fine, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)
original_copy = np.copy(original)
cv2.drawContours(original_copy, contours, -1, (0, 255, 0), 1)
plt.figure()
plt.imshow(original_copy)

plt.show()
