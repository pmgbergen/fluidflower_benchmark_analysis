import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
import scipy.ndimage as ndi
import scipy.signal as spsignal

labels = np.load("labels.npy")
signal = np.load("concentration_30.npy")

# Masks and contours
esf_sand = labels == 1
c_sand = labels == 2
rest = labels == 5

# Take the median - take into account the grain size (more or less)
if True:
    median_esf = skimage.filters.rank.median(
        signal, skimage.morphology.disk(20), mask=esf_sand
    )
    median_c = skimage.filters.rank.median(
        signal, skimage.morphology.disk(20), mask=c_sand
    )
    median_rest = skimage.filters.rank.median(
        signal, skimage.morphology.disk(20), mask=rest
    )
    median_esf[~esf_sand] = 0
    median_c[~c_sand] = 0
    median_rest[~rest] = 0

    # Median looses smoothness due to the int values.
    median = median_esf + median_c + median_rest

# Find contours, contours bands and their pairwise intersections
if False:
    print("sure?")
    assert False

    def contours_to_mask(contours):
        mask = np.zeros(signal.shape[:2], dtype=bool)
        for c in contours:
            c = (c[:, 0, 1], c[:, 0, 0])
            mask[c] = True
        return mask

    band_size = 50

    contours_esf, _ = cv2.findContours(
        100 * skimage.img_as_ubyte(esf_sand),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contour_mask_esf = contours_to_mask(contours_esf)
    contour_mask_esf = skimage.morphology.binary_dilation(
        contour_mask_esf, np.ones((band_size, band_size), np.uint8)
    )
    contour_mask_esf = skimage.img_as_bool(contour_mask_esf)

    contours_c, _ = cv2.findContours(
        100 * skimage.img_as_ubyte(c_sand),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contour_mask_c = contours_to_mask(contours_c)
    contour_mask_c = skimage.morphology.binary_dilation(
        contour_mask_c, np.ones((band_size, band_size), np.uint8)
    )
    contour_mask_c = skimage.img_as_bool(contour_mask_c)

    contours_rest, _ = cv2.findContours(
        skimage.img_as_ubyte(rest),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contour_mask_rest = contours_to_mask(contours_rest)
    contour_mask_rest = skimage.morphology.binary_dilation(
        contour_mask_rest, np.ones((band_size, band_size), np.uint8)
    )
    contour_mask_rest = skimage.img_as_bool(contour_mask_rest)

    np.save("contour_mask_esf.npy", contour_mask_esf)
    np.save("contour_mask_cf.npy", contour_mask_c)
    np.save("contour_mask_rest.npy", contour_mask_rest)

else:
    contour_mask_esf = np.load("contour_mask_esf.npy")
    contour_mask_c = np.load("contour_mask_cf.npy")
    contour_mask_rest = np.load("contour_mask_rest.npy")

# Find boundaries
boundary_esf_c = np.logical_and(contour_mask_esf, contour_mask_c)
boundary_c_rest = np.logical_and(contour_mask_c, contour_mask_rest)

# Restrict to the subdomains
roi_esf_c = np.logical_and(esf_sand, boundary_esf_c)
roi_c_esf = np.logical_and(c_sand, boundary_esf_c)
roi_c_rest = np.logical_and(c_sand, boundary_c_rest)
roi_rest_c = np.logical_and(rest, boundary_c_rest)

# ! ----  Approach I - find scaling based on equalizing mean of median
scaling_mean_esf_c = np.mean(median[roi_c_esf]) / np.mean(median[roi_esf_c])
scaling_mean_c_rest = np.mean(median[roi_c_rest]) / np.mean(median[roi_rest_c])
print(scaling_mean_esf_c)
print(scaling_mean_c_rest)

## ! ---- Approach II - find scaling based on histogram analysis
# bins = 1000
#
# range_min, range_max = 0.05 * np.max(signal[roi_c_esf]), np.max(signal[roi_c_esf])
# range_linspace = np.linspace(range_min, range_max, bins)
#
# hist_esf, _ = np.histogram(signal[roi_esf_c], bins=bins, range = (range_min, range_max))
# hist_c, _ = np.histogram(signal[roi_c_esf], bins=bins, range = (range_min, range_max))
#
# hist_esf = ndi.gaussian_filter1d(hist_esf, 10)
# hist_c = ndi.gaussian_filter1d(hist_c, 10)
#
# peak_esf, _ = spsignal.find_peaks(hist_esf)
# peak_c, _ = spsignal.find_peaks(hist_c)
# assert len(peak_esf) == 1 and len(peak_c) == 1
#
# scaling_hist_esf_c = range_linspace[peak_c[0]] / range_linspace[peak_esf[0]]
# print(scaling_hist_esf_c)
#
# range_min, range_max = 0.05 * np.max(signal[roi_rest_c]), np.max(signal[roi_rest_c])
# range_linspace = np.linspace(range_min, range_max, bins)
#
# hist_rest, _ = np.histogram(signal[roi_rest_c], bins=bins, range = (range_min, range_max))
# hist_c, _ = np.histogram(signal[roi_c_rest], bins=bins, range = (range_min, range_max))
#
# hist_rest = ndi.gaussian_filter1d(hist_rest, 10)
# hist_c = ndi.gaussian_filter1d(hist_c, 10)
#
# peak_rest, _ = spsignal.find_peaks(hist_rest)
# peak_c, _ = spsignal.find_peaks(hist_c)
# assert len(peak_rest) == 1 and len(peak_c) == 1
#
# scaling_hist_c_rest = range_linspace[peak_c[0]] / range_linspace[peak_rest[0]]
# print(scaling_hist_c_rest)

# ! ---- Test.

# Reference
median_signal = skimage.filters.rank.median(signal, skimage.morphology.disk(20))
plt.figure()
plt.imshow(median_signal)

# Rescaling based on mean of median at the boundary.
rescaled_signal = np.copy(signal)
rescaled_signal[esf_sand] *= scaling_mean_esf_c
rescaled_signal[rest] *= scaling_mean_c_rest
median_rescaled_signal = skimage.filters.rank.median(
    rescaled_signal, skimage.morphology.disk(20)
)
plt.figure()
plt.imshow(median_rescaled_signal)
tvd_rescaled_signal = skimage.restoration.denoise_tv_bregman(
    rescaled_signal, weight=1e-2, eps=1e-5, max_num_iter=200
)
plt.figure()
plt.imshow(tvd_rescaled_signal)

## Rescaling based on hist analysis at the boundary.
# rescaled_signal = np.copy(signal)
# rescaled_signal[esf_sand] *= scaling_hist_esf_c
# rescaled_signal[rest] *= scaling_hist_c_rest
# median_rescaled_signal = skimage.filters.rank.median(rescaled_signal, skimage.morphology.disk(20))
# plt.figure()
# plt.imshow(median_rescaled_signal)
# tvd_rescaled_signal = skimage.restoration.denoise_tv_bregman(rescaled_signal, weight = 1e-2, eps = 1e-5, max_num_iter = 200)
# plt.figure()
# plt.imshow(tvd_rescaled_signal)

plt.show()
