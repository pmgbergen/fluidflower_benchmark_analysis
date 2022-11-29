import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
import scipy.ndimage as ndi
import scipy.signal as spsignal

labels = np.load("labels.npy")
signal = np.load("concentration_30.npy")

if False:
    plt.figure()
    plt.imshow(labels)
    
    plt.figure()
    plt.imshow(signal)
    plt.show()

# Masks and contours
esf_sand = labels == 1
c_sand = labels == 2
rest = labels == 5


# Take the median - take into account the grain size (more or less)
if True:
    median_esf = skimage.filters.rank.median(signal, skimage.morphology.disk(20), mask=esf_sand)
    median_c = skimage.filters.rank.median(signal, skimage.morphology.disk(20), mask=c_sand)
    median_rest = skimage.filters.rank.median(signal, skimage.morphology.disk(20), mask=rest)
    median_esf[~esf_sand] = 0
    median_c[~c_sand] = 0
    median_rest[~rest] = 0

    # Median looses smoothness due to the int values.
    median = median_esf + median_c + median_rest

if False:
    plt.figure()
    plt.imshow(median_esf)
    plt.figure()
    plt.imshow(median_c)
    plt.figure()
    plt.imshow(median_rest)
    plt.figure()
    plt.imshow(median)
    plt.figure()
    plt.imshow(rescaled_median)

    plt.show()

# Take the mean - take into account the grain size (more or less)
if False:
    mean_esf = skimage.filters.rank.mean(signal, skimage.morphology.disk(20), mask=esf_sand)
    mean_c = skimage.filters.rank.mean(signal, skimage.morphology.disk(20), mask=c_sand)
    mean_rest = skimage.filters.rank.mean(signal, skimage.morphology.disk(20), mask=rest)
    mean_esf[~esf_sand] = 0
    mean_c[~c_sand] = 0
    mean_rest[~rest] = 0

    mean = mean_esf + mean_c + mean_rest
    rescaled_mean = 0.097 / 0.076 *  mean_esf + mean_c + 0.11 / 0.13 * mean_rest

if False:
    plt.figure()
    plt.imshow(mean_esf)
    plt.figure()
    plt.imshow(mean_c)
    plt.figure()
    plt.imshow(mean_rest)
    plt.figure()
    plt.imshow(mean)
    plt.figure()
    plt.imshow(rescaled_mean)

    plt.show()

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
    contour_mask_esf = skimage.morphology.binary_dilation(contour_mask_esf, np.ones((band_size,band_size),np.uint8))
    contour_mask_esf = skimage.img_as_bool(contour_mask_esf)
    
    contours_c, _ = cv2.findContours(
        100 * skimage.img_as_ubyte(c_sand),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contour_mask_c = contours_to_mask(contours_c)
    contour_mask_c = skimage.morphology.binary_dilation(contour_mask_c, np.ones((band_size,band_size),np.uint8))
    contour_mask_c = skimage.img_as_bool(contour_mask_c)
    
    contours_rest, _ = cv2.findContours(
        skimage.img_as_ubyte(rest),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contour_mask_rest = contours_to_mask(contours_rest)
    contour_mask_rest = skimage.morphology.binary_dilation(contour_mask_rest, np.ones((band_size,band_size),np.uint8))
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

# Restrict to layers
signal_esf = np.zeros(signal.shape[:2], dtype=signal.dtype)
signal_c = np.zeros(signal.shape[:2], dtype=signal.dtype)
signal_rest = np.zeros(signal.shape[:2], dtype=signal.dtype)

signal_esf[esf_sand] = signal[esf_sand]
signal_c[c_sand] = signal[c_sand]
signal_rest[rest] = signal[rest]

# Simple rescale test

original_signal = signal_esf + signal_c + signal_rest
rescaled_signal = 0.097 / 0.076 * signal_esf + signal_c + 0.11 / 0.13 * signal_rest

## Add a tiny bit of (gaussian) smoothing
#mean_esf = skimage.filters.rank.mean(signal, skimage.morphology.disk(20), mask = esf_sand)
#mean_c = skimage.filters.rank.mean(signal, skimage.morphology.disk(20), mask = c_sand)
#mean_rest = skimage.filters.rank.mean(signal, skimage.morphology.disk(20), mask = rest)
#
#if True:
#    plt.figure()
#    plt.imshow(mean_esf)
#    plt.figure()
#    plt.imshow(mean_c)
#    plt.figure()
#    plt.imshow(mean_rest)
#    plt.show()

# Histogramm analysis : ESF / C before and after scaling - gaussian
if False:
    roi_esf = np.logical_and(esf_sand, boundary_esf_c)
    roi_c = np.logical_and(c_sand, boundary_esf_c)

    original_signal_esf = skimage.filters.gaussian(original_signal[roi_esf], sigma=0.1)
    original_signal_c = skimage.filters.gaussian(original_signal[roi_c], sigma=0.1)

    rescaled_signal_esf = 0.097 / 0.076 * skimage.filters.gaussian(original_signal[roi_esf], sigma=0.1)
    rescaled_signal_c = skimage.filters.gaussian(original_signal[roi_c], sigma=0.1)

    max_value = np.max(original_signal_c)
    range_min, range_max = 0.1 * max_value, max_value
    range_linspace = np.linspace(range_min, range_max, 100)

    hist_esf_original, _ = np.histogram(original_signal_esf, bins=100, range = (range_min, range_max))
    hist_c_original, _ = np.histogram(original_signal_c, bins=100, range = (range_min, range_max))
    
    hist_esf_rescaled, _ = np.histogram(rescaled_signal_esf, bins=100, range = (range_min, range_max))
    hist_c_rescaled, _ = np.histogram(rescaled_signal_c, bins=100, range = (range_min, range_max))

    plt.figure()
    plt.plot(range_linspace, hist_esf_original)
    plt.plot(range_linspace, hist_c_original)

    plt.figure()
    plt.plot(range_linspace, hist_esf_rescaled)
    plt.plot(range_linspace, hist_c_rescaled)

    plt.show()
    assert False

# Histogramm analysis : ESF / C before and after scaling - tvd
if False:
    roi_esf = np.logical_and(esf_sand, boundary_esf_c)
    roi_c = np.logical_and(c_sand, boundary_esf_c)

    tv_signal = skimage.restoration.denoise_tv_bregman(original_signal, weight = 1)
    
    original_signal_esf = tv_signal[roi_esf]
    original_signal_c = tv_signal[roi_c]

    rescaled_signal_esf = 0.084 / 0.076 * original_signal_esf
    rescaled_signal_c = original_signal_c

    max_value = np.max(original_signal_c)
    range_min, range_max = 0.1 * max_value, max_value
    range_linspace = np.linspace(range_min, range_max, 100)

    hist_esf_original, _ = np.histogram(original_signal_esf, bins=100, range = (range_min, range_max))
    hist_c_original, _ = np.histogram(original_signal_c, bins=100, range = (range_min, range_max))
    
    hist_esf_rescaled, _ = np.histogram(rescaled_signal_esf, bins=100, range = (range_min, range_max))
    hist_c_rescaled, _ = np.histogram(rescaled_signal_c, bins=100, range = (range_min, range_max))

    plt.figure()
    plt.plot(range_linspace, hist_esf_original)
    plt.plot(range_linspace, hist_c_original)

    plt.figure()
    plt.plot(range_linspace, hist_esf_rescaled)
    plt.plot(range_linspace, hist_c_rescaled)

    plt.show()
    assert False

# Histogramm analysis : ESF / C before and after scaling - median
if True:
    roi_esf = np.logical_and(esf_sand, boundary_esf_c)
    roi_c = np.logical_and(c_sand, boundary_esf_c)

    range_min, range_max = 0.05 * np.max(median[roi_c]), np.max(median[roi_c])
    range_linspace = np.linspace(range_min, range_max, 100)

    hist_esf_original, _ = np.histogram(median[roi_esf], bins=100, range = (range_min, range_max))
    hist_c_original, _ = np.histogram(median[roi_c], bins=100, range = (range_min, range_max))

    hist_esf_original = ndi.gaussian_filter1d(hist_esf_original, 10)
    hist_c_original = ndi.gaussian_filter1d(hist_c_original, 10)

    plt.figure()
    plt.plot(range_linspace, hist_esf_original)
    plt.plot(range_linspace, hist_c_original)

    peak_esf_original = spsignal.find_peaks(hist_esf_original)
    peak_c_original = spsignal.find_peaks(hist_c_original)

    esf_arg_max = range_linspace[peak_esf_original[0]]
    c_arg_max = range_linspace[peak_c_original[0]]

    print("esf/c scaling (median)", c_arg_max / esf_arg_max)
    esf_c_scaling = c_arg_max / esf_arg_max

    plt.show()

# Histogramm analysis : C / Rest before and after scaling - median
if True:
    roi_c = np.logical_and(c_sand, boundary_c_rest)
    roi_rest = np.logical_and(rest, boundary_c_rest)

    max_value = max(np.max(median[roi_rest]), np.max(median[roi_c]))
    range_min, range_max = 0.05 * max_value, max_value
    range_linspace = np.linspace(range_min, range_max, 100)

    hist_rest_original, _ = np.histogram(median[roi_rest], bins=100, range = (range_min, range_max))
    hist_c_original, _ = np.histogram(median[roi_c], bins=100, range = (range_min, range_max))

    hist_rest_original = ndi.gaussian_filter1d(hist_rest_original, 10)
    hist_c_original = ndi.gaussian_filter1d(hist_c_original, 10)

    peak_rest_original = spsignal.find_peaks(hist_rest_original)
    peak_c_original = spsignal.find_peaks(hist_c_original)

    rest_arg_max = range_linspace[peak_rest_original[0]]
    c_arg_max = range_linspace[peak_c_original[0]]

    print("rest/c scaling (median)", c_arg_max / rest_arg_max)
    rest_c_scaling = c_arg_max / rest_arg_max

    hist_rest_rescaled, _ = np.histogram(c_arg_max / rest_arg_max * median[roi_rest], bins=100, range = (range_min, range_max))
    hist_rest_rescaled = ndi.gaussian_filter1d(hist_rest_rescaled, 10)

    plt.figure()
    plt.plot(range_linspace, hist_rest_original)
    plt.plot(range_linspace, hist_c_original)
    plt.plot(range_linspace, hist_rest_rescaled)

    plt.show()

# Histogramm analysis : ESF / C before and after scaling - mean
if False:
    roi_esf = np.logical_and(esf_sand, boundary_esf_c)
    roi_c = np.logical_and(c_sand, boundary_esf_c)

    range_min, range_max = 0.01 * np.max(mean[roi_c]), np.max(mean[roi_c])
    range_linspace = np.linspace(range_min, range_max, 100)

    hist_esf_original, _ = np.histogram(mean[roi_esf], bins=100, range = (range_min, range_max))
    hist_c_original, _ = np.histogram(mean[roi_c], bins=100, range = (range_min, range_max))
    
    hist_esf_rescaled, _ = np.histogram(rescaled_mean[roi_esf], bins=100, range = (range_min, range_max))
    hist_c_rescaled, _ = np.histogram(rescaled_mean[roi_c], bins=100, range = (range_min, range_max))

    plt.figure()
    plt.plot(range_linspace, hist_esf_original)
    plt.plot(range_linspace, hist_c_original)

    plt.figure()
    plt.plot(range_linspace, hist_esf_rescaled)
    plt.plot(range_linspace, hist_c_rescaled)


#    plt.figure()
#    plt.plot(np.linspace(np.min(original_signal[roi_esf]), np.max(original_signal[roi_esf]), 100), np.histogram(original_signal[roi_esf], bins=100)[0])
#    plt.plot(np.linspace(np.min(original_signal[roi_c]), np.max(original_signal[roi_c]), 100), np.histogram(original_signal[roi_c], bins=100)[0])
#    plt.figure()
#    plt.plot(np.linspace(np.min(rescaled_signal[roi_esf]), np.max(rescaled_signal[roi_esf]), 100), np.histogram(rescaled_signal[roi_esf], bins=100)[0])
#    plt.plot(np.linspace(np.min(rescaled_signal[roi_c]), np.max(rescaled_signal[roi_c]), 100), np.histogram(rescaled_signal[roi_c], bins=100)[0])

    plt.show()

if True:
    roi_esf = np.logical_and(esf_sand, boundary_esf_c)
    roi_c = np.logical_and(c_sand, boundary_esf_c)

    range_min, range_max = 0.05 * np.max(original_signal[roi_c]), np.max(original_signal[roi_c])
    range_linspace = np.linspace(range_min, range_max, 100)

    hist_esf_original, _ = np.histogram(original_signal[roi_esf], bins=100, range = (range_min, range_max))
    hist_c_original, _ = np.histogram(original_signal[roi_c], bins=100, range = (range_min, range_max))
    
    hist_esf_original = ndi.gaussian_filter1d(hist_esf_original, 10)
    hist_c_original = ndi.gaussian_filter1d(hist_c_original, 10)

    plt.figure()
    plt.plot(range_linspace, hist_esf_original)
    plt.plot(range_linspace, hist_c_original)

    peak_esf_original = spsignal.find_peaks(hist_esf_original)
    peak_c_original = spsignal.find_peaks(hist_c_original)

    esf_arg_max = range_linspace[peak_esf_original[0]]
    c_arg_max = range_linspace[peak_c_original[0]]

    print("esf/c scaling", c_arg_max / esf_arg_max)
    esf_c_scaling = c_arg_max / esf_arg_max

    hist_esf_rescaled, _ = np.histogram(c_arg_max / esf_arg_max * original_signal[roi_esf], bins=100, range = (range_min, range_max))
    hist_c_rescaled, _ = np.histogram(original_signal[roi_c], bins=100, range = (range_min, range_max))

    plt.figure()
    plt.plot(range_linspace, hist_esf_rescaled)
    plt.plot(range_linspace, hist_c_rescaled)

    hist_esf_rescaled = ndi.gaussian_filter1d(hist_esf_rescaled, 10)
    hist_c_rescaled = ndi.gaussian_filter1d(hist_c_rescaled, 10)

    plt.figure()
    plt.plot(range_linspace, hist_esf_rescaled)
    plt.plot(range_linspace, hist_c_rescaled)

    plt.show()

if True:
    roi_rest = np.logical_and(rest, boundary_c_rest)
    roi_c = np.logical_and(c_sand, boundary_c_rest)

    bins = 1000

    range_min, range_max = 0.05 * np.max(original_signal[rest]), np.max(original_signal[rest])
    range_linspace = np.linspace(range_min, range_max, bins)

    hist_rest_original, _ = np.histogram(original_signal[roi_rest], bins=bins, range = (range_min, range_max))
    hist_c_original, _ = np.histogram(original_signal[roi_c], bins=bins, range = (range_min, range_max))
    
    hist_rest_original = ndi.gaussian_filter1d(hist_rest_original, 10)
    hist_c_original = ndi.gaussian_filter1d(hist_c_original, 10)

    plt.figure()
    plt.plot(range_linspace, hist_rest_original)
    plt.plot(range_linspace, hist_c_original)

    peak_rest_original = spsignal.find_peaks(hist_rest_original)
    peak_c_original = spsignal.find_peaks(hist_c_original)

    rest_arg_max = range_linspace[peak_rest_original[0]]
    c_arg_max = range_linspace[peak_c_original[0]]

    print("rest/c scaling", c_arg_max / rest_arg_max)
    rest_c_scaling = c_arg_max / rest_arg_max

    hist_rest_rescaled, _ = np.histogram(c_arg_max / rest_arg_max * original_signal[roi_rest], bins=bins, range = (range_min, range_max))
    hist_c_rescaled, _ = np.histogram(original_signal[roi_c], bins=bins, range = (range_min, range_max))

    plt.figure()
    plt.plot(range_linspace, hist_rest_rescaled)
    plt.plot(range_linspace, hist_c_rescaled)

    hist_rest_rescaled = ndi.gaussian_filter1d(hist_rest_rescaled, 10)
    hist_c_rescaled = ndi.gaussian_filter1d(hist_c_rescaled, 10)

    plt.figure()
    plt.plot(range_linspace, hist_rest_rescaled)
    plt.plot(range_linspace, hist_c_rescaled)

    plt.show()


# Reference
median_signal = skimage.filters.rank.median(signal, skimage.morphology.disk(20))
plt.figure()
plt.imshow(median_signal)

# Rescaling based on mean of median at the boundary.
rescaled_signal = np.copy(signal)
rescaled_signal[esf_sand] *= 1.072
rescaled_signal[rest] *= 0.93
median_rescaled_signal = skimage.filters.rank.median(rescaled_signal, skimage.morphology.disk(20))
plt.figure()
plt.imshow(median_rescaled_signal)

# Rescaling based on mean at the boundary.
rescaled_signal = np.copy(signal)
rescaled_signal[esf_sand] *= 1.05
rescaled_signal[rest] *= 0.93
median_rescaled_signal = skimage.filters.rank.median(rescaled_signal, skimage.morphology.disk(20))
plt.figure()
plt.imshow(median_rescaled_signal)

# Rescaling based on histogram analysis using bins = 1000 (without any filter)
rescaled_signal = np.copy(signal)
rescaled_signal[esf_sand] *= 1.20
rescaled_signal[rest] *= 0.95
median_rescaled_signal = skimage.filters.rank.median(rescaled_signal, skimage.morphology.disk(20))
plt.figure()
plt.imshow(median_rescaled_signal)

plt.show()

#    plt.figure()
#    plt.plot(np.linspace(np.min(original_signal[roi_esf]), np.max(original_signal[roi_esf]), 100), np.histogram(original_signal[roi_esf], bins=100)[0])
#    plt.plot(np.linspace(np.min(original_signal[roi_c]), np.max(original_signal[roi_c]), 100), np.histogram(original_signal[roi_c], bins=100)[0])
#    plt.figure()
#    plt.plot(np.linspace(np.min(rescaled_signal[roi_esf]), np.max(rescaled_signal[roi_esf]), 100), np.histogram(rescaled_signal[roi_esf], bins=100)[0])
#    plt.plot(np.linspace(np.min(rescaled_signal[roi_c]), np.max(rescaled_signal[roi_c]), 100), np.histogram(rescaled_signal[roi_c], bins=100)[0])

# TODO continue here.
assert False

# Histogramm analysis : C / Rest
if False:
    roi_upper = (slice(1340, 1390), slice(5050, 5400))
    roi_lower = (slice(1440, 1500), slice(5050, 5400))
    plt.figure()
    plt.plot(np.linspace(np.min(signal[roi_upper]), np.max(signal[roi_upper]), 100), np.histogram(signal[roi_upper], bins=100)[0])
    plt.plot(np.linspace(np.min(signal[roi_lower]), np.max(signal[roi_lower]), 100), np.histogram(signal[roi_lower], bins=100)[0])
    plt.show()

max_original = np.max(original_signal)

if True:
    plt.figure()
    plt.imshow(original_signal + max_original * (boundary_esf_c + boundary_c_rest))
    plt.figure()
    plt.imshow(rescaled_signal)
    plt.imshow(original_signal + max_original * (boundary_esf_c + boundary_c_rest))

plt.show()

# Try to measure discontinuity - use same approach as for segmentation - try to minimize sharp edges...
median_original = skimage.filters.rank.median(original_signal, skimage.morphology.disk(20)) 
median_rescaled = skimage.filters.rank.median(rescaled_signal, skimage.morphology.disk(20))
resized_original = skimage.transform.rescale(median_original, 0.1, anti_aliasing = False)
resized_rescaled = skimage.transform.rescale(median_rescaled, 0.1, anti_aliasing = False)
edges_original = skimage.filters.scharr(resized_original)
edges_rescaled = skimage.filters.scharr(resized_rescaled)

plt.figure()
plt.imshow(median_original)
plt.figure()
plt.imshow(median_rescaled)
plt.figure()
plt.imshow(resized_original)
plt.figure()
plt.imshow(resized_rescaled)
plt.figure()
plt.imshow(edges_original)
plt.figure()
plt.imshow(edges_rescaled)



plt.show()


assert False

# Take the mean_bilateral - take into account the grain size (more or less)
mean_b_esf = skimage.filters.rank.mean_bilateral(signal_esf, skimage.morphology.disk(5), mask = esf_sand)
mean_b_c = skimage.filters.rank.mean_bilateral(signal_c, skimage.morphology.disk(10), mask = c_sand)
mean_b_rest = skimage.filters.rank.mean_bilateral(signal_rest, skimage.morphology.disk(20), mask = rest)

if True:
    plt.figure()
    plt.imshow(mean_b_esf)
    plt.figure()
    plt.imshow(mean_b_c)
    plt.figure()
    plt.imshow(mean_b_rest)
    plt.figure()
    plt.imshow(mean_b_esf + mean_b_c + mean_b_rest)

plt.show()
