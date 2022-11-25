import cv2
import daria
import matplotlib.pyplot as plt
import numpy as np
import skimage
from scipy import ndimage as ndi

images = [
    "segmentation_develop/signal/hsv_signal_10.npy",  # 0
    "segmentation_develop/signal/hsv_signal_20.npy",  # 1
    "segmentation_develop/signal/hsv_signal_30.npy",  # 2
    "segmentation_develop/signal/hsv_signal_40.npy",  # 3
    "segmentation_develop/signal/hsv_signal_50.npy",  # 4
    "segmentation_develop/signal/hsv_signal_60.npy",  # 5
    "segmentation_develop/signal/hsv_signal_70.npy",  # 6
    "segmentation_develop/signal/hsv_signal_80.npy",  # 7
    "segmentation_develop/signal/hsv_signal_90.npy",  # 8
    "segmentation_develop/signal/hsv_signal_100.npy",  # 9
    "segmentation_develop/signal/hsv_signal_110.npy",  # 10
    "segmentation_develop/signal/hsv_signal_115.npy",  # 11
    "segmentation_develop/signal/hsv_signal_120.npy",  # 12
    "segmentation_develop/signal/hsv_signal_125.npy",  # 13
]

key = 6

# Full diff
signal = np.load(images[key])
shape = signal.shape

# plt.figure()
# plt.imshow(signal)
# plt.show()

analysis = False
if analysis:
    rgb_img = np.load("segmentation_develop/signal/hsv_pre_signal_110.npy")
    plt.imshow(rgb_img)
    plt.show()


# ROI analysis for 110
if key == 10 and analysis:
    roi_dissolved = [
        (slice(1300, 2490), slice(4877, 5150)),
        (slice(3100, 3800), slice(6000, 6800)),
    ]

    roi_gas = [(slice(2935, 2955), slice(4520, 5950))]

    roi_nitro = [(slice(2900, 2920), slice(4700, 5400))]

    hist_dissolved = [
        [np.histogram(signal[roi][:, :, i], bins=100)[0] for roi in roi_dissolved]
        for i in range(3)
    ]
    hist_gas = [
        [np.histogram(signal[roi][:, :, i], bins=100)[0] for roi in roi_gas]
        for i in range(3)
    ]
    hist_nitro = [
        [np.histogram(signal[roi][:, :, i], bins=100)[0] for roi in roi_nitro]
        for i in range(3)
    ]

    vals_dissolved = [
        [
            np.linspace(np.min(signal[roi][:, :, i]), np.max(signal[roi][:, :, i]), 100)
            for roi in roi_dissolved
        ]
        for i in range(3)
    ]
    vals_gas = [
        [
            np.linspace(np.min(signal[roi][:, :, i]), np.max(signal[roi][:, :, i]), 100)
            for roi in roi_gas
        ]
        for i in range(3)
    ]
    vals_nitro = [
        [
            np.linspace(np.min(signal[roi][:, :, i]), np.max(signal[roi][:, :, i]), 100)
            for roi in roi_nitro
        ]
        for i in range(3)
    ]

    for j in range(3):

        plt.figure()
        for i in range(len(roi_dissolved)):
            plt.plot(vals_dissolved[j][i], hist_dissolved[j][i], color="b")
        plt.figure()
        for i in range(len(roi_gas)):
            plt.plot(vals_gas[j][i], hist_gas[j][i], color="r")
        plt.figure()
        for i in range(len(roi_nitro)):
            plt.plot(vals_nitro[j][i], hist_nitro[j][i], color="g")

    plt.show()

# ROI analysis for 125
if key == 13 and analysis:
    roi_dissolved = [
        (slice(3200, 3800), slice(3500, 5250)),
        (slice(3700, 4100), slice(6700, 7000)),
        (slice(1400, 2500), slice(6000, 6450)),
    ]

    roi_gas = []

    roi_nitro = [(slice(2900, 2915), slice(5100, 5500))]

    hist_dissolved = [
        [np.histogram(signal[roi][:, :, i], bins=100)[0] for roi in roi_dissolved]
        for i in range(3)
    ]
    hist_gas = [
        [np.histogram(signal[roi][:, :, i], bins=100)[0] for roi in roi_gas]
        for i in range(3)
    ]
    hist_nitro = [
        [np.histogram(signal[roi][:, :, i], bins=100)[0] for roi in roi_nitro]
        for i in range(3)
    ]

    vals_dissolved = [
        [
            np.linspace(np.min(signal[roi][:, :, i]), np.max(signal[roi][:, :, i]), 100)
            for roi in roi_dissolved
        ]
        for i in range(3)
    ]
    vals_gas = [
        [
            np.linspace(np.min(signal[roi][:, :, i]), np.max(signal[roi][:, :, i]), 100)
            for roi in roi_gas
        ]
        for i in range(3)
    ]
    vals_nitro = [
        [
            np.linspace(np.min(signal[roi][:, :, i]), np.max(signal[roi][:, :, i]), 100)
            for roi in roi_nitro
        ]
        for i in range(3)
    ]

    for j in range(3):

        plt.figure()
        for i in range(len(roi_dissolved)):
            plt.plot(vals_dissolved[j][i], hist_dissolved[j][i], color="b")
        plt.figure()
        for i in range(len(roi_gas)):
            plt.plot(vals_gas[j][i], hist_gas[j][i], color="r")
        plt.figure()
        for i in range(len(roi_nitro)):
            plt.plot(vals_nitro[j][i], hist_nitro[j][i], color="g")

    plt.show()

# plt.imshow(signal[:,:,0])
# plt.show()

h_signal = signal[:, :, 0]
s_signal = signal[:, :, 1]
v_signal = signal[:, :, 2]

# co2_mask = np.logical_and(
#    np.logical_and(
#        np.logical_and(h_signal > 14, h_signal < 255),
#        s_signal > 0.01
#    ),
#    v_signal > 0.1
# )

co2_mask = np.logical_and(h_signal > 14, h_signal < 70)

h_signal[~co2_mask] = 0
s_signal[~co2_mask] = 0
v_signal[~co2_mask] = 0

# Pick v signal
signal = v_signal

# Smoothing to signal
apply_presmoothing = True
presmoothing = {
    "resize": 0.5,
    # Chambolle
    "method": "chambolle",
    "weight": 0.5,
    "eps": 1e-4,
    "max_num_iter": 100,
    ## Bregman
    # "method": "anisotropic bregman",
    # "weight": 1,
    # "eps": 1e-4,
    # "max_num_iter": 100,
}


# Apply presmoothing
if apply_presmoothing:
    # Resize image
    smooth_signal = cv2.resize(
        signal.astype(np.float32),
        None,
        fx=presmoothing["resize"],
        fy=presmoothing["resize"],
    )

    # Apply TVD
    if presmoothing["method"] == "chambolle":
        smooth_signal = skimage.restoration.denoise_tv_chambolle(
            smooth_signal,
            weight=presmoothing["weight"],
            eps=presmoothing["eps"],
            max_num_iter=presmoothing["max_num_iter"],
        )
    elif presmoothing["method"] == "anisotropic bregman":
        smooth_signal = skimage.restoration.denoise_tv_bregman(
            smooth_signal,
            weight=presmoothing["weight"],
            eps=presmoothing["eps"],
            max_num_iter=presmoothing["max_num_iter"],
            isotropic=False,
        )
    else:
        raise ValueError(f"Method {presmoothing['method']} not supported.")

    # Resize to original size
    smooth_signal = cv2.resize(smooth_signal, tuple(reversed(shape[:2])))

else:
    smooth_signal = signal

# Apply threshold
thresh = 0.05
co2_mask = smooth_signal > thresh

plt.figure()
plt.imshow(co2_mask)

plt.figure()
plt.imshow(smooth_signal)
plt.show()

## Threshold HSV
# rgb_signal = cv2.cvtColor(signal, cv2.COLOR_HSV2RGB)
##plt.figure()
##plt.imshow(rgb_signal)
# rgb_signal = skimage.util.img_as_ubyte(rgb_signal)
##plt.figure()
##plt.imshow(rgb_signal)
##plt.show()
# signal = cv2.cvtColor(rgb_signal, cv2.COLOR_RGB2HSV)
# co2_low = (10, 1, 1)
# co2_high = (255, 255, 255)
# co2_mask = skimage.util.img_as_bool(
#    cv2.inRange(signal, co2_low, co2_high)
# )

plt.figure()
plt.imshow(co2_mask)
# Remove small objects
co2_mask = skimage.morphology.remove_small_objects(co2_mask, min_size=2)

# Define noisy co2 signal
co2 = np.copy(signal)
co2[~co2_mask] = 0

plt.figure()
plt.imshow(co2_mask)
# plt.figure()
# plt.imshow(co2)
plt.show()

# original_signal = np.load(images[key])
# original_signal[~co2_mask] = 0
# v_signal = original_signal[:,:,2]
# plt.figure()
# plt.imshow(v_signal)
# plt.show()

print("Start local covering")

# Loop through patches and fill up
covered_mask = np.zeros(co2_mask.shape[:2], dtype=bool)
size = 10  # in some sense the REV size in pixels
Ny, Nx = co2_mask.shape[:2]
for row in range(int(Ny / size)):
    for col in range(int(Nx / size)):
        roi = (
            slice(row * size, (row + 1) * size),
            slice(col * size, (col + 1) * size),
        )
        covered_mask[roi] = skimage.morphology.convex_hull_image(co2_mask[roi])
co2_mask = covered_mask

plt.figure()
plt.imshow(co2_mask)

# Fill holes
print("Fill holes")
co2_mask = skimage.morphology.remove_small_holes(co2_mask, area_threshold=20**2)

plt.figure()
plt.imshow(co2_mask)
plt.show()

# Work with mask
signal = skimage.util.img_as_float(co2_mask)

# Smoothing
apply_presmoothing = True
presmoothing = {
    "resize": 0.5,
    # Chambolle
    "method": "chambolle",
    "weight": 0.5,
    "eps": 1e-4,
    "max_num_iter": 100,
    ## Bregman
    # "method": "anisotropic bregman",
    # "weight": 1,
    # "eps": 1e-4,
    # "max_num_iter": 100,
}


# Apply presmoothing
if apply_presmoothing:
    # Resize image
    smooth_signal = cv2.resize(
        signal.astype(np.float32),
        None,
        fx=presmoothing["resize"],
        fy=presmoothing["resize"],
    )

    # Apply TVD
    if presmoothing["method"] == "chambolle":
        smooth_signal = skimage.restoration.denoise_tv_chambolle(
            smooth_signal,
            weight=presmoothing["weight"],
            eps=presmoothing["eps"],
            max_num_iter=presmoothing["max_num_iter"],
        )
    elif presmoothing["method"] == "anisotropic bregman":
        smooth_signal = skimage.restoration.denoise_tv_bregman(
            smooth_signal,
            weight=presmoothing["weight"],
            eps=presmoothing["eps"],
            max_num_iter=presmoothing["max_num_iter"],
            isotropic=False,
        )
    else:
        raise ValueError(f"Method {presmoothing['method']} not supported.")

    # Resize to original size
    smooth_signal = cv2.resize(smooth_signal, tuple(reversed(shape[:2])))

else:
    smooth_signal = signal


plt.figure()
plt.imshow(smooth_signal)
plt.show()


# Thresholding
thresh = skimage.filters.threshold_otsu(smooth_signal)
print("otsu", thresh)
thresh = 0.5
mask_co2 = smooth_signal > thresh

plt.figure()
signal = np.load(images[key])
plt.imshow(signal)
plt.figure()
plt.imshow(mask_co2)
plt.show()

assert False

## Choose: R, B, H
plt.figure()
plt.imshow(signal)
plt.show()

# Apply thresholding based on the histogram analysis
thresh_prior = 0.3
thresh_posterior = 0.5
mask = signal > thresh_prior
signal[~mask] = 0

plt.figure()
plt.imshow(signal)
plt.show()

# Smoothing
apply_presmoothing = True
presmoothing = {
    "resize": 0.5,
    ## Chambolle
    # "method": "chambolle",
    # "weight": 0.5,
    # "eps": 1e-4,
    # "max_num_iter": 100,
    # Bregman
    "method": "anisotropic bregman",
    "weight": 0.5,
    "eps": 1e-4,
    "max_num_iter": 1000,
}


# Apply presmoothing
if apply_presmoothing:
    # Resize image
    smooth_signal = cv2.resize(
        signal.astype(np.float32),
        None,
        fx=presmoothing["resize"],
        fy=presmoothing["resize"],
    )

    # Apply TVD
    if presmoothing["method"] == "chambolle":
        smooth_signal = skimage.restoration.denoise_tv_chambolle(
            smooth_signal,
            weight=presmoothing["weight"],
            eps=presmoothing["eps"],
            max_num_iter=presmoothing["max_num_iter"],
        )
    elif presmoothing["method"] == "anisotropic bregman":
        smooth_signal = skimage.restoration.denoise_tv_bregman(
            smooth_signal,
            weight=presmoothing["weight"],
            eps=presmoothing["eps"],
            max_num_iter=presmoothing["max_num_iter"],
            isotropic=False,
        )
    else:
        raise ValueError(f"Method {presmoothing['method']} not supported.")

    # Resize to original size
    smooth_signal = cv2.resize(smooth_signal, tuple(reversed(shape[:2])))

else:
    smooth_signal = signal

## Laplace / gradient info
# laplace = daria.laplace(smooth_signal)
# plt.figure()
# plt.imshow(laplace)
# plt.show()

# Apply thresholding
active_img = np.ravel(smooth_signal)[np.ravel(active)]
thresh = skimage.filters.threshold_otsu(active_img)
print(thresh)
mask = smooth_signal > 0.04

# Clean signal
clean_signal = smooth_signal
clean_signal[~active] = 0

plt.figure()
plt.imshow(clean_signal)
plt.figure()
plt.imshow(mask)

plt.show()

# Final signal
result = smooth_signal

# Resize to reported size (150 x 280)
report = cv2.resize(result, (280, 150))

plt.figure()
plt.imshow(report)
plt.show()

assert False

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
##img = skimage.restoration.denoise_tv_chambolle(img, 0.01, eps = 1e-5, max_num_iter = 1000)
##img = skimage.restoration.denoise_tv_bregman(img, weight=100, max_num_iter=1000, eps=1e-5, isotropic=False)
#
# plt.figure()
# plt.imshow(img)
#
# plt.show()
#
## Resize to original size
# img = cv2.resize(img, tuple(reversed(original.shape[:2])))

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
