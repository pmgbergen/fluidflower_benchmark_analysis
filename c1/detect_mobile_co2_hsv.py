import cv2
import skimage
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import daria

images = [
    "segmentation_develop/signal/hsv_signal_10.npy", # 0
    "segmentation_develop/signal/hsv_signal_20.npy", # 1
    "segmentation_develop/signal/hsv_signal_30.npy", # 2
    "segmentation_develop/signal/hsv_signal_40.npy", # 3
    "segmentation_develop/signal/hsv_signal_50.npy", # 4
    "segmentation_develop/signal/hsv_signal_60.npy", # 5
    "segmentation_develop/signal/hsv_signal_70.npy", # 6
    "segmentation_develop/signal/hsv_signal_80.npy", # 7
    "segmentation_develop/signal/hsv_signal_90.npy", # 8
    "segmentation_develop/signal/hsv_signal_100.npy", # 9
    "segmentation_develop/signal/hsv_signal_110.npy", # 10
    "segmentation_develop/signal/hsv_signal_115.npy", # 11
    "segmentation_develop/signal/hsv_signal_120.npy", # 12
    "segmentation_develop/signal/hsv_signal_125.npy", # 13
]


co2_images = [
    "segmentation_develop/co2/co2_10.npy",
    "segmentation_develop/co2/co2_20.npy",
    "segmentation_develop/co2/co2_30.npy",
    "segmentation_develop/co2/co2_40.npy",
    "segmentation_develop/co2/co2_50.npy",
    "segmentation_develop/co2/co2_60.npy",
    "segmentation_develop/co2/co2_70.npy",
    "segmentation_develop/co2/co2_80.npy",
    "segmentation_develop/co2/co2_90.npy",
    "segmentation_develop/co2/co2_100.npy",
    "segmentation_develop/co2/co2_110.npy",
    "segmentation_develop/co2/co2_115.npy",
    "segmentation_develop/co2/co2_120.npy",
    "segmentation_develop/co2/co2_125.npy",
]

key = 13

# Masks
esf = np.load("segmentation_develop/esf.npy").astype(bool)
co2 = np.load(co2_images[key]).astype(bool)
labels = np.load("labels.npy")

upper = labels == 2
lower = labels == 5

# Full diff
diff = np.load(images[key])
shape = diff.shape

# Active set
active = np.logical_and(co2, ~esf)

# Restrict to active set
diff[~active] = 0

# Consider full signal to restrict in HSV space
signal = diff

# Apply analysis to find threshold values
analysis = False

# ROI analysis for 50
if key == 4 and analysis:

    plt.imshow(signal[:,:,2])
    plt.show()

    roi_dissolved = [
        (slice(3280, 3420), slice(2200, 2550)),
        (slice(3240, 3320), slice(4250, 4450)),
        (slice(1750, 2050), slice(4650, 4900)),
    ]
    
    roi_gas = [
        (slice(3000, 3150), slice(4600, 6000)),
        (slice(1260, 1340), slice(5200, 6000)),
        (slice(1440, 1480), slice(5400, 5900))
    ]
    
    roi_nitro = [
    ]
    
    hist_dissolved = [[np.histogram(signal[roi][:,:,i], bins = 100)[0] for roi in roi_dissolved] for i in range(3)]
    hist_gas = [[np.histogram(signal[roi][:,:,i], bins = 100)[0] for roi in roi_gas] for i in range(3)]
    hist_nitro = [[np.histogram(signal[roi][:,:,i], bins = 100)[0] for roi in roi_nitro] for i in range(3)]
    
    vals_dissolved = [[np.linspace(np.min(signal[roi][:,:,i]), np.max(signal[roi][:,:,i]), 100) for roi in roi_dissolved] for i in range(3)]
    vals_gas = [[np.linspace(np.min(signal[roi][:,:,i]), np.max(signal[roi][:,:,i]), 100) for roi in roi_gas] for i in range(3)]
    vals_nitro = [[np.linspace(np.min(signal[roi][:,:,i]), np.max(signal[roi][:,:,i]), 100) for roi in roi_nitro] for i in range(3)]
    
    for j in range(3):
    
        plt.figure()
        for i in range(len(roi_dissolved)):
            plt.plot(vals_dissolved[j][i], hist_dissolved[j][i], color = 'b')
        plt.figure()
        for i in range(len(roi_gas)):
            plt.plot(vals_gas[j][i], hist_gas[j][i], color = 'r')
        plt.figure()
        for i in range(len(roi_nitro)):
            plt.plot(vals_nitro[j][i], hist_nitro[j][i], color = 'g')
    
    plt.show()

# ROI analysis for 110
if key == 10 and analysis:
    roi_dissolved = [
        (slice(1300, 2490), slice(4877, 5150)),
        (slice(3100, 3800), slice(6000, 6800)),
    ]
    
    roi_gas = [(slice(2935, 2955), slice(4520, 5950))]
    
    roi_nitro = [
        (slice(2900, 2920), slice(4700, 5400))
    ]
    
    hist_dissolved = [[np.histogram(signal[roi][:,:,i], bins = 100)[0] for roi in roi_dissolved] for i in range(3)]
    hist_gas = [[np.histogram(signal[roi][:,:,i], bins = 100)[0] for roi in roi_gas] for i in range(3)]
    hist_nitro = [[np.histogram(signal[roi][:,:,i], bins = 100)[0] for roi in roi_nitro] for i in range(3)]
    
    vals_dissolved = [[np.linspace(np.min(signal[roi][:,:,i]), np.max(signal[roi][:,:,i]), 100) for roi in roi_dissolved] for i in range(3)]
    vals_gas = [[np.linspace(np.min(signal[roi][:,:,i]), np.max(signal[roi][:,:,i]), 100) for roi in roi_gas] for i in range(3)]
    vals_nitro = [[np.linspace(np.min(signal[roi][:,:,i]), np.max(signal[roi][:,:,i]), 100) for roi in roi_nitro] for i in range(3)]
    
    for j in range(3):
    
        plt.figure()
        for i in range(len(roi_dissolved)):
            plt.plot(vals_dissolved[j][i], hist_dissolved[j][i], color = 'b')
        plt.figure()
        for i in range(len(roi_gas)):
            plt.plot(vals_gas[j][i], hist_gas[j][i], color = 'r')
        plt.figure()
        for i in range(len(roi_nitro)):
            plt.plot(vals_nitro[j][i], hist_nitro[j][i], color = 'g')
    
    plt.show()

# ROI analysis for 125
if key == 13 and analysis:
    roi_dissolved = [
        (slice(3200, 3800), slice(3500, 5250)),
        (slice(3700, 4100), slice(6700, 7000)),
        (slice(1400, 2500), slice(6000, 6450))
    ]
    
    roi_gas = []
    
    roi_nitro = [
        (slice(2900, 2915), slice(5100, 5500))
    ]
    
    hist_dissolved = [[np.histogram(signal[roi][:,:,i], bins = 100)[0] for roi in roi_dissolved] for i in range(3)]
    hist_gas = [[np.histogram(signal[roi][:,:,i], bins = 100)[0] for roi in roi_gas] for i in range(3)]
    hist_nitro = [[np.histogram(signal[roi][:,:,i], bins = 100)[0] for roi in roi_nitro] for i in range(3)]
    
    vals_dissolved = [[np.linspace(np.min(signal[roi][:,:,i]), np.max(signal[roi][:,:,i]), 100) for roi in roi_dissolved] for i in range(3)]
    vals_gas = [[np.linspace(np.min(signal[roi][:,:,i]), np.max(signal[roi][:,:,i]), 100) for roi in roi_gas] for i in range(3)]
    vals_nitro = [[np.linspace(np.min(signal[roi][:,:,i]), np.max(signal[roi][:,:,i]), 100) for roi in roi_nitro] for i in range(3)]
    
    for j in range(3):
    
        plt.figure()
        for i in range(len(roi_dissolved)):
            plt.plot(vals_dissolved[j][i], hist_dissolved[j][i], color = 'b')
        plt.figure()
        for i in range(len(roi_gas)):
            plt.plot(vals_gas[j][i], hist_gas[j][i], color = 'r')
        plt.figure()
        for i in range(len(roi_nitro)):
            plt.plot(vals_nitro[j][i], hist_nitro[j][i], color = 'g')
    
    plt.show()


# Restriction in HSV space - not much to be done on the full space anymore
h_signal = signal[:,:,0]
s_signal = signal[:,:,1]
v_signal = signal[:,:,2]

mobile_co2_mask = np.logical_or(
    np.logical_and(upper, v_signal > 0.3),
    np.logical_and(lower, v_signal > 0.3),
    #np.logical_and(upper, v_signal < 0.25),
    #np.logical_and(lower, v_signal < 0.25),
)

h_signal[~mobile_co2_mask] = 0
s_signal[~mobile_co2_mask] = 0
v_signal[~mobile_co2_mask] = 0

plt.figure()
plt.imshow(h_signal)
plt.figure()
plt.imshow(s_signal)
plt.figure()
plt.imshow(v_signal)
plt.show()

# For prior use v_signal, s_signal for posterior possibly
signal = v_signal
#signal = s_signal

# Smoothing
apply_presmoothing = True
#presmoothing = {
#    "resize": 0.5,
#    # Chambolle
#    "method": "chambolle",
#    "weight": 1,
#    "eps": 1e-4,
#    "max_num_iter": 100,
#    ## Bregman
#    #"method": "anisotropic bregman",
#    #"weight": 0.5,
#    #"eps": 1e-4,
#    #"max_num_iter": 1000,
#}
presmoothing = {
    "resize": 0.5,
    # Chambolle
    "method": "chambolle",
    "weight": 1,
    "eps": 1e-4,
    "max_num_iter": 100,
    ## Bregman
    #"method": "anisotropic bregman",
    #"weight": 0.5,
    #"eps": 1e-4,
    #"max_num_iter": 1000,
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

# Apply thresholding
active_img = np.ravel(smooth_signal)[np.ravel(active)]
thresh = skimage.filters.threshold_otsu(active_img)
print(thresh)
# For s_signal use 0.03
# For v_signal use 0.08
mask = smooth_signal > 0.05

# Remove small objects
mask = skimage.morphology.remove_small_objects(mask, 50**2) 

# Clean signal
clean_signal = smooth_signal
clean_signal[~active] = 0
clean_signal[~mask] = 0

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
#factor = 8
#img = cv2.resize(img, (factor * 280,factor * 150), interpolation = cv2.INTER_AREA)
#
#plt.figure()
#plt.imshow(img)
#
#plt.show()
#
#plt.figure()
#plt.imshow(img)
##img = skimage.restoration.denoise_tv_chambolle(img, 0.01, eps = 1e-5, max_num_iter = 1000)
##img = skimage.restoration.denoise_tv_bregman(img, weight=100, max_num_iter=1000, eps=1e-5, isotropic=False)
#
#plt.figure()
#plt.imshow(img)
#
#plt.show()
#
## Resize to original size
#img = cv2.resize(img, tuple(reversed(original.shape[:2])))

img = skimage.filters.rank.median(img, skimage.morphology.disk(5))
#img = skimage.restoration.denoise_tv_chambolle(img, 0.01, eps = 1e-5, max_num_iter = 1000)
img = skimage.restoration.denoise_tv_bregman(img, weight=100, max_num_iter=1000, eps=1e-5, isotropic=False)

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
outside = skimage.morphology.binary_dilation(mask, footprint=np.ones((5,5)))
outside[active] = mask[active]

# Remove small holes and small objects
mask1 = np.copy(mask)
mask1 = skimage.filters.rank.median(mask1, skimage.morphology.disk(10))
plt.figure()
plt.imshow(mask1)
mask1 = skimage.morphology.binary_closing(skimage.util.img_as_bool(mask1), footprint = np.ones((10,10)))
plt.figure()
plt.imshow(mask1)
mask1 = skimage.morphology.remove_small_holes(mask1, 50**2)
plt.figure()
plt.imshow(mask1)
mask1 = ndi.morphology.binary_opening(mask1, structure=np.ones((5,5)))
plt.figure()
plt.imshow(mask1)
mask1 = skimage.util.img_as_bool(cv2.resize(mask1.astype(np.float32), tuple(reversed(co2.shape))))
plt.figure()
plt.imshow(mask1)
plt.show()

