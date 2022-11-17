import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage

co2 = np.load("mobile_co2_70.npy")
grad = np.load("grad_70.npy")

# plt.figure()
# plt.imshow(co2)
plt.figure()
plt.imshow(grad)
plt.show()


## Extract concentration map
# mobile_co2_posterior = self.mobile_co2_analysis_posterior(img_copy)
empty_mask = np.zeros(co2.shape, dtype=bool)
# empty_array = np.zeros((*co2.shape,3), dtype=np.uint8)
empty_array = np.zeros(co2.shape, dtype=np.uint8)

# Label the connected regions first
labels, num_labels = skimage.measure.label(co2, return_num=True)
props = skimage.measure.regionprops(labels)

# plt.figure()
# plt.imshow(labels)
# plt.show()

# Investigate each labeled region separately
for label in range(1, num_labels + 1):
    labeled_region = labels == label

    contours, _ = cv2.findContours(
        skimage.util.img_as_ubyte(labeled_region),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    accept = False
    for c in contours:
        c = np.fliplr(np.squeeze(c))
        c = (c[:, 0], c[:, 1])

        print(np.max(grad[c]))

        if np.max(grad[c]) > 0.002:
            accept = True
            break
    print()

    if accept:
        empty_mask[labeled_region] = True

plt.figure()
plt.imshow(empty_mask)
plt.show()

# if np.any(np.logical_and(self.mobile_co2_prior[labeled_region], self.mobile_co2_posterior(labeled_region))):
#    empty_mask[labeled_region] = True
