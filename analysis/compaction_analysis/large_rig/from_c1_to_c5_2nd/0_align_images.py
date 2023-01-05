"""
0. Step. Move marks from c3 to c2 on pixel level. Expect an affine map is sufficient.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage

# Fetch c2 and c3 images
path_c3 = Path("corrected/c3.jpg")
path_c4 = Path("corrected/c4.jpg")
path_c5 = Path("corrected/c5.jpg")

img_c3 = cv2.imread(str(path_c3))
img_c4 = cv2.imread(str(path_c4))
img_c5 = cv2.imread(str(path_c5))

h, w = img_c3.shape[:2]

# X,Y pixels measured in inkscape (0, 0) is ine the lower left corner.
marks_pixels_c2 = np.array(
    [
        # Main marks (low uncertainty)
        [14.5, 24.5],  # White mark on the lower left corner
        #[16.5, 1526.5],  # White mark on the mid of the left frame
        # [218, 4000], # White mark on the color checker
        [7898.5, 28.5],  # Sand grain on the right lower corner
        [2738.5, 3181.5],  # White spot in the top of the fault seal
        [6912.5, 3231.5], # White spot in th eupper ESF sand - not changing for c1-2 and c3-5.
    ]
)

# Same for C4 and C5
marks_pixels_c3 = np.array(
    [
        [15.5, 24.5],
        #[17.5, 1526.5],
        # [233, 3843.5],
        [7895.5, 29.5],
        [2737.5, 3180.5],
        [6909.5, 3231.5],
    ]
)

# Convert to reverse pixel coordinates
marks_pixels_c2[:, 1] = h - marks_pixels_c2[:, 1]
marks_pixels_c3[:, 1] = h - marks_pixels_c3[:, 1]

# Find affine map to map c3 onto c2.
transformation, mask = cv2.findHomography(
    marks_pixels_c3, marks_pixels_c2, method=cv2.RANSAC
)

# Apply translation - Change the source and return it
aligned_img_c3 = cv2.warpPerspective(
    skimage.img_as_float(img_c3).astype(np.float32),
    transformation.astype(np.float32),
    (w, h),
)
aligned_img_c4 = cv2.warpPerspective(
    skimage.img_as_float(img_c4).astype(np.float32),
    transformation.astype(np.float32),
    (w, h),
)
aligned_img_c5 = cv2.warpPerspective(
    skimage.img_as_float(img_c5).astype(np.float32),
    transformation.astype(np.float32),
    (w, h),
)

if True:
    plt.figure("c3")
    plt.imshow(img_c3)
    plt.figure("aligned c3")
    plt.imshow(aligned_img_c3)
    plt.show()

# Print C3-5 to file.
cv2.imwrite(
    "aligned/c3.jpg",
    skimage.img_as_ubyte(aligned_img_c3),
    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
)
cv2.imwrite(
    "aligned/c4.jpg",
    skimage.img_as_ubyte(aligned_img_c4),
    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
)
cv2.imwrite(
    "aligned/c5.jpg",
    skimage.img_as_ubyte(aligned_img_c5),
    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
)
