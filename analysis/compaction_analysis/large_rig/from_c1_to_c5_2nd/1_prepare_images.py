"""
1. Step of compaction analysis for the FluidFlower.
"""

from pathlib import Path

import cv2
import numpy as np

# ! ----- Preliminaries - prepare two images for compaction analysis

# Paths to two images of interest.
path_c1 = Path("corrected/c1.jpg")
path_c2 = Path("corrected/c2.jpg")
path_c3 = Path("aligned/c3.jpg")
path_c4 = Path("aligned/c4.jpg")
path_c5 = Path("aligned/c5.jpg")

# Stare images unmodifed as npy arrays
img_c1 = cv2.imread(str(path_c1))
img_c1 = cv2.cvtColor(img_c1, cv2.COLOR_BGR2RGB)
np.save("images/c1.npy", img_c1)

img_c2 = cv2.imread(str(path_c2))
img_c2 = cv2.cvtColor(img_c2, cv2.COLOR_BGR2RGB)
np.save("images/c2.npy", img_c2)

img_c3 = cv2.imread(str(path_c3))
img_c3 = cv2.cvtColor(img_c3, cv2.COLOR_BGR2RGB)
np.save("images/c3.npy", img_c3)

img_c4 = cv2.imread(str(path_c4))
img_c4 = cv2.cvtColor(img_c4, cv2.COLOR_BGR2RGB)
np.save("images/c4.npy", img_c4)

img_c5 = cv2.imread(str(path_c5))
img_c5 = cv2.cvtColor(img_c5, cv2.COLOR_BGR2RGB)
np.save("images/c5.npy", img_c5)
