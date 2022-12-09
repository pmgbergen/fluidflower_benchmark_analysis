"""
Quantify the deformation.
"""

import numpy as np
import skimage
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import darsia
from mpl_toolkits.axes_grid1 import make_axes_locatable
from benchmark.rigs.largefluidflower import LargeFluidFlower

# Load segmentation -  TODO
## Load labels
#path_zero = Path("blackened/zero.png")
#fluidflower_zero = LargeFluidFlower(path_zero, "./config_compaction_zero.json", True)
#labels_zero = fluidflower_zero.labels.copy()
#
#plt.imshow(labels_zero)
#plt.show()
#
#Ny, Nx = labels_zero.shape[:2]

Ny = 4260
Nx = 7951 

# Load deformation
deformation = np.load("results/deformation_refined.npy")

deformation_x = np.reshape(deformation[:,0], (Ny, Nx))
deformation_y = np.reshape(deformation[:,1], (Ny, Nx))

# Full divergence - note the changein orientation in y-direction
full_div = np.gradient(deformation_x, axis=1) - np.gradient(deformation_y, axis=0)
y_div = -np.gradient(deformation_y, axis=0)

plt.figure("x-displacement")
plt.imshow(deformation_x)
plt.colorbar()

plt.figure("y-displacement")
plt.imshow(deformation_y)
plt.colorbar()

plt.figure("divergence")
plt.imshow(full_div)
plt.colorbar()

plt.figure("y-divergence")
plt.imshow(y_div)
plt.colorbar()

plt.show()
