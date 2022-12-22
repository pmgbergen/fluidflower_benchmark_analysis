"""
Script printing the final results to file.
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage
import cv2

# Store via matplotlib.
extent = [0, 2.8, 0, 1.5]

displacement_x = np.load("results/displacement_x.npy")
plt.figure("Displacement in x-direction")
plt.title("Displacement in x-direction")
plt.imshow(displacement_x, extent=extent)
plt.xlabel("Width in m")
plt.ylabel("Height in m")
plt.xticks([0, 0.7, 1.4, 2.1, 2.8])
plt.yticks([0, 0.75, 1.5])
plt.colorbar(fraction = 0.025)
plt.tight_layout(pad=2)
plt.savefig("results/output/displacement_x.png", dpi=1000)

displacement_y = np.load("results/displacement_y.npy")
plt.figure("Displacement in y-direction")
plt.title("Displacement in y-direction")
plt.imshow(displacement_y, extent=extent)
plt.xlabel("Width in m")
plt.ylabel("Height in m")
plt.xticks([0, 0.7, 1.4, 2.1, 2.8])
plt.yticks([0, 0.75, 1.5])
plt.colorbar(fraction = 0.025)
plt.tight_layout(pad=2)
plt.savefig("results/output/displacement_y.png", dpi=1000)

labels = np.load("results/labels_ref.npy")
plt.figure("Undeformed regions")
plt.imshow(labels, extent = extent)
plt.xlabel("Width in m")
plt.ylabel("Height in m")
plt.xticks([0, 0.7, 1.4, 2.1, 2.8])
plt.yticks([0, 0.75, 1.5])
plt.savefig("results/output/undeformed_layers.png", dpi=1000)

strain_y = np.load("results/layers_strain_y.npy")
plt.figure("Mean normal strain in y-direction")
plt.title("Mean normal strain in y-direction")
plt.imshow(strain_y, extent=extent)
plt.xlabel("Width in m")
plt.ylabel("Height in m")
plt.xticks([0, 0.7, 1.4, 2.1, 2.8])
plt.yticks([0, 0.75, 1.5])
plt.colorbar(fraction = 0.025)
plt.tight_layout(pad=2)
plt.savefig("results/output/strain/normal_strain_y.png", dpi=1000)

volume_reduction = np.load("results/layers_relative_volume_reduction.npy")
plt.figure("Relative reduction in volume")
plt.title("Relative reduction in volume")
plt.imshow(volume_reduction, extent=extent)
plt.xlabel("Width in m")
plt.ylabel("Height in m")
plt.xticks([0, 0.7, 1.4, 2.1, 2.8])
plt.yticks([0, 0.75, 1.5])
plt.colorbar(fraction = 0.025)
plt.tight_layout(pad=2)
plt.savefig("results/output/volume/relative_volume_reduction.png", dpi=1000)

layer_y_displacement = np.load("results/layers_y_mean_displacement.npy")
plt.figure("Mean displacement in y-direction")
plt.title("Mean displacement in y-direction")
plt.imshow(layer_y_displacement, extent=extent)
plt.xlabel("Width in m")
plt.ylabel("Height in m")
plt.xticks([0, 0.7, 1.4, 2.1, 2.8])
plt.yticks([0, 0.75, 1.5])
plt.colorbar(fraction = 0.025)
plt.tight_layout(pad=2)
plt.savefig("results/output/displacement/mean_y_displacement.png", dpi=1000)



