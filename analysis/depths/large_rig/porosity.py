"""
Extract porosity from segmentation.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

segmentation = cv2.cvtColor(cv2.imread("segmentation.png"), cv2.COLOR_BGR2GRAY)
segmentation = cv2.resize(segmentation, (7951, 4260), interpolation = cv2.INTER_NEAREST)

water = 0
esf = 40
c = 80  
d = 120
e = 160
f = 200
g= 240

porosity_default = 0.44
porosity_water = 0
porosity_esf = 0.435
porosity_c = 0.435
porosity_d = 0.44
porosity_e = 0.45
porosity_f = 0.44
porosity_g = 0.45

porosity = porosity_default * np.ones((4260, 7951), dtype=float)
porosity[segmentation == water] = porosity_water
porosity[segmentation == esf] = porosity_esf
porosity[segmentation == c] = porosity_c
porosity[segmentation == d] = porosity_d
porosity[segmentation == e] = porosity_e
porosity[segmentation == f] = porosity_f
porosity[segmentation == g] = porosity_g

np.save("porosity.npy", porosity)

if True:
    plt.imshow(porosity)
    plt.show()
