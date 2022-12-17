"""
Extract swi from segmentation.
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

swi_default = 0.1
swi_water = 0
swi_esf = 0.32
swi_c = 0.14
swi_d = 0.12
swi_e = 0.12
swi_f = 0.12
swi_g = 0.10

swi = swi_default * np.ones((4260, 7951), dtype=float)
swi[segmentation == water] = swi_water
swi[segmentation == esf] = swi_esf
swi[segmentation == c] = swi_c
swi[segmentation == d] = swi_d
swi[segmentation == e] = swi_e
swi[segmentation == f] = swi_f
swi[segmentation == g] = swi_g

np.save("swi.npy", swi)

if True:
    plt.imshow(swi)
    plt.show()
