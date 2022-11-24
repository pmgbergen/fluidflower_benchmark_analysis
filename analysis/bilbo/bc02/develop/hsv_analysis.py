import numpy as np
import matplotlib.pyplot as plt
import daria
import cv2

img = np.load("hsv_diff.npy")
img_rgb = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

plt.figure()
plt.imshow(img_rgb)

#plt.figure()
#plt.imshow(img[:,:,0])
#plt.figure()
#plt.imshow(img[:,:,1])
#plt.figure()
#plt.imshow(img[:,:,2])
#plt.show()

h_img = img[:,:,0]

daria.utils.coloranalysis.hsv_spectrum(img_rgb, [(slice(3380, 3480), slice(4700, 5200)), (slice(3675, 3850), slice(6050, 6400))])

plt.figure()
plt.imshow(h_img > 70)
plt.show()
