import numpy as np
import matplotlib.pyplot as plt
#import daria

img = np.load("tst.npy")
co2 = np.load("co2.npy")
esf = np.load("esf.npy")

max_val = np.max(img)
img = max_val - img
print(max_val)

img[~co2] = 0
img[esf] = 0

mask = img > 1.5

plt.figure()
plt.imshow(img)
plt.figure()
plt.imshow(mask)
#plt.figure()
#plt.imshow(co2)
#plt.figure()
#plt.imshow(esf)
plt.show()
