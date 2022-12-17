"""
Collect all results for C1-C5 and store in a single file.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

number_fingers_c1 = np.genfromtxt("../../large_rig/c1/results/number_fingers.csv", delimiter=",")
number_fingers_c2 = np.genfromtxt("../../large_rig/c2/results/number_fingers.csv", delimiter=",")
number_fingers_c3 = np.genfromtxt("../../large_rig/c3/results/number_fingers.csv", delimiter=",")
number_fingers_c4 = np.genfromtxt("../../large_rig/c4/results/number_fingers.csv", delimiter=",")
number_fingers_c5 = np.genfromtxt("../../large_rig/c5/results/number_fingers.csv", delimiter=",")

print(number_fingers_c1.shape)
print(number_fingers_c2.shape)
print(number_fingers_c3.shape)
print(number_fingers_c4.shape)
print(number_fingers_c5.shape)

# Check relative times
#print((number_fingers_c1[:,0] - number_fingers_c2[:,0]) / number_fingers_c1[:,0])
#print((number_fingers_c1[:,0] - number_fingers_c3[:,0]) / number_fingers_c1[:,0])
#print((number_fingers_c1[:,0] - number_fingers_c4[:,0]) / number_fingers_c1[:,0])
#print((number_fingers_c1[72:,0] - number_fingers_c5[68:,0]) / number_fingers_c1[72:,0])

plt.plot(number_fingers_c1[:,1])
plt.show()
# Remove indices 68, 69, 70, 71 from c1, c2, c3, c4 as they are missing in c5.
mask = np.ones(len(number_fingers_c1), dtype=bool)
mask[68:72] = 0
number_fingers_c1 = number_fingers_c1[mask, :]
number_fingers_c2 = number_fingers_c2[mask, :]
number_fingers_c3 = number_fingers_c3[mask, :]
number_fingers_c4 = number_fingers_c4[mask, :]
plt.plot(number_fingers_c1[:,1])
plt.show()

#print((number_fingers_c1[:,0] - number_fingers_c2[:,0]) / number_fingers_c1[:,0])
#print((number_fingers_c1[:,0] - number_fingers_c3[:,0]) / number_fingers_c1[:,0])
#print((number_fingers_c1[:,0] - number_fingers_c4[:,0]) / number_fingers_c1[:,0])
#print((number_fingers_c1[:,0] - number_fingers_c5[:,0]) / number_fingers_c1[:,0])

# Combine results in a single csv file
number_fingers = np.zeros((len(number_fingers_c1), 6), dtype=float)
# Time
number_fingers[:,0] = number_fingers_c1[:,0]
# C1-C5
number_fingers[:,1] = number_fingers_c1[:,1]
number_fingers[:,2] = number_fingers_c2[:,1]
number_fingers[:,3] = number_fingers_c3[:,1]
number_fingers[:,4] = number_fingers_c4[:,1]
number_fingers[:,5] = number_fingers_c5[:,1]

# Store to file
finger_header = f"Time in hours, number of finger tips in box C for C1, C2, C3, C4, C5"
fmt = "%f", "%d", "%d", "%d", "%d", "%d"
np.savetxt("number_fingers.csv", number_fingers, fmt=fmt, delimiter=",", header=finger_header)

plt.figure("number of fingers in C - 5 days")
plt.plot(number_fingers[:,0], number_fingers[:,1], label="c1")
plt.plot(number_fingers[:,0], number_fingers[:,2], label="c2")
plt.plot(number_fingers[:,0], number_fingers[:,3], label="c3")
plt.plot(number_fingers[:,0], number_fingers[:,4], label="c4")
plt.plot(number_fingers[:,0], number_fingers[:,5], label="c5")
plt.legend()

one_day = np.argmin(np.absolute(number_fingers[:,0] - 24))
print(one_day)
plt.figure("number of fingers in C - 1 day")
plt.plot(number_fingers[:one_day,0], number_fingers[:one_day,1], label="c1")
plt.plot(number_fingers[:one_day,0], number_fingers[:one_day,2], label="c2")
plt.plot(number_fingers[:one_day,0], number_fingers[:one_day,3], label="c3")
plt.plot(number_fingers[:one_day,0], number_fingers[:one_day,4], label="c4")
plt.plot(number_fingers[:one_day,0], number_fingers[:one_day,5], label="c5")
plt.legend()

ten_hours = np.argmin(np.absolute(number_fingers[:,0] - 10))
print(ten_hours)
plt.figure("number of fingers in C - 10 hours")
plt.plot(number_fingers[:ten_hours,0], number_fingers[:ten_hours,1], label="c1")
plt.plot(number_fingers[:ten_hours,0], number_fingers[:ten_hours,2], label="c2")
plt.plot(number_fingers[:ten_hours,0], number_fingers[:ten_hours,3], label="c3")
plt.plot(number_fingers[:ten_hours,0], number_fingers[:ten_hours,4], label="c4")
plt.plot(number_fingers[:ten_hours,0], number_fingers[:ten_hours,5], label="c5")
plt.legend()

four_hours = np.argmin(np.absolute(number_fingers[:,0] - 4))
seventeen_hours = np.argmin(np.absolute(number_fingers[:,0] - 17))
print(ten_hours)
plt.figure("number of fingers in C - 4-17 hours")
plt.plot(number_fingers[four_hours:seventeen_hours,0], number_fingers[four_hours:seventeen_hours,1], label="c1")
plt.plot(number_fingers[four_hours:seventeen_hours,0], number_fingers[four_hours:seventeen_hours,2], label="c2")
plt.plot(number_fingers[four_hours:seventeen_hours,0], number_fingers[four_hours:seventeen_hours,3], label="c3")
plt.plot(number_fingers[four_hours:seventeen_hours,0], number_fingers[four_hours:seventeen_hours,4], label="c4")
plt.plot(number_fingers[four_hours:seventeen_hours,0], number_fingers[four_hours:seventeen_hours,5], label="c5")
plt.legend()

plt.show()
