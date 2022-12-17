import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor

y = np.array([0.9, 0.8, 0.75, 0.6, 1.4])

y = y[3:]

x = np.arange(y.shape[0])[:, np.newaxis]

new_id = y.shape[0]

ransac = RANSACRegressor() 
ransac.fit(x, y)
 
line_X = np.array([new_id])
line_y_ransac = ransac.predict(line_X[:, np.newaxis])

print(line_y_ransac)
