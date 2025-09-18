import numpy as np
x = np.array([1, 2, 3])
y = np.array([30, 40, 50, 60, 70, 80, 90])
d = x.T[:, None] - y
print(d.shape)