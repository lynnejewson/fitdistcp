import numpy as np
from scipy.stats import genextreme
import pprint

import sys
import os
sys.path.append(os.getcwd() + '\\lib')

import P150a_gev_p1 as cp_gev_p1_a


x = []
#for i in range(len(t)):
#    x.append(genextreme.rvs(0, loc=t[i], scale=1, size=1))

# random x, with mu = t
t_example_1 = np.asarray(range(1,21))
x_example_1 = np.asarray([
    0.6379859, 2.748319, 3.080646, 4.916562, 6.895842, 7.115842, 7.170984,
    9.616542, 9.030163, 11.82559, 12.41088, 11.09098, 11.66944, 20.01093,
    16.08299, 16.87746, 17.77801, 18.7829, 19.48563, 19.74457
])

# random x, with sigma = t
t_example_2 = np.asarray(range(1,21))
x_example_2 = np.asarray([
    -1.120019, -0.915989, 0.1351531, -1.914159, 6.44977, 6.898485, 5.42345, 
    1.739186, 11.59367, 15.33829, -11.75986, 8.109923, 11.75057, -2.238912, 
    69.63034, 58.64861, 11.7273, 34.56531, 27.89557, 67.41278
])
# xi = 0.8*t
t_example_3 = np.asarray(range(-10,11))
x_example_3 = np.asarray([
  0.8002029, 1.228675, 1.327998, -0.07534215, -1.324346, -1.591088, 1.804223,
  -2.133939, -0.3588612, -0.3921269, 0.1678799, -1.073899, 0.3461773, 0.2784326, 
  4.012348, -0.8457418, 0.3015551, 3.126318, 1.076847, 9.415596, -0.7749084
])
# sigma again but different range
t_example_4 = np.asarray(range(-10,11))
x_example_4 = np.asarray([
  -11.1457, -9.562076, -8.566162, -7.086808, -5.719204, -3.626823,-2.513936, 
  -3.008662, -0.7425273, 1.824077, -0.839471, 1.408181, 1.230387, 1.298774, 
  5.491945, 4.08, 4.518696, 8.479797, 10.93809, 9.022373, 9.6516
])

x = x_example_3
t = t_example_3
t0 = max(t)

q = cp_gev_p1_a.qgev_p1_cp(x, t, t0=t0)
pprint.pprint(q['ml_quantiles'])