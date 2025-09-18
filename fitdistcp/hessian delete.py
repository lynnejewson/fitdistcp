import numpy as np
from scipy.differentiate import hessian
from scipy.optimize import rosen, rosen_hess

m = 3
x = np.full(m, 0.5)
res = hessian(rosen, x)
print(res)
print(rosen_hess)