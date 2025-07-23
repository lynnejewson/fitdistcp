import P150a_gev_p1 as cp_gev_p1_a
import numpy as np
from scipy.stats import genextreme
import pprint

t = np.asarray(range(0,21))
t0 = 21
x = []
for i in range(len(t)):
    x.append(genextreme.rvs(0, loc=t[i], scale=1, size=1))
q = cp_gev_p1_a.qgev_p1_cp(np.asarray(x), t, t0=t0)
pprint.pprint(q)