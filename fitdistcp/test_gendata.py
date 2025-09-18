import numpy as np
from pprint import pprint
from scipy.stats import norm, lognorm, gumbel_r, gamma, weibull_min, expon, genextreme, genpareto

t1 = range(1, 21)
t2 = range(11, 31)
x = []
for i in range(len(t1)):
    loc = -1 + 0.01*t1[i]
    scale = 0.01 * np.exp(0.2*t2[i])
    c = 0.2
    #x.append(genextreme.rvs(c, loc=loc, scale=scale))



#x = norm.rvs(loc=30, scale=0.001, size=5)
#x = lognorm.rvs(shape, loc=1, scale=2, size=20)
#x = gumbel_r.rvs(loc=10, scale=2, size=100)
#x = [gamma.rvs(a=0.1, scale=0.5, size=20),gamma.rvs(a=0.9, scale=12, size=15),gamma.rvs(a=1, scale=6, size=25),gamma.rvs(a=1.5, scale=0.8, size=20),gamma.rvs(a=3, scale=2, size=20),gamma.rvs(a=10,scale=10, size=20)]
#x = [weibull_min.rvs(0.5, scale=0.5, size=20),weibull_min.rvs(0.8, scale=5, size=20),weibull_min.rvs(1, scale=1.6, size=20),weibull_min.rvs(1.5, scale=2, size=20),weibull_min.rvs(2, scale=0.9, size=20),weibull_min.rvs(5, scale=2, size=20),]
#x = [expon.rvs(scale=0.01, size=20), expon.rvs(scale=0.1, size=20), expon.rvs(scale=1, size=20), expon.rvs(scale=10, size=20)]
#x = genextreme.rvs(c=0.4, loc=2, scale=2, size=20)
x = [genpareto.rvs(c=0.1, scale=1, size=50),genpareto.rvs(c=0.5, scale=2, size=25),genpareto.rvs(c=0.9, scale=0.5, size=50),genpareto.rvs(c=1.1, scale=0.9, size=25),genpareto.rvs(c=1.5, scale=1.3, size=25),genpareto.rvs(c=2, scale=6, size=25),genpareto.rvs(c=5, scale=6, size=25)]

x = [
    genextreme.rvs(c=0, loc = 1, scale=0.5, size=30),
    genextreme.rvs(c=0.1, loc = 2, scale=0.9, size=30),
    genextreme.rvs(c=-0.1, loc = 3, scale=1, size=30),
    genextreme.rvs(c=0.2, loc = 4, scale=1.2, size=30),
    genextreme.rvs(c=-0.2, loc = 5, scale=2, size=30),
    genextreme.rvs(c=0.3, loc = 6, scale=5, size=30),
    genextreme.rvs(c=-0.3, loc = 7, scale=10, size=30),
    genextreme.rvs(c=0.4, loc = 8, scale=1.1, size=30),
    genextreme.rvs(c=-0.4, loc = 9, scale=1.3, size=30),
    genextreme.rvs(c=0.5, loc = 10, scale=0.8, size=30),
    genextreme.rvs(c=-0.5, loc = 10, scale=0.8, size=30),
    genextreme.rvs(c=0.6, loc = 10, scale=0.8, size=30),
    genextreme.rvs(c=-0.6, loc = 10, scale=0.8, size=30)
]

pprint(x)