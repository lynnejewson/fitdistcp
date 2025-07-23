import numpy as np
import matplotlib.pyplot as plt
import pprint
from scipy.stats import genextreme
from scipy.optimize import minimize
import json
import warnings

import sys
import os
sys.path.append(os.getcwd() + '\\lib')

import P110a_gev_p1 as cp_gev_p1_a
import P110b_gev_p1_libs as cp_gev_p1_b

warnings.filterwarnings("ignore", category=RuntimeWarning)



def reliability_test(desired_p, ntrials, nx):
    #returns dictionary with keys actual_p_ml, actual_p_cp (lists corresponding to entries in desired_p)
    xi = 0
    mu = 0
    sigma = 1

    p_actual_ml_total = np.zeros(len(desired_p))
    p_actual_cp_total = np.zeros(len(desired_p))

    for i in range(ntrials):
        x = genextreme.rvs(-xi, loc=mu, scale=sigma, size=nx)
        
        # calculate quantile using ML and CP methods
        # ML:
        ics = [mu, sigma, xi]
        mu_ml, sigma_ml, xi_ml = minimize(lambda params: -cp_gev_p1_b.gev_loglik(params, x), ics, method='BFGS').x
        q_ml = genextreme.ppf(desired_p, -xi_ml, loc=mu_ml, scale=sigma_ml) 

        # CP:
        q_cp = cp_gev_p1_a.qgev_cp(x, desired_p)['cp_quantiles']   

        # feed back in for the actual probability
        p_actual_ml_total += genextreme.cdf(q_ml, -xi, loc=mu, scale=sigma)
        p_actual_cp_total += genextreme.cdf(q_cp, -xi, loc=mu, scale=sigma)

    p_actual_ml_avg = p_actual_ml_total / ntrials
    p_actual_cp_avg = p_actual_cp_total / ntrials

    return {'actual_p_ml' : p_actual_ml_avg, 'actual_p_cp': p_actual_cp_avg}


ntrials = 100
nx = 40
p = 0.0001 *np.asarray( range(1, 10000))
result = reliability_test(p, ntrials, nx)
result['actual_p_ml'] = np.ndarray.tolist(result['actual_p_ml'])
result['actual_p_cp'] = np.ndarray.tolist(result['actual_p_cp'])
result['p'] = np.ndarray.tolist(p)
with open('reltest 4.1 nx=40, ntrials=1000, p=[0.0001, ..., 0.9999] step=0.0001.txt', 'w') as f:
    json.dump(result, f)


#pprint.pprint(q, width=160)